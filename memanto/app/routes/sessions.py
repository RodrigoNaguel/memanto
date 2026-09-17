"""
Session and Agent Lifecycle Routes

New session-based architecture endpoints.
Replaces tenant_id with Moorcheh API key-based authentication.
"""

import asyncio

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response

from memanto.app.clients import moorcheh as moorcheh_clients
from memanto.app.config import settings
from memanto.app.models.session import (
    AgentCreate,
    AgentInfo,
    AgentList,
    Session,
    SessionInfo,
    SessionSummary,
)
from memanto.app.services.agent_service import AgentService
from memanto.app.utils.errors import (
    AgentAlreadyExistsError,
    AgentNotFoundError,
    AuthorizationError,
    SessionNotFoundError,
    map_error_to_http_exception,
)

router = APIRouter()

from memanto.app.routes import memory  # noqa: E402
from memanto.app.routes.auth_deps import (  # noqa: E402
    clear_session_cookie,
    get_current_session,
    get_session_service,
    set_session_cookie,
    verify_moorcheh_api_key,
)

router.include_router(memory.router, prefix="/agents", tags=["Memory Operations"])
agent_service = AgentService()


def get_agent_service():
    """Get agent service instance"""
    return agent_service


async def _namespace_item_counts(moorcheh_api_key: str) -> dict[str, int]:
    """Map namespace_name -> live document count from Moorcheh."""
    try:
        client = moorcheh_clients.get_moorcheh_client()
        ns_resp = await asyncio.to_thread(client.namespaces.list)
        namespaces = ns_resp.get("namespaces", []) if isinstance(ns_resp, dict) else []
        if not isinstance(namespaces, list):
            return {}
        counts: dict[str, int] = {}
        for ns in namespaces:
            if not isinstance(ns, dict):
                continue
            namespace_name = ns.get("namespace_name")
            if not isinstance(namespace_name, str) or not namespace_name:
                continue
            raw_count = ns.get("item_count", 0)
            try:
                counts[namespace_name] = int(raw_count)
            except (TypeError, ValueError):
                counts[namespace_name] = 0
        return counts
    except Exception:
        return {}


@router.post("/agents", response_model=AgentInfo, status_code=201)
async def create_agent(
    agent_create: AgentCreate, moorcheh_api_key: str = Depends(verify_moorcheh_api_key)
):
    """Create a new MEMANTO agent."""
    try:
        return agent_service.create_agent(agent_create, moorcheh_api_key)
    except AgentAlreadyExistsError as e:
        raise map_error_to_http_exception(e)


@router.get("/agents", response_model=AgentList)
async def list_agents(moorcheh_api_key: str = Depends(verify_moorcheh_api_key)):
    """List all agents for this Moorcheh account."""
    agent_list = agent_service.list_agents()
    counts = await _namespace_item_counts(moorcheh_api_key)
    for agent in agent_list.agents:
        if agent.namespace in counts:
            agent.memory_count = counts[agent.namespace]
    return agent_list


@router.get("/agents/{agent_id}", response_model=AgentInfo)
async def get_agent(
    agent_id: str, moorcheh_api_key: str = Depends(verify_moorcheh_api_key)
):
    """Get agent information."""
    agent = agent_service.get_agent(agent_id)
    if not agent:
        raise map_error_to_http_exception(
            AgentNotFoundError(f"Agent '{agent_id}' not found")
        )
    counts = await _namespace_item_counts(moorcheh_api_key)
    if agent.namespace in counts:
        agent.memory_count = counts[agent.namespace]
    return agent


@router.delete("/agents/{agent_id}", status_code=200)
async def delete_agent(
    agent_id: str,
    delete_backup_too: bool = Query(
        False, alias="delete-backup-too", description="Delete Moorcheh namespace backup"
    ),
    moorcheh_api_key: str = Depends(verify_moorcheh_api_key),
):
    """Delete an agent and, when requested, its remote memory namespace.

    Security invariant: when ``delete-backup-too`` is requested we must not
    destroy the local metadata unless deletion of the remote namespace has
    succeeded.  Otherwise a transient backend/auth failure can orphan sensitive
    memories while returning a false "all namespace memories" success message.
    Keeping the local record makes the operation retryable and prevents a false
    deletion guarantee.
    """
    try:
        agent = agent_service.get_agent(agent_id)
        if not agent:
            raise map_error_to_http_exception(
                AgentNotFoundError(f"Agent '{agent_id}' not found")
            )

        if delete_backup_too:
            moorcheh_client = moorcheh_clients.get_moorcheh_client()
            try:
                moorcheh_client.namespaces.delete(namespace_name=agent.namespace)
            except Exception as exc:
                raise HTTPException(
                    status_code=502,
                    detail=(
                        "Remote namespace deletion failed; local agent metadata "
                        "was preserved so deletion can be retried safely."
                    ),
                ) from exc

        get_session_service().delete_session(agent_id)
        agent_service.delete_agent(agent_id)
        return {
            "message": (
                f"Agent '{agent_id}' successfully deleted"
                + (
                    " with all namespace memories"
                    if delete_backup_too
                    else " (backup retained in Moorcheh)"
                )
            )
        }
    except AgentNotFoundError as e:
        raise map_error_to_http_exception(e)


@router.post("/agents/{agent_id}/activate", response_model=Session)
async def activate_agent(
    agent_id: str,
    request: Request,
    response: Response,
    moorcheh_api_key: str = Depends(verify_moorcheh_api_key),
):
    """Activate agent and start session."""
    agent = agent_service.get_agent(agent_id)
    if not agent:
        raise map_error_to_http_exception(
            AgentNotFoundError(f"Agent '{agent_id}' not found")
        )
    duration_hours = settings.SESSION_DEFAULT_DURATION_HOURS
    try:
        session = get_session_service().create_session(
            agent_id=agent_id,
            pattern=agent.pattern,
            duration_hours=duration_hours,
        )
        set_session_cookie(response, session.session_token, request)
        agent_service.update_agent_stats(
            agent_id=agent_id,
            last_session=session.started_at,
            increment_session_count=True,
        )
        return session
    except Exception as e:
        raise map_error_to_http_exception(e)


@router.post("/agents/{agent_id}/deactivate", response_model=SessionSummary)
async def deactivate_agent(
    agent_id: str,
    response: Response,
    session: Session = Depends(get_current_session),
    _server_api_key: str = Depends(verify_moorcheh_api_key),
):
    """Deactivate agent and end session."""
    if session.agent_id != agent_id:
        raise map_error_to_http_exception(
            AuthorizationError(
                f"Session is for agent '{session.agent_id}', cannot access '{agent_id}'"
            )
        )
    try:
        summary = get_session_service().end_session(agent_id)
        clear_session_cookie(response)
        return summary
    except SessionNotFoundError as e:
        raise map_error_to_http_exception(e)


@router.get("/status", response_model=SessionInfo)
async def get_status(
    _moorcheh_api_key: str = Depends(verify_moorcheh_api_key),
):
    """Get current active session status."""
    session = get_session_service().get_active_session()
    if session is None:
        raise HTTPException(status_code=404, detail="No active session")
    time_remaining = session.time_remaining()
    return SessionInfo(
        session_id=session.session_id,
        agent_id=session.agent_id,
        namespace=session.namespace,
        started_at=session.started_at,
        expires_at=session.expires_at,
        status=session.status,
        time_remaining_seconds=max(0, int(time_remaining.total_seconds())),
        pattern=session.pattern,
    )
