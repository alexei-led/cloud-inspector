"""State management for orchestration workflow."""

from datetime import datetime
from typing import Annotated, Any, Optional
from typing_extensions import TypedDict

from langgraph.graph.message import add_messages
from pydantic import BaseModel

from components.types import CloudProvider


class OrchestrationState(TypedDict):
    """State for the orchestration workflow."""
    messages: Annotated[list, add_messages]  # Chat history
    context: dict  # Stores collected data and iteration history 
    current_iteration: int
    user_request: str
    cloud: CloudProvider
    service: str
    collected_data: list[dict]
    agent_outputs: dict
    status: str  # in_progress, completed, failed
    created_at: datetime
    updated_at: datetime
    completion_reason: Optional[str]
    variables: dict[str, Any]


def create_initial_state(
    request: str,
    cloud: CloudProvider,
    service: str,
    variables: Optional[dict[str, Any]] = None,
) -> OrchestrationState:
    """Create initial state for new orchestration."""
    now = datetime.now()
    return {
        "messages": [],
        "context": {},
        "current_iteration": 0,
        "user_request": request,
        "cloud": cloud,
        "service": service,
        "collected_data": [],
        "agent_outputs": {},
        "status": "in_progress",
        "created_at": now,
        "updated_at": now,
        "completion_reason": None,
        "variables": variables or {},
    }


def add_to_context(key: str, value: Any) -> dict:
    """Add data to context."""
    current = {}
    current[key] = value
    return current
