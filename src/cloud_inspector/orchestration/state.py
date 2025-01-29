"""State management for cloud inspection workflow."""

from datetime import datetime
from typing import Annotated, Any, Optional

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from components.types import CloudProvider, WorkflowStatus


class OrchestrationState(TypedDict):
    """State for the cloud inspection workflow.

    Attributes:
        messages: Chat history with add_messages reducer
        iteration: Current iteration number
        request: Original user request
        cloud: Cloud provider being inspected
        service: Service being inspected
        discoveries: List of discovered data
        outputs: Outputs from each agent
        status: Current workflow status
        created_at: Workflow creation timestamp
        updated_at: Last update timestamp
        reason: Completion or failure reason
        params: Additional parameters and variables
    """

    messages: Annotated[list, add_messages]
    iteration: int
    request: str
    cloud: CloudProvider
    service: str
    discoveries: list[dict]
    outputs: dict[str, Any]
    status: WorkflowStatus
    created_at: datetime
    updated_at: datetime
    reason: Optional[str]
    params: dict[str, Any]


def create_initial_state(
    request: str,
    cloud: CloudProvider,
    service: str,
    params: Optional[dict[str, Any]] = None,
) -> OrchestrationState:
    """Create initial workflow state.

    Args:
        request: User's inspection request
        cloud: Target cloud provider
        service: Target cloud service
        params: Optional additional parameters

    Returns:
        New OrchestrationState instance
    """
    now = datetime.now()
    return {
        "messages": [],
        "iteration": 0,
        "request": request,
        "cloud": cloud,
        "service": service,
        "discoveries": [],
        "outputs": {},
        "status": WorkflowStatus.IN_PROGRESS,
        "created_at": now,
        "updated_at": now,
        "reason": None,
        "params": params or {},
    }
