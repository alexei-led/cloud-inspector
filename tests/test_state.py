from datetime import datetime
from cloud_inspector.orchestration.state import create_initial_state
from cloud_inspector.components.types import CloudProvider, WorkflowStatus

def test_create_initial_state():
    """Test initial state creation."""
    state = create_initial_state(
        request="List EC2 instances",
        cloud=CloudProvider.AWS,
        service="ec2"
    )
    
    assert state["iteration"] == 0
    assert state["status"] == WorkflowStatus.IN_PROGRESS
    assert isinstance(state["created_at"], datetime)
    assert len(state["discoveries"]) == 0

def test_create_initial_state_with_params():
    """Test initial state creation with parameters."""
    params = {"region": "us-west-2"}
    state = create_initial_state(
        request="List EC2 instances",
        cloud=CloudProvider.AWS,
        service="ec2",
        params=params
    )
    
    assert state["params"] == params
