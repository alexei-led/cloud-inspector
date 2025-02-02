import pytest

from cloud_inspector.components.types import CloudProvider, WorkflowStatus
from cloud_inspector.orchestration.agent import Agent


@pytest.mark.integration
def test_orchestration_workflow(prompt_generator, code_generator, code_executor):
    """Test complete orchestration workflow."""
    agent = Agent(model_name="test-model", prompt_generator=prompt_generator, code_generator=code_generator, code_executor=code_executor)

    result = agent.execute(request="List EC2 instances", cloud=CloudProvider.AWS, service="ec2", thread_id="test-123", params={"region": "us-west-2"})

    assert result["status"] in (WorkflowStatus.COMPLETED, WorkflowStatus.FAILED)
    assert "discoveries" in result
    assert "execution_metrics" in result
    assert isinstance(result["execution_metrics"].get("total_execution_time"), (int, float))


def test_orchestration_state_transitions(prompt_generator, code_generator, code_executor):
    """Test state transitions in orchestration workflow."""
    agent = Agent(model_name="test-model", prompt_generator=prompt_generator, code_generator=code_generator, code_executor=code_executor)

    result = agent.execute(request="List EC2 instances", cloud=CloudProvider.AWS, service="ec2", thread_id="test-456")

    assert "iteration" in result
    assert result["iteration"] <= 3  # Max iterations
    assert "retry_attempts" in result
    assert result["retry_attempts"] <= 2  # Max retries
