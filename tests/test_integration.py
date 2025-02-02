import pytest

from cloud_inspector.components.types import CloudProvider, WorkflowStatus
from cloud_inspector.orchestration.agent import Agent


def test_full_workflow(model_registry, prompt_generator, code_generator, code_executor):
    """Test complete workflow execution."""
    agent = Agent(model_name="test-model", prompt_generator=prompt_generator, code_generator=code_generator, code_executor=code_executor)

    result = agent.execute(request="List EC2 instances", cloud=CloudProvider.AWS, service="ec2", thread_id="test-123", params={"region": "us-west-2"})

    assert result["status"] in (WorkflowStatus.COMPLETED, WorkflowStatus.FAILED)
    assert "discoveries" in result
    assert "execution_metrics" in result


@pytest.mark.integration
def test_workflow_with_iterations(model_registry, prompt_generator, code_generator, code_executor):
    """Test workflow with multiple iterations."""
    agent = Agent(model_name="test-model", prompt_generator=prompt_generator, code_generator=code_generator, code_executor=code_executor)

    result = agent.execute(request="Analyze EC2 security", cloud=CloudProvider.AWS, service="ec2", thread_id="test-456")

    assert result["iteration"] > 0
    assert len(result["discoveries"]) > 0
