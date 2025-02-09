"""Test cases for the orchestration workflow."""

from unittest.mock import Mock

import pytest
from langgraph.checkpoint.memory import MemorySaver

from cloud_inspector.code_generator import CodeGeneratorAgent
from cloud_inspector.components.types import CloudProvider, WorkflowStatus
from cloud_inspector.execution_agent import CodeExecutionAgent
from cloud_inspector.orchestration.orchestration import OrchestrationAgent
from cloud_inspector.prompt_generator import PromptGeneratorAgent


class StubMemorySaver(MemorySaver):
    def load(self, key: str):
        return None

    def save(self, key: str, state: dict):
        pass


@pytest.fixture
def mock_prompt_generator():
    """Mock prompt generator that returns a test prompt template."""
    generator = Mock(spec=PromptGeneratorAgent)
    generator.generate_prompt.return_value = {"template": "test template", "variables": [{"name": "region", "value": "us-west-2"}], "success_criteria": "test criteria", "description": "test description"}
    return generator


@pytest.fixture
def mock_code_generator():
    """Mock code generator that returns test code files."""
    generator = Mock(spec=CodeGeneratorAgent)
    generator.generate_code.return_value = ({"model_name": "test-model", "iteration_id": "test-1", "run_id": "run-1", "generated_files": {"main.py": "print('test')", "requirements.txt": "boto3==1.26.0", "policy.json": "{}"}}, Mock())
    return generator


@pytest.fixture
def mock_model_registry():
    """Mock model registry that returns a configurable LLM model."""
    mock_registry = Mock()
    mock_model = Mock()
    mock_model.invoke.return_value = Mock(content="unique")  # default response
    mock_registry.get_model.return_value = mock_model
    return mock_registry


@pytest.fixture
def mock_code_executor():
    """Mock code executor that returns configurable execution results."""
    executor = Mock(spec=CodeExecutionAgent)
    result = Mock()
    result.success = True
    result.get_parsed_output.return_value = {"instances": [{"id": "i-123", "state": "running"}]}
    result.execution_time = 1.0
    result.resource_usage = {"memory_mb": 100, "cpu_percent": 50}
    result.error = None
    executor.execute_generated_code.return_value = result
    return executor


def test_workflow_initialization(mock_prompt_generator, mock_code_generator, mock_code_executor, mock_model_registry):
    """Test basic workflow initialization and node creation."""
    agent = OrchestrationAgent(model_name="test-model", prompt_generator=mock_prompt_generator, code_generator=mock_code_generator, code_executor=mock_code_executor, model_registry=mock_model_registry)

    workflow = agent._create_workflow()
    assert workflow is not None
    assert len(workflow.nodes) == 5  # orchestrate, prompt, code, execute, analyze


def test_successful_execution(mock_prompt_generator, mock_code_generator, mock_code_executor, mock_model_registry):
    """Test a successful workflow execution with redundancy detection."""
    mock_model = mock_model_registry.get_model.return_value
    mock_model.invoke.side_effect = [Mock(content="unique"), Mock(content="redundant")]

    agent = OrchestrationAgent(model_name="test-model", prompt_generator=mock_prompt_generator, code_generator=mock_code_generator, code_executor=mock_code_executor, model_registry=mock_model_registry)

    result = agent.execute(request="Test request", cloud=CloudProvider.AWS, service="ec2", thread_id="test-123")

    assert result["status"] == WorkflowStatus.COMPLETED
    assert result["reason"] == "no_new_information_found"
    assert len(result["discoveries"]) > 0
    assert "instances" in result["outputs"]
    assert result["outputs"]["instances"][0]["id"] == "i-123"


def test_workflow_with_empty_results(mock_prompt_generator, mock_code_generator, mock_code_executor, mock_model_registry):
    """Test workflow handling empty result sets appropriately."""
    empty_result = Mock()
    empty_result.success = True
    empty_result.error = None
    empty_result.get_parsed_output.return_value = {"instances": []}
    empty_result.execution_time = 0.5
    empty_result.resource_usage = {"memory_mb": 50, "cpu_percent": 20}
    mock_code_executor.execute_generated_code.return_value = empty_result

    mock_model = mock_model_registry.get_model.return_value
    mock_model.invoke.side_effect = [Mock(content="unique"), Mock(content="redundant")]

    agent = OrchestrationAgent(model_name="test-model", prompt_generator=mock_prompt_generator, code_generator=mock_code_generator, code_executor=mock_code_executor, model_registry=mock_model_registry)

    result = agent.execute(request="List EC2 instances", cloud=CloudProvider.AWS, service="ec2", thread_id="test-123")

    assert result["status"] == WorkflowStatus.COMPLETED
    assert len(result["discoveries"]) > 0
    assert "instances" in result["outputs"]
    assert len(result["outputs"]["instances"]) == 0


def test_workflow_with_multiple_cloud_providers(mock_prompt_generator, mock_code_generator, mock_code_executor, mock_model_registry):
    """Test workflow execution across different cloud providers."""

    def make_cloud_result(provider):
        result = Mock()
        result.success = True
        result.error = None
        result.get_parsed_output.return_value = {"provider": provider, "resources": [{"id": f"{provider}-123"}]}
        result.execution_time = 1.0
        result.resource_usage = {"memory_mb": 50, "cpu_percent": 20}
        return result

    mock_results = []
    for provider in ["aws", "azure", "gcp"]:
        mock_results.extend([make_cloud_result(provider), make_cloud_result(provider)])

    mock_code_executor.execute_generated_code.side_effect = mock_results
    mock_model = mock_model_registry.get_model.return_value
    mock_model.invoke.return_value = Mock(content="redundant")  # End each workflow quickly

    agent = OrchestrationAgent(model_name="test-model", prompt_generator=mock_prompt_generator, code_generator=mock_code_generator, code_executor=mock_code_executor, model_registry=mock_model_registry)

    for provider in [CloudProvider.AWS, CloudProvider.AZURE, CloudProvider.GCP]:
        result = agent.execute(request="List compute instances", cloud=provider, service="compute", thread_id=f"test-{provider}")

        assert result["status"] == WorkflowStatus.COMPLETED
        assert result["outputs"]["provider"].lower() == provider.value.lower()
        assert len(result["outputs"]["resources"]) > 0
        assert result["outputs"]["resources"][0]["id"].startswith(provider.value.lower())


def test_workflow_with_invalid_service(mock_prompt_generator, mock_code_generator, mock_code_executor, mock_model_registry):
    """Test error handling for invalid service requests."""
    error_result = Mock()
    error_result.success = False
    error_result.error = "Invalid service: unknown_service"
    error_result.get_parsed_output.return_value = None
    error_result.execution_time = 0.1
    error_result.resource_usage = {}
    mock_code_executor.execute_generated_code.return_value = error_result

    agent = OrchestrationAgent(model_name="test-model", prompt_generator=mock_prompt_generator, code_generator=mock_code_generator, code_executor=mock_code_executor, model_registry=mock_model_registry)

    result = agent.execute(request="List resources", cloud=CloudProvider.AWS, service="unknown_service", thread_id="test-123")

    assert result["status"] == WorkflowStatus.FAILED
    assert "Invalid service" in result["outputs"]["error"]


def test_execution_failure_after_max_retries(mock_prompt_generator, mock_code_generator, mock_code_executor, mock_model_registry):
    """Test workflow properly handles repeated execution failures."""
    error_result = Mock()
    error_result.success = False
    error_result.error = "Persistent error"
    error_result.get_parsed_output.return_value = None
    error_result.execution_time = 0.1
    error_result.resource_usage = {}
    mock_code_executor.execute_generated_code.return_value = error_result

    agent = OrchestrationAgent(model_name="test-model", prompt_generator=mock_prompt_generator, code_generator=mock_code_generator, code_executor=mock_code_executor, model_registry=mock_model_registry)

    result = agent.execute(request="Test request", cloud=CloudProvider.AWS, service="ec2", thread_id="test-123")

    assert result["status"] == WorkflowStatus.FAILED
    assert result["retry_attempts"] == 2  # Max retries reached
    assert result["outputs"]["error"] == "Persistent error"


def test_max_iterations_limit(mock_prompt_generator, mock_code_generator, mock_code_executor, mock_model_registry):
    """Test workflow respects the maximum iterations limit."""
    mock_results = []
    for i in range(5):  # More results than MAX_ITERATIONS
        result = Mock()
        result.success = True
        result.error = None
        result.get_parsed_output.return_value = {"data": f"iteration_{i}"}
        result.execution_time = 1.0
        result.resource_usage = {"memory_mb": 50}
        mock_results.append(result)

    mock_code_executor.execute_generated_code.side_effect = mock_results
    mock_model = mock_model_registry.get_model.return_value
    mock_model.invoke.return_value = Mock(content="unique")  # Always unique to force iteration

    agent = OrchestrationAgent(model_name="test-model", prompt_generator=mock_prompt_generator, code_generator=mock_code_generator, code_executor=mock_code_executor, model_registry=mock_model_registry)

    result = agent.execute(request="Test request", cloud=CloudProvider.AWS, service="ec2", thread_id="test-123")

    assert result["status"] == WorkflowStatus.COMPLETED
    assert result["reason"] == "max_iterations_reached"
    assert result["iteration"] == 3  # MAX_ITERATIONS constant from nodes.py


def test_workflow_with_region_param(mock_prompt_generator, mock_code_generator, mock_code_executor, mock_model_registry):
    """Test workflow properly handles regional parameters."""
    mock_code_executor.execute_generated_code.return_value = Mock(success=True, error=None, get_parsed_output=lambda: {"instances": [{"id": "i-123", "region": "us-west-2"}]}, execution_time=1.0, resource_usage={"memory_mb": 100})

    mock_model = mock_model_registry.get_model.return_value
    mock_model.invoke.return_value = Mock(content="redundant")  # End workflow quickly

    agent = OrchestrationAgent(model_name="test-model", prompt_generator=mock_prompt_generator, code_generator=mock_code_generator, code_executor=mock_code_executor, model_registry=mock_model_registry)

    result = agent.execute(request="List EC2 instances", cloud=CloudProvider.AWS, service="ec2", thread_id="test-123", params={"region": "us-west-2"})

    assert result["status"] == WorkflowStatus.COMPLETED
    assert result["outputs"]["instances"][0]["region"] == "us-west-2"


def test_workflow_error_handling(mock_prompt_generator, mock_code_generator, mock_code_executor, mock_model_registry):
    """Test workflow handles execution errors appropriately."""
    mock_code_executor.execute_generated_code.return_value = Mock(success=False, error="InvalidParameterError", get_parsed_output=lambda: {"error": "Invalid region specified"}, execution_time=0.1, resource_usage={})

    agent = OrchestrationAgent(model_name="test-model", prompt_generator=mock_prompt_generator, code_generator=mock_code_generator, code_executor=mock_code_executor, model_registry=mock_model_registry)

    result = agent.execute(request="List EC2 instances", cloud=CloudProvider.AWS, service="ec2", thread_id="test-123", params={"region": "invalid-region"})

    assert result["status"] == WorkflowStatus.FAILED
    assert result["outputs"]["error"] == "InvalidParameterError"


def test_workflow_with_partial_success(mock_prompt_generator, mock_code_generator, mock_code_executor, mock_model_registry):
    """Test workflow handling partial success scenarios."""
    # Configure executor to succeed first, then fail, then succeed again
    success_result = Mock(success=True, error=None, get_parsed_output=lambda: {"instances": [{"id": "i-123", "status": "running"}]}, execution_time=1.0, resource_usage={"memory_mb": 100})

    failure_result = Mock(success=False, error="Temporary failure", get_parsed_output=lambda: None, execution_time=0.5, resource_usage={"memory_mb": 50})

    mock_code_executor.execute_generated_code.side_effect = [success_result, failure_result, success_result]

    # Configure model to continue until we hit our success
    mock_model = mock_model_registry.get_model.return_value
    mock_model.invoke.side_effect = [Mock(content="unique"), Mock(content="redundant")]

    agent = OrchestrationAgent(model_name="test-model", prompt_generator=mock_prompt_generator, code_generator=mock_code_generator, code_executor=mock_code_executor, model_registry=mock_model_registry)

    result = agent.execute(request="Test request with retries", cloud=CloudProvider.AWS, service="ec2", thread_id="test-123")

    assert result["status"] == WorkflowStatus.COMPLETED
    assert len(result["discoveries"]) > 0
    assert result["error_count"] > 0  # Should have recorded the failure
    assert result["retry_attempts"] == 0  # Should have reset after final success


def test_workflow_with_model_failure(mock_prompt_generator, mock_code_generator, mock_code_executor, mock_model_registry):
    """Test workflow handling LLM model failures gracefully."""
    # Configure model to raise an exception
    mock_model = mock_model_registry.get_model.return_value
    mock_model.invoke.side_effect = Exception("Model API error")

    agent = OrchestrationAgent(model_name="test-model", prompt_generator=mock_prompt_generator, code_generator=mock_code_generator, code_executor=mock_code_executor, model_registry=mock_model_registry)

    result = agent.execute(request="Test request", cloud=CloudProvider.AWS, service="ec2", thread_id="test-123")

    assert result["status"] == WorkflowStatus.FAILED
    assert "Model API error" in result["outputs"]["error"]


def test_workflow_with_checkpointing(mock_prompt_generator, mock_code_generator, mock_code_executor, mock_model_registry):
    """Test workflow with checkpointing enabled."""
    stub_checkpointer = StubMemorySaver()
    agent = OrchestrationAgent(
        model_name="test-model",
        prompt_generator=mock_prompt_generator,
        code_generator=mock_code_generator,
        code_executor=mock_code_executor,
        model_registry=mock_model_registry,
        checkpointer=stub_checkpointer,  # Use our stub
    )
    result = agent.execute(request="Test request", cloud=CloudProvider.AWS, service="ec2", thread_id="test-123")
    assert result["status"] == WorkflowStatus.COMPLETED


def test_workflow_with_prompt_failure(mock_prompt_generator, mock_code_generator, mock_code_executor, mock_model_registry):
    """Test workflow handling prompt generation failures."""
    mock_prompt_generator.generate_prompt.side_effect = Exception("Failed to generate prompt")

    agent = OrchestrationAgent(model_name="test-model", prompt_generator=mock_prompt_generator, code_generator=mock_code_generator, code_executor=mock_code_executor, model_registry=mock_model_registry)

    result = agent.execute(request="Test request", cloud=CloudProvider.AWS, service="ec2", thread_id="test-123")

    assert result["status"] == WorkflowStatus.FAILED
    assert "Failed to generate prompt" in result["outputs"]["error"]


def test_workflow_with_code_generation_failure(mock_prompt_generator, mock_code_generator, mock_code_executor, mock_model_registry):
    """Test workflow handling code generation failures."""
    mock_code_generator.generate_code.side_effect = Exception("Failed to generate code")

    agent = OrchestrationAgent(model_name="test-model", prompt_generator=mock_prompt_generator, code_generator=mock_code_generator, code_executor=mock_code_executor, model_registry=mock_model_registry)

    result = agent.execute(request="Test request", cloud=CloudProvider.AWS, service="ec2", thread_id="test-123")

    assert result["status"] == WorkflowStatus.FAILED
    assert "Failed to generate code" in result["outputs"]["error"]


def test_workflow_completion_metrics(mock_prompt_generator, mock_code_generator, mock_code_executor, mock_model_registry):
    """Test workflow completion metrics are properly recorded."""
    success_result = Mock(success=True, error=None, get_parsed_output=lambda: {"instances": [{"id": "i-123"}]}, execution_time=2.5, resource_usage={"memory_mb": 150, "cpu_percent": 75})

    mock_code_executor.execute_generated_code.return_value = success_result
    mock_model = mock_model_registry.get_model.return_value
    mock_model.invoke.return_value = Mock(content="redundant")  # End workflow quickly

    agent = OrchestrationAgent(model_name="test-model", prompt_generator=mock_prompt_generator, code_generator=mock_code_generator, code_executor=mock_code_executor, model_registry=mock_model_registry)

    result = agent.execute(request="Test metrics", cloud=CloudProvider.AWS, service="ec2", thread_id="test-123")

    assert result["status"] == WorkflowStatus.COMPLETED
    assert result["execution_metrics"]["total_execution_time"] > 0
    assert "memory_mb" in result["execution_metrics"]["resource_usage"]
    assert "cpu_percent" in result["execution_metrics"]["resource_usage"]
    assert result["last_successful_iteration"] == 1


def test_workflow_empty_state(mock_prompt_generator, mock_code_generator, mock_code_executor, mock_model_registry):
    """Test workflow handling of empty initial state."""
    mock_code_executor.execute_generated_code.return_value = Mock(success=True, error=None, get_parsed_output=lambda: {"status": "empty", "message": "No resources found"}, execution_time=0.5, resource_usage={"memory_mb": 50})

    mock_model = mock_model_registry.get_model.return_value
    mock_model.invoke.return_value = Mock(content="redundant")  # End workflow quickly

    agent = OrchestrationAgent(model_name="test-model", prompt_generator=mock_prompt_generator, code_generator=mock_code_generator, code_executor=mock_code_executor, model_registry=mock_model_registry)

    result = agent.execute(request="Test empty state", cloud=CloudProvider.AWS, service="ec2", thread_id="test-123")

    assert result["status"] == WorkflowStatus.COMPLETED
    assert result["outputs"]["status"] == "empty"
    assert len(result["discoveries"]) == 1  # Should have one empty discovery
