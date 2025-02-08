from unittest.mock import Mock

import pytest

from cloud_inspector.code_generator import CodeGeneratorAgent
from cloud_inspector.components.types import CloudProvider, WorkflowStatus
from cloud_inspector.execution_agent import CodeExecutionAgent
from cloud_inspector.orchestration.orchestration import OrchestrationAgent
from cloud_inspector.prompt_generator import PromptGeneratorAgent


@pytest.fixture
def mock_prompt_generator():
    generator = Mock(spec=PromptGeneratorAgent)
    generator.generate_prompt.return_value = {"template": "test template", "variables": [{"name": "region", "value": "us-west-2"}], "success_criteria": "test criteria", "description": "test description"}
    return generator


@pytest.fixture
def mock_code_generator():
    generator = Mock(spec=CodeGeneratorAgent)
    generator.generate_code.return_value = ({"model_name": "test-model", "iteration_id": "test-1", "run_id": "run-1", "generated_files": {"main.py": "print('test')", "requirements.txt": "boto3==1.26.0", "policy.json": "{}"}}, Mock())
    return generator


@pytest.fixture
def mock_code_executor():
    executor = Mock(spec=CodeExecutionAgent)
    result = Mock()
    result.success = True
    result.get_parsed_output.return_value = {"instances": [{"id": "i-123", "state": "running"}]}
    result.execution_time = 1.0
    result.resource_usage = {"memory_mb": 100, "cpu_percent": 50}
    result.error = None
    executor.execute_generated_code.return_value = result
    return executor


def test_workflow_initialization(mock_prompt_generator, mock_code_generator, mock_code_executor):
    """Test OrchestrationAgent initialization and workflow creation."""
    agent = OrchestrationAgent(model_name="test-model", prompt_generator=mock_prompt_generator, code_generator=mock_code_generator, code_executor=mock_code_executor)

    workflow = agent._create_workflow()
    assert workflow is not None
    assert len(workflow.nodes) == 5  # Verify all nodes are added


def test_successful_execution(mock_prompt_generator, mock_code_generator, mock_code_executor):
    """Test successful workflow execution with all steps completing normally."""

    # Make a reusable result mock
    def make_mock_result():
        result = Mock()
        result.success = True
        result.error = None
        result.get_parsed_output.return_value = {"instances": [{"id": "i-123", "state": "running"}]}
        result.execution_time = 1.0
        result.resource_usage = {"memory_mb": 100, "cpu_percent": 50}
        return result

    # Configure executor to always return the same result
    mock_code_executor.execute_generated_code.return_value = make_mock_result()

    # Mock the model to indicate redundancy after first result
    mock_model = Mock()
    model_responses = [Mock(content="redundant")]
    mock_model.invoke.side_effect = model_responses

    # Mock the model registry
    mock_model_registry = Mock()
    mock_model_registry.get_model = Mock(return_value=mock_model)

    agent = OrchestrationAgent(model_name="test-model", prompt_generator=mock_prompt_generator, code_generator=mock_code_generator, code_executor=mock_code_executor, model_registry=mock_model_registry)

    result = agent.execute(request="Test request", cloud=CloudProvider.AWS, service="ec2", thread_id="test-123")

    assert result["status"] == WorkflowStatus.COMPLETED
    assert len(result["discoveries"]) > 0
    assert result["execution_metrics"]["total_execution_time"] >= 0
    assert not result.get("error")
    assert result["outputs"]["instances"][0]["id"] == "i-123"
    assert result["reason"] == "no_new_information_found"


def test_workflow_with_empty_results(mock_prompt_generator, mock_code_generator, mock_code_executor):
    """Test workflow when execution returns empty results."""
    empty_result = Mock()
    empty_result.success = True
    empty_result.error = None
    empty_result.get_parsed_output.return_value = {"instances": []}
    empty_result.execution_time = 0.5
    empty_result.resource_usage = {"memory_mb": 50, "cpu_percent": 20}
    mock_code_executor.execute_generated_code.return_value = empty_result

    # Mock the model to indicate unique for empty results
    mock_model = Mock()
    mock_model.invoke.return_value = Mock(content="unique")
    mock_model_registry = Mock()
    mock_model_registry.get_model.return_value = mock_model

    agent = OrchestrationAgent(model_name="test-model", prompt_generator=mock_prompt_generator, code_generator=mock_code_generator, code_executor=mock_code_executor, model_registry=mock_model_registry)

    result = agent.execute(request="List EC2 instances", cloud=CloudProvider.AWS, service="ec2", thread_id="test-123")

    assert result["status"] == WorkflowStatus.COMPLETED
    # Check that we have at least one discovery with the empty instances
    assert len(result["discoveries"]) > 0
    # Check the output in the last discovery
    last_discovery = result["discoveries"][-1]
    assert "output" in last_discovery
    assert "instances" in last_discovery["output"]
    assert len(last_discovery["output"]["instances"]) == 0
    assert result["execution_metrics"]["total_execution_time"] >= 0
    assert not result.get("error")


def test_workflow_with_multiple_cloud_providers(mock_prompt_generator, mock_code_generator, mock_code_executor):
    """Test workflow execution with different cloud providers."""

    def make_cloud_result(provider):
        result = Mock()
        result.success = True
        result.error = None
        result.get_parsed_output.return_value = {"provider": provider, "resources": [{"id": f"{provider}-123", "type": "instance"}]}
        result.execution_time = 1.0
        result.resource_usage = {"memory_mb": 100, "cpu_percent": 40}
        return result

    mock_code_executor.execute_generated_code.side_effect = [make_cloud_result("aws"), make_cloud_result("azure"), make_cloud_result("gcp")]

    agent = OrchestrationAgent(model_name="test-model", prompt_generator=mock_prompt_generator, code_generator=mock_code_generator, code_executor=mock_code_executor)

    for provider in [CloudProvider.AWS, CloudProvider.AZURE, CloudProvider.GCP]:
        result = agent.execute(request="List compute instances", cloud=provider, service="compute", thread_id=f"test-{provider}")

        assert result["status"] == WorkflowStatus.COMPLETED
        assert result["outputs"]["provider"].lower() == provider.value.lower()
        assert len(result["outputs"]["resources"]) > 0
        assert result["outputs"]["resources"][0]["id"].startswith(provider.value.lower())


def test_workflow_with_invalid_service(mock_prompt_generator, mock_code_generator, mock_code_executor):
    """Test workflow behavior with invalid service specification."""
    error_result = Mock()
    error_result.success = False
    error_result.error = "Invalid service: unknown_service"
    error_result.get_parsed_output.return_value = {"error": "Service 'unknown_service' not supported"}
    error_result.execution_time = 0.1
    error_result.resource_usage = {}
    mock_code_executor.execute_generated_code.return_value = error_result

    agent = OrchestrationAgent(model_name="test-model", prompt_generator=mock_prompt_generator, code_generator=mock_code_generator, code_executor=mock_code_executor)

    result = agent.execute(request="List resources", cloud=CloudProvider.AWS, service="unknown_service", thread_id="test-123")

    assert result["status"] == WorkflowStatus.FAILED
    assert "Invalid service" in result["error"]
    assert "unknown_service" in result["outputs"]["error"]


def test_execution_failure_after_max_retries(mock_prompt_generator, mock_code_generator, mock_code_executor):
    """Test workflow execution failing after maximum retry attempts."""
    error_result = Mock()
    error_result.success = False
    error_result.error = "Persistent error"
    error_result.get_parsed_output.return_value = None
    error_result.execution_time = 0.1
    error_result.resource_usage = {}
    mock_code_executor.execute_generated_code.return_value = error_result

    agent = OrchestrationAgent(model_name="test-model", prompt_generator=mock_prompt_generator, code_generator=mock_code_generator, code_executor=mock_code_executor)

    result = agent.execute(request="Test request", cloud=CloudProvider.AWS, service="ec2", thread_id="test-123")

    assert result["status"] == WorkflowStatus.FAILED
    assert result["retry_attempts"] == 2  # Max retries reached
    assert result["error"] == "Persistent error"
    assert "error" in result["outputs"]


def test_max_iterations_limit(mock_prompt_generator, mock_code_generator, mock_code_executor):
    """Test workflow respects maximum iterations limit."""
    # Configure successful execution but with unique results each time
    results = [
        Mock(success=True, error=None, get_parsed_output=lambda i=i: {"data": f"iteration_{i}", "new_findings": True}, execution_time=1.0, resource_usage={"memory_mb": 100})
        for i in range(5)  # More than MAX_ITERATIONS
    ]
    mock_code_executor.execute_generated_code.side_effect = results

    agent = OrchestrationAgent(model_name="test-model", prompt_generator=mock_prompt_generator, code_generator=mock_code_generator, code_executor=mock_code_executor)

    result = agent.execute(request="Test request", cloud=CloudProvider.AWS, service="ec2", thread_id="test-123")

    assert result["iteration"] <= 3  # MAX_ITERATIONS constant
    assert result["status"] == WorkflowStatus.COMPLETED
    assert result["reason"] == "max_iterations_reached"
    assert len(result["discoveries"]) > 0


def test_workflow_with_region_param(mock_prompt_generator, mock_code_generator, mock_code_executor):
    """Test workflow execution with region parameter."""
    mock_code_executor.execute_generated_code.return_value = Mock(success=True, error=None, get_parsed_output=lambda: {"instances": [{"id": "i-123", "region": "us-west-2"}]}, execution_time=1.0, resource_usage={"memory_mb": 100})

    agent = OrchestrationAgent(model_name="test-model", prompt_generator=mock_prompt_generator, code_generator=mock_code_generator, code_executor=mock_code_executor)

    result = agent.execute(request="List EC2 instances", cloud=CloudProvider.AWS, service="ec2", thread_id="test-123", params={"region": "us-west-2"})

    assert result["status"] == WorkflowStatus.COMPLETED
    assert result["outputs"]["instances"][0]["region"] == "us-west-2"


def test_workflow_error_handling(mock_prompt_generator, mock_code_generator, mock_code_executor):
    """Test workflow error handling with invalid parameters."""
    mock_code_executor.execute_generated_code.return_value = Mock(success=False, error="InvalidParameterError", get_parsed_output=lambda: {"error": "Invalid region specified"}, execution_time=0.1, resource_usage={})

    agent = OrchestrationAgent(model_name="test-model", prompt_generator=mock_prompt_generator, code_generator=mock_code_generator, code_executor=mock_code_executor)

    result = agent.execute(request="List EC2 instances", cloud=CloudProvider.AWS, service="ec2", thread_id="test-123", params={"region": "invalid-region"})

    assert result["status"] == WorkflowStatus.FAILED
    assert "Invalid region" in result["outputs"]["error"]


def test_workflow_with_complex_request(mock_prompt_generator, mock_code_generator, mock_code_executor):
    """Test workflow with a complex security analysis request."""
    security_findings = [{"severity": "HIGH", "type": "open_port", "details": "Port 22 exposed to 0.0.0.0/0"}, {"severity": "MEDIUM", "type": "weak_cipher", "details": "Weak encryption algorithm in use"}]

    mock_code_executor.execute_generated_code.return_value = Mock(success=True, error=None, get_parsed_output=lambda: {"security_findings": security_findings}, execution_time=2.0, resource_usage={"memory_mb": 150, "cpu_percent": 60})

    agent = OrchestrationAgent(model_name="test-model", prompt_generator=mock_prompt_generator, code_generator=mock_code_generator, code_executor=mock_code_executor)

    result = agent.execute(request="Analyze EC2 security", cloud=CloudProvider.AWS, service="ec2", thread_id="test-456")

    assert result["status"] == WorkflowStatus.COMPLETED
    assert len(result["outputs"]["security_findings"]) == 2
    assert result["outputs"]["security_findings"][0]["severity"] == "HIGH"
    assert "execution_metrics" in result
