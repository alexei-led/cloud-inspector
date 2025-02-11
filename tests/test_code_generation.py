import pytest
from pathlib import Path

from cloud_inspector.code_generator import CodeGeneratorResult, ParseError
from cloud_inspector.components.types import CodeGenerationPrompt, CloudProvider
from cloud_inspector.orchestration.nodes import code_generation_node, code_execution_node
from cloud_inspector.orchestration.state import create_initial_state

@pytest.fixture
def valid_code_result():
    return CodeGeneratorResult(
        generated_files={
            "main_py": "print('test')",
            "requirements_txt": "boto3==1.26.0",
            "policy_json": "{}"
        },
        output_path=Path("/tmp/test")
    )

@pytest.fixture
def mock_code_generator(mocker, valid_code_result):
    mock = mocker.Mock()
    mock.generate_code.return_value = valid_code_result
    return mock

@pytest.fixture
def mock_code_executor(mocker):
    mock = mocker.Mock()
    mock.execute_generated_code.return_value = mocker.Mock(
        success=True,
        error=None,
        execution_time=1.0,
        resource_usage={"cpu": 0.5, "memory": 100},
        get_parsed_output=lambda: {"test": "data"}
    )
    return mock

def test_code_generation_node_success(mock_code_generator):
    # Initialize state
    state = create_initial_state(
        request="list ec2 instances",
        cloud=CloudProvider.AWS,
        service="ec2"
    )
    state["outputs"]["prompt"] = CodeGenerationPrompt(
        service="ec2",
        operation="list",
        request="list instances",
        description="List EC2 instances",
        template="test template",
        cloud=CloudProvider.AWS,
        iteration=1
    )
    
    # Execute node
    updated_state = code_generation_node(
        state,
        {"code_generator": mock_code_generator, "model_name": "test-model"}
    )
    
    # Verify results
    assert "code" in updated_state["outputs"]
    assert isinstance(updated_state["outputs"]["code"], CodeGeneratorResult)
    assert "error" not in updated_state["outputs"]
    assert updated_state["status"] == "in_progress"

def test_code_execution_node_success(mock_code_executor, valid_code_result):
    # Initialize state
    state = create_initial_state(
        request="list ec2 instances",
        cloud=CloudProvider.AWS,
        service="ec2"
    )
    state["outputs"]["code"] = valid_code_result
    
    # Execute node
    updated_state = code_execution_node(
        state,
        {"code_executor": mock_code_executor}
    )
    
    # Verify results
    assert len(updated_state["discoveries"]) == 1
    assert "error" not in updated_state["outputs"]
    assert updated_state["status"] == "in_progress"
    assert "resource_usage" in updated_state["execution_metrics"]

def test_code_execution_node_invalid_result_type():
    # Initialize state with invalid code result
    state = create_initial_state(
        request="list ec2 instances",
        cloud=CloudProvider.AWS,
        service="ec2"
    )
    state["outputs"]["code"] = {"invalid": "format"}
    
    # Execute node
    updated_state = code_execution_node(
        state,
        {"code_executor": None}  # Won't be used due to type check failure
    )
    
    # Verify error handling
    assert "error" in updated_state["outputs"]
    assert "Invalid code generation result" in updated_state["outputs"]["error"]
    assert updated_state["status"] == "failed"
