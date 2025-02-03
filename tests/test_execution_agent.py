from datetime import datetime
from unittest.mock import MagicMock

import pytest

from cloud_inspector.code_execution import DockerSandbox
from cloud_inspector.components.types import GeneratedFiles
from cloud_inspector.execution_agent import CodeExecutionAgent, ExecutionResult


@pytest.fixture
def mock_sandbox():
    sandbox = MagicMock(spec=DockerSandbox)
    # Configure default success behavior
    sandbox.execute.return_value = (True, '{"test": "success"}', "", {"memory_usage_bytes": 100, "cpu_usage_percent": 50})
    return sandbox


@pytest.fixture
def executor(mock_sandbox):
    return CodeExecutionAgent(sandbox=mock_sandbox)


def test_execution_result_get_parsed_output():
    """Test ExecutionResult's get_parsed_output method."""
    # Test with valid JSON string
    result = ExecutionResult(success=True, output='{"key": "value"}', error=None, execution_time=1.0, executed_at=datetime.now(), resource_usage={}, generated_files=GeneratedFiles(main_py="", requirements_txt="", policy_json=""))
    parsed = result.get_parsed_output()
    assert parsed == {"key": "value"}
    assert result.parsed_json is True

    # Test with invalid JSON
    result = ExecutionResult(success=True, output="invalid json", error=None, execution_time=1.0, executed_at=datetime.now(), resource_usage={}, generated_files=GeneratedFiles(main_py="", requirements_txt="", policy_json=""))
    parsed = result.get_parsed_output()
    assert parsed is None
    assert result.parsed_json is False
    assert result.success is False
    assert "Invalid JSON output" in (result.error or "")


def test_successful_code_execution(executor):
    """Test successful code execution with JSON output."""
    executor.sandbox.execute.return_value = (True, '{"test": "success"}', "", {"memory_usage_bytes": 100, "cpu_usage_percent": 50})
    files = GeneratedFiles(main_py='print("{"test": "success"}")', requirements_txt="", policy_json="")

    # Execute
    result = executor.execute_generated_code(files)

    # Assert result
    assert result.success is True
    assert result.get_parsed_output() == {"test": "success"}
    assert result.error is None
    assert result.parsed_json is True


def test_failed_code_execution(executor):
    """Test failed code execution."""
    executor.sandbox.execute.return_value = (False, "", "Error: something went wrong", {"memory_usage_bytes": 100, "cpu_usage_percent": 50})
    files = GeneratedFiles(main_py="raise Exception('Test error')", requirements_txt="", policy_json="")

    result = executor.execute_generated_code(files)

    assert result.success is False
    assert "Error: something went wrong" in result.error
    assert result.parsed_json is False


def test_empty_code_execution(executor):
    """Test execution with empty code."""
    files = GeneratedFiles(main_py="", requirements_txt="", policy_json="")

    result = executor.execute_generated_code(files)

    assert result.success is False
    assert "Empty main.py content" in (result.error or "")


def test_execution_with_aws_credentials(executor):
    """Test execution with AWS credentials."""
    executor.sandbox.execute.return_value = (True, '{"status": "success"}', "", {"memory_usage_bytes": 100, "cpu_usage_percent": 50})

    files = GeneratedFiles(main_py=r"""print('{"status": "success"}')""", requirements_txt="", policy_json="")
    credentials = {"AWS_ACCESS_KEY_ID": "test-key", "AWS_SECRET_ACCESS_KEY": "test-secret"}

    result = executor.execute_generated_code(files, aws_credentials=credentials)

    assert result.success is True
    assert executor.sandbox.execute.call_args[1]["aws_credentials"] == credentials  # type: ignore


def test_cleanup(executor):
    """Test cleanup method."""
    executor.cleanup()
    executor.sandbox.cleanup.assert_called_once()  # type: ignore


def test_execution_with_requirements(executor):
    """Test execution with requirements."""
    files = GeneratedFiles(main_py=r"""print('{"status": "success"}')""", requirements_txt="requests==2.31.0\npandas==2.0.0", policy_json="")

    result = executor.execute_generated_code(files)

    assert result.success is True
    # Verify requirements were passed to sandbox
    executor.sandbox.execute.assert_called_once()  # type: ignore
    call_kwargs = executor.sandbox.execute.call_args[1]  # type: ignore
    assert "requests==2.31.0" in call_kwargs["requirements_txt"]


@pytest.mark.parametrize(
    "output,expected_success",
    [
        (b'{"valid": "json"}', True),
        (b"invalid json", False),
        (b"", False),
        (b'{"incomplete": }', False),
    ],
)
def test_various_output_formats(executor, mock_sandbox, output, expected_success):
    """Test handling of various output formats."""
    mock_sandbox.execute.return_value = (expected_success, output.decode(), "", {})

    files = GeneratedFiles(main_py="print('test')", requirements_txt="", policy_json="")

    result = executor.execute_generated_code(files)

    assert result.success is expected_success
