from datetime import datetime
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, Mock, mock_open, patch

import pytest
from black.parsing import InvalidInput
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from cloud_inspector.code_generator import CodeGeneratorAgent, CodeGeneratorResult, ParseError
from cloud_inspector.components.models import ModelCapability, ModelRegistry
from cloud_inspector.components.types import CloudProvider, CodeGenerationPrompt, GeneratedFiles


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
def mock_model_registry():
    registry = Mock(spec=ModelRegistry)
    registry.validate_model_capability.return_value = True
    registry.get_structured_output_params.return_value = {"include_raw": True}
    return registry


@pytest.fixture
def generator(mock_model_registry):
    return CodeGeneratorAgent(model_registry=mock_model_registry)


@pytest.fixture
def test_prompt():
    return CodeGenerationPrompt(
        service="lambda",
        operation="create",
        request="create a lambda function",
        generated_by="test",
        generated_at=datetime.now(),
        success_criteria="function created",
        feedback={"feedback": "good"},
        iteration=1,
        description="Test prompt",
        cloud=CloudProvider.AWS,
        template="Generate code for ${service} with ${environment}",
        variables=[
            {"name": "service", "description": "Service name"},
            {"name": "environment", "description": "Environment name", "value": "dev"},
        ],
    )


# Test prompt formatting and validation
def test_format_prompt_with_default_values(generator, test_prompt):
    """Test that prompt formatting uses default values from template variables"""
    # Find variables with default values
    default_vars = {var["name"]: var["value"] for var in test_prompt.variables if "value" in var}

    # Combine provided and default variables
    variables = {"service": "lambda"}
    variables.update(default_vars)

    messages = generator.format_prompt(
        test_prompt,
        variables=variables,
        supports_system_prompt=True,
    )
    assert any("dev" in str(msg.content) for msg in messages)
    assert any("lambda" in str(msg.content) for msg in messages)
    assert len(messages) == 2
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)


def test_format_prompt_without_system_prompt(generator, test_prompt):
    """Test prompt formatting when system prompts aren't supported"""
    messages = generator.format_prompt(test_prompt, variables={"service": "lambda", "environment": "prod"}, supports_system_prompt=False)
    assert len(messages) == 1
    assert isinstance(messages[0], HumanMessage)
    assert "<instructions>" in messages[0].content
    assert "<question>" in messages[0].content


def test_format_prompt_validation(generator, test_prompt):
    """Test prompt validation with missing required variables"""
    test_prompt.variables.append({"name": "region", "description": "AWS region"})
    with pytest.raises(ValueError) as exc:
        generator.format_prompt(test_prompt, variables={"service": "lambda"})
    assert "region" in str(exc.value)
    assert "AWS region" in str(exc.value)


# Test code generation and response handling
def test_generate_code_model_validation(generator, test_prompt):
    """Test model capability validation"""
    generator.model_registry.validate_model_capability.return_value = False
    with pytest.raises(ValueError) as exc:
        generator.generate_code(prompt=test_prompt, model_name="test-model", variables={"service": "lambda"}, iteration_id="test-1")
    assert "model" in str(exc.value)
    generator.model_registry.validate_model_capability.assert_called_once_with("test-model", ModelCapability.CODE_GENERATION)


def test_generate_code_structured_output_parsing(generator, test_prompt):
    """Test the complete workflow of code generation with structured output."""
    # Setup mock model and response
    mock_model = Mock()
    mock_structured_model = Mock()
    mock_model.with_structured_output.return_value = mock_structured_model

    # Setup expected response
    expected_files = GeneratedFiles(main_py="def handler(event, context):\n    return {'statusCode': 200}", requirements_txt="aws-lambda-powertools==2.0.0", policy_json='{"Version": "2012-10-17"}')
    mock_structured_model.invoke.return_value = {"parsed": expected_files}
    generator.model_registry.get_model.return_value = mock_model

    # Setup run tracking mock
    mock_run = Mock()
    mock_run.id = "test-run-id"
    mock_runs_cb = Mock()
    mock_runs_cb.traced_runs = [mock_run]

    variables = {"service": "lambda", "environment": "dev"}

    # Create a proper context manager mock for collect_runs
    mock_ctx_manager = MagicMock()
    mock_ctx_manager.__enter__.return_value = mock_runs_cb
    mock_ctx_manager.__exit__.return_value = None

    with patch("langchain_core.tracers.context.collect_runs", return_value=mock_ctx_manager):
        result = generator.generate_code(prompt=test_prompt, model_name="test-model", variables=variables, iteration_id="test-1")

        # Verify model validation was called
        generator.model_registry.validate_model_capability.assert_called_once_with("test-model", ModelCapability.CODE_GENERATION)

        # Verify structured output params were checked
        generator.model_registry.get_structured_output_params.assert_called_once()

        # Verify model was configured correctly
        mock_model.with_structured_output.assert_called_once()

        # Verify result contains expected data
        assert result.model_name == "test-model"
        assert result.iteration_id == "test-1"
        assert result.run_id == "test-run-id"
        assert isinstance(result.generated_at, datetime)

        # Verify files were processed and formatted
        assert "handler" in result.generated_files["main_py"]
        assert "aws-lambda-powertools" in result.generated_files["requirements_txt"]
        assert "2012-10-17" in result.generated_files["policy_json"]

        # Verify output directory was set and exists
        assert result.output_path is not None
        assert result.output_path.exists()
        assert (result.output_path / "main.py").exists()
        assert (result.output_path / "requirements.txt").exists()
        assert (result.output_path / "policy.json").exists()
        assert (result.output_path / "metadata.json").exists()


# Test code formatting and validation
def test_reformat_code_python_syntax_error(generator):
    """Test handling of Python syntax errors during formatting"""
    invalid_code = "def test(:"
    formatted = generator._reformat_code(invalid_code, code=True)
    assert formatted == invalid_code  # Should return original if can't parse


def test_reformat_code_import_cleanup(generator):
    """Test cleanup of Python imports"""
    messy_code = """
from os import *
import sys
import os.path
from datetime import datetime, datetime

def test():
    print("hello")
"""
    formatted = generator._reformat_code(messy_code, code=True)
    assert "from os import *" not in formatted
    assert formatted.count("import") < 4  # Should combine/remove duplicate imports


@patch("cloud_inspector.code_generator.format_str")
def test_reformat_code_black_error(mock_black, generator):
    """Test handling of black formatting errors"""
    mock_black.side_effect = InvalidInput("Bad format")
    code = "def test():\n  pass"
    formatted = generator._reformat_code(code, code=True)
    assert formatted == code  # Should return original if black fails


# Test file saving
def test_save_result_file_error(generator, tmp_path):
    """Test handling of file write errors"""
    result = CodeGeneratorResult(
        generated_files={
            "main_py": "def test(): pass",
            "requirements_txt": "requests",
            "policy_json": "{}"
        },
        model_name="test-model",
        generated_at=datetime.now(),
        iteration_id="test-1"
    )

    # Create mock file handlers
    main_file = mock_open().return_value
    requirements_file = mock_open()
    requirements_file.return_value.write.side_effect = PermissionError("Access denied")
    policy_file = mock_open().return_value
    metadata_file = mock_open().return_value

    mock_files = {
        'main.py': main_file,
        'requirements.txt': requirements_file(),
        'policy.json': policy_file,
        'metadata.json': metadata_file
    }

    def mock_open_func(file_path, mode='r'):
        # Extract filename from path
        filename = Path(file_path).name
        if filename in mock_files:
            return mock_files[filename]
        return mock_open()()

    # Set up the output directory for the generator
    generator.output_dir = tmp_path

    # Patch both mkdir and open
    with patch('pathlib.Path.mkdir', return_value=None), \
         patch('builtins.open', mock_open_func), \
         pytest.raises(RuntimeError) as exc:
        generator._save_result(result)

    assert "Failed to save generated file" in str(exc.value)
    assert "Access denied" in str(exc.value)


def test_process_model_response_with_empty_files(generator):
    """Test processing model response with empty file content"""
    response = {"parsed": GeneratedFiles(main_py="", requirements_txt="", policy_json="")}
    files = generator._process_model_response(response)
    assert all(content == "" for content in files.values())


def test_prepare_messages_json_format(generator, test_prompt):
    """Test that prepared messages include JSON format requirements"""
    messages = generator._prepare_messages(test_prompt, variables={"service": "lambda", "environment": "dev"})

    # Check system message contains JSON structure
    assert any('"main_py"' in msg["content"] for msg in messages)
    assert any('"requirements_txt"' in msg["content"] for msg in messages)
    assert any('"policy_json"' in msg["content"] for msg in messages)

    # Check user message requests JSON response
    assert any("respond with a valid JSON object" in msg["content"] for msg in messages)


def test_parse_raw_response_with_mock(generator):
    """Test parsing raw response from Mock object."""
    mock_response = Mock()
    mock_response.content = '{"key": "value"}'

    result = generator._parse_raw_response(mock_response)
    assert isinstance(result, dict)
    assert result["key"] == "value"

def test_parse_raw_response_with_dict(generator):
    """Test parsing raw response that's already a dict."""
    input_dict = {"key": "value"}
    result = generator._parse_raw_response(input_dict)
    assert result == input_dict

def test_parse_raw_response_invalid_json(generator):
    """Test parsing invalid JSON string."""
    with pytest.raises(ParseError) as exc:
        generator._parse_raw_response("invalid json")
    assert "Invalid JSON response" in str(exc.value)

def test_extract_files_from_messages_direct_dict(generator):
    """Test extracting files from direct dictionary format."""
    messages = {
        "main_py": "def test(): pass",
        "requirements_txt": "requests==2.0.0",
        "policy_json": "{}"
    }
    result = generator._extract_files_from_messages(messages)
    assert result == messages

def test_extract_files_from_messages_missing_keys(generator):
    """Test handling dictionary with missing required keys."""
    messages = {
        "main_py": "def test(): pass",
        "requirements_txt": "requests==2.0.0"
        # missing policy_json
    }
    with pytest.raises(ParseError) as exc:
        generator._extract_files_from_messages(messages)
    assert "missing required file keys" in str(exc.value).lower()

def test_extract_files_from_messages_list_format(generator):
    """Test extracting files from list of messages format."""
    messages = [
        {
            "type": "tool_use",
            "name": "GeneratedFiles",
            "input": {
                "main_py": "def test1(): pass",
                "requirements_txt": "requests==1.0.0",
                "policy_json": "{}"
            }
        },
        {
            "type": "tool_use",
            "name": "GeneratedFiles",
            "input": {
                "main_py": "def test2(): pass",
                "requirements_txt": "requests==2.0.0",
                "policy_json": "{}"
            }
        }
    ]
    result = generator._extract_files_from_messages(messages)
    # Should contain the latest values
    assert "test2()" in result["main_py"]
    assert "2.0.0" in result["requirements_txt"]

def test_extract_files_from_messages_empty_list(generator):
    """Test handling empty list of messages."""
    with pytest.raises(ParseError) as exc:
        generator._extract_files_from_messages([])
    assert "No GeneratedFiles content found" in str(exc.value)

def test_is_valid_generated_files_message(generator):
    """Test validation of GeneratedFiles messages."""
    valid_msg = {
        "type": "tool_use",
        "name": "GeneratedFiles",
        "input": {}
    }

    assert generator._is_valid_generated_files_message(valid_msg) is True
    invalid_msg = {
        "type": "tool_use",
        "name": "OtherTool",
        "input": {}
    }
    assert generator._is_valid_generated_files_message(invalid_msg) is False

def test_update_latest_files(generator):
    """Test updating latest files with new content."""
    latest_files = {
        "main_py": "old code",
        "requirements_txt": "old req",
        "policy_json": "old policy"
    }
    new_input = {
        "main_py": "new code",
        "requirements_txt": "new req",
        # policy_json not included
    }
    generator._update_latest_files(latest_files, new_input)

    assert latest_files["main_py"] == "new code"
    assert latest_files["requirements_txt"] == "new req"
    assert latest_files["policy_json"] == "old policy"  # Should retain old value

def test_process_model_response_json_format(generator):
    """Test processing of JSON formatted model response."""
    # Test with raw response
    mock_response = Mock()
    mock_response.content = '''{
        "main_py": "def test(): pass",
        "requirements_txt": "requests==2.0.0",
        "policy_json": "{}"
    }'''
    response = {"raw": mock_response, "parsed": None}
    files = generator._process_model_response(response)
    assert "def test()" in files["main_py"]
    assert "requests==2.0.0" in files["requirements_txt"]
    assert files["policy_json"] == "{}"

    # Test with parsed response
    parsed_response = {
        "parsed": GeneratedFiles(
            main_py="def test2(): pass",
            requirements_txt="requests==2.1.0",
            policy_json="{}"
        )
    }
    files = generator._process_model_response(parsed_response)
    assert "def test2()" in files["main_py"]
    assert "requests==2.1.0" in files["requirements_txt"]

def test_process_model_response_invalid_json(generator):
    """Test handling of invalid JSON response."""
    # Test with invalid JSON string
    mock_response = Mock()
    mock_response.content = "Invalid JSON content"
    response = {"raw": mock_response, "parsed": None}
    with pytest.raises(ParseError) as exc:
        generator._process_model_response(response)
    assert "Invalid JSON response" in str(exc.value)

    # Test with invalid response type
    with pytest.raises(ParseError) as exc:
        generator._process_model_response("not a dict or BaseModel")
    assert "Expected dictionary or BaseModel" in str(exc.value)

    # Test with missing required keys
    mock_response.content = '{"main_py": "code"}'  # Missing other required keys
    response = {"raw": mock_response, "parsed": None}
    with pytest.raises(ParseError) as exc:
        generator._process_model_response(response)
    assert "missing required file keys" in str(exc.value).lower()


def test_process_model_response_base_model(generator):
    """Test processing of BaseModel response"""

    class TestResponse(BaseModel):
        parsed: GeneratedFiles
        raw: Optional[str] = None

    # Create a response with GeneratedFiles that will be converted to dict
    files_content = GeneratedFiles(main_py="def test(): pass", requirements_txt="requests==2.0.0", policy_json="{}")
    response = TestResponse(parsed=files_content)

    files = generator._process_model_response(response)
    assert "def test():" in files["main_py"]
    assert "requests==2.0.0" in files["requirements_txt"]
    assert files["policy_json"] == "{}"


def test_code_generator_result_validation():
    """Test CodeGeneratorResult validation of required keys."""
    # Test valid case
    valid_result = CodeGeneratorResult(
        generated_files={
            "main_py": "print('test')",
            "requirements_txt": "boto3==1.26.0",
            "policy_json": "{}"
        }
    )
    assert valid_result is not None

    # Test missing required key
    with pytest.raises(ValueError) as exc:
        CodeGeneratorResult(
            generated_files={
                "main_py": "print('test')",
                "requirements_txt": "boto3==1.26.0"
                # missing policy_json
            }
        )
    assert "missing required keys" in str(exc.value).lower()


def test_code_generator_result_optional_output_path():
    """Test CodeGeneratorResult with and without output_path."""
    # Without output path
    result1 = CodeGeneratorResult(
        generated_files={
            "main_py": "print('test')",
            "requirements_txt": "boto3==1.26.0",
            "policy_json": "{}"
        }
    )
    assert result1.output_path is None

    # With output path
    result2 = CodeGeneratorResult(
        generated_files={
            "main_py": "print('test')",
            "requirements_txt": "boto3==1.26.0",
            "policy_json": "{}"
        },
        output_path=Path("/tmp/test")
    )
    assert result2.output_path == Path("/tmp/test")
