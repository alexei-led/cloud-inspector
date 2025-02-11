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
        result, output_dir = generator.generate_code(prompt=test_prompt, model_name="test-model", variables=variables, iteration_id="test-1")

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
        assert "handler" in result.generated_files["main.py"]
        assert "aws-lambda-powertools" in result.generated_files["requirements.txt"]
        assert "2012-10-17" in result.generated_files["policy.json"]

        # Verify output directory was created
        assert output_dir.exists()
        assert (output_dir / "main.py").exists()
        assert (output_dir / "requirements.txt").exists()
        assert (output_dir / "policy.json").exists()
        assert (output_dir / "metadata.json").exists()


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
def test_save_result_file_error(generator):
    """Test handling of file write errors"""
    result = CodeGeneratorResult(
        prompt_template="test", model_name="test-model", iteration_id="test-1", run_id="run-1", generated_at=datetime.now(), generated_files={"main.py": "def test(): pass", "requirements.txt": "requests", "policy.json": "{}"}
    )

    with patch("builtins.open", mock_open()) as mock_file:
        mock_file.side_effect = [None, PermissionError("Access denied"), None, None]
        output_dir = generator._save_result(result)
        assert isinstance(output_dir, Path)


def test_process_model_response_with_empty_files(generator):
    """Test processing model response with empty file content"""
    response = {"parsed": GeneratedFiles(main_py="", requirements_txt="", policy_json="")}
    files = generator._process_model_response(response)
    assert all(content == "" for content in files.values())


def test_prepare_messages_json_format(generator, test_prompt):
    """Test that prepared messages include JSON format requirements"""
    messages = generator._prepare_messages(
        test_prompt,
        variables={"service": "lambda", "environment": "dev"}
    )
    
    # Check system message contains JSON structure
    assert any('"main_py"' in msg["content"] for msg in messages)
    assert any('"requirements_txt"' in msg["content"] for msg in messages)
    assert any('"policy_json"' in msg["content"] for msg in messages)
    
    # Check user message requests JSON response
    assert any("respond with a valid JSON object" in msg["content"] for msg in messages)

def test_process_model_response_json_format(generator):
    """Test processing of JSON formatted model response"""
    # Create a mock with content property that returns the JSON string
    mock_response = Mock()
    mock_response.content = '''{
        "main_py": "def test(): pass",
        "requirements_txt": "requests==2.0.0",
        "policy_json": "{}"
    }'''
    
    json_response = {
        "raw": mock_response
    }
    
    files = generator._process_model_response(json_response)
    assert "def test():" in files["main.py"]
    assert "requests==2.0.0" in files["requirements.txt"]
    assert files["policy.json"] == "{}"

def test_process_model_response_invalid_json(generator):
    """Test handling of invalid JSON response"""
    # Create a mock with content property that returns invalid JSON
    mock_response = Mock()
    mock_response.content = "Invalid JSON content"
    
    invalid_response = {
        "raw": mock_response
    }
    
    with pytest.raises(ParseError):
        generator._process_model_response(invalid_response)

def test_process_model_response_base_model(generator):
    """Test processing of BaseModel response"""

    class TestResponse(BaseModel):
        parsed: GeneratedFiles
        raw: Optional[str] = None

    # Create a response with GeneratedFiles that will be converted to dict
    files_content = GeneratedFiles(main_py="def test(): pass", requirements_txt="requests==2.0.0", policy_json="{}")
    response = TestResponse(parsed=files_content)

    files = generator._process_model_response(response)
    assert "def test():" in files["main.py"]
    assert "requests==2.0.0" in files["requirements.txt"]
    assert files["policy.json"] == "{}"
