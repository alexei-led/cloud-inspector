from datetime import datetime
from pathlib import Path
from typing import Optional
from unittest.mock import Mock, mock_open, patch

import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from black.parsing import InvalidInput
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
    messages = generator.format_prompt(
        test_prompt,
        variables={"service": "lambda"},  # environment has default value
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
    """Test end-to-end code generation with structured output"""
    mock_model = Mock()
    mock_model.with_structured_output.return_value.invoke.return_value = {"parsed": GeneratedFiles(main_py="def test(): pass", requirements_txt="requests==2.0.0", policy_json="{}")}
    generator.model_registry.get_model.return_value = mock_model

    result, _ = generator.generate_code(prompt=test_prompt, model_name="test-model", variables={"service": "lambda"}, iteration_id="test-1")

    assert "def test():" in result.generated_files["main.py"]
    assert "requests==2.0.0" in result.generated_files["requirements.txt"]


def test_generate_code_raw_response_parsing(generator, test_prompt):
    """Test parsing raw response when structured parsing fails"""
    mock_model = Mock()
    mock_model.with_structured_output.return_value.invoke.return_value = {
        "parsed": None,
        "raw": [
            {"type": "tool_use", "name": "GeneratedFiles", "input": {"main_py": "def v1(): pass", "requirements_txt": "pkg1", "policy_json": "{}"}},
            {"type": "tool_use", "name": "GeneratedFiles", "input": {"main_py": "def v2(): pass", "requirements_txt": "pkg2", "policy_json": "{}"}},
        ],
    }
    generator.model_registry.get_model.return_value = mock_model

    result, _ = generator.generate_code(prompt=test_prompt, model_name="test-model", variables={"service": "lambda"}, iteration_id="test-1")

    assert "def v2():" in result.generated_files["main.py"]
    assert result.generated_files["requirements.txt"] == "pkg2"


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


def test_process_model_response_base_model(generator):
    """Test processing of BaseModel response"""

    class TestResponse(BaseModel):
        parsed: GeneratedFiles
        raw: Optional[str] = None

    response = TestResponse(parsed=GeneratedFiles(main_py="def test(): pass", requirements_txt="requests==2.0.0", policy_json="{}"))

    files = generator._process_model_response(response)
    assert "def test():" in files["main.py"]
    assert "requests==2.0.0" in files["requirements.txt"]
    assert files["policy.json"] == "{}"
