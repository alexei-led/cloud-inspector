import pytest
from datetime import datetime
from components.types import GeneratedFiles

def test_execution_agent_basic(code_executor):
    """Test basic code execution."""
    files = GeneratedFiles(
        main_py="import json\nprint(json.dumps({'test': 'success'}))",
        requirements_txt="",
        policy_json=""
    )
    
    result = code_executor.execute_generated_code(
        generated_files=files,
        execution_id="test-123"
    )
    
    assert result.success
    assert result.parsed_json
    assert result.output == {"test": "success"}
    assert isinstance(result.executed_at, datetime)
    assert "execution_time" in result.resource_usage

def test_execution_agent_invalid_json(code_executor):
    """Test handling of invalid JSON output."""
    files = GeneratedFiles(
        main_py="print('not json')",
        requirements_txt="",
        policy_json=""
    )
    
    result = code_executor.execute_generated_code(
        generated_files=files,
        execution_id="test-456"
    )
    
    assert not result.success
    assert not result.parsed_json
    assert "Invalid JSON output" in (result.error or "")

@pytest.mark.integration
def test_execution_agent_with_requirements(code_executor):
    """Test execution with package requirements."""
    files = GeneratedFiles(
        main_py="""
import requests
import json
print(json.dumps({"status": "success"}))
""",
        requirements_txt="requests==2.31.0",
        policy_json=""
    )
    
    result = code_executor.execute_generated_code(
        generated_files=files,
        execution_id="test-789"
    )
    
    assert result.success
    assert result.parsed_json
    assert result.output.get("status") == "success"
