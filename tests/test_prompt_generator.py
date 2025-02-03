from unittest.mock import Mock

import pytest
import yaml

from cloud_inspector.components.models import ModelRegistry
from cloud_inspector.prompt_generator import PromptGeneratorAgent


@pytest.fixture
def mock_model():
    model = Mock()
    model.model_kwargs = {"response_format": "something"}
    return model


@pytest.fixture
def model_registry(mock_model):
    registry = Mock(spec=ModelRegistry)
    registry.get_model.return_value = mock_model
    return registry


@pytest.fixture
def prompt_generator(model_registry):
    return PromptGeneratorAgent(model_registry=model_registry)


def test_format_data_handles_complex_data(prompt_generator):
    """Test _format_data handles complex nested data structures"""
    complex_data = {"instances": [{"id": "i-123", "state": "running", "tags": {"Name": "web-1"}}, {"id": "i-456", "state": "stopped", "tags": {"Name": "web-2"}}], "metadata": {"region": "us-west-2", "vpc_id": "vpc-789"}}
    formatted = prompt_generator._format_data(complex_data)
    assert yaml.safe_load(formatted) == complex_data
    assert "instances:" in formatted
    assert "metadata:" in formatted

    # Test handling of special characters
    special_chars = {"key": 'value with \n newline and "quotes"'}
    formatted_special = prompt_generator._format_data(special_chars)
    assert yaml.safe_load(formatted_special) == special_chars


def test_determine_next_focus_logic(prompt_generator):
    """Test _determine_next_focus returns appropriate focus based on previous results"""
    # Test initial state
    assert prompt_generator._determine_next_focus(None) == "Initial basic configuration"
    assert prompt_generator._determine_next_focus({}) == "Initial basic configuration"

    # Test with various discovery results
    discoveries = [{"instance": "id12345", "state": "running"}, {"security_groups": [{"id": "sg-123", "rules": []}]}, {"vpc": {"id": "vpc-456", "cidr": "10.0.0.0/16"}}]

    for discovery in discoveries:
        result = prompt_generator._determine_next_focus(discovery)
        assert result == "Next logical configuration based on previous findings"
        assert isinstance(result, str)
        assert len(result) > 0


def test_get_iteration_goals_different_iterations(prompt_generator):
    """Test _get_iteration_goals returns different goals based on iteration number"""
    first_iteration = prompt_generator._get_iteration_goals(1, "ec2")
    assert "FIRST most relevant piece" in first_iteration
    assert "ec2" in first_iteration

    later_iteration = prompt_generator._get_iteration_goals(2, "ec2")
    assert "previously discovered information" in later_iteration
    assert "next logical check" in later_iteration


def test_parse_response_handles_complex_yaml(prompt_generator):
    """Test _parse_response handles complex YAML responses with various formats and content"""
    # Test response with markdown code blocks and complex template
    response_with_blocks = Mock(
        content="""```yaml
template: |
    Generate code to analyze EC2 instances in ${region}:
    1. List all instances with tag ${tag_key}=${tag_value}
    2. Check their security groups
    3. Output findings in JSON format
description: Analyze EC2 instances matching specific tags
variables:
  - name: region
    description: AWS region to analyze
    default_value: us-west-2
  - name: tag_key
    description: Tag key to filter instances
  - name: tag_value
    description: Expected tag value
success_criteria: |
    - All instances matching tags are found
    - Security group rules are analyzed
    - Results are properly formatted
```"""
    )

    template, vars, criteria, description = prompt_generator._parse_response(response_with_blocks)
    assert "Generate code to analyze EC2 instances" in template
    assert len(vars) == 3
    assert any(v["name"] == "region" and v["value"] == "us-west-2" for v in vars)
    assert "All instances matching tags are found" in criteria

    # Test response with minimal required fields
    minimal_response = Mock(content="template: Simple template\ndescription: Brief\nvariables: []\nsuccess_criteria: Works")
    template, vars, criteria, description = prompt_generator._parse_response(minimal_response)
    assert template == "Simple template"
    assert vars == []
    assert criteria == "Works"
    assert description == "Brief"


def test_parse_response_error_handling(prompt_generator):
    """Test _parse_response handles various error cases and malformed responses"""
    # Test completely invalid YAML
    invalid_yaml = Mock(content="{invalid: [yaml: :]}")
    with pytest.raises(ValueError, match="Failed to extract template"):
        prompt_generator._parse_response(invalid_yaml)

    # Test missing required fields
    missing_template = Mock(content="description: Test\nvariables: []\nsuccess_criteria: Test")
    with pytest.raises(ValueError):
        prompt_generator._parse_response(missing_template)

    # Test malformed variables section
    malformed_vars = Mock(
        content="""template: Test
description: Test
variables:
  - invalid_var
  - name: valid_var
    description: test
success_criteria: Test"""
    )
    template, vars, _, _ = prompt_generator._parse_response(malformed_vars)
    assert len(vars) == 1  # Should only parse the valid variable
    assert vars[0]["name"] == "valid_var"


def test_generate_prompt_removes_response_format(prompt_generator, mock_model):
    """Test generate_prompt removes response_format from model_kwargs"""
    mock_model.invoke.return_value = Mock(
        content="""
template: Test
description: Test
variables: []
success_criteria: Test
"""
    )

    prompt_generator.generate_prompt(model_name="test", service="ec2", operation="test", request="test", variables=[])

    assert "response_format" not in mock_model.model_kwargs


def test_generate_prompt_variable_handling(prompt_generator, mock_model):
    """Test generate_prompt handles complex variable scenarios"""
    # Test merging variables with defaults and overrides
    mock_model.invoke.return_value = Mock(
        content="""template: Test
description: Test
variables:
  - name: existing_var
    description: updated description
    default_value: new_default
  - name: new_var
    description: new variable
    default_value: default_value
  - name: required_var
    description: required variable
success_criteria: Test"""
    )

    existing_vars = [{"name": "existing_var", "value": "custom_value"}, {"name": "other_var", "value": "keep_this"}]

    result = prompt_generator.generate_prompt(model_name="test", service="test", operation="test", request="test", variables=existing_vars)

    # Verify variable merging logic
    assert len(result.variables) == 4
    for var in result.variables:
        if var["name"] == "existing_var":
            assert var["value"] == "custom_value"  # Should keep existing value
        elif var["name"] == "new_var":
            assert var["value"] == "default_value"  # Should use default
        elif var["name"] == "required_var":
            assert var["value"] == ""  # Should have empty value
        elif var["name"] == "other_var":
            assert var["value"] == "keep_this"  # Should preserve unrelated var


def test_generate_prompt_handles_feedback_and_previous_results(prompt_generator, mock_model):
    """Test generate_prompt incorporates feedback and previous results into the prompt"""
    mock_model.invoke.return_value = Mock(
        content="""
template: Test template with {previous_results} and {feedback}
description: Test
variables: []
success_criteria: Test
"""
    )

    previous_results = {"key": "value"}
    feedback = {"feedback": "test"}

    result = prompt_generator.generate_prompt(model_name="test", service="test", operation="test", request="test", variables=[], previous_results=previous_results, feedback=feedback)

    assert result.template is not None
    # Verify that the context was passed to the model
    messages = mock_model.invoke.call_args[0][0]
    assert yaml.dump(previous_results) in messages[1]["content"]
    assert yaml.dump(feedback) in messages[1]["content"]
