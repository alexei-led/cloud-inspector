from datetime import datetime

import pytest

from cloud_inspector.components.types import CloudProvider


def test_prompt_generator_basic(prompt_generator, mock_llm_response):
    """Test basic prompt generation."""
    variables = [{"name": "region", "value": "us-west-2"}]

    prompt = prompt_generator.generate_prompt(model_name="test-model", service="ec2", operation="list", request="List EC2 instances", variables=variables, cloud=CloudProvider.AWS, iteration=1)

    assert prompt.service == "ec2"
    assert prompt.cloud == CloudProvider.AWS
    assert prompt.generated_by == "test-model"
    assert isinstance(prompt.generated_at, datetime)
    assert any(v["name"] == "region" for v in prompt.variables)
    assert "Generate Python code" in prompt.template


def test_prompt_generator_with_previous_results(prompt_generator):
    """Test prompt generation with previous results."""
    previous_results = {"instances": [{"id": "i-123", "state": "running"}]}

    prompt = prompt_generator.generate_prompt(model_name="test-model", service="ec2", operation="list", request="List EC2 instances", variables=[], cloud=CloudProvider.AWS, previous_results=previous_results, iteration=2)

    assert prompt.template
    assert prompt.success_criteria
    assert prompt.description


def test_prompt_generator_error_handling(prompt_generator, model_registry, mock_model):
    """Test error handling in prompt generation."""
    mock_model.invoke.side_effect = Exception("API Error")
    model_registry.get_model.return_value = mock_model

    with pytest.raises(ValueError, match="Failed to extract template"):
        prompt_generator.generate_prompt(model_name="test-model", service="ec2", operation="list", request="List EC2 instances", variables=[], cloud=CloudProvider.AWS)
