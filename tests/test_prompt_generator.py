from datetime import datetime
from unittest.mock import Mock

import pytest

from cloud_inspector.components.models import ModelRegistry
from cloud_inspector.components.types import CloudProvider, CodeGenerationPrompt
from cloud_inspector.prompt_generator import PromptGeneratorAgent


@pytest.fixture
def mock_model():
    model = Mock()
    model.invoke.return_value = Mock(
        content="""
template: |
    Discover EC2 instances in the specified region
    Previous findings: {previous_results}
    Focus on: {next_focus}
description: Initial EC2 instance discovery
variables:
    - name: region
      description: AWS region to search in
success_criteria: Successfully retrieved list of EC2 instances
"""
    )
    return model


@pytest.fixture
def model_registry(mock_model):
    registry = Mock(spec=ModelRegistry)
    registry.get_model.return_value = mock_model
    return registry


@pytest.fixture
def prompt_generator(model_registry):
    return PromptGeneratorAgent(model_registry=model_registry)


def test_generate_prompt_initial(prompt_generator):
    """Test initial prompt generation without previous results."""
    variables = [{"name": "region", "value": "us-west-2"}]

    result = prompt_generator.generate_prompt(
        model_name="test-model",
        service="ec2",
        operation="describe_instances",
        request="List all EC2 instances",
        variables=variables,
        cloud=CloudProvider.AWS,
    )

    assert isinstance(result, CodeGenerationPrompt)
    assert result.service == "ec2"
    assert result.operation == "describe_instances"
    assert result.cloud == CloudProvider.AWS
    assert result.generated_at <= datetime.now()  # type: ignore
    assert result.generated_by == "test-model"
    assert len(result.variables) >= 1
    assert any(v["name"] == "region" for v in result.variables)


def test_generate_prompt_with_previous_results(prompt_generator):
    """Test prompt generation with previous discovery results."""
    previous_results = {
        "Instances": [
            {"InstanceId": "i-1234", "State": {"Name": "running"}},
        ]
    }
    variables = [{"name": "region", "value": "us-west-2"}]

    result = prompt_generator.generate_prompt(
        model_name="test-model",
        service="ec2",
        operation="describe_instances",
        request="Get instance details",
        variables=variables,
        previous_results=previous_results,
        iteration=2,
    )

    assert result.template is not None
    assert "Previous findings" in result.template
    assert result.success_criteria is not None
    assert result.iteration == 2


def test_generate_prompt_with_feedback(prompt_generator):
    """Test prompt generation with user feedback."""
    feedback = {"focus": "security_groups"}
    variables = [{"name": "instance_id", "value": "i-1234"}]

    result = prompt_generator.generate_prompt(
        model_name="test-model",
        service="ec2",
        operation="describe_instances",
        request="Check instance security",
        variables=variables,
        feedback=feedback,
    )

    assert result.template is not None
    assert result.feedback == feedback


def test_variable_merging(prompt_generator):
    """Test that new variables are properly merged with existing ones."""
    existing_variables = [
        {"name": "region", "value": "us-west-2"},
    ]

    result = prompt_generator.generate_prompt(
        model_name="test-model",
        service="ec2",
        operation="describe_instances",
        request="List instances",
        variables=existing_variables,
    )

    assert len(result.variables) >= len(existing_variables)
    assert any(v["name"] == "region" for v in result.variables)


def test_invalid_response_handling(prompt_generator, mock_model):
    """Test handling of invalid model responses."""
    mock_model.invoke.return_value = Mock(content="invalid yaml: :")

    with pytest.raises(ValueError, match="Failed to extract template"):
        prompt_generator.generate_prompt(
            model_name="test-model",
            service="ec2",
            operation="describe_instances",
            request="List instances",
            variables=[],
        )


def test_empty_variables_handling(prompt_generator):
    """Test handling of empty variables list."""
    result = prompt_generator.generate_prompt(
        model_name="test-model",
        service="ec2",
        operation="describe_instances",
        request="List instances",
        variables=[],
    )

    assert isinstance(result.variables, list)
    assert result.template is not None


def test_different_cloud_provider(prompt_generator):
    """Test prompt generation for different cloud provider."""
    result = prompt_generator.generate_prompt(
        model_name="test-model",
        service="compute",
        operation="list",
        request="List VMs",
        variables=[],
        cloud=CloudProvider.AZURE,
    )

    assert result.cloud == CloudProvider.AZURE
    assert result.service == "compute"
