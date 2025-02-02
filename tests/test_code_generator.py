from datetime import datetime

import pytest

from cloud_inspector.components.types import CloudProvider, CodeGenerationPrompt


def test_format_prompt(code_generator):
    """Test prompt formatting with valid inputs."""
    prompt = CodeGenerationPrompt(
        service="ec2",
        operation="list",
        request="List EC2 instances",
        description="List EC2 instances",
        template="List EC2 instances in region {region}",
        variables=[{"name": "region", "description": "AWS region"}],
        cloud=CloudProvider.AWS,
        generated_by="test",
        generated_at=datetime.strptime("2024-01-01", "%Y-%m-%d"),
        success_criteria="Instances are listed successfully",
    )

    messages = code_generator.format_prompt(prompt=prompt, variables={"region": "us-west-2"}, supports_system_prompt=True)

    assert len(messages) == 2
    assert "us-west-2" in messages[1].content


def test_format_prompt_missing_variables(code_generator):
    """Test prompt formatting with missing required variables."""
    prompt = CodeGenerationPrompt(
        service="ec2",
        operation="list",
        request="List EC2 instances",
        template="List EC2 instances in region {region}",
        variables=[{"name": "region", "description": "AWS region"}],
        cloud=CloudProvider.AWS,
        description="List EC2 instances",
        generated_by="test",
        generated_at=datetime.strptime("2024-01-01", "%Y-%m-%d"),
        success_criteria="Instances are listed successfully",
    )

    with pytest.raises(ValueError, match="Missing required variables"):
        code_generator.format_prompt(prompt, variables={})
