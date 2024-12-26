"""Prompt management system for Cloud Inspector."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field


class PromptTemplate(BaseModel):
    """Single prompt template definition."""

    service: str = Field(..., description="AWS service (e.g., ec2, s3)")
    operation: str = Field(..., description="Operation type (e.g., list, analyze)")
    description: str = Field(
        ..., description="Brief description of what the prompt does"
    )
    template: str = Field(..., description="The actual prompt template")
    variables: List[str] = Field(
        default_factory=list, description="Variables required in the template"
    )
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")

    def format_messages(self, variables: Dict[str, Any]) -> List[Any]:
        """Format the prompt into chat messages."""
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert AWS developer. Your task is to generate Python code using boto3 for AWS operations.

For any code generation task, you MUST provide three files:
1. main.py - The main Python script using the latest boto3
2. requirements.txt - All required dependencies with versions
3. policy.json - IAM Policy with minimum required permissions

Requirements for code generation:
1. Use proper error handling with informative error messages
2. Include all necessary imports
3. Follow AWS security best practices
4. Add comprehensive type hints
5. Include detailed docstrings with parameters and return types
6. Use clear variable names and add comments for complex logic
7. Make code modular and reusable
8. Include logging for important operations
9. Validate input parameters
10. Handle AWS credentials properly (never hardcode)

Additional Considerations:
- The code will be used by both humans and AI agents
- Output should be clear and well-documented
- Include example usage in docstring
- Add any relevant security or cost warnings
- Specify required IAM permissions""",
                ),
                ("user", self.template),
            ]
        )
        return prompt.format_messages(**variables)


class PromptCollection(BaseModel):
    """Collection of prompt templates."""

    prompts: Dict[str, PromptTemplate]


class PromptManager:
    """Manager for handling prompt templates."""

    def __init__(self, prompt_dir: Optional[Path] = None):
        self.prompt_dir = prompt_dir or Path("prompts")
        self.prompts: Dict[str, PromptTemplate] = {}
        self._load_prompts()

    def _load_prompts(self) -> None:
        """Load all prompt files from the prompt directory."""
        if not self.prompt_dir.exists():
            return

        for file in self.prompt_dir.glob("*.yaml"):
            try:
                with file.open("r") as f:
                    data = yaml.safe_load(f)
                    collection = PromptCollection(prompts=data.get("prompts", {}))
                    self.prompts.update(collection.prompts)
            except Exception as e:
                print(f"Error loading prompts from {file}: {e}")

    def get_prompt(self, name: str) -> Optional[PromptTemplate]:
        """Get a prompt template by name."""
        return self.prompts.get(name)

    def list_prompts(self) -> List[Dict[str, Any]]:
        """List all available prompts with their details."""
        return [
            {
                "name": name,
                "service": prompt.service,
                "operation": prompt.operation,
                "description": prompt.description,
                "variables": prompt.variables,
                "tags": prompt.tags,
            }
            for name, prompt in self.prompts.items()
        ]

    def get_prompts_by_service(self, service: str) -> List[str]:
        """Get all prompt names for a specific service."""
        return [
            name
            for name, prompt in self.prompts.items()
            if prompt.service.lower() == service.lower()
        ]

    def get_prompts_by_tag(self, tag: str) -> List[str]:
        """Get all prompt names with a specific tag."""
        return [
            name
            for name, prompt in self.prompts.items()
            if tag.lower() in [t.lower() for t in prompt.tags]
        ]

    def get_all_services(self) -> Set[str]:
        """Get all unique AWS services in the prompts."""
        return {prompt.service for prompt in self.prompts.values()}

    def get_all_tags(self) -> Set[str]:
        """Get all unique tags from all prompts."""
        return {tag for prompt in self.prompts.values() for tag in prompt.tags}

    def format_prompt(
        self, name: str, variables: Dict[str, Any]
    ) -> Optional[List[Any]]:
        """Format a prompt template with provided variables."""
        prompt = self.get_prompt(name)
        if not prompt:
            return None

        # Validate all required variables are provided
        missing_vars = set(prompt.variables) - set(variables.keys())
        if missing_vars:
            raise ValueError(f"Missing required variables: {missing_vars}")

        try:
            return prompt.format_messages(variables)
        except KeyError as e:
            raise ValueError(f"Invalid variable in template: {e}")
        except Exception as e:
            raise ValueError(f"Error formatting prompt: {e}")

    def validate_prompt_file(self, file_path: Path) -> List[str]:
        """Validate a prompt file format and content."""
        errors = []
        try:
            with file_path.open("r") as f:
                data = yaml.safe_load(f)

            if not isinstance(data, dict):
                errors.append("Root element must be a dictionary")
                return errors

            if "prompts" not in data:
                errors.append("Missing 'prompts' key in root")
                return errors

            prompts = data["prompts"]
            if not isinstance(prompts, dict):
                errors.append("'prompts' must be a dictionary")
                return errors

            for name, prompt_data in prompts.items():
                try:
                    PromptTemplate(**prompt_data)
                except Exception as e:
                    errors.append(f"Invalid prompt '{name}': {str(e)}")

        except yaml.YAMLError as e:
            errors.append(f"Invalid YAML format: {str(e)}")
        except Exception as e:
            errors.append(f"Error validating file: {str(e)}")

        return errors
