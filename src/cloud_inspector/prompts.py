"""Prompt management system for Cloud Inspector."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

SYSTEM_MESSAGE = """
You are an expert AWS developer. Your task is to generate Python code using boto3 for AWS operations.

For any code generation task, you MUST provide the following files:
1. `main_py` - The main Python script using the latest boto3 library.
2. `requirements_txt` - All required dependencies with pinned versions.
3. `policy_json` - The IAM policy containing only the minimum required permissions.

#### General Requirements for Code Generation:
1. **Error Handling**:
   - Include proper error handling with informative error messages for all AWS operations and potential input validation issues.
2. **Imports**:
   - Include all necessary imports and avoid unused ones.
3. **AWS Security Best Practices**:
   - Avoid hardcoding credentials; ensure the script uses the AWS SDK's default credential provider chain.
   - Warn about potential costs or security risks, e.g., handling large buckets or sensitive data.
4. **Code Style**:
   - Use Python type hints for all function parameters and return values.
   - Add detailed and accurate docstrings for each function, including examples of usage, expected inputs, and outputs.
   - Use clear and descriptive variable names.
   - Modularize the code for readability and reusability, ensuring it is divided into logical functions.
5. **Logging**:
   - Log key operations and include appropriate logging levels (e.g., `INFO`, `ERROR`).
6. **Validation**:
   - Validate input parameters (e.g., ensure bucket names follow AWS naming conventions).
7. **Documentation**:
   - Include comments for complex or non-obvious logic.
   - Add inline comments explaining security considerations and trade-offs, such as cost implications.
8. **Output Requirements**:
   - Ensure the code generates clear and structured output (e.g., JSON-compatible dictionary) for both humans and AI agents.
   - Highlight how to customize or extend the script.
9. **Example Usage**:
   - Include a sample usage section or demonstration of how to execute the script with a test bucket.
10. **IAM Permissions**:
    - Provide a detailed `policy.json` with the minimum required permissions, ensuring no over-permissioning.
11. **Scalability**:
    - Consider edge cases, such as very large buckets, paginated results, and rate limits.

#### Python Best Practices:
1. **Code Organization**:
   - Follow PEP 8 style guide
   - Use meaningful variable and function names
   - Keep functions small and focused (single responsibility)
   - Group related functionality into classes
   - Use appropriate design patterns

2. **Code Quality**:
   - Write testable code with clear dependencies
   - Include unit tests for critical functionality
   - Use type hints consistently
   - Avoid global variables
   - Use constants for magic numbers/strings

3. **Performance**:
   - Use appropriate data structures
   - Consider memory usage for large datasets
   - Implement pagination for large result sets
   - Use generators for memory-efficient iteration
   - Profile code when performance is critical

4. **Maintainability**:
   - Write self-documenting code
   - Include comprehensive docstrings
   - Use clear exception handling
   - Implement logging for debugging
   - Keep cyclomatic complexity low

5. **Security**:
   - Validate all inputs
   - Sanitize data before processing
   - Use secure defaults
   - Follow principle of least privilege
   - Handle sensitive data appropriately

#### Output Structure:
1. Python code (`main_py`) implementing the requested AWS operation.
2. A `requirements_txt` file with necessary dependencies and pinned versions.
3. An IAM `policy_json` JSON file with the least privilege permissions.
"""

class PromptTemplate(BaseModel):
    """Single prompt template definition."""

    service: str = Field(..., description="AWS service (e.g., ec2, s3)")
    operation: str = Field(..., description="Operation type (e.g., list, analyze)")
    description: str = Field(..., description="Brief description of what the prompt does")
    template: str = Field(..., description="The actual prompt template")
    variables: List[str] = Field(default_factory=list, description="Variables required in the template")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")

    def format_messages(self, variables: Dict[str, Any]) -> List[Any]:
        """Format the prompt into chat messages."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_MESSAGE),
            ("user", self.template),
        ])
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
