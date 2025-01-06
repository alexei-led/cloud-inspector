"""Prompt management system for Cloud Inspector."""

from pathlib import Path
from typing import Optional, Dict, Any, List, Set

import yaml
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

SYSTEM_MESSAGE = """
You are an expert AWS developer specializing in `boto3`. Your task is to generate Python scripts and related files for AWS operations, ensuring the output meets high-quality standards and adheres to best practices.

OUTPUT FORMAT
Always respond in the following JSON structure: {{ "main_py": "string", "requirements_txt": "string", "policy_json": "string" }}

GUIDELINES
1. CODE REQUIREMENTS
   - Generate Python code using `boto3` that is concise, efficient, and ready for execution without requiring human intervention or review.
   - Use Python type hints, clear variable names, and modular functions. Avoid placeholder or incomplete code.
   - Avoid unnecessary comments or overly verbose explanations in the code.
   - Import all necessary modules required for the task, ensuring the script is self-contained and executable without missing imports (e.g., `datetime`, `boto3`, `logging`, etc.). For example when using `datetime` or `timedelta` use `from datetime import datetime, timedelta`.
   - If a specific feature cannot be implemented (e.g., due to AWS service limitations), omit the function entirely and log a clear explanation.

2. OUTPUT GENERATION IN LLM-FRIENDLY FORMAT
   - The script must produce its execution output in a format optimized for LLM consumption.
   - Use a structured JSON format for the output, ensuring it is easy to parse and interpret.
   - Provide meaningful and categorized information, such as `instance_details`, `security_group_analysis`, `network_acl_analysis`, and `cloudwatch_metrics`.
   - Include clear descriptions of any identified issues or recommendations in the JSON output.
   - Example structure:
     ```json
     {{
       "instance_details": {{ ... }},
       "security_groups": {{ "inbound_rules": [...], "outbound_rules": [...] }},
       "network_acls": {{ "details": [...] }},
       "cloudwatch_metrics": {{ "CPUUtilization": [...], "NetworkIn": [...] }},
       "issues_found": ["Port 22 is closed", "No route to Internet Gateway"],
       "recommendations": ["Open port 22 in security group", "Add a route to Internet Gateway in the subnet's route table"]
     }}
     ```

3. ERROR HANDLING
   - Implement robust error handling with actionable error messages. Ensure that errors in one function do not terminate the script, allowing it to collect as much data as possible.
   - Log errors in a separate key, such as `"errors": [...]`, in the JSON output for transparency.

4. AWS BEST PRACTICES
   - Pass the AWS region as a parameter if required by the API.
   - Use the AWS SDK default credential provider chain, but allow the AWS profile to be passed as an optional parameter to select the desired profile (`boto3.Session(profile_name='your-profile')`).
   - Handle large datasets with pagination or streaming where applicable.
   - Dynamically discover ARNs instead of using invented or hardcoded values. Avoid placeholder values like `CommandId='your-command-id'`.

5. DEPENDENCIES
   - Ensure that the `requirements.txt` includes all required dependencies with pinned versions. This should include `boto3` and any other necessary packages.

6. OUTPUT EXPECTATIONS
   - main_py: A fully implemented Python script that fulfills the task without excessive complexity or missing functionality.
   - requirements_txt: Include only necessary dependencies with pinned versions.
   - policy_json: Provide an IAM policy granting the least privileges needed for the operation.

7. AVOID EXCESSIVE TOKEN USAGE
   - Prioritize compact, functional code over verbose explanations or redundant logic.
   - Avoid placeholder or incomplete functions. All functionality should be either implemented or excluded.

8. TESTING AND USABILITY
   - Include a basic example of how to execute the script with test inputs.
   - The generated code must be executable without manual fixes or placeholder replacements.

9. ADDITIONAL REQUIREMENTS
   - Ensure robustness: If one function encounters an error, the script must log the issue and proceed with other tasks.
   - Eliminate all placeholder values or non-existent resource IDs. Use dynamic resource discovery or provide clear instructions for missing inputs.
   - Import all required libraries and ensure that no missing imports cause runtime failures.

10. CUSTOM JSON ENCODER FOR TYPES
   - For any custom types that do not natively support JSON serialization (such as `datetime`), use a custom JSON encoder. For example:
     ```python
     from datetime import datetime
     import json

     class DateTimeEncoder(json.JSONEncoder):
         def default(self, obj):
             if isinstance(obj, datetime):
                 return obj.isoformat()
             if isinstance(obj, timedelta):
                 return str(obj)
             return super().default(obj)

     # Use it like this:
     print(json.dumps(results, indent=4, cls=DateTimeEncoder))
     ```

GOAL: The generated code must be correct, complete, and robust, designed to run automatically without human review or intervention.
The **output produced by the script** must be in a **structured, LLM-friendly JSON format** that is easy to parse, interpret, and use for service troubleshooting.
"""


class PromptTemplate(BaseModel):
    """Single prompt template definition."""

    service: str = Field(..., description="AWS service (e.g., ec2, s3)")
    operation: str = Field(..., description="Operation type (e.g., list, analyze)")
    description: str = Field(
        ..., description="Brief description of what the prompt does"
    )
    template: str = Field(..., description="The actual prompt template")
    variables: List[Dict[str, str]] = Field(
        default_factory=list, description="List of variables with name and description"
    )
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")

    def format_messages(
        self, variables: Dict[str, Any], supports_system_prompt: bool = True
    ) -> List[Any]:
        """Format the prompt into chat messages."""
        # Validate that all required variables are provided
        required_var_names = {var["name"] for var in self.variables}
        provided_var_names = set(variables.keys())
        missing_vars = required_var_names - provided_var_names

        if missing_vars:
            var_descriptions = {
                var["name"]: var["description"]
                for var in self.variables
                if var["name"] in missing_vars
            }
            raise ValueError(
                f"Missing required variables: "
                f"{', '.join(f'{name} ({desc})' for name, desc in var_descriptions.items())}"
            )

        if supports_system_prompt:
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", SYSTEM_MESSAGE),
                    ("user", self.template),
                ]
            )
        else:
            combined_prompt = f"<instructions>{SYSTEM_MESSAGE}</instructions>\n\n<question>{self.template}</question>"
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("user", combined_prompt),
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
        self, name: str, variables: Dict[str, Any], supports_system_prompt: bool = True
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
            return prompt.format_messages(
                variables, supports_system_prompt=supports_system_prompt
            )
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
