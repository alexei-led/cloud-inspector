import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml
from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from cloud_inspector.prompts import CloudProvider, PromptType
from langchain_components.models import ModelCapability, ModelRegistry


class GeneratedPrompt(BaseModel):
    """Structure for generated prompt."""

    service: str
    operation: str
    description: str
    template: str
    variables: list[dict[str, str]]  # Each dict has 'name' and 'description' keys
    tags: list[str]
    cloud: CloudProvider
    prompt_type: PromptType = PromptType.GENERATED
    generated_by: Optional[str] = None
    generated_at: Optional[datetime] = None


class PromptGenerator:
    """Generates prompts for cloud service operations."""

    def __init__(self, model_registry: ModelRegistry, output_dir: Optional[Path] = None):
        self.model_registry = model_registry
        self.output_dir = output_dir or Path("generated_prompts")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_prompt(
        self,
        model_name: str,
        service: str,
        request: str,
        cloud: CloudProvider = CloudProvider.AWS,
        **kwargs,
    ) -> tuple[GeneratedPrompt, Path]:
        """Generate a prompt template from a natural language request."""

        # Validate model can generate prompts
        if not self.model_registry.validate_model_capability(model_name, ModelCapability.PROMPT_GENERATION):
            raise ValueError(f"Model '{model_name}' does not support prompt generation")

        # Get the model and its config
        model = self.model_registry.get_model(model_name)
        model_config = self.model_registry.models.get(model_name)

        # remove response_format from kwargs
        model.model_kwargs = {k: v for k, v in model.model_kwargs.items() if k != "response_format"}

        if not model_config:
            raise ValueError(f"Model config not found for '{model_name}'")

        # Create the system message
        system_message = f"""
Act as a professional {cloud} Python DevOps engineer.
Your task is to inspect the customer request and generate a prompt that will:
1. Use provided information without making assumptions
2. Include code logic to discover missing but required information
3. Handle multiple possible configurations/services

IMPORTANT: Always use ${{variable_name}} syntax for variables in description and template sections.
Example: "Investigate ${{bucket_name}} in ${{region}}" (NOT using {{variable_name}})

Variable Handling Rules:
1. Only create variables for explicitly provided values in the request (IDs, names, ARNs, regions, zones) or those that are needed but cannot be discovered
2. For missing but required information, include discovery logic in the template
2. Replace values with ${{variable_name}} format in description and template
3. Do NOT include variables for cloud credentials (handled by runtime environment)
4. Define all required variables in the variables section

The prompt should help generate code that:
1. Uses {cloud} API (Python) to fetch all relevant configuration, logs, and monitoring data
2. Produces correct, working, production-ready Python code that can be executed directly
3. Follows best practices for error handling and logging
4. Returns data in a structured JSON format for easy analysis

The generated prompt should follow this structure (YAML format):
service: The {cloud} service (e.g., ec2, s3)
operation: Type of operation (e.g., troubleshoot, analyze)
description: Clear description using {{variable_name}} syntax for known variables
template: |
  A clear, concise list of requirements using {{variable_name}} syntax:
  1. Core Functionality
     - First list steps to discover missing information
     - Then list steps to analyze the discovered configuration
     - Use ${{variable_name}} syntax for variables
     - Use cloud API calls to fetch data
     - Collect all relevant data for analysis
  2. Error Handling & Logging
     - Handle errors for each discovery and analysis step
     - Log all discovered information
  3. Output Format
     - JSON structure showing both discovered and analyzed information
variables:
  - name: variable_name # Without ${{ }} syntax in name
    description: Clear description of what this variable represents and any constraints
  # Add all required inputs specific to this service and operation
  # Example:
  # - name: instance_id
  #   description: ID of the instance to analyze
  # - name: region
  #   description: Region where the resources are located
tags: Relevant tags for categorization (as a list)
"""

        # Create the user message
        user_message = f"""
Cloud Service: {service}
Customer Request: {request}

Generate a prompt template for a Python script that will:
1. Connect to {cloud} {service} service
2. Collect and analyze relevant data
3. Return findings in JSON format

Focus on:
1. Required permissions
2. Key data points to collect
3. Success/failure criteria
4. Output structure
5. Use of variables (like {{variable}}) for dynamic values in template
"""

        # Format messages based on system prompt support
        if model_config.supports_system_prompt:
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ]
        else:
            combined_message = f"""
<instructions>
{system_message}
</instructions>

<request>
{user_message}
</request>
"""
            messages = [{"role": "user", "content": combined_message}]

        # Generate the prompt
        response = model.invoke(messages)

        # Parse the response into a GeneratedPrompt
        parsed_prompt = self._parse_response(response)

        # Save the generated prompt and get the path
        saved_path = self._save_prompt(model_name, parsed_prompt)

        return parsed_prompt, saved_path

    def _parse_response(self, response: BaseMessage) -> GeneratedPrompt:
        """Parse the model response into a GeneratedPrompt structure."""
        try:
            # Extract content from the response
            content = response.content if hasattr(response, "content") else str(response)

            # Remove yaml code block markers if present
            content = content.replace("```yaml", "").replace("```", "").strip()

            # Parse YAML content
            data = yaml.safe_load(content)

            # Validate variable usage
            description = data.get("description", "").strip()
            template = data.get("template", "").strip()
            variables = data.get("variables", [])

            # Extract all {{variable}} patterns from description and template
            variable_pattern = r"\$\{([^}]+)\}"
            used_variables = set(re.findall(variable_pattern, description + template))

            # Validate all used variables are defined
            defined_variables = {v["name"] for v in variables}
            undefined_vars = used_variables - defined_variables
            if undefined_vars:
                raise ValueError(f"Variables used but not defined: {undefined_vars}")

            # Convert to GeneratedPrompt
            return GeneratedPrompt(
                service=data.get("service", "").strip(),
                operation=data.get("operation", "").strip(),
                description=data.get("description", "").strip(),
                template=data.get("template", "").strip(),
                variables=data.get("variables", []),
                tags=data.get("tags", []),
                cloud=CloudProvider(data.get("cloud", "aws").lower()),
                prompt_type=PromptType.GENERATED,
                generated_by=data.get("generated_by", "").strip(),
                generated_at=datetime.now().isoformat(),
            )
        except Exception as e:
            raise ValueError(f"Failed to parse model response: {e}") from e

    def _save_prompt(self, model_name: str, prompt: GeneratedPrompt) -> Path:
        """Save the generated prompt to a YAML file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create a unique name for the prompt
        prompt_name = f"{prompt.service}_{prompt.operation}_{model_name}_{timestamp}"
        filename = f"prompt_{prompt_name}.yaml"

        prompt_data = {
            "prompts": {
                prompt_name: {
                    "service": prompt.service,
                    "operation": prompt.operation,
                    "description": prompt.description,
                    "template": prompt.template,
                    "variables": prompt.variables,
                    "tags": prompt.tags,
                    "cloud": prompt.cloud.value,
                    "prompt_type": "generated",
                    "generated_by": model_name,
                    "generated_at": datetime.now().isoformat(),
                }
            }
        }

        # Add custom representer for str to handle multiline strings
        yaml.SafeDumper.org_represent_str = yaml.SafeDumper.represent_str

        def repr_str(dumper, data):
            if "\n" in data:
                return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
            return dumper.org_represent_str(data)

        yaml.add_representer(str, repr_str, Dumper=yaml.SafeDumper)

        output_path = self.output_dir / filename
        with output_path.open("w") as f:
            yaml.safe_dump(prompt_data, f, sort_keys=False, allow_unicode=True)

        return output_path
