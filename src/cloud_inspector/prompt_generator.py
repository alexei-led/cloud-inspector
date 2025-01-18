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
    """Structure for generated prompt with discovery tracking capabilities."""

    # Core prompt fields
    service: str
    operation: str
    description: str
    template: str
    variables: list[dict[str, str]]  # Each dict has 'name' and 'description' keys
    tags: list[str]
    cloud: CloudProvider
    prompt_type: PromptType = PromptType.GENERATED

    # Discovery tracking
    discovered_resources: list[dict] = []  # List of resources discovered by this prompt
    dependencies: list[str] = []  # List of resource IDs this prompt depends on
    next_discovery_targets: list[str] = []  # Suggested next resources to discover
    discovery_complete: bool = False  # True if no more discoveries needed for this context

    # Metadata
    generated_by: Optional[str] = None
    generated_at: Optional[str] = None
    iteration: int = 1  # Track which iteration this prompt belongs to
    parent_request_id: Optional[str] = None  # Link to original user request


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
        discovered_data: Optional[dict] = None,
        iteration: int = 1,
        parent_request_id: Optional[str] = None,
        cloud: CloudProvider = CloudProvider.AWS,
        **kwargs,
    ) -> tuple[GeneratedPrompt, Path]:
        """Generate a prompt template from a natural language request
        Args:
            model_name: Name of the model to use
            service: Cloud service to interact with
            request: Original user request
            discovered_data: Previously discovered information (optional)
            iteration: Current iteration number
            parent_request_id: ID of the original user request
            cloud: Cloud provider to use
            **kwargs: Additional model-specific arguments
        """

        # Validate model can generate prompts
        if not self.model_registry.validate_model_capability(model_name, ModelCapability.PROMPT_GENERATION):
            raise ValueError(f"Model '{model_name}' does not support prompt generation")

        # Get the model and its config
        model = self.model_registry.get_model(model_name)
        model_config = self.model_registry.models.get(model_name)

        # remove response_format from kwargs
        model.model_kwargs = {k: v for k, v in model.model_kwargs.items() if k != "response_format"}  # type: ignore

        if not model_config:
            raise ValueError(f"Model config not found for '{model_name}'")

        # Create discovery context from previously discovered data
        discovery_context = ""
        if discovered_data:
            discovery_context = "Previously Discovered Information:\n"
            for resource_type, data in discovered_data.items():
                discovery_context += f"- {resource_type}:\n"
                if isinstance(data, list):
                    for item in data:
                        discovery_context += f"  - {str(item)}\n"
                else:
                    discovery_context += f"  {str(data)}\n"

        # Create the system message
        system_message = f"""
Act as a professional {cloud} Python DevOps engineer specializing in troubleshooting and discovery.
Your task is to generate the NEXT most relevant prompt for discovering cloud resources and their state.

Context Awareness:
1. Consider already discovered information (if any) before suggesting next steps
2. Follow logical troubleshooting progression (e.g., for connectivity: instance → security groups → VPC → routing)
3. Focus on ONE specific discovery task at a time - don't try to discover everything at once

Discovery Priorities:
1. Core Configuration - Essential resource settings and state
2. Related Resources - Directly connected or dependent resources
3. Recent Changes - Audit logs showing relevant modifications
4. Performance Data - Key metrics indicating resource health
5. Operational Logs - Detailed logs for troubleshooting

IMPORTANT: Always use ${{variable_name}} syntax for variables in description and template sections.
Example: "Investigate ${{bucket_name}} in ${{region}}" (NOT using {{variable_name}})

Variable Handling Rules:
1. Only create variables for explicitly provided values or those that cannot be discovered
2. For missing but required information, include discovery logic in the template
3. Replace values with ${{variable_name}} format in description and template
4. Do NOT include variables for cloud credentials (handled by runtime)

The prompt should help generate code that:
1. Uses {cloud} API (Python) to fetch specific data points
2. Produces correct, working, production-ready Python code
3. Follows best practices for error handling and logging
4. Returns data in a structured JSON format for analysis

The generated prompt must follow this structure (YAML format):
service: The {cloud} service (e.g., ec2, s3)
operation: Type of operation (e.g., discover_config, fetch_logs)
description: Clear description using ${{variable_name}} syntax for variables
template: |
  A clear, focused list of requirements:
  1. Core Functionality
     - List specific data points to collect
     - Use ${{variable_name}} syntax for variables
     - Use cloud API calls to fetch data
  2. Error Handling & Logging
     - Handle API errors gracefully
     - Log all discovered information
  3. Output Format
     - JSON structure for discovered information
variables:
  - name: variable_name # Without ${{ }} syntax
    description: Clear description of the variable
dependencies: # List of resource IDs this discovery depends on
  - "resource_id_1"
  - "resource_id_2"
next_discovery_targets: # Suggested next resources to investigate
  - "target_1"
  - "target_2"
discovery_complete: false # Set to true if no more discoveries needed for this context
tags: Relevant tags for categorization
"""

        # Create the user message
        user_message = f"""
Cloud Service: {service}
Customer Request: {request}
Iteration: {iteration}

{discovery_context}

Generate the NEXT most valuable prompt for discovering and analyzing cloud resources.
The prompt should:
1. Focus on collecting specific, targeted information from {cloud} {service}
2. Consider already discovered data when choosing what to investigate next
3. Follow logical troubleshooting progression
4. Return findings in structured JSON format

Remember:
1. Focus on ONE specific discovery task
2. Include error handling
3. Use variables for dynamic values
4. Suggest next discovery targets
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
        parsed_prompt = self._parse_response(response, iteration=iteration, parent_request_id=parent_request_id)

        # Save the generated prompt and get the path
        saved_path = self._save_prompt(model_name, parsed_prompt)

        return parsed_prompt, saved_path

    def _parse_response(self, response: BaseMessage, iteration: int = 1, parent_request_id: Optional[str] = None) -> GeneratedPrompt:
        """Parse the model response into a GeneratedPrompt structure."""
        try:
            # Extract content from the response
            content = response.content if hasattr(response, "content") else str(response)

            # Remove yaml code block markers if present
            content = content.replace("```yaml", "").replace("```", "").strip()  # type: ignore

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
                iteration=iteration,
                parent_request_id=parent_request_id,
                discovered_resources=[],
                dependencies=data.get("dependencies", []),
                next_discovery_targets=data.get("next_discovery_targets", []),
                discovery_complete=data.get("discovery_complete", False),
            )
        except Exception as e:
            raise ValueError(f"Failed to parse model response: {e}") from e

    def _save_prompt(self, model_name: str, prompt: GeneratedPrompt) -> Path:
        """Save the generated prompt to a YAML file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create a unique name for the prompt
        prompt_name = f"{prompt.service}_{prompt.operation}_{model_name}_{prompt.iteration}_{timestamp}"
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
                    "generated_at": prompt.generated_at,
                    "iteration": prompt.iteration,
                    "parent_request_id": prompt.parent_request_id,
                    "discovered_resources": prompt.discovered_resources,
                    "dependencies": prompt.dependencies,
                    "next_discovery_targets": prompt.next_discovery_targets,
                    "discovery_complete": prompt.discovery_complete,
                }
            }
        }

        # Add custom representer for str to handle multiline strings
        original_represent_str = yaml.SafeDumper.represent_str

        def repr_str(dumper, data):
            if "\n" in data:
                return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
            return original_represent_str(dumper, data)

        yaml.add_representer(str, repr_str, Dumper=yaml.SafeDumper)

        output_path = self.output_dir / filename
        with output_path.open("w") as f:
            yaml.safe_dump(prompt_data, f, sort_keys=False, allow_unicode=True)

        return output_path
