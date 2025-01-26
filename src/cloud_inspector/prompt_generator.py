from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import yaml
from langchain_core.messages import BaseMessage

from cloud_inspector.prompts import CloudProvider, PromptTemplate
from langchain_components.models import ModelRegistry


class PromptGenerator:
    """Generates prompts for cloud service operations."""

    def __init__(self, model_registry: ModelRegistry, output_dir: Optional[Path] = None, history_dir: Optional[Path] = None):
        self.model_registry = model_registry
        self.output_dir = output_dir or Path("generated_prompts")
        self.history_dir = history_dir or Path("prompt_history")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history_dir.mkdir(parents=True, exist_ok=True)

    def _format_data(self, data: Any) -> str:
        """Format data appropriately based on its type."""
        if not data:
            return "None"
        if isinstance(data, (dict, list)):
            return yaml.dump(data, default_flow_style=False, sort_keys=False)
        return str(data)

    def generate_prompt(
        self,
        model_name: str,
        service: str,
        operation: str,
        description: str,
        variables: list[dict[str, str]],
        tags: list[str],
        cloud: CloudProvider = CloudProvider.AWS,
        previous_results: Optional[dict[str, Any]] = None,
        feedback: Optional[dict[str, Any]] = None,
        iteration: int = 1,
    ) -> PromptTemplate:
        """Generate a new prompt focused on iterative discovery."""
        model = self.model_registry.get_model(model_name)

        # Format variables as simple name-value pairs
        vars_formatted = "\n".join(f"  {v['name']}: {v['value']}" for v in variables)

        # Determine iteration phase and goals
        iteration_phase = self._get_iteration_phase(iteration, previous_results)
        iteration_goals = self._get_iteration_goals(iteration_phase, service, operation)

        system_prompt = f"""You are an expert prompt engineer specializing in creating prompts for code generation.
Your task is to create a prompt that will be used by another AI model to generate Python code for cloud operations.

The prompt you create should guide the code generation model to:
1. Focus on the specific cloud operation needed
2. Consider previously discovered data
3. Generate precise, secure, and efficient code
4. Include proper error handling and logging
5. Return structured data that can be used in subsequent iterations

=== CONTEXT ===
Original Request: {description}
Cloud Service: {cloud.value} {service}
Operation: {operation}
Current Iteration: {iteration}
Phase: {iteration_phase}

=== GOALS ===
{iteration_goals}

=== STATE ===
Previously Discovered Data:
{self._format_data(previous_results)}

User Feedback:
{self._format_data(feedback)}

Variables:
{vars_formatted}

=== VARIABLES ===
The generated prompt should specify any additional variables needed for code generation.
Format: YAML list under 'variables' with fields:
- name: Variable name
- description: What the variable is used for
- default_value: Optional default value

=== OUTPUT FORMAT ===
Return a YAML document containing:
1. 'template' field with your generated prompt
2. 'variables' field listing any new variables needed
3. 'completion_criteria' field describing what defines success for this iteration

The prompt should be clear, structured, and focused on the current phase goals."""

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": f"Request: {description}\nOperation: {operation}"},
        ]

        # Remove response_format from model kwargs
        model_kwargs = getattr(model, "model_kwargs", {})
        if "response_format" in model_kwargs:
            model_kwargs.pop("response_format")
            setattr(model, "model_kwargs", model_kwargs)  # noqa: B010

        response = model.invoke(messages)
        template, new_variables = self._extract_template(response)

        return PromptTemplate(
            service=service,
            operation=operation,
            description=description,
            template=template,
            variables=new_variables,
            tags=tags,
            cloud=cloud,
            generated_by=model_name,
            generated_at=datetime.now(),
            # Track discovery progress using previous_results
            discovered_resources=[str(previous_results)] if previous_results else [],
            discovery_complete=(iteration > 3),  # Simple example threshold
            history=None,
        )

    def _get_iteration_phase(self, iteration: int, previous_results: Optional[dict[str, Any]]) -> str:
        """Determine the current iteration phase based on progress."""
        phases = {1: "Initial Discovery", 2: "Detailed Analysis", 3: "Validation and Enhancement"}
        return phases.get(iteration, "Refinement and Completion")

    def _get_iteration_goals(self, phase: str, service: str, operation: str) -> str:
        """Get specific goals for the current iteration phase."""
        goals = {
            "Initial Discovery": f"""
- List basic {service} resources and their configurations
- Identify key parameters and settings
- Establish baseline inventory""",
            "Detailed Analysis": """
- Analyze relationships between discovered resources
- Gather detailed configuration and status information
- Identify potential security or compliance issues""",
            "Validation and Enhancement": """
- Validate collected data against best practices
- Enhance error handling and logging
- Implement comprehensive resource filtering""",
            "Refinement and Completion": """
- Address any gaps in collected data
- Optimize code performance and reliability
- Ensure all error cases are handled""",
        }
        return goals.get(phase, "Complete remaining discovery tasks")

    def _extract_template(self, response: BaseMessage) -> tuple[str, list[dict[str, str]]]:
        """Extract the prompt template from the model response."""
        try:
            # Extract content from the response
            content = response.content if hasattr(response, "content") else str(response)

            # Remove yaml code block markers if present
            content = content.replace("```yaml", "").replace("```", "").strip()  # type: ignore

            # Parse YAML content
            data = yaml.safe_load(content)

            # Extract template and variables
            template = data.get("template", "").strip()

            # Parse any new variables from response
            new_vars = []
            for var in data.get("variables", []):
                if isinstance(var, dict) and "name" in var and "description" in var:
                    new_vars.append({"name": var["name"], "description": var["description"], "value": var.get("default_value", "")})

            # Merge new variables with existing ones
            return template, new_vars
        except Exception as e:
            raise ValueError(f"Failed to extract template from model response: {e}") from e
