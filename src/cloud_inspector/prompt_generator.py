from datetime import datetime
from typing import Any, Optional

import yaml
from langchain_core.messages import BaseMessage

from components.models import ModelRegistry
from components.types import CloudProvider, CodeGenerationPrompt


class PromptGeneratorAgent:
    """Generates prompts for cloud service operations."""

    def __init__(self, model_registry: ModelRegistry):
        self.model_registry = model_registry

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
        request: str,
        variables: list[dict[str, str]],
        cloud: CloudProvider = CloudProvider.AWS,
        previous_results: Optional[dict[str, Any]] = None,
        feedback: Optional[dict[str, Any]] = None,
        iteration: int = 1,
    ) -> CodeGenerationPrompt:
        """Generate a new prompt focused on iterative discovery."""
        model = self.model_registry.get_model(model_name)

        # Format variables as simple name-value pairs
        vars_formatted = "\n".join(f"  {v['name']}: {v['value']}" for v in variables)

        # Determine iteration goal
        iteration_goals = self._get_iteration_goals(iteration, service)

        next_focus = self._determine_next_focus(previous_results)

        system_message = """
You are an expert prompt engineer specializing in cloud infrastructure.
Your task is to create a clear, detailed prompt that will guide another AI model in generating focused code 
for discovering specific cloud infrastructure information."""

        user_message = f"""
The prompt you create should help the code generation model understand:
1. The NEXT specific piece of information to discover
2. Context from previously discovered information
3. How to use the context to guide the next discovery step
4. Required inputs for this specific discovery step
5. Expected output format for the discovered information

=== CONTEXT ===
Original Request: {request}
Cloud Service: {cloud.value} {service}
Operation: {operation}
Current Iteration: {iteration}
Next Focus: {next_focus}

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
The generated prompt should specify any missing variables needed for code generation.
Format: YAML list under 'variables' with fields:
- name: Variable name
- description: What the variable is used for

=== OUTPUT FORMAT ===
Return a YAML document containing:
1. 'template' field with your generated prompt
2. 'description' field with a brief summary of the prompt
2. 'variables' field listing any new variables needed
3. 'success_criteria' field describing what defines success for this iteration

The prompt should be clear, structured, and focused on the current phase goals."""

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

        # Remove response_format from model kwargs
        model_kwargs = getattr(model, "model_kwargs", {})
        if "response_format" in model_kwargs:
            model_kwargs.pop("response_format")
            setattr(model, "model_kwargs", model_kwargs)  # noqa: B010

        response = model.invoke(messages)
        template, new_variables, success_criteria, description = self._parse_response(response)

        # Merge existing variables with new ones, avoiding duplicates by name
        existing_var_names = {v["name"] for v in variables}
        merged_variables = variables + [v for v in new_variables if v["name"] not in existing_var_names]

        return CodeGenerationPrompt(
            service=service,
            operation=operation,
            request=request,
            template=template,
            variables=merged_variables,  # Use merged variables here
            cloud=cloud,
            generated_by=model_name,
            generated_at=datetime.now(),
            success_criteria=success_criteria,
            description=description,
        )

    def _determine_next_focus(self, previous_results: Optional[dict[str, Any]]) -> str:
        """Determine what should be the next focus area based on previous discoveries."""
        if not previous_results:
            return "Initial basic configuration"

        return "Next logical configuration based on previous findings"

    def _get_iteration_goals(self, iteration: int, service: str) -> str:
        if iteration == 1:
            return f"""
- Identify the FIRST most relevant piece of information to check
- Focus on a single, specific aspect of {service}
- Keep the discovery logic simple and focused"""
        return """
- Use previously discovered information to determine the next logical check
- Focus on one specific relationship or configuration
- Keep the scope narrow and targeted"""

    def _parse_response(self, response: BaseMessage) -> tuple[str, list[dict[str, str]], str, str]:
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
            vars = []
            for var in data.get("variables", []):
                if isinstance(var, dict) and "name" in var and "description" in var:
                    vars.append({"name": var["name"], "description": var["description"], "value": var.get("default_value", "")})

            # Merge new variables with existing ones
            return template, vars, data.get("success_criteria", ""), data.get("description", "")
        except Exception as e:
            raise ValueError(f"Failed to extract template from model response: {e}") from e
