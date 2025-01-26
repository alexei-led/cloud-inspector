from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import yaml
from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from cloud_inspector.prompts import CloudProvider, PromptType, PromptTemplate
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

=== STATE ===
Previously Discovered Data:
{self._format_data(previous_results)}

User Feedback:
{self._format_data(feedback)}

Variables:
{vars_formatted}

=== OUTPUT FORMAT ===
Return a YAML document containing only a 'template' field with your generated prompt.
The prompt should be clear, structured, and focused on the next piece of information to discover."""

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": f"Request: {description}\nOperation: {operation}"},
        ]

        # Remove response_format from model kwargs
        if hasattr(model, "model_kwargs") and "response_format" in model.model_kwargs:
            model.model_kwargs.pop("response_format")

        response = model.invoke(messages)
        template = self._extract_template(response)

        return PromptTemplate(
            service=service,
            operation=operation,
            description=description,
            template=template,
            variables=variables,
            tags=tags,
            cloud=cloud,
            generated_by=model_name,
            generated_at=datetime.now().isoformat(),
            # Track discovery progress using previous_results
            discovered_resources=[str(previous_results)] if previous_results else [],
            discovery_complete=(iteration > 3),  # Simple example threshold
        )

    def _extract_template(self, response: BaseMessage) -> str:
        """Extract the prompt template from the model response."""
        try:
            # Extract content from the response
            content = response.content if hasattr(response, "content") else str(response)

            # Remove yaml code block markers if present
            content = content.replace("```yaml", "").replace("```", "").strip()  # type: ignore

            # Parse YAML content
            data = yaml.safe_load(content)

            # Extract the template
            template = data.get("template", "").strip()

            return template
        except Exception as e:
            raise ValueError(f"Failed to extract template from model response: {e}") from e
