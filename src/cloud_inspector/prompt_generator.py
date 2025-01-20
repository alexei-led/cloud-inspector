from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import yaml
from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from cloud_inspector.prompts import CloudProvider, PromptType
from langchain_components.models import ModelRegistry


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


class PromptGenerator:
    """Generates prompts for cloud service operations."""

    def __init__(self, model_registry: ModelRegistry, output_dir: Optional[Path] = None, history_dir: Optional[Path] = None):
        self.model_registry = model_registry
        self.output_dir = output_dir or Path("generated_prompts")
        self.history_dir = history_dir or Path("prompt_history")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history_dir.mkdir(parents=True, exist_ok=True)

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
    ) -> GeneratedPrompt:
        """Generate a new prompt."""
        model = self.model_registry.get_model(model_name)

        # Build context from previous results and feedback if available
        context = ""
        if previous_results:
            context += "\nPrevious execution results:\n" + yaml.dump(previous_results)
        if feedback:
            context += "\nUser feedback:\n" + yaml.dump(feedback)

        # Generate the prompt template
        messages = [
            {
                "role": "system",
                "content": f"You are an expert prompt engineer. Generate a prompt template for AWS service: {service}, operation: {operation}.\n{context}",
            },
            {"role": "user", "content": f"Description: {description}\nRequired variables: {variables}"},
        ]

        response = model.invoke(messages)
        template = self._extract_template(response)

        # Create the prompt
        prompt = GeneratedPrompt(
            service=service,
            operation=operation,
            description=description,
            template=template,
            variables=variables,
            tags=tags,
            cloud=cloud,
            generated_by=model_name,
            generated_at=datetime.now().isoformat(),
        )

        return prompt

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
