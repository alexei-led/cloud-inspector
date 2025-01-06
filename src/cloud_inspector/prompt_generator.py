from pathlib import Path
from typing import Optional
from langchain_core.messages import BaseMessage
from pydantic import BaseModel
from langchain_components.models import ModelRegistry, ModelCapability


class GeneratedPrompt(BaseModel):
    """Structure for generated prompt."""

    service: str
    operation: str
    description: str
    template: str
    variables: list[str]
    tags: list[str]


class PromptGenerator:
    """Generates prompts for cloud service operations."""

    def __init__(
        self, model_registry: ModelRegistry, output_dir: Optional[Path] = None
    ):
        self.model_registry = model_registry
        self.output_dir = output_dir or Path("generated_prompts")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_prompt(
        self, model_name: str, service: str, request: str, **kwargs
    ) -> GeneratedPrompt:
        """Generate a prompt template from a natural language request."""

        # Validate model can generate prompts
        if not self.model_registry.validate_model_capability(
            model_name, ModelCapability.PROMPT_GENERATION
        ):
            raise ValueError(f"Model '{model_name}' does not support prompt generation")

        # Get the model
        model = self.model_registry.get_model(model_name)

        # Create the system message
        system_message = """You are a prompt engineer specializing in cloud services.
        Generate a detailed prompt template that will help code generation models create
        Python scripts for cloud operations. The prompt should follow the format used
        in aws_prompts.yaml."""

        # Create the user message
        user_message = f"""
        Cloud Service: {service}
        Request: {request}

        Generate a prompt template that includes:
        1. A descriptive operation name
        2. Clear description
        3. Detailed template with requirements
        4. Required variables
        5. Relevant tags
        """

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

        # Generate the prompt
        response = model.invoke(messages)

        # Parse the response into a GeneratedPrompt
        # (You'll need to implement the parsing logic based on your model's output format)
        parsed_prompt = self._parse_response(response)

        # Save the generated prompt
        self._save_prompt(parsed_prompt)

        return parsed_prompt

    def _parse_response(self, response: BaseMessage) -> GeneratedPrompt:
        """Parse the model response into a GeneratedPrompt structure."""
        # Implement parsing logic here
        pass

    def _save_prompt(self, prompt: GeneratedPrompt) -> None:
        """Save the generated prompt to a YAML file."""
        # Implement saving logic here
        pass
