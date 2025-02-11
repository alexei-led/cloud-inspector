"""Workflow system for code generation."""

# Standard library imports
import ast
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any, Optional, Union
from unittest.mock import Mock

# Third-party imports
import autopep8
from autoflake import fix_code
from black import FileMode, format_str
from langchain_core.prompts.chat import ChatPromptTemplate
from pydantic import BaseModel
from pyflakes.api import check
from pyflakes.reporter import Reporter

# Local imports
from cloud_inspector.components.models import ModelCapability, ModelRegistry
from cloud_inspector.components.types import CodeGenerationPrompt, GeneratedFiles

logger = logging.getLogger(__name__)


@dataclass
class CodeGeneratorResult:
    """Result of a code generation."""

    prompt_template: str
    model_name: str
    iteration_id: str
    run_id: str
    generated_at: datetime
    generated_files: Optional[dict[str, str]] = None


class ParseError(Exception):
    """Raised when response parsing fails."""

    pass


class CodeGeneratorAgent:
    """Generating code from prompts."""

    def __init__(
        self,
        model_registry: ModelRegistry,
        output_dir: Optional[Path] = None,
    ):
        self.model_registry = model_registry
        self.output_dir = output_dir or Path("generated_code")
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def format_prompt(self, prompt: CodeGenerationPrompt, variables: Optional[dict[str, Any]] = None, supports_system_prompt: bool = True) -> list[Any]:
        """Format a prompt for code generation.

        Args:
            prompt: The prompt template and metadata
            variables: Variables to inject into the prompt
            supports_system_prompt: Whether the model supports system prompts

        Returns:
            List of formatted messages for the model
        """
        # Validate required variables are provided
        required_var_names = {var["name"] for var in prompt.variables}
        variables = variables or {}
        provided_var_names = set(variables.keys())
        missing_vars = required_var_names - provided_var_names

        if missing_vars:
            var_descriptions = {var["name"]: var["description"] for var in prompt.variables if var["name"] in missing_vars}
            raise ValueError("Missing required variables: " + ", ".join(f"{name} ({desc})" for name, desc in var_descriptions.items()))

        system_message = f"""You are an expert {prompt.cloud.value} DevOps engineer..."""

        if supports_system_prompt:
            chat_prompt = ChatPromptTemplate.from_messages([("system", system_message), ("user", prompt.template)])
        else:
            combined_prompt = f"<instructions>{system_message}</instructions>\n\n<question>{prompt.template}</question>"
            chat_prompt = ChatPromptTemplate.from_messages([("user", combined_prompt)])

        return chat_prompt.format_messages(**variables)

    def _extract_latest_generated_files(self, raw_response: Union[str, dict, Mock, None]) -> dict[str, str]:
        """Extract latest GeneratedFiles content from model response."""
        if raw_response is None:
            raise ParseError("Response cannot be None")

        messages = self._parse_raw_response(raw_response)
        return self._extract_files_from_messages(messages)

    def _parse_raw_response(self, raw_response: Union[str, dict, Mock]) -> Union[dict, list]:
        """Parse raw response into a structured format."""
        if isinstance(raw_response, Mock) and hasattr(raw_response, "content"):
            raw_response = raw_response.content

        if isinstance(raw_response, str):
            try:
                return json.loads(raw_response)
            except json.JSONDecodeError as e:
                raise ParseError("Invalid JSON response") from e

        return raw_response

    def _extract_files_from_messages(self, messages: Union[dict, list]) -> dict[str, str]:
        """Extract files content from parsed messages."""
        required_keys = {"main_py", "requirements_txt", "policy_json"}
        default_files = {key: "" for key in required_keys}

        # Handle direct dictionary format
        if isinstance(messages, dict):
            if all(key in messages for key in required_keys):
                return messages
            raise ParseError("Response missing required file keys")

        # Handle list of messages format
        if isinstance(messages, list):
            latest_files = default_files.copy()
            for msg in messages:
                if self._is_valid_generated_files_message(msg):
                    self._update_latest_files(latest_files, msg.get("input", {}))

            if any(latest_files.values()):
                return latest_files

        raise ParseError("No GeneratedFiles content found")

    def _is_valid_generated_files_message(self, msg: dict) -> bool:
        """Check if message is a valid GeneratedFiles message."""
        return msg.get("type") == "tool_use" and msg.get("name") == "GeneratedFiles"

    def _update_latest_files(self, latest_files: dict[str, str], input_data: dict) -> None:
        """Update latest files with new content if available."""
        for key in latest_files:
            if key in input_data and input_data[key]:
                latest_files[key] = input_data[key]

    def _validate_model(self, model_name: str) -> None:
        """Validate model capabilities for code generation."""
        if not self.model_registry.validate_model_capability(model_name, ModelCapability.CODE_GENERATION):
            raise ValueError(f"Model '{model_name}' does not support code generation")

    def _prepare_messages(self, prompt: CodeGenerationPrompt, variables: dict[str, Any], previous_results: Optional[dict[str, Any]] = None, feedback: Optional[dict[str, Any]] = None) -> list[Any]:
        """Prepare messages for the model including context and feedback."""
        # Add JSON format requirement to the system message
        system_message = f"""You are an expert {prompt.cloud.value} DevOps engineer.
You must provide your response as a JSON object that follows this exact structure:
{{
    "main_py": "string containing Python code",
    "requirements_txt": "string containing requirements",
    "policy_json": "string containing JSON policy"
}}"""

        messages = [{"role": "system", "content": system_message}]

        # Add the main prompt with JSON format requirement
        user_prompt = f"{prompt.template}\n\nYou must respond with a valid JSON object containing the generated files. Format your entire response as a JSON object."
        messages.append({"role": "user", "content": user_prompt})

        if previous_results:
            context = "Previous execution results (format your new response as JSON):\n" + json.dumps(previous_results, indent=2)
            messages.append({"role": "user", "content": context})
        if feedback:
            feedback_msg = "User feedback (remember to respond with JSON):\n" + json.dumps(feedback, indent=2)
            messages.append({"role": "user", "content": feedback_msg})

        return messages

    def _process_model_response(self, response: Union[dict[str, Any], BaseModel]) -> dict[str, str]:
        """Process and format the model's response."""
        try:
            # Convert BaseModel to dict if needed
            if isinstance(response, BaseModel):
                response = response.model_dump()
            elif not isinstance(response, dict):
                raise ParseError("Expected dictionary or BaseModel response format")

            if response.get("parsed") is not None:
                parsed = response["parsed"].model_dump() if isinstance(response["parsed"], GeneratedFiles) else response["parsed"]
                return {
                    "main.py": self._reformat_code(parsed["main_py"], code=True),
                    "requirements.txt": self._reformat_code(parsed["requirements_txt"]),
                    "policy.json": self._reformat_code(parsed["policy_json"]),
                }

            raw_response = response.get("raw")
            latest_files = self._extract_latest_generated_files(raw_response)

            return {
                "main.py": self._reformat_code(latest_files["main_py"], code=True),
                "requirements.txt": self._reformat_code(latest_files["requirements_txt"]),
                "policy.json": self._reformat_code(latest_files["policy_json"]),
            }
        except Exception as e:
            if isinstance(e, ParseError):
                raise
            raise ParseError(f"Failed to process response: {str(e)}") from e

    def generate_code(
        self,
        prompt: CodeGenerationPrompt,
        model_name: str,
        variables: dict[str, Any],
        iteration_id: str,
        previous_results: Optional[dict[str, Any]] = None,
        feedback: Optional[dict[str, Any]] = None,
    ) -> tuple[CodeGeneratorResult, Path]:
        """Execute the code generation workflow."""
        try:
            from langchain_core.tracers.context import collect_runs

            self._validate_model(model_name)

            with collect_runs() as runs_cb:
                model = self.model_registry.get_model(model_name)
                structured_output_params = self.model_registry.get_structured_output_params(model_name, GeneratedFiles)

                if not structured_output_params.get("include_raw"):
                    raise ValueError("include_raw must be True in model definition for token limit handling")

                messages = self._prepare_messages(prompt, variables, previous_results, feedback)
                structured_model = model.with_structured_output(GeneratedFiles, **structured_output_params)
                response = structured_model.invoke(messages)

                generated_files = self._process_model_response(response)

                result = CodeGeneratorResult(
                    prompt_template=prompt.template,
                    model_name=model_name,
                    iteration_id=iteration_id,
                    run_id=str(runs_cb.traced_runs[0].id),
                    generated_at=datetime.now(),
                    generated_files=generated_files,
                )

                output_dir = self._save_result(result)
                return result, output_dir

        except Exception as e:
            raise e

    def _reformat_code(self, model_response: str, code: bool = False) -> str:
        """Reformat code by properly handling escaped characters and fix common Python issues."""
        decoded = bytes(model_response.encode("utf-8").decode("unicode-escape").encode("utf-8")).decode("utf-8")

        # Only process Python files
        if not code:
            return decoded

        try:
            # Validate Python syntax
            ast.parse(decoded)

            # Run pyflakes to detect issues
            error_output = StringIO()
            reporter = Reporter(StringIO(), error_output)
            check(decoded, "generated_code", reporter)

            if error_output.getvalue():
                logger.warning("Pyflakes detected issues:\n%s", error_output.getvalue())

            try:
                # Try black formatting first
                formatted = format_str(decoded, mode=FileMode())
                # Only apply additional formatting if black succeeds
                decoded = fix_code(
                    formatted,
                    remove_all_unused_imports=True,
                    remove_unused_variables=True,
                    expand_star_imports=True,  # Expand * imports for better clarity
                )

                # Apply autopep8 fixes for other issues
                return autopep8.fix_code(
                    decoded,
                    options={
                        "aggressive": 2,  # More aggressive fixes
                        "max_line_length": 100,
                    },
                )
            except ValueError as e:
                logger.warning("Black formatting failed: %s", e)
                return model_response  # Return original code if black fails

        except SyntaxError as e:
            logger.warning("Generated Python code has syntax errors: %s", e)
        except Exception as e:
            logger.warning("Code formatting failed: %s", e)

        return decoded

    def _save_result(self, result: CodeGeneratorResult) -> Path:
        """Save workflow result to file."""
        # Create timestamp-based filename
        timestamp = result.generated_at.timestamp()
        base_name = f"{result.model_name}_{timestamp}"

        # Create a directory for this run
        run_dir = self.output_dir / base_name
        run_dir.mkdir(parents=True, exist_ok=True)

        # Save generated files
        if result.generated_files:
            for filename, content in result.generated_files.items():
                try:
                    file_path = run_dir / filename
                    with open(file_path, "w") as f:
                        f.write(content)
                        f.flush()
                except Exception as e:
                    print(f"Error saving {filename}: {e}")

        # Save metadata
        meta_file = run_dir / "metadata.json"
        with meta_file.open("w") as f:
            json.dump(
                {
                    "prompt_template": result.prompt_template,
                    "model_name": result.model_name,
                    "generated_at": result.generated_at.isoformat(),
                    "run_id": str(result.run_id) if result.run_id else None,
                    "iteration_id": str(result.iteration_id) if result.iteration_id else None,
                },
                f,
                indent=2,
            )

        return run_dir
