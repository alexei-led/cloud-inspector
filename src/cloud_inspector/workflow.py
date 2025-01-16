"""Workflow system for code generation."""

# Standard library imports
import ast
import json
from dataclasses import dataclass
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any, Optional, Union

# Third-party imports
import autopep8
from autoflake import fix_code
from black import FileMode, format_str
from langchain_core.messages.base import BaseMessage
from langchain_core.tracers.context import collect_runs
from pyflakes.api import check
from pyflakes.reporter import Reporter

# Local imports
from cloud_inspector.prompts import PromptManager
from langchain_components.models import ModelCapability, ModelRegistry
from langchain_components.templates import GeneratedFiles


@dataclass
class WorkflowResult:
    """Result of a workflow execution."""

    prompt_name: str
    model_name: str
    timestamp: datetime
    execution_time: float
    success: bool
    run_id: Optional[str] = None
    error: Optional[str] = None
    generated_files: dict[str, str] = None


# Constants for token limit detection
TOKEN_LIMIT_MARKERS = ("maximum context length", "maximum token limit")
MAX_RETRIES = 5
CONTINUE_PROMPT = "Please continue providing the remaining files and code."


class TokenLimitError(Exception):
    """Raised when max token limit is hit and retries are exhausted."""

    pass


class ParseError(Exception):
    """Raised when response parsing fails."""

    pass


class CodeGenerationWorkflow:
    """Workflow for generating code from prompts."""

    def __init__(
        self,
        prompt_manager: PromptManager,
        model_registry: ModelRegistry,
        project_name: str,
        output_dir: Optional[Path] = None,
    ):
        self.prompt_manager = prompt_manager
        self.model_registry = model_registry
        self.project_name = project_name
        self.output_dir = output_dir or Path("generated_code")

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _handle_token_limit_response(
        self,
        structured_model: Any,
        messages: list[dict[str, str]],
        initial_response: str,
    ) -> dict[str, str]:
        """Handle responses that hit token limits by continuing the conversation."""
        retry_count = 0
        accumulated_response = initial_response

        while retry_count < MAX_RETRIES:
            messages.extend(
                [
                    {"role": "assistant", "content": accumulated_response},
                    {"role": "user", "content": CONTINUE_PROMPT},
                ]
            )

            continuation = structured_model.invoke(messages)
            if not isinstance(continuation, dict):
                raise ParseError("Expected dictionary response format")

            if continuation.get("parsed") is not None:
                return {
                    "main.py": self._reformat_code(continuation["parsed"].main_py, Code=True),
                    "requirements.txt": self._reformat_code(continuation["parsed"].requirements_txt),
                    "policy.json": self._reformat_code(continuation["parsed"].policy_json),
                }

            raw_response = (
                continuation.get("raw", "").content
                if isinstance(continuation.get("raw"), BaseMessage)
                else str(continuation.get("raw", ""))
            )
            accumulated_response += f"\n{raw_response}"

            # Check if this is still a token limit issue
            if not any(marker in raw_response.lower() for marker in TOKEN_LIMIT_MARKERS):
                raise ParseError(f"Failed to parse response after continuation: {raw_response}")

            retry_count += 1

        raise TokenLimitError(
            f"Failed to get complete response after {MAX_RETRIES} retries.\nAccumulated response: {accumulated_response}"
        )

    def _extract_latest_generated_files(self, raw_response: Union[str, list[dict]]) -> dict[str, str]:
        """Extract latest GeneratedFiles content from Nova model response."""
        # Parse response if string
        messages = json.loads(raw_response) if isinstance(raw_response, str) else raw_response
        # Track latest content
        latest_files = {"main_py": "", "requirements_txt": "", "policy_json": ""}
        # Scan all messages, overriding with latest content
        for msg in messages:
            if msg.get("type") == "tool_use" and msg.get("name") == "GeneratedFiles":
                input_data = msg.get("input", {})
                for key in latest_files:
                    if key in input_data and input_data[key]:
                        latest_files[key] = input_data[key]

        if not any(latest_files.values()):
            raise ParseError("No GeneratedFiles content found")
        return latest_files

    def execute(self, prompt_name: str, model_name: str, variables: dict[str, Any]) -> tuple[WorkflowResult, Path]:
        """Execute the code generation workflow."""
        start_time = datetime.now()

        try:
            # Validate model can generate code
            if not self.model_registry.validate_model_capability(model_name, ModelCapability.CODE_GENERATION):
                raise ValueError(f"Model '{model_name}' does not support code generation")

            with collect_runs() as runs_cb:
                run_name = f"code_generation_{prompt_name}"
                tags = ["code_generation", prompt_name, model_name]

                # Get model config to check system prompt support
                model_config = self.model_registry.models[model_name]
                supports_system_prompt = model_config.supports_system_prompt

                messages = self.prompt_manager.format_prompt(
                    prompt_name,
                    variables,
                    supports_system_prompt=supports_system_prompt,
                )
                if not messages:
                    raise ValueError(f"Prompt '{prompt_name}' not found")

                model = self.model_registry.get_model(model_name)
                model = model.with_config({"run_name": run_name, "tags": tags})

                structured_output_params = self.model_registry.get_structured_output_params(model_name, GeneratedFiles)
                if not structured_output_params.get("include_raw"):
                    raise ValueError("include_raw must be True in model definition for token limit handling")

                structured_model = model.with_structured_output(GeneratedFiles, **structured_output_params)

                response = structured_model.invoke(messages)
                if not isinstance(response, dict):
                    raise ParseError("Expected dictionary response format")

                if response.get("parsed") is not None:
                    generated_files = {
                        "main.py": self._reformat_code(response["parsed"].main_py, code=True),
                        "requirements.txt": self._reformat_code(response["parsed"].requirements_txt),
                        "policy.json": (
                            response["parsed"].policy_json
                            if isinstance(response["parsed"].policy_json, dict)
                            else self._reformat_code(response["parsed"].policy_json)
                        ),
                    }
                else:
                    raw_response = (
                        response.get("raw", "").content
                        if isinstance(response.get("raw"), BaseMessage)
                        else str(response.get("raw", ""))
                    )
                    latest_files = self._extract_latest_generated_files(raw_response)
                    generated_files = {
                        "main.py": self._reformat_code(latest_files["main_py"], code=True),
                        "requirements.txt": self._reformat_code(latest_files["requirements_txt"]),
                        "policy.json": (
                            latest_files["policy_json"]
                            if isinstance(latest_files["policy_json"], dict)
                            else self._reformat_code(latest_files["policy_json"])
                        ),
                    }

                result = WorkflowResult(
                    prompt_name=prompt_name,
                    model_name=model_name,
                    timestamp=start_time,
                    execution_time=(datetime.now() - start_time).total_seconds(),
                    success=True,
                    run_id=runs_cb.traced_runs[0].id,
                    generated_files=generated_files,
                )

                output_dir = self._save_result(result)
                return result, output_dir

        except Exception as e:
            result = WorkflowResult(
                prompt_name=prompt_name,
                model_name=model_name,
                timestamp=start_time,
                execution_time=(datetime.now() - start_time).total_seconds(),
                success=False,
                error=str(e),
                generated_files={},
            )
            output_dir = self._save_result(result)
            return result, output_dir

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
                print(f"Pyflakes detected issues:\n{error_output.getvalue()}")

            # Fix imports automatically
            decoded = fix_code(
                decoded,
                remove_all_unused_imports=True,
                remove_unused_variables=True,
                expand_star_imports=True,  # Expand * imports for better clarity
            )

            # Apply autopep8 fixes for other issues
            decoded = autopep8.fix_code(
                decoded,
                options={
                    "aggressive": 2,  # More aggressive fixes
                    "max_line_length": 100,
                },
            )

            # Final formatting with black
            decoded = format_str(decoded, mode=FileMode())

        except SyntaxError as e:
            print(f"Warning: Generated Python code has syntax errors: {e}")
        except Exception as e:
            print(f"Warning: Code formatting failed: {e}")

        return decoded

    def _save_result(self, result: WorkflowResult) -> Path:
        """Save workflow result to file."""
        # Create timestamp-based filename
        timestamp = result.timestamp.strftime("%Y%m%d_%H%M%S")
        base_name = f"{result.prompt_name}_{result.model_name}_{timestamp}"

        # Create a directory for this run
        run_dir = self.output_dir / base_name
        run_dir.mkdir(parents=True, exist_ok=True)

        # Save generated files
        if result.generated_files:
            for filename, content in result.generated_files.items():
                try:
                    file_path = run_dir / filename
                    if filename.endswith(".json") and isinstance(content, dict):
                        with open(file_path, "w") as f:
                            json.dump(content, f, indent=2)
                    else:
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
                    "prompt_name": result.prompt_name,
                    "model_name": result.model_name,
                    "timestamp": result.timestamp.isoformat(),
                    "execution_time": result.execution_time,
                    "success": result.success,
                    "error": result.error,
                    "run_id": str(result.run_id) if result.run_id else None,
                },
                f,
                indent=2,
            )

        return run_dir


class WorkflowManager:
    """Manager for handling workflow executions."""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("generated_code")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def list_results(
        self,
        prompt_name: Optional[str] = None,
        model_name: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> list[dict[str, Any]]:
        """list workflow results with optional filtering."""
        results = []

        # Look for metadata files
        for meta_file in self.output_dir.rglob("metadata.json"):
            try:
                with meta_file.open("r") as f:
                    data = json.load(f)

                # Apply filters
                timestamp = datetime.fromisoformat(data["timestamp"])
                if prompt_name and data["prompt_name"] != prompt_name:
                    continue
                if model_name and data["model_name"] != model_name:
                    continue
                if start_time and timestamp < start_time:
                    continue
                if end_time and timestamp > end_time:
                    continue

                results.append(data)

            except Exception as e:
                print(f"Error reading {meta_file}: {e}")

        return sorted(results, key=lambda x: x["timestamp"], reverse=True)

    def get_result(self, prompt_name: str, timestamp: str) -> Optional[dict[str, Any]]:
        """Get a specific workflow result."""
        for meta_file in self.output_dir.glob(f"{prompt_name}_*_{timestamp}_meta.json"):
            try:
                with meta_file.open("r") as f:
                    return json.load(f)
            except Exception:
                continue
        return None

    def get_statistics(self) -> dict[str, Any]:
        """Get execution statistics."""
        results = self.list_results()

        stats = {
            "total_executions": len(results),
            "successful_executions": sum(1 for r in results if r["success"]),
            "failed_executions": sum(1 for r in results if not r["success"]),
            "average_execution_time": (sum(r["execution_time"] for r in results) / len(results) if results else 0),
            "by_model": {},
            "by_prompt": {},
            "common_errors": {},
        }

        # Collect detailed stats
        for result in results:
            # Stats by model
            model = result["model_name"]
            if model not in stats["by_model"]:
                stats["by_model"][model] = {"total": 0, "successful": 0}
            stats["by_model"][model]["total"] += 1
            if result["success"]:
                stats["by_model"][model]["successful"] += 1

            # Stats by prompt
            prompt = result["prompt_name"]
            if prompt not in stats["by_prompt"]:
                stats["by_prompt"][prompt] = {"total": 0, "successful": 0}
            stats["by_prompt"][prompt]["total"] += 1
            if result["success"]:
                stats["by_prompt"][prompt]["successful"] += 1

            # Error tracking
            if not result["success"] and result["error"]:
                error_type = result["error"].split(":")[0]
                stats["common_errors"][error_type] = stats["common_errors"].get(error_type, 0) + 1

        return stats
