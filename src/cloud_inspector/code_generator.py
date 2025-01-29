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
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.tracers.context import collect_runs
from pyflakes.api import check
from pyflakes.reporter import Reporter

# Local imports
from components.models import ModelCapability, ModelRegistry
from components.types import CodeGenerationPrompt, GeneratedFiles


def format_prompt(self, prompt: CodeGenerationPrompt, variables: Optional[dict[str, Any]] = None, supports_system_prompt: bool = True) -> list[Any]:
    # Validate that all required variables are provided
    required_var_names = {var["name"] for var in prompt.variables}
    variables = variables or {}
    provided_var_names = set(variables.keys())
    missing_vars = required_var_names - provided_var_names

    if missing_vars:
        var_descriptions = {var["name"]: var["description"] for var in prompt.variables if var["name"] in missing_vars}
        raise ValueError(f"Missing required variables: {', '.join(f'{name} ({desc})' for name, desc in var_descriptions.items())}")

    system_message = f"""
You are an expert {prompt.cloud.value} DevOps engineer. Your task is to generate Python scripts and related files for {prompt.cloud.value} operations, ensuring the output meets high-quality standards and adheres to best practices.

OUTPUT FORMAT
Always respond in the following JSON structure: {{ "main_py": "string", "requirements_txt": "string", "policy_json": "string" }}

GUIDELINES
1. CODE REQUIREMENTS
   - Generate Python code using `boto3` that is concise, efficient, and ready for execution without requiring human intervention or review.
   - Use Python type hints, clear variable names, and modular functions. Avoid placeholder or incomplete code.
   - Avoid unnecessary comments or overly verbose explanations in the code.
   - Import all necessary modules required for the task, ensuring the script is self-contained and executable without missing imports (e.g., `datetime`, `boto3`, `logging`, etc.). For example when using `datetime` or `timedelta` use `from datetime import datetime, timedelta`.
   - If a specific feature cannot be implemented (e.g., due to {prompt.cloud.value} service limitations), omit the function entirely and log a clear explanation.

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

4. {prompt.cloud.value} BEST PRACTICES
   - Pass the {prompt.cloud.value} region as a parameter if required by the API.
   - Use the {prompt.cloud.value} SDK default credential provider chain, but allow passing custom credentials if necessary.
   - Handle large datasets with pagination or streaming where applicable.
   - Dynamically discover resource IDs instead of using invented or hardcoded values. Avoid placeholder values like `CommandId='your-command-id'`.

5. DEPENDENCIES
   - Ensure that the `requirements.txt` includes all required dependencies with pinned versions.

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

    if supports_system_prompt:
        chat_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_message),
                ("user", prompt.template),
            ]
        )
    else:
        combined_prompt = f"<instructions>{system_message}</instructions>\n\n<question>{prompt.template}</question>"
        chat_prompt = ChatPromptTemplate.from_messages(
            [
                ("user", combined_prompt),
            ]
        )

    return chat_prompt.format_messages(**variables)


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

    def generate_code(
        self,
        prompt: CodeGenerationPrompt,
        model_name: str,
        variables: dict[str, Any],
        iteration_id: str,
        previous_results: Optional[dict[str, Any]] = None,
        feedback: Optional[dict[str, Any]] = None,
    ) -> tuple[CodeGeneratorResult, Path]:
        """Execute the code generation workflow.

        Args:
            prompt_template: The prompt template to use for code generation
            model_name: Name of the model to use for code generation
            variables: Variables to inject into the prompt (request, service, iteration)
            iteration_id: Optional ID to track this iteration
            previous_results: Optional dict containing data discovered in previous iterations
            feedback: Optional dict containing user feedback/direction for this iteration
        """

        try:
            # Validate model can generate code
            if not self.model_registry.validate_model_capability(model_name, ModelCapability.CODE_GENERATION):
                raise ValueError(f"Model '{model_name}' does not support code generation")

            with collect_runs() as runs_cb:
                tags = ["code_generation", variables.get("service", ""), model_name]
                if iteration_id:
                    tags.append(f"iteration_{iteration_id}")

                model = self.model_registry.get_model(model_name)

                # Validate all variables are provided
                missing_vars = []
                for var in prompt.variables:
                    if var["name"] not in variables:
                        if var.get("value"):  # Use value from template if available
                            variables[var["name"]] = var["value"]
                        else:
                            missing_vars.append(f"{var['name']} ({var['description']})")

                if missing_vars:
                    raise ValueError("Missing variables:\n" + "\n".join(f"- {var}" for var in missing_vars))

                # Format prompt using PromptManager
                messages = format_prompt("", prompt, variables)

                # Add context from previous results and feedback if available
                if previous_results:
                    context = "Previous execution results:\n" + json.dumps(previous_results, indent=2)
                    messages.append({"role": "user", "content": context})
                if feedback:
                    feedback_msg = "User feedback:\n" + json.dumps(feedback, indent=2)
                    messages.append({"role": "user", "content": feedback_msg})

                structured_output_params = self.model_registry.get_structured_output_params(model_name, GeneratedFiles)
                if not structured_output_params.get("include_raw"):
                    raise ValueError("include_raw must be True in model definition for token limit handling")

                structured_model = model.with_structured_output(GeneratedFiles, **structured_output_params)  # type: ignore

                response = structured_model.invoke(messages)
                if not isinstance(response, dict):
                    raise ParseError("Expected dictionary response format")

                if response.get("parsed") is not None:
                    generated_files = {
                        "main.py": self._reformat_code(response["parsed"].main_py, code=True),
                        "requirements.txt": self._reformat_code(response["parsed"].requirements_txt),
                        "policy.json": self._reformat_code(response["parsed"].policy_json),
                    }
                else:
                    raw_response = response.get("raw", "").content if isinstance(response.get("raw"), BaseMessage) else str(response.get("raw", ""))
                    latest_files = self._extract_latest_generated_files(raw_response)
                    generated_files = {
                        "main.py": self._reformat_code(latest_files["main_py"], code=True),
                        "requirements.txt": self._reformat_code(latest_files["requirements_txt"]),
                        "policy.json": self._reformat_code(latest_files["policy_json"]),
                    }

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
