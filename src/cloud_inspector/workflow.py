"""Workflow system for code generation."""

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.tracers.context import collect_runs
from pydantic import BaseModel

from cloud_inspector.prompts import PromptManager
from langchain_components.models import ModelRegistry
from langchain_components.parsers import (
    CodeParseResult,
    PythonCodeParser,
    MetadataParser,
)
from langchain_components.templates import GeneratedFiles


@dataclass
class WorkflowResult:
    """Result of a workflow execution."""

    prompt_name: str
    model_name: str
    code_result: CodeParseResult
    metadata: Dict[str, Any]
    timestamp: datetime
    execution_time: float
    success: bool
    run_id: Optional[str] = None
    error: Optional[str] = None
    generated_files: Dict[str, str] = None


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
        self.code_parser = PythonCodeParser()
        self.metadata_parser = MetadataParser()

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def execute(
        self, prompt_name: str, model_name: str, variables: Dict[str, Any]
    ) -> WorkflowResult:
        """Execute the code generation workflow."""
        start_time = datetime.now()

        try:
            with collect_runs() as runs_cb:
                # run name is code_generation_{prompt_name}
                run_name = f"code_generation_{prompt_name}"
                tags = ["code_generation", prompt_name, model_name]

                # Load and format prompt
                messages = self.prompt_manager.format_prompt(prompt_name, variables)
                if not messages:
                    raise ValueError(f"Prompt '{prompt_name}' not found")

                # Get model and generate code
                model = self.model_registry.get_model(model_name)

                # pass run_name and tags to the model
                model = model.with_config(
                    {"run_name": run_name, "tags": tags}
                )

                # Create a structured output model with proper tracing
                structured_output_params = self.model_registry.get_structured_output_params(
                    model_name, GeneratedFiles
                )
                structured_model = model.with_structured_output(
                    GeneratedFiles, **structured_output_params
                )

                # Invoke the model with chat messages
                response = structured_model.invoke(messages)
                # Convert response to dictionary format
                generated_files = {
                    "main.py": response.main_py,
                    "requirements.txt": response.requirements_txt,
                    "policy.json": response.policy_json,
                }

                # Parse the main Python code
                code_result = self.code_parser.parse(generated_files["main.py"])
                metadata = self.metadata_parser.parse(str(response))

                # Create result
                result = WorkflowResult(
                    prompt_name=prompt_name,
                    model_name=model_name,
                    code_result=code_result,
                    metadata=metadata,
                    timestamp=start_time,
                    execution_time=(datetime.now() - start_time).total_seconds(),
                    success=True,
                    run_id=runs_cb.traced_runs[0].id,
                    generated_files=generated_files,
                )

                # Save result
                self._save_result(result)

                return result

        except Exception as e:
            return WorkflowResult(
                prompt_name=prompt_name,
                model_name=model_name,
                code_result=CodeParseResult(
                    code="",
                    imports=set(),
                    boto3_services=set(),
                    syntax_valid=False,
                    errors=[str(e)],
                    security_risks=[],
                    dependencies=set(),
                ),
                metadata={},
                timestamp=start_time,
                execution_time=(datetime.now() - start_time).total_seconds(),
                success=False,
                error=str(e),
                generated_files={},
            )

    def _save_result(self, result: WorkflowResult) -> None:
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
                file_path = run_dir / filename
                with file_path.open("w") as f:
                    f.write(content)

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
                    "metadata": result.metadata,
                    "code_info": {
                        "imports": list(result.code_result.imports),
                        "boto3_services": list(result.code_result.boto3_services),
                        "syntax_valid": result.code_result.syntax_valid,
                        "errors": result.code_result.errors,
                        "security_risks": result.code_result.security_risks,
                        "dependencies": list(result.code_result.dependencies),
                    },
                },
                f,
                indent=2,
            )


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
    ) -> List[Dict[str, Any]]:
        """List workflow results with optional filtering."""
        results = []

        # Look for metadata files
        for meta_file in self.output_dir.glob("*_meta.json"):
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

    def get_result(self, prompt_name: str, timestamp: str) -> Optional[Dict[str, Any]]:
        """Get a specific workflow result."""
        for meta_file in self.output_dir.glob(f"{prompt_name}_*_{timestamp}_meta.json"):
            try:
                with meta_file.open("r") as f:
                    return json.load(f)
            except Exception:
                continue
        return None

    def get_statistics(self) -> Dict[str, Any]:
        """Get execution statistics."""
        results = self.list_results()

        stats = {
            "total_executions": len(results),
            "successful_executions": sum(1 for r in results if r["success"]),
            "failed_executions": sum(1 for r in results if not r["success"]),
            "average_execution_time": (
                sum(r["execution_time"] for r in results) / len(results)
                if results
                else 0
            ),
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
                stats["common_errors"][error_type] = (
                    stats["common_errors"].get(error_type, 0) + 1
                )

        return stats
