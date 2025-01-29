"""Agent for executing generated code in isolated environment."""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from cloud_inspector.code_execution import DockerSandbox
from components.types import GeneratedFiles

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of code execution."""

    success: bool
    output: str
    error: Optional[str]
    execution_time: float
    executed_at: datetime
    resource_usage: dict[str, Any]
    generated_files: GeneratedFiles


class CodeExecutionAgent:
    """Agent responsible for safely executing generated code."""

    def __init__(
        self,
        sandbox: Optional[DockerSandbox] = None,
        max_execution_time: int = 30,
    ):
        """Initialize the execution agent.

        Args:
            sandbox: Optional preconfigured DockerSandbox instance
            max_execution_time: Maximum execution time in seconds
        """
        self.sandbox = sandbox or DockerSandbox(
            image="python:3.12-slim",
            cpu_limit=1.0,
            memory_limit="512m",
            timeout=max_execution_time,
        )

    def execute_generated_code(
        self,
        generated_files: GeneratedFiles,
        aws_credentials: Optional[dict[str, str]] = None,
        execution_id: Optional[str] = None,
    ) -> ExecutionResult:
        """Execute code generated by CodeGeneratorAgent.

        Args:
            generated_files: GeneratedFiles from CodeGeneratorAgent
            aws_credentials: Optional AWS credentials
            execution_id: Optional identifier for this execution

        Returns:
            ExecutionResult containing output and metadata
        """
        start_time = datetime.now()

        try:
            # Validate generated files
            if not generated_files.main_py.strip():
                raise ValueError("Empty main.py content")

            # Log execution attempt
            logger.info(
                "Executing generated code",
                extra={
                    "execution_id": execution_id,
                    "requirements": generated_files.requirements_txt.split("\n"),
                    "has_policy": bool(generated_files.policy_json.strip()),
                },
            )

            # Execute in sandbox
            success, stdout, stderr, resource_usage = self.sandbox.execute(
                main_py=generated_files.main_py,
                requirements_txt=generated_files.requirements_txt,
                aws_credentials=aws_credentials,
            )

            execution_time = (datetime.now() - start_time).total_seconds()

            # Add execution time to resource usage
            resource_usage["execution_time"] = execution_time

            return ExecutionResult(
                success=success,
                output=stdout,
                error=stderr if stderr else None,
                execution_time=execution_time,
                executed_at=start_time,
                resource_usage=resource_usage,
                generated_files=generated_files,
            )

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Code execution failed: {e}", exc_info=True)
            return ExecutionResult(
                success=False,
                output="",
                error=str(e),
                execution_time=execution_time,
                executed_at=start_time,
                resource_usage={"execution_time": execution_time},
                generated_files=generated_files,
            )

    def cleanup(self):
        """Cleanup any resources used by the agent."""
        try:
            self.sandbox.cleanup()
        except Exception as e:
            logger.error(f"Failed to cleanup sandbox: {e}")
