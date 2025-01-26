"""Manages iterative data collection process."""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

from pydantic import BaseModel

from cloud_inspector.prompt_generator import PromptGenerator
from cloud_inspector.prompts import CloudProvider, PromptManager
from cloud_inspector.workflow import CodeGenerationWorkflow, WorkflowResult

from .execution import DockerSandbox


@dataclass
class CollectedData:
    """Data collected from manual code execution."""

    iteration_id: str
    timestamp: datetime
    data: dict[str, Any]
    source_files: list[str]
    feedback: Optional[dict[str, str]] = None


class IterationState(BaseModel):
    """State of an iteration process."""

    request_id: str
    original_request: str
    cloud: CloudProvider
    service: str
    current_iteration: int
    collected_data: list[dict]  # List of CollectedData as dicts
    status: str = "in_progress"  # in_progress, completed, failed
    created_at: datetime
    updated_at: datetime
    completion_reason: Optional[str] = None


class IterationManager:
    """Manages the iterative data collection process."""

    def __init__(
        self,
        prompt_manager: PromptManager,
        workflow: CodeGenerationWorkflow,
        prompt_generator: PromptGenerator,
        data_dir: Optional[Path] = None,
    ):
        self.prompt_manager = prompt_manager
        self.workflow = workflow
        self.prompt_generator = prompt_generator
        self.data_dir = data_dir or Path("collected_data")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.sandbox = DockerSandbox()

    def execute_iteration(
        self,
        model_name: str,
        request_id: Optional[str] = None,
        request: Optional[str] = None,
        cloud: Optional[CloudProvider] = None,
        service: Optional[str] = None,
    ) -> tuple[str, WorkflowResult, Path]:
        """Execute an iteration, either starting new or continuing existing.

        Args:
            model_name: Name of the model to use
            request_id: Optional ID of existing iteration process
            request: Required for new iterations - the user's original request
            cloud: Required for new iterations - the cloud provider
            service: Required for new iterations - the service to inspect

        Returns:
            A tuple containing:
            - request_id: Unique identifier for this iteration process
            - result: The workflow execution result
            - output_path: Path where the generated files are saved

        Raises:
            ValueError: If parameters are invalid for new/existing iteration
        """
        if request_id:
            # Continue existing iteration
            state = self._load_state(request_id)
            if state.status != "in_progress":
                raise ValueError(f"Iteration {request_id} is not in progress")
            
            # Increment iteration counter
            state.current_iteration += 1
        else:
            # Start new iteration
            if not all([request, cloud, service]):
                raise ValueError("request, cloud, and service are required for new iterations")
            
            # Create new state
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            request_id = f"{service}_{timestamp}"
            state = IterationState(
                request_id=request_id,
                original_request=request,
                cloud=cloud,
                service=service,
                current_iteration=1,
                collected_data=[],
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )

        # Aggregate previous data and feedback if any
        aggregated_data = {}
        latest_feedback = None
        if state.collected_data:
            for collected in state.collected_data:
                aggregated_data.update(collected["data"])
            latest_feedback = state.collected_data[-1].get("feedback")

        # Generate prompt
        variables = [
            {"name": "request", "value": state.original_request},
            {"name": "service", "value": state.service},
            {"name": "iteration", "value": str(state.current_iteration)}
        ]
        if state.current_iteration == 1:
            variables.append({"name": "region", "value": "us-west-2"})

        prompt = self.prompt_generator.generate_prompt(
            model_name=model_name,
            cloud=state.cloud,
            service=state.service,
            operation="inspect",
            description=state.original_request,
            variables=variables,
            tags=["cloud_inspection", state.service],
            previous_results=aggregated_data if aggregated_data else None,
            feedback=latest_feedback,
            iteration=state.current_iteration,
        )

        # Execute workflow
        result, output_path = self.workflow.execute(
            prompt=prompt,
            model_name=model_name,
            variables={
                "request": state.original_request,
                "service": state.service,
                "iteration": state.current_iteration,
            },
            iteration_id=request_id,
            iteration_number=state.current_iteration,
            previous_results=aggregated_data if aggregated_data else None,
            feedback=latest_feedback,
        )

        # Update and save state
        state.updated_at = datetime.now()
        self._save_state(state)

        return request_id, result, output_path

    def save_collected_data(
        self,
        request_id: str,
        data: Union[dict, str, Path],
        source_files: list[str],
        feedback: Optional[dict[str, str]] = None,
    ) -> None:
        """Save data collected from manual code execution."""
        state = self._load_state(request_id)

        # Convert data to dictionary if it's a string or Path
        if isinstance(data, (str, Path)):
            with open(data) as f:
                processed_data = json.load(f)
        else:
            processed_data = data

        # Create collected data entry
        collected = CollectedData(
            iteration_id=f"{request_id}_{state.current_iteration}",
            timestamp=datetime.now(),
            data=processed_data,
            source_files=source_files,
            feedback=feedback,
        )

        # Update state
        state.collected_data.append(asdict(collected))
        state.updated_at = datetime.now()
        self._save_state(state)

        # Save collected data to a separate file
        data_file = self.data_dir / f"{collected.iteration_id}_data.json"
        with open(data_file, "w") as f:
            json.dump(processed_data, f, indent=2)

    def complete_iteration(
        self,
        request_id: str,
        reason: str = "completed",
    ) -> None:
        """Mark an iteration process as complete."""
        state = self._load_state(request_id)
        state.status = "completed"
        state.completion_reason = reason
        state.updated_at = datetime.now()
        self._save_state(state)

    def get_collected_data(
        self,
        request_id: str,
    ) -> dict[str, Any]:
        """Get all collected data for a request."""
        state = self._load_state(request_id)

        # Aggregate all collected data
        aggregated_data = {}
        for collected in state.collected_data:
            aggregated_data.update(collected["data"])

        return aggregated_data

    def _save_state(self, state: IterationState) -> None:
        """Save iteration state to file."""
        state_file = self.data_dir / f"{state.request_id}_state.json"
        with open(state_file, "w") as f:
            f.write(state.json(indent=2))

    def _load_state(self, request_id: str) -> IterationState:
        """Load iteration state from file."""
        state_file = self.data_dir / f"{request_id}_state.json"
        if not state_file.exists():
            raise ValueError(f"No state found for request {request_id}")

        with open(state_file) as f:
            return IterationState.parse_raw(f.read())

    def _validate_execution_result(self, stdout: str, stderr: str) -> tuple[bool, str]:
        """Validate execution result.

        Args:
            stdout: Standard output from execution
            stderr: Standard error from execution

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check for common error patterns
        error_patterns = [
            "AccessDenied",
            "NoCredentialsError",
            "ClientError",
            "BotoCoreError",
            "Exception",
            "Error",
        ]

        for pattern in error_patterns:
            if pattern in stderr:
                return False, f"Execution failed: {pattern} found in stderr"

        if stderr and not stdout:
            return False, "Execution produced only errors"

        return True, ""

    def execute_code(
        self,
        code: str,
        aws_credentials: Optional[dict[str, str]] = None,
    ) -> tuple[bool, dict[str, Any]]:
        """Execute code in sandbox and validate results.

        Args:
            code: Python code to execute
            aws_credentials: Optional AWS credentials

        Returns:
            Tuple of (success, results)
        """
        success, stdout, stderr = self.sandbox.execute(code, aws_credentials)

        # Validate execution results
        is_valid, error_msg = self._validate_execution_result(stdout, stderr)

        results = {
            "stdout": stdout,
            "stderr": stderr,
            "success": success and is_valid,
            "error": error_msg if not is_valid else None,
        }

        return success and is_valid, results

    def cleanup(self):
        """Cleanup resources."""
        if hasattr(self, "sandbox"):
            self.sandbox.cleanup()
