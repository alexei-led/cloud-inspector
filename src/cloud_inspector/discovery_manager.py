"""Manages iterative cloud resource discovery process."""

from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel

from cloud_inspector.prompt_generator import PromptGenerator
from cloud_inspector.prompts import CloudProvider, PromptManager
from cloud_inspector.workflow import CodeGenerationWorkflow, WorkflowResult


class DiscoveryState(BaseModel):
    """State of a discovery process."""

    request_id: str  # Format: "{service}_discovery_{timestamp}"
    original_request: str
    cloud: CloudProvider
    service: str
    variables: dict[str, Any] = {}  # Configuration variables
    discovered_data: dict[str, Any] = {}  # All discovered data
    status: str = "in_progress"  # in_progress, completed, failed
    created_at: datetime
    updated_at: datetime
    completion_reason: Optional[str] = None
    discovery_complete: bool = False
    max_iterations: int = 3

    @property
    def iteration_count(self) -> int:
        """Number of discovery iterations performed."""
        return len(self.discovered_data)

    @property
    def should_continue(self) -> bool:
        """Whether discovery should continue."""
        return self.status == "in_progress" and not self.discovery_complete and self.iteration_count < self.max_iterations


class DiscoveryManager:
    """Manages the iterative cloud resource discovery process."""

    def __init__(
        self,
        prompt_manager: PromptManager,
        workflow: CodeGenerationWorkflow,
        prompt_generator: PromptGenerator,
        state_dir: Optional[Path] = None,
    ):
        self.prompt_manager = prompt_manager
        self.workflow = workflow
        self.prompt_generator = prompt_generator
        self.state_dir = state_dir or Path("discovery_states")
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def execute_discovery(
        self,
        model_name: str,
        request: Optional[str] = None,
        cloud: Optional[CloudProvider] = None,
        service: Optional[str] = None,
        request_id: Optional[str] = None,
        variables: Optional[dict[str, Any]] = None,
    ) -> tuple[str, WorkflowResult, Path]:
        """Execute a discovery iteration, either starting new or continuing existing.

        Args:
            model_name: Name of the model to use for code generation
            request: Description of what to discover (required for new discovery)
            cloud: Cloud provider to inspect (required for new discovery)
            service: Service to inspect (required for new discovery)
            request_id: ID of existing discovery to continue
            variables: Optional variables for the discovery process

        Returns:
            Tuple of (request_id, execution_result, output_path)

        Raises:
            ValueError: If parameters are invalid or discovery cannot continue
        """
        # Load or create discovery state
        state = self._get_or_create_state(
            request_id=request_id,
            request=request,
            cloud=cloud,
            service=service,
            variables=variables or {},
        )

        # Check if we can continue
        if not state.should_continue:
            raise ValueError(f"Discovery {state.request_id} cannot continue: Status={state.status}, Complete={state.discovery_complete}, Iterations={state.iteration_count}/{state.max_iterations}")

        # Generate prompt using current state
        prompt = self.prompt_generator.generate_prompt(
            model_name=model_name,
            cloud=state.cloud,
            service=state.service,
            operation="inspect",
            description=state.original_request,
            variables=[{"name": k, "value": v} for k, v in state.variables.items()],
            tags=["cloud_inspection", state.service],
            previous_results=state.discovered_data,
            iteration=state.iteration_count + 1,
        )

        # Update state with any new variables from prompt
        if hasattr(prompt, "variables") and prompt.variables:
            state.variables.update({var["name"]: var["value"] for var in prompt.variables})

        # Validate all required variables have values
        self._validate_variables(state.variables)

        # Execute workflow
        result, output_path = self.workflow.execute(
            prompt=prompt,
            model_name=model_name,
            variables=state.variables,
            iteration_id=state.request_id,
            previous_results=state.discovered_data,
        )

        # Update state with new discoveries
        if result.execution_results:
            state.discovered_data.update(result.execution_results)

        # Check if discovery is complete
        if hasattr(prompt, "discovery_complete"):
            state.discovery_complete = prompt.discovery_complete

        # Save updated state
        state.updated_at = datetime.now()
        self._save_state(state)

        return state.request_id, result, output_path

    def complete_discovery(self, request_id: str, reason: str = "completed") -> None:
        """Mark a discovery process as complete."""
        state = self._load_state(request_id)
        state.status = "completed"
        state.completion_reason = reason
        state.discovery_complete = True
        state.updated_at = datetime.now()
        self._save_state(state)

    def get_discovered_data(self, request_id: str) -> dict[str, Any]:
        """Get all discovered data for a request."""
        state = self._load_state(request_id)
        return state.discovered_data

    def _get_or_create_state(
        self,
        request_id: Optional[str],
        request: Optional[str],
        cloud: Optional[CloudProvider],
        service: Optional[str],
        variables: dict[str, Any],
    ) -> DiscoveryState:
        """Load existing state or create new one."""
        if request_id:
            return self._load_state(request_id)

        # Validate parameters for new discovery
        if not all([request, cloud, service]):
            raise ValueError("request, cloud, and service are required for new discovery")

        # Create new state with unique ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_request_id = f"{service}_discovery_{timestamp}"

        state = DiscoveryState(
            request_id=new_request_id,
            original_request=request,  # type: ignore
            cloud=cloud,  # type: ignore
            service=service,  # type: ignore
            variables=variables,
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        self._save_state(state)
        return state

    def _validate_variables(self, variables: dict[str, Any]) -> None:
        """Validate all required variables have values."""
        missing = {k for k, v in variables.items() if v is None}
        if missing:
            raise ValueError(f"Missing values for variables: {missing}")

    def _load_state(self, request_id: str) -> DiscoveryState:
        """Load discovery state from file."""
        state_file = self.state_dir / f"{request_id}.json"
        if not state_file.exists():
            raise ValueError(f"No state found for request {request_id}")
        return DiscoveryState.parse_file(state_file)

    def _save_state(self, state: DiscoveryState) -> None:
        """Save discovery state to file."""
        state_file = self.state_dir / f"{state.request_id}.json"
        state_file.write_text(state.model_dump_json(indent=2))
