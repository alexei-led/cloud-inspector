"""Node functions for orchestration workflow."""

import logging
from datetime import datetime
from typing import Any, Optional, cast

from langchain.prompts import ChatPromptTemplate

from cloud_inspector.code_generator import CodeGeneratorAgent, CodeGeneratorResult, ParseError
from cloud_inspector.components.types import CloudProvider, CodeGenerationPrompt, WorkflowStatus
from cloud_inspector.execution_agent import CodeExecutionAgent, ExecutionResult
from cloud_inspector.prompt_generator import PromptGeneratorAgent

from .state import OrchestrationState

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 3


def discovery_analysis_node(state: OrchestrationState, agents: dict[str, Any]) -> OrchestrationState:
    """Analyzes new discoveries against all previous findings to determine uniqueness."""
    try:
        if not state["discoveries"] or state.get("status") != WorkflowStatus.IN_PROGRESS:
            return state

        current_discovery = state["discoveries"][-1]

        # Skip analysis if this is the first discovery
        if len(state["discoveries"]) == 1:
            return state

        # Compare current discovery with all previous discoveries
        previous_discoveries = state["discoveries"][:-1]
        is_redundant = _is_discovery_redundant(current_discovery, previous_discoveries, agents)

        if is_redundant:
            state["status"] = WorkflowStatus.COMPLETED
            state["reason"] = "no_new_information_found"
            # Remove the redundant discovery
            state["discoveries"].pop()
            # Copy last successful discovery output to outputs
            if state["discoveries"]:
                state["outputs"].update(state["discoveries"][-1]["output"])

        state["updated_at"] = datetime.now()
    except Exception as e:
        state["status"] = WorkflowStatus.FAILED
        state["outputs"]["error"] = str(e)
    return state


def _is_discovery_redundant(current: dict, previous_discoveries: list[dict], agents: dict[str, Any]) -> bool:
    """Determine if current discovery adds new information compared to all previous discoveries.

    Uses AI to analyze if the current discovery's information is already contained within
    the collective knowledge from previous discoveries.

    Returns:
        bool: True if current discovery is redundant (contained within previous discoveries),
              False if it provides new information.
    """
    from cloud_inspector.components.models import ModelRegistry

    model_registry: ModelRegistry = agents["model_registry"]
    model = model_registry.get_model(agents["model_name"])

    template = """Analyze if the new discovery provides unique information compared to the existing discoveries.

Existing Discoveries:
{previous}

New Discovery:
{current}

Determine if the new discovery's information is already contained within the existing discoveries.
Respond with 'redundant' if the new information is already covered by existing discoveries,
or 'unique' if it provides genuinely new insights.
Just output 'redundant' or 'unique' without any other text."""

    prompt = ChatPromptTemplate.from_template(template)

    # Convert discoveries to string format for comparison
    current_str = str(current.get("output", {}))
    previous_str = str([d.get("output", {}) for d in previous_discoveries])

    messages = prompt.format_messages(previous=previous_str, current=current_str)
    result = model.invoke(messages).content.strip().lower()

    return result == "redundant"


def orchestration_node(state: OrchestrationState, agents: dict[str, Any]) -> OrchestrationState:
    """Decides next steps based on current state and analysis of discoveries."""
    now = datetime.now()

    # Update execution metrics
    start_time = datetime.fromisoformat(state["execution_metrics"]["start_time"])
    state["execution_metrics"]["total_execution_time"] = (now - start_time).total_seconds()

    # If status is already final, do nothing
    if state.get("status") in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED]:
        return state

    # Validate workflow state
    if state.get("status") != WorkflowStatus.IN_PROGRESS:
        state["status"] = WorkflowStatus.FAILED
        state["reason"] = "invalid_workflow_state"
        state["updated_at"] = now
        return state

    # Handle errors and retries
    if state["outputs"].get("error"):
        state["error_count"] += 1
        state["retry_attempts"] += 1

        if state["retry_attempts"] < 2:  # Allow up to 2 retries
            # Clear error for retry
            state["outputs"].pop("error")
            state["updated_at"] = now
            return state

        # Max retries reached - update final state with error
        state["status"] = WorkflowStatus.FAILED
        state["reason"] = f"error_in_iteration_{state['iteration']}_after_{state['retry_attempts']}_retries"
        # Keep error in outputs for final state
        error_msg = state["outputs"]["error"]
        state["outputs"].clear()  # Clear other outputs
        state["outputs"]["error"] = error_msg  # Preserve just the error
        state["updated_at"] = now
        return state

    # Check maximum iterations
    if state["iteration"] >= MAX_ITERATIONS:
        state["status"] = WorkflowStatus.COMPLETED
        state["reason"] = "max_iterations_reached"
        state["updated_at"] = now
        return state

    # Update successful iteration state
    state["last_successful_iteration"] = state["iteration"]
    state["retry_attempts"] = 0  # Reset retry counter
    state["iteration"] += 1
    state["updated_at"] = now
    return state


def prompt_generation_node(state: OrchestrationState, agents: dict[str, Any]) -> OrchestrationState:
    """Generates prompts using PromptGeneratorAgent."""
    try:
        prompt_generator: PromptGeneratorAgent = agents["prompt_generator"]

        # Convert variables to list format expected by prompt generator
        var_list = [{"name": k, "value": v} for k, v in state["params"].items()]

        # Generate prompt based on current state
        prompt = prompt_generator.generate_prompt(
            model_name=agents["model_name"],
            cloud=cast(CloudProvider, state["cloud"]),
            service=state["service"],
            operation="inspect",
            request=state["request"],
            variables=var_list,
            previous_results=state["discoveries"][-1] if state["discoveries"] else None,
            iteration=state["iteration"],
        )

        # Update variables from prompt
        if isinstance(prompt, CodeGenerationPrompt) and prompt.variables:
            state["params"].update({var["name"]: var["value"] for var in prompt.variables if var["name"] not in state["params"]})

        state["outputs"]["prompt"] = prompt
        state["updated_at"] = datetime.now()
    except Exception as e:
        state["status"] = WorkflowStatus.FAILED
        state["outputs"]["error"] = str(e)
        state["error_count"] += 1  # Add this line to increment error count
        return state
    return state


def code_generation_node(state: OrchestrationState, agents: dict[str, Any]) -> OrchestrationState:
    """Generates code using CodeGeneratorAgent.

    Args:
        state: Current workflow state
        agents: Dictionary containing required agents and configuration

    Returns:
        Updated workflow state
    """
    if state.get("status") != WorkflowStatus.IN_PROGRESS or "prompt" not in state["outputs"]:
        return state

    try:
        code_generator: CodeGeneratorAgent = agents["code_generator"]
        prompt = state["outputs"]["prompt"]

        logger.debug("Generating code for prompt in iteration %d", state["iteration"])

        result: CodeGeneratorResult = code_generator.generate_code(
            prompt=prompt,
            model_name=agents["model_name"],
            variables=state["params"],
            iteration_id=f"iter_{state['iteration']}"
        )

        # Store the CodeGeneratorResult directly
        state["outputs"]["code"] = result
        state["updated_at"] = datetime.now()

        logger.debug("Code generation successful for iteration %d", state["iteration"])

    except ParseError as e:
        logger.error("JSON parsing error in code generation: %s", str(e))
        state["status"] = WorkflowStatus.FAILED
        state["outputs"]["error"] = f"Invalid JSON format: {str(e)}"
        state["error_count"] += 1
        return state
    except Exception as e:
        logger.error("Unexpected error in code generation: %s", str(e), exc_info=True)
        state["status"] = WorkflowStatus.FAILED
        state["outputs"]["error"] = str(e)
        state["error_count"] += 1
        return state

    return state


def code_execution_node(
    state: OrchestrationState,
    agents: dict[str, Any],
    credentials: Optional[dict[str, str]],
    cloud_context: Optional[str]
) -> OrchestrationState:
    """Executes generated code using CodeExecutionAgent.

    Args:
        state: Current workflow state
        agents: Dictionary containing required agents
        credentials: Cloud provider credentials
        cloud_context: Cloud account/project/subscription identifier

    Returns:
        Updated workflow state
    """
    if state.get("status") == WorkflowStatus.FAILED or "code" not in state["outputs"]:
        return state

    code_executor: CodeExecutionAgent = agents["code_executor"]

    try:
        # Get the code result
        code_result = state["outputs"]["code"]
        if isinstance(code_result, tuple):
            code_result = code_result[0]  # Extract CodeGeneratorResult from tuple
        if not isinstance(code_result, CodeGeneratorResult):
            raise TypeError(f"Expected CodeGeneratorResult, got {type(code_result)}")

        logger.debug("Executing generated code for iteration %d", state["iteration"])

        # Add cloud context to credentials if provided
        execution_credentials = credentials.copy() if credentials else {}
        if cloud_context:
            execution_credentials["cloud_context"] = cloud_context

        execution_result: ExecutionResult = code_executor.execute_generated_code(
            generated_files=code_result.generated_files,
            credentials=execution_credentials,
            execution_id=f"exec_{state['iteration']}"
        )

        if execution_result.success:
            parsed_output = execution_result.get_parsed_output()
            if parsed_output is not None:
                discovery = {
                    "output": parsed_output,
                    "timestamp": datetime.now().isoformat(),
                    "iteration": state["iteration"],
                    "execution_time": execution_result.execution_time,
                    "resource_usage": execution_result.resource_usage
                }
                state["discoveries"].append(discovery)
                logger.info("Successfully added discovery for iteration %d", state["iteration"])
            else:
                error_msg = execution_result.error or "Code execution succeeded but did not produce valid JSON output"
                logger.warning("No valid JSON output: %s", error_msg)
                state["outputs"]["error"] = error_msg
        else:
            error_msg = execution_result.error or "Code execution failed"
            logger.error("Code execution failed: %s", error_msg)
            state["outputs"]["error"] = error_msg

        # Update execution metrics
        state["execution_metrics"]["resource_usage"].update(execution_result.resource_usage)

    except TypeError as e:
        logger.error("Type error in code execution: %s", str(e))
        state["status"] = WorkflowStatus.FAILED
        state["outputs"]["error"] = f"Invalid code generation result: {str(e)}"
        state["error_count"] += 1
    except Exception as e:
        logger.exception("Unexpected error in code execution")
        state["outputs"]["error"] = f"Code execution error: {str(e)}"
        state["error_count"] += 1

    state["updated_at"] = datetime.now()
    return state
