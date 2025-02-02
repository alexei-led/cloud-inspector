"""Node functions for orchestration workflow."""

import logging
from datetime import datetime
from typing import Any, cast

from langchain.prompts import ChatPromptTemplate

from cloud_inspector.code_generator import CodeGeneratorAgent
from cloud_inspector.components.types import CloudProvider, CodeGenerationPrompt, WorkflowStatus
from cloud_inspector.execution_agent import CodeExecutionAgent
from cloud_inspector.prompt_generator import PromptGeneratorAgent

from .state import OrchestrationState

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 3


def discovery_analysis_node(state: OrchestrationState, agents: dict[str, Any]) -> OrchestrationState:
    """Analyzes new discoveries against all previous findings to determine uniqueness."""
    if not state["discoveries"] or state.get("status") != "in_progress":
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

    state["updated_at"] = datetime.now()
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

    if state.get("status") != "in_progress":
        return state

    # Handle errors and retries
    if state["outputs"].get("error"):
        state["error_count"] += 1

        # Check if we can retry
        if state["retry_attempts"] < 2:  # Allow up to 2 retries
            state["retry_attempts"] += 1
            state["outputs"]["error"] = None  # Clear error for retry
            state["updated_at"] = now
            return state

        state["status"] = WorkflowStatus.FAILED
        state["reason"] = f"error_in_iteration_{state['iteration']}_after_{state['retry_attempts']}_retries"
        return state

    # Check maximum iterations as a safety limit
    if state["iteration"] >= MAX_ITERATIONS:
        state["status"] = WorkflowStatus.COMPLETED
        state["reason"] = "max_iterations_reached"
        return state

    # Successful iteration
    state["last_successful_iteration"] = state["iteration"]
    state["retry_attempts"] = 0  # Reset retry counter
    state["iteration"] += 1
    state["updated_at"] = now
    return state


def prompt_generation_node(state: OrchestrationState, agents: dict[str, Any]) -> OrchestrationState:
    """Generates prompts using PromptGeneratorAgent."""
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
    return state


def code_generation_node(state: OrchestrationState, agents: dict[str, Any]) -> OrchestrationState:
    """Generates code using CodeGeneratorAgent."""
    code_generator: CodeGeneratorAgent = agents["code_generator"]
    prompt = state["outputs"]["prompt"]

    result = code_generator.generate_code(prompt=prompt, model_name=agents["model_name"], variables=state["params"], iteration_id=f"iter_{state['iteration']}")

    state["outputs"]["code"] = result
    state["updated_at"] = datetime.now()
    return state


def code_execution_node(state: OrchestrationState, agents: dict[str, Any]) -> OrchestrationState:
    """Executes code using CodeExecutionAgent."""
    code_executor: CodeExecutionAgent = agents["code_executor"]
    code_result = state["outputs"]["code"]

    try:
        # Execute the generated code
        execution_result = code_executor.execute_generated_code(generated_files=code_result.generated_files, aws_credentials=agents.get("aws_credentials"), execution_id=f"exec_{state['iteration']}")

        if execution_result.success:
            # Try to parse output as JSON
            parsed_output = execution_result.get_parsed_output()
            if parsed_output is not None:
                # Add parsed discovery to state
                discovery = {"output": parsed_output, "timestamp": datetime.now().isoformat(), "iteration": state["iteration"], "execution_time": execution_result.execution_time, "resource_usage": execution_result.resource_usage}
                state["discoveries"].append(discovery)
            else:
                state["outputs"]["error"] = execution_result.error or "Code execution succeeded but did not produce valid JSON output"
        else:
            state["outputs"]["error"] = execution_result.error or "Code execution failed"

        # Update execution metrics
        state["execution_metrics"]["resource_usage"].update(execution_result.resource_usage)

    except Exception as e:
        logger.exception("Code execution node failed")
        state["outputs"]["error"] = f"Code execution error: {str(e)}"

    state["updated_at"] = datetime.now()
    return state
