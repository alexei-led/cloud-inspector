"""Node functions for orchestration workflow."""

from datetime import datetime
from typing import Any, cast

from cloud_inspector.code_generator import CodeGeneratorAgent
from cloud_inspector.execution_agent import CodeExecutionAgent
from cloud_inspector.prompt_generator import PromptGeneratorAgent
from components.types import CloudProvider, CodeGenerationPrompt, WorkflowStatus

from .state import OrchestrationState

MAX_ITERATIONS = 3


def orchestration_node(state: OrchestrationState, agents: dict[str, Any]) -> OrchestrationState:
    """Decides next steps based on current state."""
    if state["iteration"] >= MAX_ITERATIONS:
        state["status"] = WorkflowStatus.COMPLETED
        state["reason"] = "max_iterations_reached"
        return state

    if state.get("status") != "in_progress":
        return state

    # Check if we had an error in the last iteration
    if state["outputs"].get("error"):
        state["status"] = WorkflowStatus.FAILED
        state["reason"] = f"error_in_iteration_{state['iteration']}"
        return state

    state["iteration"] += 1
    state["updated_at"] = datetime.now()
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

    # Execute the generated code
    execution_result = code_executor.execute_generated_code(generated_files=code_result.generated_files, aws_credentials=agents.get("aws_credentials"), execution_id=f"exec_{state['iteration']}")

    if execution_result.success:
        # Convert output to dictionary if it's not already
        discovery = {"output": execution_result.output} if isinstance(execution_result.output, str) else execution_result.output
        state["discoveries"].append(discovery)
    else:
        state["outputs"]["error"] = execution_result.error

    state["updated_at"] = datetime.now()
    return state
