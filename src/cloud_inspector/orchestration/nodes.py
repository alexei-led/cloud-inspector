"""Node functions for orchestration workflow."""

from typing import Any, cast

from components.types import CloudProvider, CodeGenerationPrompt
from cloud_inspector.code_generator import CodeGeneratorAgent
from cloud_inspector.prompt_generator import PromptGeneratorAgent
from cloud_inspector.execution_agent import CodeExecutionAgent
from .state import OrchestrationState

MAX_ITERATIONS = 3


def orchestration_node(
    state: OrchestrationState,
    agents: dict[str, Any]
) -> OrchestrationState:
    """Decides next steps based on current state."""
    if state["current_iteration"] >= MAX_ITERATIONS:
        state["status"] = "completed"
        state["completion_reason"] = "max_iterations_reached"
        return state

    if state.get("status") != "in_progress":
        return state

    # Check if we had an error in the last iteration
    if state["agent_outputs"].get("error"):
        state["status"] = "failed"
        state["completion_reason"] = f"error_in_iteration_{state['current_iteration']}"
        return state

    state["current_iteration"] += 1
    state["updated_at"] = datetime.now()
    return state


def prompt_generation_node(
    state: OrchestrationState,
    agents: dict[str, Any]
) -> OrchestrationState:
    """Generates prompts using PromptGeneratorAgent."""
    prompt_generator: PromptGeneratorAgent = agents["prompt_generator"]
    
    # Convert variables to list format expected by prompt generator
    var_list = [{"name": k, "value": v} for k, v in state["variables"].items()]
    
    # Generate prompt based on current state
    prompt = prompt_generator.generate_prompt(
        model_name=agents["model_name"],
        cloud=cast(CloudProvider, state["cloud"]),
        service=state["service"],
        operation="inspect",
        request=state["user_request"],
        variables=var_list,
        previous_results=state["collected_data"][-1] if state["collected_data"] else None,
        iteration=state["current_iteration"]
    )
    
    # Update variables from prompt
    if isinstance(prompt, CodeGenerationPrompt) and prompt.variables:
        state["variables"].update({
            var["name"]: var["value"] 
            for var in prompt.variables 
            if var["name"] not in state["variables"]
        })
    
    state["agent_outputs"]["prompt"] = prompt
    state["updated_at"] = datetime.now()
    return state


def code_generation_node(
    state: OrchestrationState,
    agents: dict[str, Any]
) -> OrchestrationState:
    """Generates code using CodeGeneratorAgent."""
    code_generator: CodeGeneratorAgent = agents["code_generator"]
    prompt = state["agent_outputs"]["prompt"]
    
    result = code_generator.generate_code(
        prompt=prompt,
        model_name=agents["model_name"],
        variables=state["variables"],
        iteration_id=f"iter_{state['current_iteration']}"
    )
    
    state["agent_outputs"]["code"] = result
    state["updated_at"] = datetime.now()
    return state


def code_execution_node(
    state: OrchestrationState,
    agents: dict[str, Any]
) -> OrchestrationState:
    """Executes code using CodeExecutionAgent."""
    code_executor: CodeExecutionAgent = agents["code_executor"]
    code_result = state["agent_outputs"]["code"]
    
    # Execute the generated code
    execution_result = code_executor.execute_generated_code(
        generated_files=code_result.generated_files,
        aws_credentials=agents.get("aws_credentials"),
        execution_id=f"exec_{state['current_iteration']}"
    )
    
    if execution_result.success:
        state["collected_data"].append(execution_result.output)
    else:
        state["agent_outputs"]["error"] = execution_result.error
        
    state["updated_at"] = datetime.now()
    return state
