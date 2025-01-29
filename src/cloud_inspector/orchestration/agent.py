"""Orchestration agent for managing cloud inspection workflow."""

from typing import Optional
from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END

from cloud_inspector.code_generator import CodeGeneratorAgent
from cloud_inspector.prompt_generator import PromptGeneratorAgent
from cloud_inspector.execution_agent import CodeExecutionAgent
from components.types import CloudProvider
from .nodes import orchestration_node, prompt_generation_node, code_generation_node, code_execution_node
from .state import OrchestrationState, create_initial_state

class OrchestrationAgent:
    """Manages the iterative cloud inspection workflow."""

    def __init__(
        self,
        code_generator: CodeGeneratorAgent,
        prompt_generator: PromptGeneratorAgent,
        code_executor: CodeExecutionAgent,
        model_name: str,
        checkpointer: Optional[MemorySaver] = None,
        state_dir: Optional[Path] = None,
    ):
        self.code_generator = code_generator
        self.prompt_generator = prompt_generator
        self.code_executor = code_executor
        self.model_name = model_name
        self.checkpointer = checkpointer or MemorySaver()
        self.state_dir = state_dir or Path("orchestration_states")
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.workflow = self._create_workflow()

    def _create_workflow(self) -> StateGraph:
        """Create and configure the workflow graph."""
        workflow = StateGraph(OrchestrationState)
        
        # Add nodes with agent context
        workflow.add_node("orchestrate", 
            lambda state: orchestration_node(state, {
                "model_name": self.model_name
            }))
        
        workflow.add_node("generate_prompt", 
            lambda state: prompt_generation_node(state, {
                "model_name": self.model_name,
                "prompt_generator": self.prompt_generator
            }))
            
        workflow.add_node("generate_code",
            lambda state: code_generation_node(state, {
                "model_name": self.model_name,
                "code_generator": self.code_generator
            }))
            
        workflow.add_node("execute_code",
            lambda state: code_execution_node(state, {
                "code_executor": self.code_executor
            }))
        
        # Add edges
        workflow.add_edge(START, "orchestrate")
        workflow.add_edge("orchestrate", "generate_prompt")
        workflow.add_edge("generate_prompt", "generate_code")
        workflow.add_edge("generate_code", "execute_code")
        workflow.add_edge("execute_code", "orchestrate")
        workflow.add_edge("orchestrate", END)
        
        return workflow.compile(checkpointer=self.checkpointer)

    def execute(
        self,
        request: str,
        cloud: CloudProvider,
        service: str,
        thread_id: str,
        variables: Optional[dict] = None
    ) -> dict:
        """Execute the orchestration workflow."""
        initial_state = create_initial_state(
            request=request,
            cloud=cloud,
            service=service,
            variables=variables
        )
        
        config = {"configurable": {"thread_id": thread_id}}
        return self.workflow.invoke(initial_state, config)
