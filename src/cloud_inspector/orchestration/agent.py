"""Orchestration agent for managing cloud inspection workflow."""

from typing import Optional

from langchain.schema.runnable import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from cloud_inspector.code_generator import CodeGeneratorAgent
from cloud_inspector.components.types import CloudProvider
from cloud_inspector.execution_agent import CodeExecutionAgent
from cloud_inspector.prompt_generator import PromptGeneratorAgent

from .nodes import code_execution_node, code_generation_node, discovery_analysis_node, orchestration_node, prompt_generation_node
from .state import create_initial_state


class OrchestrationAgent:
    def __init__(self, model_name: str, prompt_generator: PromptGeneratorAgent, code_generator: CodeGeneratorAgent, code_executor: CodeExecutionAgent, checkpointer: Optional[MemorySaver] = None):
        self.model_name = model_name
        self.prompt_generator = prompt_generator
        self.code_generator = code_generator
        self.code_executor = code_executor
        self.checkpointer = checkpointer

    def _create_workflow(self) -> StateGraph:
        workflow = StateGraph()

        # Add nodes with agent context
        workflow.add_node("orchestrate", lambda state: orchestration_node(state, {"model_name": self.model_name}))  # type: ignore
        workflow.add_node("generate_prompt", lambda state: prompt_generation_node(state, {"model_name": self.model_name, "prompt_generator": self.prompt_generator}))  # type: ignore
        workflow.add_node("generate_code", lambda state: code_generation_node(state, {"model_name": self.model_name, "code_generator": self.code_generator}))  # type: ignore
        workflow.add_node("execute_code", lambda state: code_execution_node(state, {"code_executor": self.code_executor}))  # type: ignore
        workflow.add_node("analyze_discovery", lambda state: discovery_analysis_node(state, {"model_name": self.model_name}))  # type: ignore

        # Add edges
        workflow.add_edge(START, "orchestrate")
        workflow.add_edge("orchestrate", "generate_prompt")
        workflow.add_edge("generate_prompt", "generate_code")
        workflow.add_edge("generate_code", "execute_code")
        workflow.add_edge("execute_code", "analyze_discovery")
        workflow.add_edge("analyze_discovery", "orchestrate")
        workflow.add_edge("orchestrate", END)

        return workflow

    def execute(self, request: str, cloud: CloudProvider, service: str, thread_id: str, params: Optional[dict] = None) -> dict:
        """Execute the orchestration workflow."""
        workflow = self._create_workflow()
        compiled_workflow = workflow.compile(checkpointer=self.checkpointer)
        initial_state = create_initial_state(request=request, cloud=cloud, service=service, params=params)
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

        return compiled_workflow.invoke(initial_state, config)
