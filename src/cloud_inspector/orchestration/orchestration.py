"""Orchestration agent for managing cloud inspection workflow."""

from typing import Any, Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from cloud_inspector.code_generator import CodeGeneratorAgent
from cloud_inspector.components.types import CloudProvider, WorkflowStatus
from cloud_inspector.execution_agent import CodeExecutionAgent
from cloud_inspector.prompt_generator import PromptGeneratorAgent

from .nodes import code_execution_node, code_generation_node, discovery_analysis_node, orchestration_node, prompt_generation_node
from .state import OrchestrationState, create_initial_state


class OrchestrationAgent:
    def __init__(
        self,
        model_name: str,
        prompt_generator: PromptGeneratorAgent,
        code_generator: CodeGeneratorAgent,
        code_executor: CodeExecutionAgent,
        model_registry: Any = None,
        checkpointer: Optional[MemorySaver] = None,
        credentials: Optional[dict[str, str]] = None,
        cloud_context: Optional[str] = None,
    ):
        self.model_name = model_name
        self.prompt_generator = prompt_generator
        self.code_generator = code_generator
        self.code_executor = code_executor
        self.model_registry = model_registry
        self.checkpointer = checkpointer
        self.credentials = credentials
        self.cloud_context = cloud_context

    def _create_workflow(self) -> StateGraph:
        workflow = StateGraph(state_schema=OrchestrationState)

        # Add nodes with agent context
        workflow.add_node("orchestrate", lambda state: orchestration_node(state, {"model_name": self.model_name}))  # type: ignore
        workflow.add_node("generate_prompt", lambda state: prompt_generation_node(state, {"model_name": self.model_name, "prompt_generator": self.prompt_generator}))  # type: ignore
        workflow.add_node("generate_code", lambda state: code_generation_node(state, {"model_name": self.model_name, "code_generator": self.code_generator}))  # type: ignore
        workflow.add_node("execute_code", lambda state: code_execution_node(
            state,
            {"code_executor": self.code_executor},
            self.credentials,
            self.cloud_context
        ))  # type: ignore
        workflow.add_node("analyze_discovery", lambda state: discovery_analysis_node(state, {"model_name": self.model_name, "model_registry": self.model_registry}))  # type: ignore

        # Add edges
        def is_complete(state):
            """Check if workflow should end."""
            return state["status"] in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED]

        # Build workflow graph
        workflow.add_edge(START, "orchestrate")
        workflow.add_edge("orchestrate", "generate_prompt")
        workflow.add_edge("generate_prompt", "generate_code")
        workflow.add_edge("generate_code", "execute_code")
        workflow.add_edge("execute_code", "analyze_discovery")
        # Add conditional edge after analysis to either end or continue
        workflow.add_conditional_edges(
            "analyze_discovery",
            is_complete,
            {
                True: END,  # If complete/failed, end workflow
                False: "orchestrate",  # Otherwise continue to next iteration
            },
        )

        return workflow

    def execute(self, request: str, cloud: CloudProvider, service: str, params: Optional[dict] = None) -> dict:
        """Execute the orchestration workflow."""
        credentials = None
        if params and "credentials" in params:
            credentials = params.pop("credentials")

        workflow = self._create_workflow()
        compiled_workflow = workflow.compile()

        initial_state = create_initial_state(request=request, cloud=cloud, service=service, params=params)

        return compiled_workflow.invoke(initial_state, {"configurable": {"__credentials": credentials}})
