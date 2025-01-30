# PROJECT: Iterative, Agentic LLM-Driven Cloud Inspection and Troubleshooting System

## 1. OVERVIEW AND OBJECTIVES

This project aims to create an intelligent, iterative system that leverages Language Models (LLMs) to interact with cloud services, initially focusing on Amazon Web Services (AWS). The system will dynamically generate, validate, and execute Python scripts to collect helpful information from AWS services based on user request. The goal is to provide actionable insights and context to address user requests by methodically generating and executing small Python scripts. The system will capture, store, and track every piece of discovered data to inform subsequent steps or iterations.

- Build an end-to-end “agentic” system that uses multiple AI models to iteratively collect data from the cloud.
- Provide actionable insights and context to address user requests by methodically generating and executing small Python scripts.
- Capture, store, and track every piece of discovered data to inform subsequent steps or iterations.
- Leverage LangChain, LangGraph, and LangSmith to create a structured workflow, define model configurations, and log results.

## 2. HIGH-LEVEL WORKFLOW

The system will follow a structured, iterative process to collect data from AWS services based on user queries. The workflow is designed to be adaptive, with each step informing the next based on the data collected so far. The system will use a combination of LLMs, code generation, validation, and execution agents to interact with AWS services securely and efficiently.

1. **Orchestration Agent**
   - Input: User request, global state, iteration data
   - Manages the workflow state and iteration cycles
   - Tracks collected data and merges new discoveries
   - Makes decisions on:
     - Whether more data collection is needed
     - When to terminate the collection process
     - How to handle errors and retries
   - Maximum 3 iterations with up to 2 retries per iteration

2. **Prompt Generator Agent**
   - Input: Current state, user request, iteration count
   - Generates focused prompts for code generation
   - Analyzes previous discoveries to refine prompts
   - Defines required variables and success criteria
   - Ensures instructions are minimal and focused

3. **Code Generation Agent**
   - Input: CodeGenerationPrompt
   - Generates Python code for AWS SDK interactions
   - Creates complete code packages including:
     - Python scripts using boto3
     - Required dependencies with versions
     - Minimal IAM policies
   - Implements code validation and error handling
   - Uses linter to ensure code quality

4. **Code Execution Agent**
   - Input: Validated code, dependencies, IAM policy
   - Sets up isolated execution environment (Docker)
   - Installs dependencies and manages AWS credentials
   - Executes code safely with timeout limits
   - Captures outputs, errors, and resource metrics
   - Handles execution errors and retries

5. **Discovery Analysis Agent**
   - Input: Execution results, current context
   - Analyzes collected data for patterns and insights
   - Identifies missing information and gaps
   - Provides recommendations for next iteration
   - Updates global context with analyzed data
   - Helps guide the orchestration process

## 3. LANGGRAPH & LANGSMITH INTEGRATION

### LangGraph Integration

- Implement StateGraph to manage the iterative workflow
- Define nodes for each agent (Orchestration, Prompt Generator, Code Generation, Code Execution)
- Configure state management:

  ```python
  class WorkflowState(TypedDict):
      context: dict  # Stores collected data and iteration history
      current_iteration: int
      user_request: str
      collected_data: list[dict]
      agent_outputs: dict
  ```

- Define edges between nodes with conditional logic:
  - Orchestration → Prompt Generator (when more data needed)
  - Prompt Generator → Code Generation
  - Code Generation → Code Execution
  - Code Execution → Orchestration (for data aggregation)
  - Orchestration → Final Output (when complete)

### LangSmith Integration

- Track agent performance and workflow metrics
- Log each iteration's:
  - Generated prompts
  - Code snippets
  - Execution results
  - State changes
- Enable debugging and optimization through:
  - Trace visualization
  - Performance analytics
  - Error tracking

## 4. DETAILED GOALS

### A. Iterative Code Generation and Execution

### B. State Management and Context Tracking

### C. Agent-like Debugging and Error Feedback

### D. Orchestration and Decision Logic

### E. Validation and Security

### F. Data Output and Storage

## 5. SYSTEM COMPONENTS & AGENTS

1. **Orchestration Agent**

- Input: User request, global state, iteration data
- Responsibilities:
  - Analyze user requests to identify required cloud information
  - Manage the workflow state and iteration cycles
  - Track collected data and merge new discoveries
  - Make decisions on:
    - Whether more data collection is needed
    - When to terminate the collection process
    - How to handle errors and retries
  - Provide feedback for subsequent iterations
  - Maintain global context and state
- Output:
  - Updated workflow state
  - Next action decision (continue/complete)
  - Aggregated results
  - Error handling directives

1. **Prompt Generator Agent**

- Input: Current state, user request, iteration count
- Responsibilities:
  - Analyze previous discoveries
  - Generate focused prompts for next data collection
  - Define required variables and success criteria
- Output: CodeGenerationPrompt object with:
  - Template
  - Variables
  - Success criteria
  - Description

1. **Code Generation Agent**

- Input: CodeGenerationPrompt
- Responsibilities:
  - Generate Python code for AWS SDK interactions
  - Include error handling
  - Generate minimal IAM policies
  - Implement code validation
- Output:
  - Validated Python script
  - Dependencies list
  - IAM policy (if needed)

1. **Code Execution Agent**

- Input: Validated code, dependencies, IAM policy
- Responsibilities:
  - Set up isolated execution environment
  - Install dependencies
  - Execute code safely
  - Capture outputs and errors
  - Handle timeouts and resource limits
- Output:
  - Execution results
  - Error details (if any)
  - Resource usage metrics

1. **Discovery Analysis Agent**

- Input: Execution results, current context
- Responsibilities:
  - Analyze collected data for patterns and insights
  - Identify missing information and gaps
  - Evaluate data quality and completeness
  - Generate recommendations for next iteration
  - Update global context with new findings
  - Guide the orchestration process with insights
- Output:
  - Analysis results and insights
  - Data quality assessment
  - Recommendations for next steps
  - Updated context information

## 5.1 REQUIRED COMPONENTS

1. Core Agents:
   - Orchestration Agent
   - Prompt Generator Agent
   - Code Generation Agent
   - Code Execution Agent

2. State Management:
   - WorkflowState class
   - Context manager
   - Data aggregation system

3. Execution Environment:
   - Docker container system
   - Dependency manager
   - IAM policy handler

4. Integration Components:
   - LangGraph workflow manager
   - LangSmith logging system
   - AWS SDK interface

5. Support Systems:
   - Code validation/linting system
   - Error handling system
   - Security validation system
   - Metrics collection system

6. Storage & Persistence:
   - State storage (Redis/similar)
   - Results storage
   - Logging storage

7. API Components (for service mode):
   - FastAPI server
   - Authentication system
   - Rate limiter
   - Job queue

## 6. ILLUSTRATIVE WORKFLOW DIAGRAM

```mermaid
graph TD
    Start([Start]) --> UserRequest[User Request]
    UserRequest --> Orchestrator{Orchestration Agent}
    
    subgraph IterativeLoop[Iterative Discovery Loop]
        Orchestrator -->|Need More Data| PromptGen[Prompt Generator Agent]
        PromptGen -->|Generate Prompt| CodeGen[Code Generation Agent]
        CodeGen -->|Generate Script| Validation{Code Validation}
        Validation -->|Failed| CodeGen
        Validation -->|Passed| Executor[Code Execution Agent]
        Executor -->|Error| CodeGen
        Executor -->|Success| Discovery[Discovery Analysis Agent]
        Discovery -->|Analysis Complete| DataAggregation[Data Aggregation]
        DataAggregation --> StateUpdate[Update Global State]
        StateUpdate --> Orchestrator
    end
    
    Orchestrator -->|Complete| ResultsProcessing[Process Final Results]
    ResultsProcessing --> End([End])
    
    subgraph States[State Management]
        GlobalContext[(Global Context)]
        IterationData[(Iteration Data)]
        CollectedData[(Collected Data)]
    end
    
    StateUpdate -.->|Update| GlobalContext
    StateUpdate -.->|Update| IterationData
    StateUpdate -.->|Store| CollectedData
    
    style Start fill:#90EE90
    style End fill:#FFB6C1
    style IterativeLoop fill:#f0f0f0,stroke:#333,stroke-width:2px
    style States fill:#e6f3ff,stroke:#333,stroke-width:2px
    style Orchestrator fill:#FFA07A
    style PromptGen fill:#98FB98
    style CodeGen fill:#87CEEB
    style Executor fill:#DDA0DD
    style Discovery fill:#FFD700
```

## 7. ADDITIONAL ENHANCEMENTS

### Multi-Cloud Support

- Azure integration
  - Azure SDK implementation
  - Azure-specific IAM handling
- GCP integration
  - Google Cloud SDK implementation
  - GCP service account management

### REST API Service

- FastAPI implementation
- Endpoints:
  - /inspect: Start new inspection
  - /status: Check inspection status
  - /results: Get inspection results
- Authentication & authorization
- Rate limiting
- API documentation (OpenAPI)

### Scalability Features

- Kubernetes deployment support
- Horizontal scaling of execution agents
- Redis-based state management
- Job queuing system

### Cost Optimization

- Resource pooling
- Execution time limits
- Cloud resource cleanup
- Cost tracking per request

### Additional Features

- Custom plugin system
- Template library for common scenarios
- Export results in multiple formats
- Integration with monitoring tools

## 8. TRACKING & LOGGING

### Local Logging

- Standard Python logging configuration
  - File and console handlers
  - JSON formatting for structured logs
  - Log levels for different components
  - Rotation policy for log files

### LangSmith Integration

- Trace model executions
  - Prompt generation steps
  - Code generation attempts
  - Success/failure rates
- Graph flow monitoring
  - State transitions
  - Agent interactions
  - Iteration cycles
- Basic metrics
  - Response times
  - Token usage
  - Success rates

## 9. CONCLUSION

This project presents a sophisticated approach to cloud infrastructure inspection using a combination of LLMs and automated agents. The system's key strengths lie in its:

1. **Iterative Discovery**
   - Progressive data collection
   - Context-aware decision making
   - Adaptive workflow management

2. **Modular Architecture**
   - Clear separation of concerns
   - Easily extensible components
   - Pluggable agent system

3. **Safety & Reliability**
   - Secure code execution
   - Comprehensive validation
   - Error handling and recovery

4. **Future-Ready Design**
   - Multi-cloud extensibility
   - API-first approach
   - Scalability considerations

The initial AWS focus provides a solid foundation for future enhancements while delivering immediate value for cloud infrastructure inspection and troubleshooting tasks.
