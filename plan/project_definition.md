# Project: LLM Code Generation Comparison for Cloud Inspection (AWS focus initially)

**Objective:** Develop a Python project to compare the code generation capabilities of various LLMs (OpenAI, Gemini, Titan, Claude, Llama, and potentially others) specifically for generating Python scripts that interact with AWS services using boto3. The project should facilitate prompt engineering, code generation, automated code review/correction, and organized output saving. The ultimate goal is to identify the best model for generating reliable and efficient cloud inspection scripts, initially focusing on AWS, with future expansion to GCP and Azure.

**Detailed Requirements:**

1. **Prompt Management:**
    * Accept user-provided prompts (natural language requests for AWS actions).
    * Optionally allow for prompt engineering (modifying user prompts for better LLM performance).
    * Support a list of prompts for batch code generation.

2. **LLM Integration:**
    * Modular design to easily integrate new LLMs.
    * Handle API authentication and requests for each LLM.
    * Support configuration of LLM parameters (e.g., temperature, max tokens).

3. **Code Generation:**
    * Generate Python code using boto3 based on the provided prompts.
    * Handle potential errors in generated code gracefully.

4. **Automated Code Review and Correction:**
    * Implement a mechanism to review the generated code for correctness, security vulnerabilities, and adherence to best practices.
    * Use an LLM or other static analysis tools for this review.
    * Attempt to automatically fix identified issues.

5. **Output Management:**
    * Save generated code in separate files, organized by LLM and prompt.
    * Use descriptive file names (e.g., `prompt_name_llm_name.py`).
    * Maintain a log of the generation process, including prompts, LLM responses, and review results.

6. **AWS Focus (Initial):**
    * Prioritize generating code for common AWS inspection tasks (e.g., listing EC2 instances, checking S3 bucket configurations, retrieving CloudWatch metrics).

7. **Extensibility:**
    * Design the project to be easily extensible to support GCP and Azure in the future.

**Deliverables:**

* A well-structured Python project with clear documentation.
* A set of example prompts and generated code for AWS inspection tasks.
* A comparative analysis of the LLM performance based on code correctness, efficiency, and adherence to best practices.

**Task List Generation Request:**

Generate a detailed task list for implementing this project. The task list should include specific coding tasks, testing tasks, and documentation tasks. Organize the tasks into logical phases (e.g., setup, prompt management, LLM integration, code generation, review/correction, output management, testing, documentation). For each task provide short description and expected outcome.

**Output format:**

Use Markdown format for the task list and project structure.

**Important Considerations:**

* Error handling should be a priority.
* Security best practices should be followed when generating code that interacts with cloud APIs.
* The project should be easy to set up and run.
* Clear documentation is essential.
