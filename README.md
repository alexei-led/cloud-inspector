# Cloud Inspector

A tool for generating and analyzing AWS cloud operations code using various LLMs.

## Features

- Generate Python code for AWS operations using multiple LLM providers
- Validate and analyze generated code for security and best practices
- Track and compare LLM performance using LangSmith
- Manage and customize prompt templates
- Command-line interface for easy interaction
- Configurable model support for multiple providers

## Supported Models

- **OpenAI**
  - GPT-4o - OpenAI's versatile, high-intelligence flagship model 
  - GPT-4o-mini - OpenAI's fast, affordable small model for focused tasks 
  - o1 - reasoning model designed to solve hard problems across domains
  - o1-mini - fast and affordable reasoning model for specialized tasks
- **Anthropic**
  - Claude 3.5 Sonnet - Latest and most intelligent model
  - Claude 3.5 Haiku - Latest, fastest, and most efficient model
- **Local Models (via Ollama)**
  - Llama 3.3 (70B) - Latest state-of-the-art
  - Qwen 2.5 Coder (72B) - Specialized for code
- **Google**
  - Gemini Flash 2.0 - Latest
- **AWS Bedrock**
  - Nova Micro - Amazon Nova Micro is a text-only model that delivers the lowest latency responses at very low cost.
  - Nova Pro - Amazon Nova Pro is a highly capable multimodal model with the best combination of accuracy, speed, and cost for a wide range of tasks. 

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/cloud-inspector.git
cd cloud-inspector
```

2. Install dependencies:

```bash
# Install production dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install -r requirements-dev.txt

# Install the package in editable mode
pip install -e .
```

3. Set up environment variables:

```bash
# OpenAI
export OPENAI_API_KEY=your_openai_key
export OPENAI_ORG_ID=your_org_id  # Optional

# Anthropic
export ANTHROPIC_API_KEY=your_anthropic_key

# Google
export GOOGLE_APPLICATION_CREDENTIALS=path_to_google_credentials.json

# Ollama (for local models)
export OLLAMA_BASE_URL=http://localhost:11434  # Default

# AWS Bedrock
export AWS_DEFAULT_REGION=us-west-2  # Default
export AWS_PROFILE=your_aws_profile  # Optional

# LangSmith Configuration
export LANGCHAIN_API_KEY=your_langsmith_key
export LANGCHAIN_PROJECT=your_project_name  # Optional, defaults to "cloud-inspector"
```

## Model Configuration

Models are configured in `config/models.yaml`. You can customize:

- Model parameters (temperature, max tokens, etc.)
- Provider-specific settings
- Environment variable mappings

Example configuration:

```yaml
models:
  gpt-4-turbo:
    provider: openai
    model_id: gpt-4-0125-preview
    max_tokens: 4096
    temperature: 0.2
    top_p: 0.95

  claude-3-sonnet:
    provider: anthropic
    model_id: claude-3-sonnet-20240229
    max_tokens: 4096
    temperature: 0.2
    top_p: 0.95
```

## Usage

### Basic Commands

Generate AWS code:

```bash
cloud-inspector workflow generate ec2_list_instances --model gpt-4-turbo -v region=us-west-2
```

List available models:

```bash
cloud-inspector models list
```

List available prompts:

```bash
cloud-inspector prompts list
```

Show prompt details:

```bash
cloud-inspector prompts show ec2_list_instances
```

### Filtering and Analysis

Filter prompts by service:

```bash
cloud-inspector prompts by-service ec2
```

Filter prompts by tag:

```bash
cloud-inspector prompts by-tag security
```

View execution statistics:

```bash
cloud-inspector workflow stats
```

### Performance Monitoring

All code generation runs are automatically tracked in LangSmith. You can:

- View detailed traces of each run
- Monitor token usage and costs
- Compare performance across different models
- Analyze error patterns
- Track success rates

Visit [LangSmith Dashboard](https://smith.langchain.com) to access these metrics.

## Project Structure

```shell
cloud-inspector/
├── src/
│   ├── cloud_inspector/      # Core application code
│   ├── langchain_components/ # LangChain integration
│   └── examples/            # Example usage
├── config/                  # Configuration files
│   └── models.yaml         # Model configurations
├── prompts/                 # Prompt templates
├── tests/                  # Test suite
└── generated_code/         # Generated code output
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests: `pytest`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
