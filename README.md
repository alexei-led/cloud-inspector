# Cloud Inspector

A tool for generating and analyzing AWS code using various LLMs.

## Features

- Generate AWS code using different LLM models
- Structured output with multiple files:
  - `main.py` - Python script using boto3
  - `requirements.txt` - Dependencies with versions
  - `policy.json` - IAM policy with required permissions
- Automatic code analysis and validation
- Security risk detection
- AWS service and permission tracking
- LangSmith integration for tracing and monitoring

## Installation

```bash
pip install -e .
```

## Configuration

Create a `.env` file based on `.env.example`:

```bash
cp .env.example .env
```

Required environment variables:
- `OPENAI_API_KEY` - Your OpenAI API key
- `LANGCHAIN_API_KEY` - Your LangSmith API key
- `LANGCHAIN_PROJECT` - LangSmith project name (default: "cloud-inspector")

## Usage

### Global Options

```bash
cloud-inspector [OPTIONS] COMMAND [ARGS]...

Options:
  --log-level [DEBUG|INFO|WARNING|ERROR|CRITICAL]  Set the logging level (default: INFO)
  --project TEXT                                   LangSmith project name (default: cloud-inspector)
```

### Prompt Management

List available prompts:
```bash
# Basic list
cloud-inspector prompts list

# Filter by tag
cloud-inspector prompts list --tag security

# Filter by AWS service
cloud-inspector prompts list --service ec2

# Different output formats
cloud-inspector prompts list --format json
cloud-inspector prompts list --format table
```

Show prompt details:
```bash
cloud-inspector prompts show PROMPT_NAME
```

Validate a prompt file:
```bash
cloud-inspector prompts validate path/to/prompt.yaml
```

### Code Generation

Generate AWS code:
```bash
# Using default model (gpt-4o-mini)
cloud-inspector workflow generate list_instances -v region=us-west-2

# Using a specific model with multiple variables
cloud-inspector workflow generate create_bucket --model gpt-4-turbo -v bucket_name=my-bucket -v region=us-west-2
```

List generation results:
```bash
# List all results
cloud-inspector workflow list-results

# Filter results
cloud-inspector workflow list-results --prompt list_instances
cloud-inspector workflow list-results --model gpt-4-turbo
cloud-inspector workflow list-results --start "2024-01-01T00:00:00"
cloud-inspector workflow list-results --end "2024-12-31T23:59:59"
```

View execution statistics:
```bash
cloud-inspector workflow stats
```

### Model Management

List available models:
```bash
cloud-inspector models list
```

### Output Structure

Each code generation creates a timestamped directory containing:
- `main.py` - The main Python script using boto3
- `requirements.txt` - All required dependencies
- `policy.json` - IAM policy with minimum required permissions
- `metadata.json` - Generation metadata and analysis results

Example output directory:
```
generated_code/
  list_instances_gpt-4o-mini_20240315_123456/
    main.py
    requirements.txt
    policy.json
    metadata.json
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

```
cloud-inspector/
├── prompts/              # YAML prompt templates
├── generated_code/       # Generated code output
├── src/
│   ├── cloud_inspector/
│   │   ├── cli.py       # Command-line interface
│   │   ├── prompts.py   # Prompt management
│   │   └── workflow.py  # Code generation workflow
│   └── langchain_components/
│       ├── models.py    # Model registry
│       ├── parsers.py   # Code parsing
│       └── templates.py # Chat templates
├── tests/               # Test suite
├── setup.py            # Package configuration
└── README.md
```

## Development

Install development dependencies:

```bash
pip install -e ".[dev]"
```

Run tests:

```bash
pytest
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## License

MIT License
