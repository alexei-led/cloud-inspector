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

```bash
# List prompts
cloud-inspector prompt list [OPTIONS]
Options:
  --tag TEXT          Filter prompts by tag
  --service TEXT      Filter prompts by AWS service
  --format [text|json|table]  Output format (default: text)

# Show prompt details
cloud-inspector prompt show NAME

# Validate prompt file
cloud-inspector prompt validate FILE

# Generate new prompt
cloud-inspector prompt generate [OPTIONS]
Options:
  --cloud TEXT    Cloud provider (e.g., aws, gcp, azure) [required]
  --service TEXT  Service name within the cloud provider [required]
  --request TEXT  Description of the prompt to generate [required]
  --model TEXT    Name of the LLM model to use (default: gpt-4o)
```

### Model Management

```bash
# List available models
cloud-inspector model list
```

### Code Generation

```bash
# Generate code
cloud-inspector code generate [OPTIONS] PROMPT_NAME
Options:
  --model TEXT        Name of the LLM model to use (default: gpt-4o-mini)
  --var, -v TEXT     Variables in key=value format (multiple allowed)

# List generation results
cloud-inspector code list [OPTIONS]
Options:
  --prompt TEXT      Filter by prompt name
  --model TEXT       Filter by model name
  --start DATETIME   Filter from this start time
  --end DATETIME     Filter until this end time

# View statistics
cloud-inspector code stats
```

### Example Usage

Generate code with variables:

```bash
cloud-inspector code generate list_instances \
  --model gpt-4-turbo \
  -v region=us-west-2 \
  -v instance_id=i-1234abcd
```

Generate a new prompt:

```bash
cloud-inspector prompt generate \
  --cloud aws \
  --service ec2 \
  --request "Create a script to monitor EC2 instance CPU and memory usage" \
  --model gpt-4o
```

List prompts in table format:

```bash
cloud-inspector prompt list --format table --service ec2
```

View code generation statistics:

```bash
cloud-inspector code stats
```

### Output Structure

Each code generation creates a timestamped directory containing:

- `main.py` - The main Python script using boto3
- `requirements.txt` - All required dependencies
- `policy.json` - IAM policy with minimum required permissions
- `metadata.json` - Generation metadata and analysis results

Example output directory:

```text
generated_code/
  list_instances_gpt-4o-mini_20240315_123456/
    main.py
    requirements.txt
    policy.json
    metadata.json
```

### Performance Monitoring

All code generation runs are automatically tracked in LangSmith. Statistics available via `code stats` include:

- Total executions and success rates
- Model-specific performance metrics
- Prompt-specific success rates
- Common error patterns
- Execution time statistics

Visit [LangSmith Dashboard](https://smith.langchain.com) for detailed metrics.

## Project Structure

```text
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
