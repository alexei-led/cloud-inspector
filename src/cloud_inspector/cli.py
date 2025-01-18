"""Command-line interface for Cloud Inspector."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import click

from cloud_inspector.prompt_generator import PromptGenerator
from cloud_inspector.prompts import CloudProvider, PromptManager, PromptType
from cloud_inspector.workflow import CodeGenerationWorkflow, WorkflowManager
from langchain_components.models import ModelRegistry


class DateTimeEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, datetime):
            return o.isoformat()
        return super().default(o)


def setup_logging(log_level: str) -> None:
    """Set up logging configuration."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


@click.group()
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False),
    default="INFO",
    help="Set the logging level.",
)
@click.option("--project", default="cloud-inspector", help="LangSmith project name for tracing.")
@click.pass_context
def cli(ctx: click.Context, log_level: str, project: str) -> None:
    """Cloud Inspector - AWS Code Generation Tool.

    Generate and analyze Python code for AWS operations using various LLMs.
    """
    setup_logging(log_level)

    # Store common objects in context
    ctx.ensure_object(dict)
    ctx.obj["project"] = project
    ctx.obj["registry"] = ModelRegistry()
    ctx.obj["prompt_manager"] = PromptManager()
    ctx.obj["workflow"] = CodeGenerationWorkflow(
        prompt_manager=ctx.obj["prompt_manager"],
        model_registry=ctx.obj["registry"],
        project_name=project,
    )
    ctx.obj["workflow_manager"] = WorkflowManager()


# Prompt Management Commands


@cli.group()
def prompt():
    """Manage prompt templates."""
    pass


@prompt.command(name="list")
@click.option("--tag", help="Filter prompts by tag")
@click.option("--service", help="Filter prompts by service")
@click.option("--cloud", help="Filter prompts by cloud provider")
@click.option("--prompt-type", help="Filter prompts by type (predefined/generated)")
@click.option("--discovery-complete", type=bool, help="Filter prompts by discovery status")
@click.option("--parent-request", help="Filter prompts by parent request ID")
@click.option(
    "--format",
    type=click.Choice(["table", "json"], case_sensitive=False),
    default="table",
    help="Output format",
)
@click.pass_context
def list_prompts(
    ctx: click.Context,
    tag: Optional[str],
    service: Optional[str],
    cloud: Optional[str],
    prompt_type: Optional[str],
    discovery_complete: Optional[bool],
    parent_request: Optional[str],
    format: str,
):
    """List all available prompts. Optionally filter by various criteria."""
    prompt_manager = ctx.obj["prompt_manager"]
    prompts = prompt_manager.list_prompts()

    # Apply filters
    if tag:
        prompts = {k: v for k, v in prompts.items() if tag in v.tags}
    if service:
        prompts = {k: v for k, v in prompts.items() if v.service == service}
    if cloud:
        prompts = {k: v for k, v in prompts.items() if v.cloud == cloud}
    if prompt_type:
        prompts = {k: v for k, v in prompts.items() if v.prompt_type == prompt_type}
    if discovery_complete is not None:
        prompts = {k: v for k, v in prompts.items() if getattr(v, "discovery_complete", None) == discovery_complete}
    if parent_request:
        prompts = {k: v for k, v in prompts.items() if getattr(v, "parent_request_id", None) == parent_request}

    if not prompts:
        click.echo("No prompts found matching the criteria.")
        return

    if format == "json":
        click.echo(json.dumps({k: v.model_dump() for k, v in prompts.items()}, indent=2, cls=DateTimeEncoder))
        return

    # Table format
    headers = ["Name", "Service", "Operation", "Type", "Discovery", "Parent Request"]

    # Emoji mapping for prompt types
    PROMPT_TYPE_EMOJI = {"predefined": "📝", "generated": "🤖", None: "-"}

    rows = []
    for name, prompt in prompts.items():
        discovery_status = getattr(prompt, "discovery_complete", None)
        discovery_str = "✓" if discovery_status else "..." if discovery_status is False else "-"
        parent_req = getattr(prompt, "parent_request_id", "-")
        service = prompt.service if prompt.service is not None else "-"
        operation = prompt.operation if prompt.operation is not None else "-"
        prompt_type = PROMPT_TYPE_EMOJI.get(prompt.prompt_type, "-")
        parent_req_display = parent_req[:8] + "..." if parent_req and parent_req != "-" and len(parent_req) > 8 else parent_req
        rows.append([name, service, operation, prompt_type, discovery_str, parent_req_display])

    # Sort rows by name
    rows.sort(key=lambda x: x[0])

    # Print table
    click.echo("\nAvailable Prompts:")
    click.echo("=" * 120)

    # Calculate column widths
    widths = [max(len(str(row[i])) for row in [headers] + rows) for i in range(len(headers))]

    # Print headers
    header_format = "  ".join(f"{{:<{w}}}" for w in widths)
    click.echo(header_format.format(*headers))
    click.echo("-" * 120)

    # Print rows
    row_format = "  ".join(f"{{:<{w}}}" for w in widths)
    for row in rows:
        # Replace None with "-" during formatting
        formatted_row = ["-" if x is None else x for x in row]
        click.echo(row_format.format(*formatted_row))


@prompt.command()
@click.argument("name")
@click.pass_context
def show(ctx: click.Context, name: str):
    """Show details of a specific prompt."""
    prompt_manager = ctx.obj["prompt_manager"]
    prompt = prompt_manager.get_prompt(name)

    if not prompt:
        click.echo(f"Prompt '{name}' not found.")
        return

    click.echo("\nPrompt Details:")
    click.echo("=" * 40)
    click.echo(f"Name: {name}")
    click.echo(f"Service: {prompt.service}")
    click.echo(f"Operation: {prompt.operation}")
    click.echo(f"Cloud: {prompt.cloud}")
    click.echo(f"Type: {prompt.prompt_type}")
    click.echo(f"Tags: {', '.join(prompt.tags)}")

    if prompt.prompt_type == PromptType.GENERATED:
        click.echo("\nGeneration Info:")
        click.echo("-" * 40)
        click.echo(f"Generated By: {prompt.generated_by}")
        click.echo(f"Generated At: {prompt.generated_at}")
        click.echo(f"Iteration: {getattr(prompt, 'iteration', 1)}")
        click.echo(f"Parent Request: {getattr(prompt, 'parent_request_id', '-')}")

        click.echo("\nDiscovery Status:")
        click.echo("-" * 40)
        discovery_complete = getattr(prompt, "discovery_complete", None)
        click.echo(f"Discovery Complete: {'✓' if discovery_complete else '...' if discovery_complete is False else '-'}")

        if hasattr(prompt, "discovered_resources") and prompt.discovered_resources:
            click.echo("\nDiscovered Resources:")
            for resource in prompt.discovered_resources:
                click.echo(f"  - {json.dumps(resource)}")

        if hasattr(prompt, "dependencies") and prompt.dependencies:
            click.echo("\nDependencies:")
            for dep in prompt.dependencies:
                click.echo(f"  - {dep}")

        if hasattr(prompt, "next_discovery_targets") and prompt.next_discovery_targets:
            click.echo("\nNext Discovery Targets:")
            for target in prompt.next_discovery_targets:
                click.echo(f"  - {target}")

    click.echo("\nDescription:")
    click.echo("-" * 40)
    click.echo(prompt.description)

    if prompt.variables:
        click.echo("\nVariables:")
        click.echo("-" * 40)
        for var in prompt.variables:
            click.echo(f"  {var['name']}: {var['description']}")

    click.echo("\nTemplate:")
    click.echo("-" * 40)
    click.echo(prompt.template)


@prompt.command()
@click.argument("file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.pass_context
def validate(ctx: click.Context, file: Path):
    """Validate a prompt file."""
    prompt_manager = ctx.obj["prompt_manager"]
    errors = prompt_manager.validate_prompt_file(file)

    if not errors:
        click.echo(f"✅ Prompt file '{file}' is valid.")
    else:
        click.echo(f"❌ Found {len(errors)} errors in '{file}':")
        for error in errors:
            click.echo(f"  - {error}")


# Models Commands
@cli.group()
def model():
    """Manage model configurations."""
    pass


@model.command(name="list")
@click.pass_context
def list_models(ctx: click.Context):
    """List available models."""
    registry = ctx.obj["registry"]
    models = registry.list_models()

    if not models:
        click.echo("No models configured.")
        return

    # Find the longest name for alignment
    max_name_length = max(len(name) for name in models)

    click.echo("\nAvailable Models:")
    click.echo("=" * (max_name_length + 40))

    # Print each model with aligned columns
    for name, config in models.items():
        click.echo(f"{name:<{max_name_length}}    {config['model_id']}")


@prompt.command(name="generate")
@click.option("--cloud", required=True, help="Cloud provider (e.g., aws, gcp, azure)")
@click.option("--service", required=True, help="Service name within the cloud provider")
@click.option("--request", required=True, help="Description of the prompt to generate")
@click.option("--model", default="gpt-4o", help="Name of the LLM model to use.")
@click.option("--discovered-data", type=click.Path(exists=True, dir_okay=False), help="Path to JSON file containing previously discovered data")
@click.option("--iteration", type=int, default=1, help="Current iteration number")
@click.option("--parent-request-id", help="ID of the original user request")
@click.pass_context
def generate_prompt(
    ctx: click.Context,
    cloud: str,
    service: str,
    request: str,
    model: str,
    discovered_data: Optional[str],
    iteration: int,
    parent_request_id: Optional[str],
):
    """Generate a new prompt template from a request."""
    # Load discovered data from JSON file if provided
    discovered_data_dict = None
    if discovered_data:
        with open(discovered_data) as f:
            discovered_data_dict = json.load(f)

    generator = PromptGenerator(ctx.obj["registry"])
    result, saved_path = generator.generate_prompt(
        model,
        service,
        request,
        discovered_data=discovered_data_dict,
        iteration=iteration,
        parent_request_id=parent_request_id,
        cloud=CloudProvider(cloud),
    )

    # Display the generated prompt
    click.echo("\nGenerated Prompt:")
    click.echo("=" * 120)
    click.echo(f"Service: {result.service}")
    click.echo(f"Operation: {result.operation}")
    click.echo(f"\nPrompt saved to: {saved_path}")


@cli.group()
def code():
    """Generate and manage code generation results."""
    pass


@code.command(name="generate")
@click.option("--prompt", "prompt_name", required=True, help="Name of the prompt to use.")
@click.option("--model", default="gpt-4o-mini", help="Name of the LLM model to use.")
@click.option("--var", "-v", multiple=True, help="Variables in key=value format.")
@click.pass_context
def generate_code(ctx: click.Context, prompt_name: str, model: str, var: tuple[str, ...]):
    """Generate code using a specified prompt and model."""
    prompt_manager = ctx.obj["prompt_manager"]
    prompt = prompt_manager.get_prompt(prompt_name)

    if not prompt:
        click.echo(f"Error: Prompt '{prompt_name}' not found.")
        return

    # Parse variables
    variables = {}
    for v in var:
        try:
            key, value = v.split("=", 1)
            variables[key.strip()] = value.strip()
        except ValueError:
            click.echo(f"Invalid variable format: {v}")
            click.echo("Use format: key=value")
            return

    # Show required variables if none provided
    if not var and prompt.variables:
        click.echo("\nRequired variables for this prompt:")
        for var_info in prompt.variables:
            click.echo(f"  - {var_info['name']}: {var_info['description']}")
        click.echo("\nUse --var/-v option to provide values (e.g., -v name=value)")
        return

    def get_workflow(ctx) -> CodeGenerationWorkflow:
        """Get CodeGenerationWorkflow instance from Click context."""
        workflow = ctx.obj.get("workflow")
        if not workflow:
            raise RuntimeError("Workflow not found in context")

        if not isinstance(workflow, CodeGenerationWorkflow):
            raise TypeError("Invalid workflow type")

        return workflow

    # Execute workflow
    flow = get_workflow(ctx)
    try:
        result, output_dir = flow.execute(prompt_name, model, variables)

        if result.success:
            click.echo("\nCode Generation Successful!")
            click.echo("=" * 120)
            click.echo(f"\nFiles saved to: {output_dir}")
        else:
            click.echo("\nCode Generation Failed!")
            click.echo("=" * 120)
            click.echo(f"Error: {result.error}")
    except ValueError as e:
        click.echo(f"\nError: {str(e)}")


@code.command(name="list")
@click.option("--prompt", help="Filter by prompt name")
@click.option("--model", help="Filter by model name")
@click.option("--start", type=click.DateTime(), help="Filter from this start time")
@click.option("--end", type=click.DateTime(), help="Filter until this end time")
@click.pass_context
def list_code_results(
    ctx: click.Context,
    prompt: Optional[str],
    model: Optional[str],
    start: Optional[datetime],
    end: Optional[datetime],
):
    """List previous code generation results."""
    results = ctx.obj["workflow_manager"].list_results(prompt, model, start, end)

    if not results:
        click.echo("No results found.")
        return

    click.echo("\nWorkflow Results:")
    click.echo("=" * 80)
    for result in results:
        click.echo(f"\nPrompt: {result['prompt_name']}")
        click.echo(f"Model: {result['model_name']}")
        click.echo(f"Timestamp: {result['timestamp']}")
        click.echo(f"Success: {'✅' if result['success'] else '❌'}")
        click.echo(f"Execution Time: {result['execution_time']:.2f}s")
        if not result["success"]:
            click.echo(f"Error: {result['error']}")
        click.echo("-" * 40)


@code.command(name="stats")
@click.pass_context
def code_stats(ctx: click.Context):
    """Show statistics about code generation executions."""
    workflow_manager = ctx.obj["workflow_manager"]
    wf_stats = workflow_manager.get_statistics()

    click.echo("\nWorkflow Statistics:")
    click.echo("=" * 80)
    click.echo(f"Total Executions: {wf_stats['total_executions']}")
    click.echo(f"Successful: {wf_stats['successful_executions']}")
    click.echo(f"Failed: {wf_stats['failed_executions']}")
    click.echo(f"Average Execution Time: {wf_stats['average_execution_time']:.2f}s")

    click.echo("\nBy Model:")
    click.echo("-" * 40)
    for model, data in wf_stats["by_model"].items():
        success_rate = (data["successful"] / data["total"]) * 100 if data["total"] > 0 else 0
        click.echo(f"{model}: {data['successful']}/{data['total']} ({success_rate:.1f}% success)")

    click.echo("\nBy Prompt:")
    click.echo("-" * 40)
    for prompt, data in wf_stats["by_prompt"].items():
        success_rate = (data["successful"] / data["total"]) * 100 if data["total"] > 0 else 0
        click.echo(f"{prompt}: {data['successful']}/{data['total']} ({success_rate:.1f}% success)")

    if wf_stats["common_errors"]:
        click.echo("\nCommon Errors:")
        click.echo("-" * 40)
        for error, count in wf_stats["common_errors"].items():
            click.echo(f"{error}: {count} occurrences")


if __name__ == "__main__":
    cli()
