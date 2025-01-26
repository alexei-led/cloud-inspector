"""Command-line interface for Cloud Inspector."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import click

from cloud_inspector.iteration_manager import IterationManager
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
    ctx.obj["prompt_generator"] = PromptGenerator(model_registry=ctx.obj["registry"])
    ctx.obj["iteration_manager"] = IterationManager(ctx.obj["prompt_manager"], ctx.obj["workflow"], ctx.obj["prompt_generator"])


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
    prompt_type_emoji = {"predefined": "📝", "generated": "🤖", None: "-"}

    rows = []
    for name, prompt in prompts.items():
        discovery_status = getattr(prompt, "discovery_complete", None)
        discovery_str = "✓" if discovery_status else "..." if discovery_status is False else "-"
        parent_req = getattr(prompt, "parent_request_id", "-")
        service = prompt.service if prompt.service is not None else "-"
        operation = prompt.operation if prompt.operation is not None else "-"
        prompt_type = prompt_type_emoji.get(prompt.prompt_type, "-")
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


@prompt.command(name="show")
@click.argument("name")
@click.pass_context
def show_prompt(ctx: click.Context, name: str):
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


@cli.group()
def code():
    """View and manage code generation results from iterations."""
    pass


@code.command(name="list")
@click.option("--prompt", help="Filter by prompt name")
@click.option("--model", help="Filter by model name")
@click.option("--start", type=click.DateTime(), help="Filter by start time")
@click.option("--end", type=click.DateTime(), help="Filter by end time")
@click.pass_context
def list_code_results(
    ctx: click.Context,
    prompt: Optional[str],
    model: Optional[str],
    start: Optional[datetime],
    end: Optional[datetime],
):
    """List code generation results from iterations."""
    iteration_manager = ctx.obj["iteration_manager"]
    results = iteration_manager.list_results(prompt=prompt, model=model, start_time=start, end_time=end)

    if not results:
        click.echo("No code generation results found.")
        return

    click.echo("\nCode Generation Results:")
    click.echo("=" * 100)

    headers = ["Request ID", "Prompt", "Model", "Status", "Created At"]
    rows = []

    for result in results:
        rows.append([result.request_id, result.prompt_name, result.model_name, result.status, result.created_at.strftime("%Y-%m-%d %H:%M:%S")])

    # Calculate column widths
    widths = [max(len(str(row[i])) for row in [headers] + rows) for i in range(len(headers))]
    row_format = "  ".join(f"{{:<{w}}}" for w in widths)

    # Print headers
    click.echo(row_format.format(*headers))
    click.echo("-" * 100)

    # Print rows
    for row in rows:
        click.echo(row_format.format(*row))


@cli.group()
def iterate():
    """Manage iterative data collection process."""
    pass


class CloudProviderParamType(click.ParamType):
    """Click parameter type for CloudProvider enum."""
    name = "cloud_provider"

    def convert(self, value, param, ctx):
        try:
            return CloudProvider(value.lower())
        except ValueError:
            valid_providers = [p.value for p in CloudProvider]
            self.fail(f"Invalid cloud provider '{value}'. Valid options are: {', '.join(valid_providers)}", param, ctx)


@iterate.command()
@click.argument("request")
@click.option("--cloud", type=CloudProviderParamType(), default=CloudProvider.AWS, help="Name of the cloud provider.")
@click.option("--service", help="Name of the service to interact with.")
@click.option("--model", default="gpt-4o", help="Name of the LLM model to use.")
@click.pass_context
def start(ctx: click.Context, request: str, cloud: CloudProvider, service: str, model: str):
    """Start a new iteration process."""
    manager = ctx.obj["iteration_manager"]
    request_id, result, output_path = manager.start_iteration(request, cloud, service, model)

    click.echo(f"\nStarted iteration process: {request_id}")
    if result.generated_files:
        click.echo("\nGenerated files:")
        for filename, _content in result.generated_files.items():
            click.echo(f"- {filename}")
        click.echo(f"\nFiles saved to: {output_path}")
    else:
        click.echo("No files were generated")


@iterate.command()
@click.argument("request_id")
@click.argument("data_file", type=click.Path(exists=True))
@click.option("--source", multiple=True, help="Source files used to collect data.")
@click.option("--feedback", help="Feedback for next iteration in JSON format.")
@click.pass_context
def collect(
    ctx: click.Context,
    request_id: str,
    data_file: str,
    source: tuple[str, ...],
    feedback: Optional[str],
) -> None:
    """Save collected data from manual code execution."""
    prompt_manager = ctx.obj["prompt_manager"]
    workflow = ctx.obj["workflow"]
    prompt_generator = ctx.obj["prompt_generator"]
    iteration_manager = IterationManager(prompt_manager, workflow, prompt_generator)

    try:
        # Parse feedback if provided
        feedback_dict = json.loads(feedback) if feedback else None

        # Save collected data
        iteration_manager.save_collected_data(
            request_id,
            Path(data_file),
            list(source),
            feedback_dict,
        )
        click.echo(f"Saved collected data for {request_id}")
    except Exception as e:
        click.echo(f"Error saving collected data: {e}", err=True)


@iterate.command()
@click.argument("request_id")
@click.option("--model", default="gpt-4o", help="Name of the LLM model to use.")
@click.pass_context
def next(ctx: click.Context, request_id: str, model: str):
    """Start next iteration for data collection."""
    manager = ctx.obj["iteration_manager"]
    result = manager.next_iteration(request_id, model)

    if result:
        workflow_result, output_path = result
        click.echo(f"Started next iteration for {request_id}")
        if workflow_result.generated_files:
            click.echo("\nGenerated files:")
            for filename, content in workflow_result.generated_files.items():
                click.echo(f"\n{filename}:")
                click.echo(content)
            click.echo(f"\nFiles saved to: {output_path}")
        else:
            click.echo("No files were generated")
    else:
        click.echo("No more iterations needed or request ID not found")


@iterate.command()
@click.argument("request_id")
@click.argument("reason")
@click.pass_context
def complete(ctx: click.Context, request_id: str, reason: str) -> None:
    """Mark an iteration process as complete."""
    prompt_manager = ctx.obj["prompt_manager"]
    workflow = ctx.obj["workflow"]
    prompt_generator = ctx.obj["prompt_generator"]
    iteration_manager = IterationManager(prompt_manager, workflow, prompt_generator)

    try:
        iteration_manager.complete_iteration(request_id, reason)
        click.echo(f"Marked {request_id} as complete: {reason}")
    except Exception as e:
        click.echo(f"Error completing iteration: {e}", err=True)


@iterate.command()
@click.argument("request_id")
@click.pass_context
def show(ctx: click.Context, request_id: str) -> None:
    """Show collected data for a request."""
    prompt_manager = ctx.obj["prompt_manager"]
    workflow = ctx.obj["workflow"]
    prompt_generator = ctx.obj["prompt_generator"]
    iteration_manager = IterationManager(prompt_manager, workflow, prompt_generator)

    try:
        data = iteration_manager.get_collected_data(request_id)
        click.echo(json.dumps(data, indent=2, cls=DateTimeEncoder))
    except Exception as e:
        click.echo(f"Error showing collected data: {e}", err=True)


if __name__ == "__main__":
    cli()
