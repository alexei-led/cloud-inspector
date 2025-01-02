"""Command-line interface for Cloud Inspector."""

from datetime import datetime
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import json
from tabulate import tabulate

import click

from cloud_inspector.prompts import PromptManager
from cloud_inspector.workflow import CodeGenerationWorkflow, WorkflowManager
from langchain_components.models import ModelRegistry


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
    type=click.Choice(
        ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False
    ),
    default="INFO",
    help="Set the logging level.",
)
@click.option(
    "--project", default="cloud-inspector", help="LangSmith project name for tracing."
)
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
def prompts():
    """Manage prompt templates."""
    pass


@prompts.command(name="list")
@click.option("--tag", help="Filter prompts by tag")
@click.option("--service", help="Filter prompts by AWS service")
@click.option(
    "--format",
    type=click.Choice(["text", "json", "table"], case_sensitive=False),
    default="text",
    help="Output format",
)
@click.pass_context
def list_prompts(
    ctx: click.Context, tag: Optional[str], service: Optional[str], format: str
):
    """List all available prompts. Optionally filter by tag and/or service."""
    prompt_manager = ctx.obj["prompt_manager"]

    # Get all prompts first
    available_prompts = prompt_manager.list_prompts()

    if not available_prompts:
        click.echo("No prompts found.")
        return

    # Apply filters if specified
    if tag:
        available_prompts = [p for p in available_prompts if tag in p["tags"]]
    if service:
        available_prompts = [
            p for p in available_prompts if p["service"].lower() == service.lower()
        ]

    if not available_prompts:
        click.echo("No prompts found matching the specified filters.")
        return

    # Format output based on selected format
    if format == "json":
        click.echo(json.dumps(available_prompts, indent=2))

    elif format == "table":
        # Prepare data for tabulate
        headers = ["Name", "Service", "Operation", "Description"]
        table_data = [
            [p["name"], p["service"], p["operation"], p["description"]]
            for p in available_prompts
        ]
        click.echo(
            tabulate(
                table_data,
                headers=headers,
                tablefmt="pretty",
                colalign=("left", "left", "left", "left"),
            )
        )

    else:  # text format (default)
        # Find maximum name length for alignment
        max_name_length = max(len(p["name"]) for p in available_prompts)

        click.echo("\nAvailable Prompts:")
        click.echo("=" * (max_name_length + 40))

        for prompt in available_prompts:
            # Format with aligned description
            formatted_line = (
                f"{prompt['name']:<{max_name_length}}    {prompt['description']}"
            )
            click.echo(formatted_line)


@prompts.command()
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
    click.echo("=" * 80)
    click.echo(f"Name: {name}")
    click.echo(f"Service: {prompt.service}")
    click.echo(f"Operation: {prompt.operation}")
    click.echo(f"Description: {prompt.description}")
    click.echo(f"Variables: {', '.join(prompt.variables)}")
    click.echo(f"Tags: {', '.join(prompt.tags)}")
    click.echo("\nTemplate:")
    click.echo("-" * 40)
    click.echo(prompt.template)


@prompts.command()
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


# Workflow Commands


@cli.group()
def workflow():
    """Manage code generation workflow."""
    pass


@workflow.command()
@click.argument("prompt_name")
@click.option("--model", default="gpt-4o-mini", help="Name of the LLM model to use.")
@click.option("--var", "-v", multiple=True, help="Variables in key=value format.")
@click.pass_context
def generate(ctx: click.Context, prompt_name: str, model: str, var: tuple[str, ...]):
    """Generate code using a prompt."""
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
    result = flow.execute(prompt_name, model, variables)

    if result.success:
        click.echo("\nCode Generation Successful!")
        click.echo("=" * 80)

        # Display each generated file
        for filename, content in result.generated_files.items():
            click.echo(f"\n{filename}:")
            click.echo("-" * 80)
            click.echo(content)

        # Show where files were saved
        output_dir = (
            flow.output_dir
            / f"{prompt_name}_{model}_{result.timestamp.strftime('%Y%m%d_%H%M%S')}"
        )
        click.echo(f"\nFiles saved to: {output_dir}")
    else:
        click.echo("\nCode Generation Failed!")
        click.echo("=" * 80)
        click.echo(f"Error: {result.error}")


@workflow.command()
@click.option("--prompt", help="Filter by prompt name")
@click.option("--model", help="Filter by model name")
@click.option("--start", type=click.DateTime(), help="Filter from this start time")
@click.option("--end", type=click.DateTime(), help="Filter until this end time")
@click.pass_context
def list_results(
    ctx: click.Context,
    prompt: Optional[str],
    model: Optional[str],
    start: Optional[datetime],
    end: Optional[datetime],
):
    """List workflow execution results."""
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


@workflow.command()
@click.pass_context
def stats(ctx: click.Context):
    """Show workflow execution statistics."""
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
        success_rate = (
            (data["successful"] / data["total"]) * 100 if data["total"] > 0 else 0
        )
        click.echo(
            f"{model}: {data['successful']}/{data['total']} ({success_rate:.1f}% success)"
        )

    click.echo("\nBy Prompt:")
    click.echo("-" * 40)
    for prompt, data in wf_stats["by_prompt"].items():
        success_rate = (
            (data["successful"] / data["total"]) * 100 if data["total"] > 0 else 0
        )
        click.echo(
            f"{prompt}: {data['successful']}/{data['total']} ({success_rate:.1f}% success)"
        )

    if wf_stats["common_errors"]:
        click.echo("\nCommon Errors:")
        click.echo("-" * 40)
        for error, count in wf_stats["common_errors"].items():
            click.echo(f"{error}: {count} occurrences")


# Models Commands
@cli.group()
def models():
    """Manage model configurations."""
    pass


@models.command(name="list")
@click.pass_context
def list_models(ctx: click.Context):
    """List available models."""
    registry = ctx.obj["registry"]
    models = registry.list_models()

    if not models:
        click.echo("No models configured.")
        return

    # Find the longest name for alignment
    max_name_length = max(len(name) for name in models.keys())

    click.echo("\nAvailable Models:")
    click.echo("=" * (max_name_length + 40))

    # Print each model with aligned columns
    for name, config in models.items():
        click.echo(f"{name:<{max_name_length}}    {config['model_id']}")


if __name__ == "__main__":
    cli()
