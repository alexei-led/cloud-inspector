"""Command-line interface for Cloud Inspector."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
from tabulate import tabulate

from cloud_inspector.prompt_generator import PromptGenerator
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
@click.option("--service", help="Filter prompts by AWS service")
@click.option("--cloud", help="Filter by cloud provider (aws, gcp, azure)")
@click.option("--type", "prompt_type", help="Filter by type (predefined, generated)")
@click.option(
    "--format",
    type=click.Choice(["text", "json", "table"], case_sensitive=False),
    default="text",
    help="Output format",
)
@click.pass_context
def list_prompts(
    ctx: click.Context,
    tag: Optional[str],
    service: Optional[str],
    cloud: Optional[str],
    prompt_type: Optional[str],
    format: str,
):
    """List all available prompts. Optionally filter by tag, service, cloud, and type."""
    prompt_manager = ctx.obj["prompt_manager"]
    available_prompts = prompt_manager.list_prompts()

    if not available_prompts:
        click.echo("No prompts available.")
        return

    # Apply filters
    if tag:
        available_prompts = [p for p in available_prompts if tag in p["tags"]]
    if service:
        available_prompts = [p for p in available_prompts if p["service"] == service]
    if cloud:
        available_prompts = [p for p in available_prompts if p["cloud"] == cloud]
    if prompt_type:
        available_prompts = [p for p in available_prompts if p["prompt_type"] == prompt_type]

    if not available_prompts:
        click.echo("No prompts found matching the specified filters.")
        return

    if format == "json":
        click.echo(json.dumps(available_prompts, indent=2, default=str))

    elif format == "table":
        headers = [
            "Name",
            "Cloud",
            "Service",
            "Operation",
            "Type",
            "Source",
            "Description",
        ]
        table_data = [
            [
                p["name"],
                p["cloud"].value,
                p["service"],
                p["operation"],
                "🤖" if p["prompt_type"] == "generated" else "📋",
                p["generated_by"] or "manual",
                ((p["description"][:60] + "...") if len(p["description"]) > 60 else p["description"]),
            ]
            for p in available_prompts
        ]
        click.echo(
            tabulate(
                table_data,
                headers=headers,
                tablefmt="pretty",
                colalign=("left", "left", "left", "left", "center", "left", "left"),
            )
        )

    else:  # text format
        max_name_length = max(len(p["name"]) for p in available_prompts)
        max_source_length = max(len(p["generated_by"] or "manual") for p in available_prompts)

        click.echo("\nAvailable Prompts:")
        click.echo("=" * (max_name_length + 60))

        for prompt in available_prompts:
            type_icon = "🤖" if prompt["prompt_type"] == "generated" else "📋"
            source = prompt["generated_by"] or "manual"
            description = prompt["description"]
            if len(description) > 60:
                description = description[:60] + "..."

            formatted_line = (
                f"{prompt['name']:<{max_name_length}} "
                f"{type_icon} "
                f"[{prompt['cloud'].value}] "
                f"[{source:<{max_source_length}}]    "
                f"{description}"
            )
            click.echo(formatted_line)


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
    click.echo("=" * 80)
    click.echo(f"Name: {name}")
    click.echo(f"Cloud: {prompt.cloud}")
    click.echo(f"Service: {prompt.service}")
    click.echo(f"Operation: {prompt.operation}")
    click.echo(f"Description: {prompt.description}")
    click.echo(f"Type: {'Generated' if prompt.prompt_type == 'generated' else 'Predefined'}")

    if prompt.generated_by:
        click.echo(f"Generated By: {prompt.generated_by}")
    if prompt.generated_at:
        click.echo(f"Generated At: {prompt.generated_at}")

    if prompt.variables:
        click.echo("\nVariables:")
        for var in prompt.variables:
            click.echo(f"  - {var['name']}: {var['description']}")

    click.echo(f"\nTags: {', '.join(prompt.tags)}")
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
    max_name_length = max(len(name) for name in models.keys())

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
@click.pass_context
def generate_prompt(ctx: click.Context, cloud: str, service: str, request: str, model: str):
    """Generate a new prompt template from a request."""
    generator = PromptGenerator(ctx.obj["registry"])
    result, saved_path = generator.generate_prompt(model, service, request, cloud=cloud)

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
