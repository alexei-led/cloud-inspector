"""Command-line interface for Cloud Inspector."""

import json
import logging
from datetime import datetime

import click

from cloud_inspector.code_generator import CodeGeneratorAgent
from cloud_inspector.components.models import ModelRegistry
from cloud_inspector.components.types import CloudProvider
from cloud_inspector.execution_agent import CodeExecutionAgent
from cloud_inspector.orchestration.orchestration import OrchestrationAgent
from cloud_inspector.prompt_generator import PromptGeneratorAgent


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
    model_registry = ModelRegistry()
    ctx.obj["registry"] = model_registry
    code_generator = CodeGeneratorAgent(model_registry)
    prompt_generator = PromptGeneratorAgent(model_registry)
    code_executor = CodeExecutionAgent()
    ctx.obj["code_generator"] = code_generator
    ctx.obj["prompt_generator"] = prompt_generator
    ctx.obj["code_executor"] = code_executor


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
def discovery():
    """Manage cloud resource discovery process."""
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


@discovery.command()
@click.argument("request", required=True)
@click.option("--cloud", type=CloudProviderParamType(), default=CloudProvider.AWS)
@click.option("--service", required=True)
@click.option("--model", default="gpt-4-turbo")
@click.option("--thread-id", required=True)
@click.pass_context
def execute(
    ctx: click.Context,
    request: str,
    cloud: CloudProvider,
    service: str,
    model: str,
    thread_id: str,
):
    """Execute cloud inspection workflow."""
    agent = OrchestrationAgent(code_generator=ctx.obj["code_generator"], prompt_generator=ctx.obj["prompt_generator"], code_executor=ctx.obj["code_executor"], model_name=model)

    try:
        result = agent.execute(request=request, cloud=cloud, service=service, thread_id=thread_id)

        click.echo(json.dumps(result, indent=2, cls=DateTimeEncoder))
    except ValueError as e:
        click.echo(f"Error: {str(e)}", err=True)
        ctx.exit(1)


if __name__ == "__main__":
    cli()
