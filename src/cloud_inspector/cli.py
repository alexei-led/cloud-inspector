"""Command-line interface for Cloud Inspector."""

import json
import logging
from datetime import datetime
from typing import Optional

import click

from cloud_inspector.code_generator import CodeGeneratorAgent
from cloud_inspector.discovery_manager import DiscoveryManager
from cloud_inspector.prompt_generator import PromptGeneratorAgent
from components.models import ModelRegistry
from components.types import CloudProvider


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
    ctx.obj["discovery_manager"] = DiscoveryManager(code_generator=code_generator, prompt_generator=prompt_generator)


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
@click.argument("request", required=True, help="User request that triggered the discovery")
@click.option("--cloud", type=CloudProviderParamType(), default=CloudProvider.AWS, help="Name of the cloud provider (for new iterations)")
@click.option("--service", help="Name of the service to interact with (for new iterations)")
@click.option("--model", default="gpt-4o-mini", help="Name of the LLM model to use")
@click.pass_context
def execute(
    ctx: click.Context,
    request: str,
    cloud: CloudProvider,
    service: Optional[str],
    model: str,
):
    """Execute a discovery - either start new or continue existing.

    If --request-id is provided, continues an existing discovery.
    Otherwise, starts a new discovery using the request argument and options.
    """
    manager = ctx.obj["discovery_manager"]

    try:
        request_id, result, output_path = manager.execute_discovery(
            model_name=model,
            request=request,
            cloud=cloud,
            service=service,
        )

        # Show results
        if request_id:
            click.echo(f"\nIteration process: {request_id}")
            if result.generated_files:
                click.echo("\nGenerated files:")
                for filename, _content in result.generated_files.items():
                    click.echo(f"- {filename}")
                click.echo(f"\nFiles saved to: {output_path}")
            else:
                click.echo("No files were generated")
    except ValueError as e:
        click.echo(f"Error: {str(e)}", err=True)
        ctx.exit(1)


if __name__ == "__main__":
    cli()
