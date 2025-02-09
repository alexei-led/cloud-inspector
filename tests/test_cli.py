import json
from click.testing import CliRunner
import pytest
import os
import tempfile

from cloud_inspector.cli import cli
from cloud_inspector.orchestration.orchestration import OrchestrationAgent

def test_model_list_cli(monkeypatch):
    # Create a fake registry object with a list_models method.
    class FakeRegistry:
        def list_models(self):
            return {
                "modelA": {"model_id": "idA"},
                "modelB": {"model_id": "idB"}
            }
    # Patch the ModelRegistry used in the CLI initialization.
    monkeypatch.setattr("cloud_inspector.cli.ModelRegistry", lambda: FakeRegistry())
    
    runner = CliRunner()
    result = runner.invoke(cli, ["model", "list"])
    assert result.exit_code == 0
    # Check that fake model names are printed
    assert "modelA" in result.output
    assert "modelB" in result.output

def fake_execute_success(self, request, cloud, service, thread_id, params=None):
    # Return a fake result as a dictionary.
    return {"fake": "result"}

def test_discovery_execute_success(monkeypatch):
    # Patch OrchestrationAgent.execute so it returns a fake result.
    monkeypatch.setattr(OrchestrationAgent, "execute", fake_execute_success)
    
    # Create a temporary credentials file
    credentials = {"aws_access_key_id": "FAKE", "aws_secret_access_key": "FAKE"}
    with tempfile.NamedTemporaryFile("w+", delete=False) as tf:
        tf.write(json.dumps(credentials))
        tf.flush()
        credentials_path = tf.name

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "discovery", "execute",
            "test request",
            "--service", "s3",
            "--thread-id", "123",
            "--credentials-file", credentials_path,
            "--cloud-context", "aws-account-1"
        ]
    )
    # Clean up temporary file
    os.remove(credentials_path)

    assert result.exit_code == 0
    # Output should contain the fake result
    assert '"fake": "result"' in result.output

def fake_execute_failure(self, request, cloud, service, thread_id, params=None):
    raise ValueError("Test error")

def test_discovery_execute_failure(monkeypatch):
    # Patch OrchestrationAgent.execute so it raises an error.
    monkeypatch.setattr(OrchestrationAgent, "execute", fake_execute_failure)
    
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "discovery", "execute",
            "test request",
            "--service", "s3",
            "--thread-id", "123"
        ]
    )
    # Exit code should be nonzero and error message included.
    assert result.exit_code != 0
    assert "Test error" in result.output
