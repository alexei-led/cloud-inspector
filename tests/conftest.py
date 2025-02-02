import pytest
from pathlib import Path
from components.models import ModelRegistry
from cloud_inspector.code_generator import CodeGeneratorAgent
from cloud_inspector.execution_agent import CodeExecutionAgent
from cloud_inspector.prompt_generator import PromptGeneratorAgent
from components.types import CloudProvider

@pytest.fixture
def model_registry():
    """Create a test model registry with mock models."""
    return ModelRegistry(config_path=Path("tests/fixtures/test_models.yaml"))

@pytest.fixture
def prompt_generator(model_registry):
    """Create a prompt generator for testing."""
    return PromptGeneratorAgent(model_registry)

@pytest.fixture
def code_generator(model_registry):
    """Create a code generator for testing."""
    return CodeGeneratorAgent(model_registry, output_dir=Path("tests/output"))

@pytest.fixture
def code_executor():
    """Create a code executor with test settings."""
    return CodeExecutionAgent(max_execution_time=5)
