import pytest
from pathlib import Path
from unittest.mock import MagicMock
import sys
from pathlib import Path
from unittest.mock import MagicMock
from langchain_core.messages import AIMessage

# Add src directory to Python path
src_path = str(Path(__file__).parent.parent / "src")
if src_path not in sys.path:
    sys.path.append(src_path)

from cloud_inspector.components.models import ModelRegistry
from cloud_inspector.code_generator import CodeGeneratorAgent
from cloud_inspector.execution_agent import CodeExecutionAgent
from cloud_inspector.prompt_generator import PromptGeneratorAgent
from cloud_inspector.components.types import CloudProvider
from langchain_core.messages import AIMessage

@pytest.fixture
def mock_llm_response():
    """Create a mock LLM response."""
    return AIMessage(content="""
template: |
    Generate Python code to list EC2 instances in {region}.
    The code should:
    1. Use boto3 to connect to AWS
    2. List all instances in the specified region
    3. Output instance details as JSON
description: List EC2 instances in a specific region
variables:
    - name: region
      description: AWS region to inspect
success_criteria: Code successfully lists EC2 instances and outputs JSON
""")

@pytest.fixture
def mock_model():
    """Create a mock LLM model."""
    mock = MagicMock()
    mock.invoke.return_value = mock_llm_response()
    return mock

@pytest.fixture
def model_registry(mock_model):
    """Create a test model registry with mock model."""
    registry = MagicMock(spec=ModelRegistry)
    registry.get_model.return_value = mock_model
    registry.validate_model_capability.return_value = True
    return registry

@pytest.fixture
def prompt_generator(model_registry):
    """Create a prompt generator for testing."""
    return PromptGeneratorAgent(model_registry)

@pytest.fixture
def code_generator(model_registry, tmp_path):
    """Create a code generator for testing."""
    return CodeGeneratorAgent(model_registry, output_dir=tmp_path / "output")

@pytest.fixture
def code_executor():
    """Create a code executor with test settings."""
    return CodeExecutionAgent(max_execution_time=5)
