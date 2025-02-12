"""Unit tests for DockerSandbox class."""

from unittest import mock
from unittest.mock import MagicMock, patch

import pytest
from docker.errors import DockerException

from cloud_inspector.code_execution import DockerSandbox


@pytest.fixture
def mock_docker_client():
    """Create a mock Docker client."""
    with patch("docker.from_env") as mock_client:
        client = MagicMock()
        mock_client.return_value = client
        yield client


@pytest.fixture
def mock_container():
    """Create a mock Docker container."""
    container = MagicMock()
    container.wait.return_value = {"StatusCode": 0}
    container.logs.return_value = b""
    return container

@pytest.fixture
def mock_container_with_stats():
    """Create a mock Docker container with realistic stats."""
    container = MagicMock()
    container.wait.return_value = {"StatusCode": 0}
    container.logs.return_value = b""

    # Add realistic stats
    container.stats.return_value = {
        "cpu_stats": {
            "cpu_usage": {"total_usage": 100000},
            "system_cpu_usage": 1000000,
            "online_cpus": 4
        },
        "precpu_stats": {
            "cpu_usage": {"total_usage": 90000},
            "system_cpu_usage": 900000
        },
        "memory_stats": {
            "usage": 1024 * 1024,  # 1MB
            "limit": 512 * 1024 * 1024  # 512MB
        }
    }
    return container


def test_init_docker_success(mock_docker_client):
    """Test successful Docker initialization and image pulling."""
    sandbox = DockerSandbox()
    mock_docker_client.images.get.side_effect = [DockerException("Image not found"), None]
    mock_docker_client.images.pull.return_value = None

    assert sandbox._init_docker()
    assert sandbox.docker == mock_docker_client
    mock_docker_client.images.get.assert_has_calls([mock.call(sandbox.image), mock.call(sandbox.image)])
    mock_docker_client.images.pull.assert_called_once_with(sandbox.image)


def test_init_docker_failure(mock_docker_client):
    """Test Docker initialization failure."""
    mock_docker_client.images.get.side_effect = DockerException("Connection failed")
    mock_docker_client.images.pull.side_effect = DockerException("Pull failed")

    sandbox = DockerSandbox()
    assert not sandbox._init_docker()
    assert sandbox.docker is None


def test_container_stats_calculation(mock_docker_client, mock_container):
    """Test container resource usage statistics calculation."""
    mock_stats = {"cpu_stats": {"cpu_usage": {"total_usage": 100000}, "system_cpu_usage": 1000000}, "precpu_stats": {"cpu_usage": {"total_usage": 90000}, "system_cpu_usage": 900000}, "memory_stats": {"usage": 1024 * 1024}}
    mock_container.stats.return_value = mock_stats

    sandbox = DockerSandbox()
    sandbox.docker = mock_docker_client

    stats = sandbox._get_container_stats(mock_container)
    assert stats["memory_usage_bytes"] == 1024 * 1024
    assert stats["cpu_usage_percent"] == pytest.approx(10.0)


def test_execute_with_aws_credentials(mock_docker_client, mock_container):
    """Test code execution with AWS credentials handling."""
    mock_docker_client.containers.create.return_value = mock_container
    mock_container.logs.return_value = b'{"result": "success"}'

    sandbox = DockerSandbox()
    sandbox.docker = mock_docker_client

    aws_credentials = {"aws_access_key_id": "test_key", "aws_secret_access_key": "test_secret", "aws_session_token": "test_token"}

    success, stdout, stderr, usage = sandbox.execute("print('test')", "requests==2.28.0", aws_credentials)

    assert success
    create_call = mock_docker_client.containers.create.call_args
    assert create_call is not None

    # Verify container configuration
    container_config = create_call[1]
    assert container_config["network_mode"] == "none"
    assert container_config["cpu_quota"] == int(sandbox.cpu_limit * 100000)
    assert container_config["mem_limit"] == sandbox.memory_limit

    # Verify AWS credentials handling
    env_vars = container_config["environment"]
    assert "AWS_SHARED_CREDENTIALS_FILE" in env_vars
    assert env_vars["AWS_SHARED_CREDENTIALS_FILE"] == "/code/credentials"

    # Verify volume mounting
    volumes = container_config["volumes"]
    assert len(volumes) == 1
    mount_point = list(volumes.values())[0]
    assert mount_point["mode"] == "ro"


def test_execute_with_invalid_json_output(mock_docker_client, mock_container):
    """Test handling of invalid JSON output from executed code."""
    mock_docker_client.containers.create.return_value = mock_container
    mock_container.logs.return_value = b"Invalid JSON output"

    sandbox = DockerSandbox()
    sandbox.docker = mock_docker_client

    success, stdout, stderr, usage = sandbox.execute("print('invalid json')", "# no requirements")

    assert not success
    assert "not in valid JSON format" in stderr


def test_execute_with_timeout(mock_docker_client, mock_container):
    """Test execution timeout handling."""
    mock_docker_client.containers.create.return_value = mock_container
    mock_container.wait.side_effect = DockerException("Container wait timeout")

    sandbox = DockerSandbox(timeout=1)
    sandbox.docker = mock_docker_client

    success, stdout, stderr, usage = sandbox.execute("import time; time.sleep(2)", "# no requirements")

    assert not success
    assert "Docker execution error" in stderr
    mock_container.remove.assert_called_once_with(force=True)


def test_cleanup_containers(mock_docker_client):
    """Test cleanup of leftover containers."""
    mock_containers = [MagicMock(status="exited"), MagicMock(status="dead")]
    mock_docker_client.containers.list.return_value = mock_containers

    sandbox = DockerSandbox()
    sandbox.docker = mock_docker_client
    sandbox.cleanup()

    mock_docker_client.containers.list.assert_called_once_with(all=True, filters={"ancestor": sandbox.image, "status": ["exited", "dead"]})
    assert all(container.remove.call_count == 1 for container in mock_containers)


def test_execute_with_container_error(mock_docker_client, mock_container):
    """Test handling of container execution errors."""
    mock_docker_client.containers.create.return_value = mock_container
    mock_container.wait.return_value = {"StatusCode": 1, "Error": {"Message": "Runtime error occurred"}}
    mock_container.logs.side_effect = [
        b"",  # stdout
        b"Error: Division by zero",  # stderr
    ]

    sandbox = DockerSandbox()
    sandbox.docker = mock_docker_client

    success, stdout, stderr, usage = sandbox.execute(
        "1/0",  # Code that will raise an exception
        "# no requirements",
    )

    assert not success
    assert "Division by zero" in stderr
    assert "Runtime error occurred" in stderr

    # Verify container cleanup
    mock_container.remove.assert_called_once_with(force=True)


def test_execute_with_pip_error(mock_docker_client, mock_container):
    """Test handling of pip installation errors."""
    mock_docker_client.containers.create.return_value = mock_container
    mock_container.wait.return_value = {"StatusCode": 1}
    mock_container.logs.side_effect = [
        b"",  # stdout
        b"ERROR: Could not find a version that satisfies the requirement invalid-package==99.99.99",  # stderr
    ]

    sandbox = DockerSandbox()
    sandbox.docker = mock_docker_client

    success, stdout, stderr, usage = sandbox.execute(
        "print('test')",
        "invalid-package==99.99.99",
    )

    assert not success
    assert "Could not find a version" in stderr
    mock_container.remove.assert_called_once_with(force=True)


def test_execute_with_network_access(mock_docker_client, mock_container):
    """Test that network access is properly configured."""
    mock_docker_client.containers.create.return_value = mock_container
    mock_container.logs.return_value = b'{"result": "success"}'

    sandbox = DockerSandbox()
    sandbox.docker = mock_docker_client

    success, stdout, stderr, usage = sandbox.execute(
        'import requests\nprint(\'{"result": "success"}\')',
        "requests==2.28.0"
    )

    create_call = mock_docker_client.containers.create.call_args
    container_config = create_call[1]

    # Verify network access is enabled
    assert container_config["network_mode"] == "bridge"

    # Verify other configurations remain secure
    volumes = container_config["volumes"]
    assert list(volumes.values())[0]["mode"] == "ro"  # Read-only mount

    # Verify environment variables
    env_vars = container_config["environment"]
    assert env_vars["PIP_NO_CACHE_DIR"] == "1"
    assert env_vars["PYTHONUNBUFFERED"] == "1"

def test_execute_with_pip_network_failure(mock_docker_client, mock_container):
    """Test handling of network failures during pip install."""
    mock_docker_client.containers.create.return_value = mock_container
    mock_container.wait.return_value = {"StatusCode": 1}
    mock_container.logs.side_effect = [
        b"",  # stdout
        b"ERROR: Connection failed: Could not connect to PyPI",  # stderr
    ]

    sandbox = DockerSandbox()
    sandbox.docker = mock_docker_client

    success, stdout, stderr, usage = sandbox.execute(
        "print('test')",
        "requests==2.28.0"
    )

    assert not success
    assert "Connection failed" in stderr
    mock_container.remove.assert_called_once_with(force=True)

def test_execute_aws_api_call(mock_docker_client, mock_container):
    """Test execution of code making AWS API calls."""
    mock_docker_client.containers.create.return_value = mock_container
    mock_container.logs.return_value = b'{"aws_response": "success"}'

    sandbox = DockerSandbox()
    sandbox.docker = mock_docker_client

    aws_code = '''
import boto3
client = boto3.client('s3')
response = client.list_buckets()
print('{"aws_response": "success"}')
'''

    success, stdout, stderr, usage = sandbox.execute(
        aws_code,
        "boto3",
        credentials={"aws_access_key_id": "test", "aws_secret_access_key": "test"}
    )

    assert success
    assert "aws_response" in stdout

    # Verify AWS credentials were properly mounted
    create_call = mock_docker_client.containers.create.call_args
    container_config = create_call[1]
    assert "AWS_SHARED_CREDENTIALS_FILE" in container_config["environment"]

def test_execute_with_entrypoint_permissions(mock_docker_client, mock_container):
    """Test that entrypoint script has correct permissions."""
    from unittest.mock import patch

    mock_docker_client.containers.create.return_value = mock_container
    mock_container.logs.return_value = b'{"result": "success"}'

    with patch('pathlib.Path.chmod') as mock_chmod:
        sandbox = DockerSandbox()
        sandbox.docker = mock_docker_client

        success, stdout, stderr, usage = sandbox.execute(
            "print('test')",
            "# no requirements"
        )

        # Verify entrypoint script was made executable
        mock_chmod.assert_called_once_with(0o755)

def test_execute_with_resource_limits(mock_docker_client, mock_container):
    """Test that resource limits are properly set."""
    mock_docker_client.containers.create.return_value = mock_container
    mock_container.logs.return_value = b'{"result": "success"}'

    custom_cpu_limit = 0.5
    custom_memory_limit = "256m"

    sandbox = DockerSandbox(
        cpu_limit=custom_cpu_limit,
        memory_limit=custom_memory_limit
    )
    sandbox.docker = mock_docker_client

    success, stdout, stderr, usage = sandbox.execute(
        "print('test')",
        "# no requirements"
    )

    create_call = mock_docker_client.containers.create.call_args
    container_config = create_call[1]

    # Verify resource limits
    assert container_config["cpu_quota"] == int(custom_cpu_limit * 100000)
    assert container_config["mem_limit"] == custom_memory_limit
