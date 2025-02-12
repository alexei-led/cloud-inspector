"""Unit tests for DockerSandbox class."""

import json
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
        "cpu_stats": {"cpu_usage": {"total_usage": 100000}, "system_cpu_usage": 1000000, "online_cpus": 4},
        "precpu_stats": {"cpu_usage": {"total_usage": 90000}, "system_cpu_usage": 900000},
        "memory_stats": {
            "usage": 1024 * 1024,  # 1MB
            "limit": 512 * 1024 * 1024,  # 512MB
        },
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

    aws_credentials = {"Version": 1, "AccessKeyId": "test_key", "SecretAccessKey": "test_secret", "SessionToken": "test_token"}

    success, stdout, stderr, usage = sandbox.execute("print('test')", "requests==2.28.0", aws_credentials)

    assert success
    create_call = mock_docker_client.containers.create.call_args
    assert create_call is not None

    # Verify container configuration
    container_config = create_call[1]
    assert container_config["network_mode"] == "bridge"
    assert container_config["cpu_quota"] == int(sandbox.cpu_limit * 100000)
    assert container_config["mem_limit"] == sandbox.memory_limit
    assert container_config["user"] == "65534:65534"  # nobody:nogroup

    # verify AWS config file
    volumes = container_config["volumes"]
    volume_config = list(volumes.values())[0]  # Get the first (and only) volume configuration
    assert volume_config["bind"] == "/code"
    assert volume_config["mode"] == "rw"




def test_json_parsing_scenarios(mock_docker_client, mock_container):
    """Test comprehensive JSON parsing scenarios."""
    mock_docker_client.containers.create.return_value = mock_container

    # Test direct parsing function
    sandbox = DockerSandbox()
    sandbox.docker = mock_docker_client

    parsing_test_cases = [
        # Valid JSON cases
        ('{"key": "value"}', True, '{"key":"value"}'),  # Note: json.dumps removes spaces
        (' {"key": "value"} ', True, '{"key":"value"}'),
        ('"{"key": "value"}"', True, '{"key":"value"}'),
        ('\'{"key": "value"}\'', True, '{"key":"value"}'),
        ('{"key": "value"}\n', True, '{"key":"value"}'),
        ('null', True, 'null'),
        ('[]', True, '[]'),
        ('{}', True, '{}'),
        ('true', True, 'true'),
        ('123', True, '123'),
        ('"string"', True, '"string"'),
        ('[1,2,3]', True, '[1,2,3]'),
        ('{"nested": {"key": "value"}}', True, '{"nested":{"key":"value"}}'),
        ('"{\"escaped\": \"json\"}"', True, '{"escaped": "json"}'),

        # Invalid JSON cases
        ('not json', False, 'not json'),
        ('{"incomplete": "json"', False, '{"incomplete": "json"'),
        ('', False, ''),
        (' ', False, ' '),
        ('Error: {"error": "message"}', False, 'Error: {"error": "message"}'),
    ]

    for input_text, expected_success, expected_output in parsing_test_cases:
        success, cleaned = sandbox._try_parse_json(input_text)
        assert success == expected_success, f"Failed for input: {input_text}"
        if success:
            # Don't check if result is None - it's valid for 'null'
            # Instead verify we can parse it as JSON and it matches expected
            parsed = json.loads(cleaned)
            if parsed is None:
                assert input_text.strip() == 'null', f"Got None from non-null input: {input_text}"
            assert cleaned == expected_output, f"Unexpected output for input: {input_text}"
        else:
            assert cleaned == input_text

    # Test execution with various outputs
    execution_test_cases = [
        (b'"{"result": "success"}"', True),  # Double-quoted
        (b'\'{"result": "success"}\'', True),  # Single-quoted
        (b'{"result": "success"}\n', True),  # With newline
        (b' {"result": "success"} ', True),  # With whitespace
        (b'\t{"result": "success"}\n', True),  # With tab
        (b'"\\"{\\"result\\": \\"success\\"}"', True),  # Escaped quotes
        (b'Debug: Starting\n{"result": "success"}', True),  # Non-JSON prefix
        (b'{"result": "success"}\nDebug: Done', True),  # Non-JSON suffix
        (b'Log: {"nested": "json"} continue', False),  # JSON embedded in text
        (b'not json at all', False),  # Invalid JSON
    ]

    for output, should_succeed in execution_test_cases:
        mock_container.logs.side_effect = [output, b""]  # stdout, stderr
        mock_container.wait.return_value = {"StatusCode": 0}

        success, stdout, stderr, usage = sandbox.execute(
            'print(\'test output\')',
            "# no requirements"
        )

        assert success == should_succeed
        if should_succeed:
            try:
                parsed = json.loads(stdout)
                assert isinstance(parsed, (dict, list, str, int, float, bool)) or parsed is None
            except json.JSONDecodeError:
                pytest.fail(f"Failed to parse JSON output: {stdout}")


def test_execute_with_mixed_stderr_output(mock_docker_client, mock_container):
    """Test handling of mixed JSON and non-JSON stderr output."""
    mock_docker_client.containers.create.return_value = mock_container

    test_cases = [
        (b"", b'Regular error\n{"error": "json error"}'),  # Mixed content
        (b"", b'{"error": "json error"}\nRegular error'),  # JSON first
        (b"", b'Error: {"error": "wrapped json"}'),  # Embedded JSON
        (b"", b'Multiple\nLine\nError\nMessage'),  # Pure text error
        (b"", b'{"error": {"nested": "error message"}}'),  # Nested JSON error
    ]

    sandbox = DockerSandbox()
    sandbox.docker = mock_docker_client

    for stdout_output, stderr_output in test_cases:
        mock_container.wait.return_value = {"StatusCode": 1, "Error": {"Message": "Runtime error"}}
        mock_container.logs.side_effect = [stdout_output, stderr_output]

        success, stdout, stderr, usage = sandbox.execute(
            'import sys; sys.stderr.write(\'error output\')',
            "# no requirements"
        )

        assert not success
        assert stderr, "Stderr should not be empty"
        # Either it should be valid JSON or contain the original error message
        try:
            json.loads(stderr)
        except json.JSONDecodeError:
            assert "Runtime error" in stderr or "Error:" in stderr


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

    # Simulate pip install error in stderr
    mock_container.logs.side_effect = [
        b"",  # stdout
        b"pip install --no-cache-dir -r /code/requirements.txt\nERROR: Could not find a version that satisfies the requirement invalid-package==99.99.99",  # stderr
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

    success, stdout, stderr, usage = sandbox.execute('import requests\nprint(\'{"result": "success"}\')', "requests==2.28.0")

    create_call = mock_docker_client.containers.create.call_args
    container_config = create_call[1]

    # Verify network access is enabled
    assert container_config["network_mode"] == "bridge"

    # Verify other configurations remain secure
    volumes = container_config["volumes"]
    assert list(volumes.values())[0]["mode"] == "rw"  # Changed from "ro" to "rw"

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

    success, stdout, stderr, usage = sandbox.execute("print('test')", "requests==2.28.0")

    assert not success
    assert "Connection failed" in stderr
    mock_container.remove.assert_called_once_with(force=True)


def test_execute_aws_api_call(mock_docker_client, mock_container):
    """Test execution of code making AWS API calls."""
    mock_docker_client.containers.create.return_value = mock_container
    mock_container.logs.return_value = b'{"aws_response": "success"}'

    sandbox = DockerSandbox()
    sandbox.docker = mock_docker_client

    credentials = {"aws_access_key_id": "test_key", "aws_secret_access_key": "test_secret", "aws_session_token": "test_token"}

    aws_code = """
import boto3
client = boto3.client('s3')
response = client.list_buckets()
print('{"aws_response": "success"}')
"""

    success, stdout, stderr, usage = sandbox.execute(aws_code, "boto3", credentials=credentials)

    assert success
    assert "aws_response" in stdout

    # Verify container configuration
    create_call = mock_docker_client.containers.create.call_args
    container_config = create_call[1]

    # Verify environment setup
    env_vars = container_config["environment"]
    assert env_vars["HOME"] == "/code"
    assert env_vars["PYTHONPATH"] == "/code"

    # Verify volume mounting
    volumes = container_config["volumes"]
    assert list(volumes.values())[0]["mode"] == "rw"


def test_execute_with_entrypoint_permissions(mock_docker_client, mock_container):
    """Test that entrypoint script has correct permissions."""
    from unittest.mock import patch

    mock_docker_client.containers.create.return_value = mock_container
    mock_container.logs.return_value = b'{"result": "success"}'

    with patch("pathlib.Path.chmod") as mock_chmod:
        sandbox = DockerSandbox()
        sandbox.docker = mock_docker_client

        success, stdout, stderr, usage = sandbox.execute("print('test')", "# no requirements")

        # Verify entrypoint script was made executable
        mock_chmod.assert_called_once_with(0o755)


def test_execute_with_resource_limits(mock_docker_client, mock_container):
    """Test that resource limits are properly set."""
    mock_docker_client.containers.create.return_value = mock_container
    mock_container.logs.return_value = b'{"result": "success"}'

    custom_cpu_limit = 0.5
    custom_memory_limit = "256m"

    sandbox = DockerSandbox(cpu_limit=custom_cpu_limit, memory_limit=custom_memory_limit)
    sandbox.docker = mock_docker_client

    success, stdout, stderr, usage = sandbox.execute("print('test')", "# no requirements")

    create_call = mock_docker_client.containers.create.call_args
    container_config = create_call[1]

    # Verify resource limits
    assert container_config["cpu_quota"] == int(custom_cpu_limit * 100000)
    assert container_config["mem_limit"] == custom_memory_limit


def test_pip_output_redirection(mock_docker_client, mock_container):
    """Test that pip output is properly redirected to stderr."""
    mock_docker_client.containers.create.return_value = mock_container
    # Simulate pip output in stderr and actual program output in stdout
    mock_container.logs.side_effect = [
        b'{"result": "success"}',  # stdout
        b"Installing collected packages: boto3",  # stderr
    ]

    sandbox = DockerSandbox()
    sandbox.docker = mock_docker_client

    success, stdout, stderr, usage = sandbox.execute('print(\'{"result": "success"}\')', "boto3")

    assert success
    assert "Installing collected packages" in stderr
    assert '{"result": "success"}' in stdout
