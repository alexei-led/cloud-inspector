"""Docker-based sandbox for safe code execution."""

import json
import logging
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Optional

import docker
from docker.errors import DockerException
from docker.models.containers import Container

from cloud_inspector.constants import DEFAULT_CPU_LIMIT, DEFAULT_DOCKER_IMAGE, DEFAULT_MEMORY_LIMIT, DEFAULT_TIMEOUT

logger = logging.getLogger(__name__)


class DockerSandbox:
    """Docker-based sandbox for safe code execution."""

    def __init__(
        self,
        image: str = DEFAULT_DOCKER_IMAGE,
        cpu_limit: float = DEFAULT_CPU_LIMIT,
        memory_limit: str = DEFAULT_MEMORY_LIMIT,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        """Initialize Docker sandbox.

        Args:
            image: Docker image to use
            cpu_limit: CPU limit (cores)
            memory_limit: Memory limit (e.g. "512m")
            timeout: Execution timeout in seconds
        """
        self.image = image
        self.cpu_limit = cpu_limit
        self.memory_limit = memory_limit
        self.timeout = timeout
        self.docker = None

    def _ensure_image(self):
        """Ensure Docker image is available."""
        try:
            if self.docker:
                self.docker.images.get(self.image)
        except DockerException as e:
            logger.info(f"Pulling image {self.image}: {e}")
            try:
                if self.docker:
                    self.docker.images.pull(self.image)
                    # Verify the image is now available
                    self.docker.images.get(self.image)
            except DockerException as e:
                logger.error(f"Failed to pull Docker image {self.image}: {e}")
                raise

    def _init_docker(self) -> bool:
        """Initialize Docker client and ensure image is available.

        Returns:
            bool: True if Docker is available and initialized successfully
        """
        if self.docker is None:
            try:
                self.docker = docker.from_env()
                self._ensure_image()
                return True
            except DockerException as e:
                logger.error(f"Failed to initialize Docker: {e}")
                self.docker = None
                return False
        return True

    def _get_container_stats(self, container: Container) -> dict:
        """Get container resource usage statistics."""
        try:
            stats = container.stats(stream=False)
            return {
                "memory_usage_bytes": stats.get("memory_stats", {}).get("usage", 0),
                "cpu_usage_percent": self._calculate_cpu_percent(stats),
            }
        except DockerException as e:
            logger.warning(f"Failed to get container stats: {e}")
            return {}

    def _calculate_cpu_percent(self, stats: dict) -> float:
        """Calculate CPU usage percentage from stats."""
        try:
            cpu_delta = stats["cpu_stats"]["cpu_usage"]["total_usage"] - stats["precpu_stats"]["cpu_usage"]["total_usage"]
            system_delta = stats["cpu_stats"]["system_cpu_usage"] - stats["precpu_stats"]["system_cpu_usage"]
            if system_delta > 0:
                return (cpu_delta / system_delta) * 100.0
        except (KeyError, TypeError):
            pass
        return 0.0

    def execute(
        self,
        main_py: str,
        requirements_txt: str,
        aws_credentials: Optional[dict[str, str]] = None,
    ) -> tuple[bool, str, str, dict]:
        """Execute code in sandbox.

        Args:
            main_py: Python code to execute
            requirements_txt: Requirements file content
            aws_credentials: Optional AWS credentials

        Returns:
            Tuple of (success, stdout, stderr, resource_usage)
        """
        if not self._init_docker():
            return False, "", "Docker is not available", {}

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Write code and requirements files
            (temp_path / "main.py").write_text(main_py)
            (temp_path / "requirements.txt").write_text(requirements_txt)

            # Write AWS credentials if provided
            if aws_credentials:
                creds_file = temp_path / "credentials"
                creds_content = "[default]\n"
                for key, value in aws_credentials.items():
                    creds_content += f"{key} = {value}\n"
                creds_file.write_text(creds_content)

            # Create entrypoint script that installs requirements first
            entrypoint_script = """#!/bin/sh
set -e  # Exit on any error
pip install --no-cache-dir -r /code/requirements.txt
python /code/main.py
"""
            (temp_path / "entrypoint.sh").write_text(entrypoint_script)
            (temp_path / "entrypoint.sh").chmod(0o755)

            # Create and run container
            try:
                container = self.docker.containers.create(  # type: ignore
                    self.image,
                    command=["/code/entrypoint.sh"],
                    volumes={
                        str(temp_path.absolute()): {
                            "bind": "/code",
                            "mode": "ro",  # Keep as read-only for security
                        }
                    },
                    cpu_quota=int(self.cpu_limit * 100000),
                    mem_limit=self.memory_limit,
                    environment={
                        "PYTHONPATH": "/code",
                        "AWS_SHARED_CREDENTIALS_FILE": "/code/credentials" if aws_credentials else "",
                        "PYTHONUNBUFFERED": "1",
                        "PIP_NO_CACHE_DIR": "1",  # Prevent pip from trying to write cache
                    },
                    network_mode="none",  # Isolate network
                    detach=True,
                    working_dir="/code",
                )

                container.start()

                try:
                    logger.debug("Starting container execution")
                    # Wait for completion with timeout
                    result = container.wait(timeout=self.timeout)
                    status_code = result.get("StatusCode", -1)
                    error_msg = result.get("Error", {}).get("Message", "")

                    # Get resource usage statistics
                    resource_usage = self._get_container_stats(container)

                    # Get logs with separate stdout and stderr
                    stdout_logs = container.logs(stdout=True, stderr=False)
                    stderr_logs = container.logs(stdout=False, stderr=True)

                    stdout = stdout_logs.decode() if stdout_logs else ""
                    stderr = stderr_logs.decode() if stderr_logs else ""

                    # Combine stderr with error message if both exist
                    if error_msg and stderr:
                        stderr = f"{stderr}\nContainer Error: {error_msg}"
                    elif error_msg:
                        stderr = error_msg

                    success = status_code == 0

                    # Try to parse stdout as JSON if execution was successful
                    if success and stdout.strip():
                        try:
                            json.loads(stdout)
                            logger.debug("Successfully parsed output as JSON")
                        except json.JSONDecodeError as e:
                            success = False
                            stderr = f"Output is not in valid JSON format: {str(e)}"
                            logger.warning("Code execution output was not valid JSON: %s", str(e))

                    if not success:
                        logger.error("Container execution failed with status %d: %s", status_code, stderr or "No error message provided")

                except DockerException as e:
                    error_msg = f"Docker execution error: {str(e)}"
                    logger.error(error_msg)
                    return False, "", error_msg, {}
                finally:
                    with suppress(DockerException):
                        container.remove(force=True)
                        logger.debug("Container cleaned up")

                return success, stdout, stderr, resource_usage

            except (OSError, DockerException) as e:
                error_msg = f"Failed to prepare or execute container: {str(e)}"
                logger.error(error_msg)
                return False, "", error_msg, {}

    def cleanup(self):
        """Cleanup any leftover containers."""
        if not self._init_docker():
            return

        try:
            containers = self.docker.containers.list(  # type: ignore
                all=True, filters={"ancestor": self.image, "status": ["exited", "dead"]}
            )
            removed = 0
            for container in containers:
                with suppress(DockerException):
                    container.remove(force=True)
                    removed += 1
            if removed:
                logger.info("Cleaned up %d containers", removed)
        except DockerException as e:
            logger.error(f"Failed to cleanup containers: {e}")
