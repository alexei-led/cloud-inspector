"""Docker-based sandbox for safe code execution."""

import logging
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Optional

import docker
from docker.errors import DockerException

logger = logging.getLogger(__name__)


class DockerSandbox:
    """Docker-based sandbox for safe code execution."""

    def __init__(
        self,
        image: str = "python:3.12-slim",
        cpu_limit: float = 1,
        memory_limit: str = "512m",
        timeout: int = 30,
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
        self.docker = docker.from_env()
        self._ensure_image()

    def _ensure_image(self):
        """Ensure Docker image is available."""
        try:
            self.docker.images.get(self.image)
        except DockerException:
            logger.info(f"Pulling image {self.image}")
            self.docker.images.pull(self.image)

    def execute(
        self,
        code: str,
        aws_credentials: Optional[dict[str, str]] = None,
    ) -> tuple[bool, str, str]:
        """Execute code in sandbox.

        Args:
            code: Python code to execute
            aws_credentials: Optional AWS credentials to mount

        Returns:
            Tuple of (success, stdout, stderr)
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Write code to file
            code_file = temp_path / "code.py"
            code_file.write_text(code)

            # Write AWS credentials if provided
            if aws_credentials:
                creds_file = temp_path / "credentials"
                creds_content = "[default]\n"
                for key, value in aws_credentials.items():
                    creds_content += f"{key} = {value}\n"
                creds_file.write_text(creds_content)

            # Create and run container
            try:
                container = self.docker.containers.create(
                    self.image,
                    command=["python", "/code/code.py"],
                    volumes={str(temp_path.absolute()): {"bind": "/code", "mode": "ro"}},
                    cpu_quota=int(self.cpu_limit * 100000),
                    mem_limit=self.memory_limit,
                    environment={"PYTHONPATH": "/code", "AWS_SHARED_CREDENTIALS_FILE": "/code/credentials" if aws_credentials else ""},
                    network_mode="none",
                    detach=True,
                )

                container.start()

                try:
                    # Wait for completion with timeout
                    result = container.wait(timeout=self.timeout)
                    logs = container.logs(stdout=True, stderr=True)
                    stdout = logs.decode() if logs else ""
                    stderr = ""  # In docker-py, stderr is included in stdout when using logs()
                    success = result["StatusCode"] == 0
                except DockerException as e:
                    success = False
                    stdout = ""
                    stderr = f"Execution failed: {str(e)}"
                finally:
                    with suppress(DockerException):
                        container.remove(force=True)

                return success, stdout, stderr

            except DockerException as e:
                return False, "", f"Failed to create container: {str(e)}"

    def cleanup(self):
        """Cleanup any leftover containers."""
        try:
            containers = self.docker.containers.list(all=True, filters={"ancestor": self.image})
            for container in containers:
                try:
                    container.remove(force=True)
                finally:
                    suppress(DockerException)
        except DockerException as e:
            logger.error(f"Failed to cleanup containers: {e}")
