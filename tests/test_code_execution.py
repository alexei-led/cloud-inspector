from cloud_inspector.code_execution import DockerSandbox


def test_docker_sandbox_execution():
    """Test basic code execution in sandbox."""
    sandbox = DockerSandbox(timeout=5)

    main_py = """
import json
print(json.dumps({"test": "success"}))
"""
    requirements_txt = "# no requirements"

    success, stdout, stderr, usage = sandbox.execute(main_py, requirements_txt)

    assert success
    assert "test" in stdout
    assert not stderr
    assert "memory_usage_bytes" in usage


def test_docker_sandbox_timeout():
    """Test execution timeout."""
    sandbox = DockerSandbox(timeout=1)

    main_py = """
import time
time.sleep(2)
"""
    requirements_txt = "# no requirements"

    success, stdout, stderr, usage = sandbox.execute(main_py, requirements_txt)

    assert not success
    assert "timeout" in stderr.lower()
