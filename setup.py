from setuptools import setup, find_packages

setup(
    name="cloud-inspector",
    version="0.1.0",
    description="A tool for comparing LLM code generation capabilities for cloud inspection tasks",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'cloud-inspector=cloud_inspector.cli:cli',
        ],
    },
    python_requires=">=3.9",
) 