"""Pydantic models for structured output."""

from pydantic import BaseModel, Field


class GeneratedFiles(BaseModel):
    """Structure for generated code files and configuration."""

    main_py: str = Field(..., description="The main Python script containing the core application logic")
    requirements_txt: str = Field(..., description="List of Python package dependencies in requirements.txt format")
    policy_json: str = Field(..., description="JSON configuration file containing policy settings and rules")
