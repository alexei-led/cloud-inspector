"""Pydantic models for structured output."""
from langchain_core.pydantic_v1 import BaseModel, Field 


class GeneratedFiles(BaseModel):
    """Structure for generated code files and configuration."""
    main_py: str = Field(
        ...,
        alias="main.py",
        description="The main Python script containing the core application logic"
    )
    requirements_txt: str = Field(
        ...,
        alias="requirements.txt",
        description="List of Python package dependencies in requirements.txt format"
    )
    policy_json: str = Field(
        ...,
        alias="policy.json",
        description="JSON configuration file containing policy settings and rules"
    ) 
