from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class CloudProvider(str, Enum):
    """Supported cloud providers."""

    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"


class WorkflowStatus(str, Enum):
    """Status of the workflow execution."""

    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    WAITING_FOR_FEEDBACK = "waiting_for_feedback"


class CodeGenerationPrompt(BaseModel):
    """Prompt for code generation."""

    service: str = Field(..., description="Cloud service (e.g., ec2, s3)")
    operation: str = Field(..., description="Operation type (e.g., list, analyze)")
    request: str = Field(..., description="User request that triggered the prompt")
    description: str = Field(..., description="Brief description of what the prompt does")
    template: str = Field(..., description="The actual prompt template")
    variables: list[dict[str, str]] = Field(default_factory=list, description="List of variables with name, description")
    cloud: CloudProvider = Field(..., description="Cloud provider")
    generated_by: Optional[str] = Field(None, description="Model used to generate the prompt")
    generated_at: Optional[datetime] = Field(None, description="Timestamp when the prompt was generated")
    success_criteria: Optional[str] = Field(None, description="Criteria for success in this iteration")


class GeneratedFiles(BaseModel):
    """Generated code files and configuration."""

    main_py: str = Field(..., description="Main Python script containing core logic")
    requirements_txt: str = Field(..., description="Python package dependencies")
    policy_json: str = Field(..., description="IAM policy or configuration settings")


