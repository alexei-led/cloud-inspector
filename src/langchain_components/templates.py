"""Cloud Inspector prompt templates."""
from typing import Any, Dict, List, Optional

from langchain.prompts import PromptTemplate


class CloudTemplate:
    """Template for AWS cloud operations code generation."""

    DEFAULT_TEMPLATE = """You are an expert AWS developer. Generate Python code using boto3 to perform the following operation:

Operation Description:
{operation_description}

Requirements:
1. Use proper error handling with informative error messages
2. Include all necessary imports
3. Follow AWS security best practices
4. Add comprehensive type hints
5. Include detailed docstrings with parameters and return types
6. Use clear variable names and add comments for complex logic
7. Make code modular and reusable
8. Include logging for important operations
9. Validate input parameters
10. Handle AWS credentials properly (never hardcode)

Additional Considerations:
- The code will be used by both humans and AI agents
- Output should be clear and well-documented
- Include example usage in docstring
- Add any relevant security or cost warnings
- Specify required IAM permissions

Code:"""

    def __init__(
        self,
        template: Optional[str] = None,
        model_specific_params: Optional[Dict[str, Any]] = None
    ):
        """Initialize the cloud template.
        
        Args:
            template: Optional custom template string. If not provided, uses DEFAULT_TEMPLATE.
            model_specific_params: Optional model-specific parameters for generation.
        """
        self._template = PromptTemplate(
            template=template or self.DEFAULT_TEMPLATE,
            input_variables=["operation_description"]
        )
        self.model_specific_params = model_specific_params or {}

    def format(self, operation_description: str) -> str:
        """Format the template with the operation description.
        
        Args:
            operation_description: Description of the AWS operation to perform.
        
        Returns:
            Formatted prompt string.
        """
        return self._template.format(operation_description=operation_description)

    def get_model_params(self, model_name: str) -> Dict[str, Any]:
        """Get model-specific parameters for this template.
        
        Args:
            model_name: Name of the model to get parameters for.
        
        Returns:
            Dictionary of model-specific parameters.
        """
        return self.model_specific_params.get(model_name, {}) 