"""Model registry for Cloud Inspector."""
import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional

from langchain_openai.chat_models.base import ChatOpenAI
from langchain_anthropic.chat_models import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_aws.chat_models.bedrock import ChatBedrock
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain.schema.language_model import BaseLanguageModel
from pydantic import BaseModel, SecretStr


class ModelConfig(BaseModel):
    """Model configuration."""
    provider: str
    model_id: str
    max_tokens: int
    temperature: float
    top_p: float
    top_k: Optional[int] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    repeat_penalty: Optional[float] = None
    stop: Optional[list[str]] = None
    response_format: Optional[Dict[str, Any]] = None


class ProviderConfig(BaseModel):
    """Provider configuration."""
    api_key_env: Optional[str] = None
    organization_env: Optional[str] = None
    base_url_env: Optional[str] = None
    default_base_url: Optional[str] = None
    region_env: Optional[str] = None
    profile_env: Optional[str] = None
    default_region: Optional[str] = None


class ModelRegistry:
    """Registry for managing LLM models."""

    def __init__(self, config_path: Optional[Path] = None):
        """Initialize the model registry.
        
        Args:
            config_path: Path to the models configuration file.
        """
        self.config_path = config_path or Path("config/models.yaml")
        self.models: Dict[str, ModelConfig] = {}
        self.provider_configs: Dict[str, ProviderConfig] = {}
        self._load_config()

    def _load_config(self) -> None:
        """Load model configurations from file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Model config file not found: {self.config_path}")

        with self.config_path.open("r") as f:
            config = yaml.safe_load(f)

        # Load models
        for name, model_data in config["models"].items():
            self.models[name] = ModelConfig(**model_data)

        # Load provider configs
        for provider, provider_data in config["provider_configs"].items():
            self.provider_configs[provider] = ProviderConfig(**provider_data)

    def get_model(self, name: str) -> BaseLanguageModel:
        """Get a model instance by name.
        
        Args:
            name: Name of the model to get.
            
        Returns:
            Configured language model instance.
            
        Raises:
            ValueError: If model name is not found or provider is not supported.
        """
        if name not in self.models:
            raise ValueError(f"Model '{name}' not found in configuration")

        model_config = self.models[name]
        provider_config = self.provider_configs[model_config.provider]

        # Common parameters that are supported by all LangChain chat models
        common_params = {
            "model": model_config.model_id,
            "max_tokens": model_config.max_tokens,
            "temperature": model_config.temperature,
            "top_p": model_config.top_p,
        }

        # Model-specific parameters
        model_kwargs = {}
        if model_config.top_k is not None:
            model_kwargs["top_k"] = model_config.top_k
        if model_config.frequency_penalty is not None:
            model_kwargs["frequency_penalty"] = model_config.frequency_penalty
        if model_config.presence_penalty is not None:
            model_kwargs["presence_penalty"] = model_config.presence_penalty
        if model_config.repeat_penalty is not None:
            model_kwargs["repeat_penalty"] = model_config.repeat_penalty
        if model_config.stop is not None:
            model_kwargs["stop"] = model_config.stop
        if model_config.response_format is not None:
            model_kwargs["response_format"] = model_config.response_format

        # Provider-specific initialization
        if model_config.provider == "openai":
            return ChatOpenAI(
                openai_api_key=os.getenv(provider_config.api_key_env),
                openai_organization=os.getenv(provider_config.organization_env),
                model_kwargs=model_kwargs,
                **common_params
            )

        elif model_config.provider == "anthropic":
            return ChatAnthropic(
                anthropic_api_key=SecretStr(os.getenv(provider_config.api_key_env)),
                model_kwargs=model_kwargs,
                **common_params
            )

        elif model_config.provider == "google":
            return ChatGoogleGenerativeAI(
                google_api_key=SecretStr(os.getenv(provider_config.api_key_env)),
                **common_params
            )

        elif model_config.provider == "ollama":
            base_url = os.getenv(
                provider_config.base_url_env,
                provider_config.default_base_url
            )
            return ChatOllama(
                base_url=base_url,
                **common_params
            )

        elif model_config.provider == "bedrock":
            return ChatBedrock(
                region_name=os.getenv(
                    provider_config.region_env,
                    provider_config.default_region
                ),
                credentials_profile_name=os.getenv(provider_config.profile_env),
                model_id=model_config.model_id,
                model_kwargs={**common_params, **model_kwargs}
            )

        else:
            raise ValueError(f"Unsupported provider: {model_config.provider}")

    def list_models(self) -> Dict[str, Dict[str, Any]]:
        """List all available models with their configurations.
        
        Returns:
            Dictionary of model names and their configurations.
        """
        return {
            name: {
                "provider": model.provider,
                "model_id": model.model_id,
                "max_tokens": model.max_tokens,
                "temperature": model.temperature,
                "top_p": model.top_p
            }
            for name, model in self.models.items()
        }

    def list_providers(self) -> Dict[str, Dict[str, Any]]:
        """List all configured providers.
        
        Returns:
            Dictionary of provider names and their configurations.
        """
        return {
            name: provider.model_dump(exclude_none=True)
            for name, provider in self.provider_configs.items()
        } 