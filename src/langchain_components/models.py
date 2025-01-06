"""Model registry for Cloud Inspector."""

import os
import yaml
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Type, List
from enum import Enum

from langchain_openai.chat_models.base import ChatOpenAI
from langchain_anthropic.chat_models import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_aws.chat_models.bedrock import ChatBedrock
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain.schema.language_model import BaseLanguageModel
from pydantic import BaseModel, SecretStr


class ModelCapability(Enum):
    CODE_GENERATION = "code_generation"
    PROMPT_GENERATION = "prompt_generation"


class CommonModelParams(BaseModel):
    """Common parameters for all models."""

    model_id: str
    capabilities: List[ModelCapability]
    max_tokens: int
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stop: Optional[list[str]] = None


class OpenAIParams(BaseModel):
    """OpenAI-specific parameters."""

    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    response_format: Optional[Dict[str, Any]] = None


class OllamaParams(BaseModel):
    """Ollama-specific parameters."""

    repeat_penalty: Optional[float] = None


class GoogleParams(BaseModel):
    """Google-specific parameters."""

    response_schema: Optional[Dict[str, Any]] = None
    response_mime_type: Optional[str] = None


class ModelConfig(BaseModel):
    """Model configuration."""

    provider: str
    common: CommonModelParams
    openai: Optional[OpenAIParams] = None
    ollama: Optional[OllamaParams] = None
    google: Optional[GoogleParams] = None
    supports_system_prompt: bool = True


class ProviderConfig(BaseModel):
    """Provider configuration."""

    api_key_env: Optional[str] = None
    organization_env: Optional[str] = None
    base_url_env: Optional[str] = None
    default_base_url: Optional[str] = None
    region_env: Optional[str] = None
    profile_env: Optional[str] = None
    default_region: Optional[str] = None


class ProviderStrategy(ABC):
    """Abstract base class for provider-specific strategies."""

    @abstractmethod
    def create_model(
        self,
        model_config: ModelConfig,
        provider_config: ProviderConfig,
    ) -> BaseLanguageModel:
        """Create a model instance for this provider."""
        pass

    @abstractmethod
    def get_structured_output_params(self, output_type: Type) -> Dict[str, Any]:
        """Get provider-specific structured output parameters."""
        pass

    def _get_common_params(self, common: CommonModelParams) -> Dict[str, Any]:
        """Get common parameters supported by all LangChain chat models."""
        params = {
            "model": common.model_id,
            "max_tokens": common.max_tokens,
        }
        if common.temperature is not None:
            params["temperature"] = common.temperature
        if common.top_p is not None:
            params["top_p"] = common.top_p
        if common.top_k is not None:
            params["top_k"] = common.top_k
        if common.stop is not None:
            params["stop"] = common.stop
        return params


class OpenAIStrategy(ProviderStrategy):
    def create_model(
        self,
        model_config: ModelConfig,
        provider_config: ProviderConfig,
    ) -> BaseLanguageModel:
        params = self._get_common_params(model_config.common)
        model_kwargs = {}

        if model_config.openai and model_config.openai.response_format:
            model_kwargs["response_format"] = model_config.openai.response_format

        return ChatOpenAI(
            openai_api_key=os.getenv(provider_config.api_key_env),
            openai_organization=os.getenv(provider_config.organization_env),
            frequency_penalty=(
                model_config.openai.frequency_penalty if model_config.openai else None
            ),
            presence_penalty=(
                model_config.openai.presence_penalty if model_config.openai else None
            ),
            model_kwargs=model_kwargs,
            **params,
        )

    def get_structured_output_params(self, output_type: Type) -> Dict[str, Any]:
        return {"method": "json_mode", "include_raw": True}


class AnthropicStrategy(ProviderStrategy):
    def create_model(
        self,
        model_config: ModelConfig,
        provider_config: ProviderConfig,
    ) -> BaseLanguageModel:
        params = self._get_common_params(model_config.common)
        return ChatAnthropic(
            anthropic_api_key=SecretStr(os.getenv(provider_config.api_key_env)),
            **params,
        )

    def get_structured_output_params(self, output_type: Type) -> Dict[str, Any]:
        return {}


class GoogleStrategy(ProviderStrategy):
    def create_model(
        self,
        model_config: ModelConfig,
        provider_config: ProviderConfig,
    ) -> BaseLanguageModel:
        params = self._get_common_params(model_config.common)
        if model_config.google:
            if model_config.google.response_schema:
                params["response_schema"] = model_config.google.response_schema
            if model_config.google.response_mime_type:
                params["response_mime_type"] = model_config.google.response_mime_type

        return ChatGoogleGenerativeAI(
            google_api_key=SecretStr(os.getenv(provider_config.api_key_env)), **params
        )

    def get_structured_output_params(self, output_type: Type) -> Dict[str, Any]:
        return {"include_raw": True}


class OllamaStrategy(ProviderStrategy):
    def create_model(
        self,
        model_config: ModelConfig,
        provider_config: ProviderConfig,
    ) -> BaseLanguageModel:
        params = self._get_common_params(model_config.common)
        if model_config.ollama and model_config.ollama.repeat_penalty is not None:
            params["repeat_penalty"] = model_config.ollama.repeat_penalty

        base_url = os.getenv(
            provider_config.base_url_env, provider_config.default_base_url
        )
        return ChatOllama(base_url=base_url, **params)

    def get_structured_output_params(self, output_type: Type) -> Dict[str, Any]:
        return {"method": "json_mode"}


class BedrockStrategy(ProviderStrategy):
    def create_model(
        self,
        model_config: ModelConfig,
        provider_config: ProviderConfig,
    ) -> BaseLanguageModel:
        params = self._get_common_params(model_config.common)
        return ChatBedrock(
            region_name=os.getenv(
                provider_config.region_env, provider_config.default_region
            ),
            credentials_profile_name=os.getenv(provider_config.profile_env),
            model_id=model_config.common.model_id,
            model_kwargs=params,
            beta_use_converse_api=True,
        )

    def get_structured_output_params(self, output_type: Type) -> Dict[str, Any]:
        return {"include_raw": True}


class ModelRegistry:
    """Registry for managing LLM models."""

    _PROVIDER_STRATEGIES: Dict[str, ProviderStrategy] = {
        "openai": OpenAIStrategy(),
        "anthropic": AnthropicStrategy(),
        "google": GoogleStrategy(),
        "ollama": OllamaStrategy(),
        "bedrock": BedrockStrategy(),
    }

    def __init__(self, config_path: Optional[Path] = None):
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

        # Convert flat config structure to nested
        for name, model_data in config["models"].items():
            common_params = {
                "model_id": model_data["model_id"],
                "capabilities": model_data["capabilities"],
                "max_tokens": model_data["max_tokens"],
            }

            # Only add optional parameters if they exist
            if "temperature" in model_data:
                common_params["temperature"] = model_data["temperature"]
            if "top_p" in model_data:
                common_params["top_p"] = model_data["top_p"]
            if "top_k" in model_data:
                common_params["top_k"] = model_data["top_k"]
            if "stop" in model_data:
                common_params["stop"] = model_data["stop"]

            provider_specific = {}
            if model_data["provider"] == "openai":
                provider_specific["openai"] = OpenAIParams(
                    frequency_penalty=model_data.get("frequency_penalty"),
                    presence_penalty=model_data.get("presence_penalty"),
                    response_format=model_data.get("response_format"),
                )
            elif model_data["provider"] == "ollama":
                provider_specific["ollama"] = OllamaParams(
                    repeat_penalty=model_data.get("repeat_penalty"),
                )
            elif model_data["provider"] == "google":
                provider_specific["google"] = GoogleParams(
                    response_schema=model_data.get("response_schema"),
                    response_mime_type=model_data.get("response_mime_type"),
                )

            self.models[name] = ModelConfig(
                provider=model_data["provider"],
                common=CommonModelParams(**common_params),
                **provider_specific,
                supports_system_prompt=model_data.get("supports_system_prompt", True),
            )

        # Load provider configs
        for provider, provider_data in config["provider_configs"].items():
            self.provider_configs[provider] = ProviderConfig(**provider_data)

    def get_model(self, name: str) -> BaseLanguageModel:
        """Get a model instance by name."""
        if name not in self.models:
            raise ValueError(f"Model '{name}' not found in configuration")

        model_config = self.models[name]
        provider_config = self.provider_configs[model_config.provider]

        strategy = self._PROVIDER_STRATEGIES.get(model_config.provider)
        if not strategy:
            raise ValueError(f"Unsupported provider: {model_config.provider}")

        return strategy.create_model(model_config, provider_config)

    def get_structured_output_params(
        self, name: str, output_type: Type
    ) -> Dict[str, Any]:
        """Get structured output parameters for a specific model."""
        if name not in self.models:
            raise ValueError(f"Model '{name}' not found in configuration")

        model_config = self.models[name]
        strategy = self._PROVIDER_STRATEGIES.get(model_config.provider)
        if not strategy:
            raise ValueError(f"Unsupported provider: {model_config.provider}")

        return strategy.get_structured_output_params(output_type)

    def list_models(self) -> Dict[str, Dict[str, Any]]:
        """List all available models with their configurations."""
        return {
            name: {
                "provider": model.provider,
                "model_id": model.common.model_id,
                "max_tokens": model.common.max_tokens,
                "temperature": model.common.temperature,
                "top_p": model.common.top_p,
            }
            for name, model in self.models.items()
        }

    def list_providers(self) -> Dict[str, Dict[str, Any]]:
        """List all configured providers."""
        return {
            name: provider.model_dump(exclude_none=True)
            for name, provider in self.provider_configs.items()
        }

    def get_models_by_capability(
        self, capability: ModelCapability
    ) -> Dict[str, ModelConfig]:
        """Get all models that have a specific capability."""
        return {
            name: config
            for name, config in self.models.items()
            if capability in config.common.capabilities
        }

    def validate_model_capability(
        self, model_name: str, required_capability: ModelCapability
    ) -> bool:
        """Check if a model has a specific capability."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")

        return required_capability in self.models[model_name].common.capabilities
