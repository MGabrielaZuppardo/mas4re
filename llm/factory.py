import logging
from collections.abc import Callable

from langchain_core.language_models import BaseChatModel

from config.settings import settings

logger = logging.getLogger(__name__)

Builder = Callable[[str, float], BaseChatModel]

_PROVIDERS: dict[str, Builder] = {}

# Names without a provider prefix that have always been routed to a provider.
_BARE_PREFIXES: dict[str, str] = {
    "claude": "anthropic",
    "llama": "groq",
    "qwen": "groq",
    "mixtral": "groq",
    "gemma": "groq",
}


def provider(prefix: str) -> Callable[[Builder], Builder]:
    """Register how to build a chat model for `<prefix>/<name>`. A new provider is one decorated
    function; `build_llm` does not change."""

    def register(builder: Builder) -> Builder:
        _PROVIDERS[prefix] = builder
        return builder

    return register


def available_providers() -> list[str]:
    return sorted(_PROVIDERS)


@provider("ollama")
def _ollama(name: str, temperature: float) -> BaseChatModel:
    from langchain_ollama import ChatOllama

    logger.info("LLM → Ollama local | model=%s", name)
    return ChatOllama(model=name, base_url=settings.ollama_base_url, temperature=temperature)


@provider("azure")
def _azure_openai(deployment: str, temperature: float) -> BaseChatModel:
    from langchain_openai import AzureChatOpenAI

    logger.info("LLM → Azure OpenAI | deployment=%s", deployment)
    return AzureChatOpenAI(
        azure_deployment=deployment,
        azure_endpoint=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        api_version=settings.azure_openai_api_version,
        temperature=temperature,
    )


@provider("foundry")
def _foundry(deployment: str, temperature: float) -> BaseChatModel:
    from langchain_openai import AzureChatOpenAI

    logger.info("LLM → Azure AI Foundry | deployment=%s", deployment)
    return AzureChatOpenAI(
        azure_deployment=deployment,
        azure_endpoint=settings.foundry_endpoint,
        api_key=settings.foundry_api_key,
        api_version="2025-01-01-preview",
        temperature=temperature,
    )


@provider("deepseek")
def _deepseek(name: str, temperature: float) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    logger.info("LLM → DeepSeek | model=%s", name)
    return ChatOpenAI(
        model=name,
        base_url=settings.deepseek_base_url,
        api_key=settings.deepseek_api_key,
        temperature=temperature,
    )


@provider("anthropic")
def _anthropic(name: str, temperature: float) -> BaseChatModel:
    from langchain_anthropic import ChatAnthropic

    logger.info("LLM → Anthropic | model=%s", name)
    return ChatAnthropic(model=name, temperature=temperature, api_key=settings.anthropic_api_key)


@provider("groq")
def _groq(name: str, temperature: float) -> BaseChatModel:
    from langchain_groq import ChatGroq

    logger.info("LLM → Groq | model=%s", name)
    return ChatGroq(model_name=name, temperature=temperature, api_key=settings.groq_api_key)


def _resolve(model: str) -> tuple[Builder, str] | None:
    prefix, separator, name = model.partition("/")
    if separator and prefix in _PROVIDERS:
        return _PROVIDERS[prefix], name
    for bare, target in _BARE_PREFIXES.items():
        if model.startswith(bare):
            return _PROVIDERS[target], model
    return None


def build_llm(model: str, temperature: float = 0.0) -> BaseChatModel:
    """Chat model for `model`, written `<provider>/<name>`: ollama/qwen2.5:7b,
    foundry/gpt-4.1-mini, azure/<deployment>, deepseek/deepseek-chat, anthropic/<model>,
    groq/<model>. Bare `claude-*` and Groq-hosted names keep working. Providers are registered
    with `@provider`."""
    resolved = _resolve(model)
    if resolved is None:
        raise ValueError(
            f"Modelo não reconhecido: {model!r}\n"
            f"Prefixos válidos: {', '.join(f'{p}/' for p in available_providers())}"
        )
    builder, name = resolved
    return builder(name, temperature)
