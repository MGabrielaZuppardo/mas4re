import logging

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

from config.settings import settings

logger = logging.getLogger(__name__)


def build_llm(model: str, temperature: float = 0.0) -> BaseChatModel:
    """
    Factory de LLMs com suporte a Ollama (local), Anthropic, Groq e Azure OpenAI.

    Prefixos:
        ollama/<nome>      → Ollama local (sem custo)
        azure/<deployment> → Azure OpenAI (deployment name no Foundry)
        claude-*           → Anthropic (produção)
        llama-* | qwen-* | mixtral-* → Groq (produção)
    """

    if model.startswith("ollama/"):
        model_name = model.removeprefix("ollama/")
        logger.info("LLM → Ollama local | model=%s", model_name)
        return ChatOllama(
            model=model_name,
            base_url=settings.ollama_base_url,
            temperature=temperature,
        )

    elif model.startswith("azure/"):
        from langchain_openai import AzureChatOpenAI

        deployment = model.removeprefix("azure/")
        logger.info("LLM → Azure OpenAI | deployment=%s", deployment)
        return AzureChatOpenAI(
            azure_deployment=deployment,
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            api_version=settings.azure_openai_api_version,
            temperature=temperature,
        )

    elif model.startswith("claude"):
        logger.info("LLM → Anthropic | model=%s", model)
        return ChatAnthropic(
            model=model,
            temperature=temperature,
            api_key=settings.anthropic_api_key,
        )

    elif any(model.startswith(p) for p in ("llama", "qwen", "mixtral", "gemma")):
        logger.info("LLM → Groq | model=%s", model)
        return ChatGroq(
            model_name=model,
            temperature=temperature,
            api_key=settings.groq_api_key,
        )

    raise ValueError(
        f"Modelo não reconhecido: {model!r}\n"
        "Prefixos válidos: ollama/, azure/, claude-, llama-, qwen-, mixtral-, gemma-"
    )
