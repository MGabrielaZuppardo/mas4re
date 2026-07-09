from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    anthropic_api_key: str = Field(default="")
    groq_api_key: str = Field(default="")

    azure_openai_api_key: str = Field(default="")
    azure_openai_endpoint: str = Field(default="")
    azure_openai_api_version: str = Field(default="2025-01-01-preview")

    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_classifier_model: str = Field(default="qwen2.5:7b")
    ollama_prioritizer_model: str = Field(default="llama3.1:8b")
    ollama_fast_model: str = Field(default="phi3.5:3.8b")

    model_opus: str = Field(default="claude-opus-4-6")
    model_sonnet: str = Field(default="claude-sonnet-4-6")
    model_haiku: str = Field(default="claude-haiku-4-5")

    model_groq_default: str = Field(default="llama-3.3-70b-versatile")
    model_qwen_groq: str = Field(default="qwen2.5-72b-instruct")

    classifier_model: str = Field(default="ollama/qwen2.5:7b")
    prioritizer_model: str = Field(default="ollama/llama3.1:8b")

    agent_temperature: float = Field(default=0.0)
    max_workers: int = Field(default=3)
    max_retries: int = Field(default=3)

    promise_dataset_path: str = Field(default="datasets/data/promise_nfr/promise_nfr_pt.csv")
    nfric_dataset_path: str = Field(default="datasets/data/nfric/")


settings = Settings()
