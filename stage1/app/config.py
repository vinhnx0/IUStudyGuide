# config.py
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    # Ollama
    ollama_host: str = Field(default="http://localhost:11434", description="Ollama base URL")
    model: str = Field(default="llama3", description="Default Ollama model name")

    # Inference defaults (deterministic-ish)
    temperature: float = Field(default=0.2)
    max_tokens: int = Field(default=512)
    seed: int = Field(default=42)
    request_timeout_s: int = Field(default=60)

    # API
    api_title: str = Field(default="Stage1 API")
    api_version: str = Field(default="1.0.0")

    # Logging
    log_level: str = Field(default="INFO")
    log_json: bool = Field(default=True)
    service_name: str = Field(default="stage1")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
