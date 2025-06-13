"""Settings configuration for the Ironhide framework, including API endpoints, keys, models, and general options."""

from pydantic_settings import BaseSettings

from ironhide.models import Provider


class Settings(BaseSettings):
    """Configuration settings for the Ironhide framework, including API endpoints, keys, models, and general options."""

    # OpenAI
    openai_url: str = "https://api.openai.com/v1"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_audio_to_text_model_id: str = "whisper-1"

    # Gemini
    gemini_url: str = "https://generativelanguage.googleapis.com/v1beta/models/"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash-preview-05-20"

    # General
    # default_provider: Provider = Provider.openai
    # default_model: str = "gpt-4o-mini"
    # default_api_key: str = ""
    log_level: str = "INFO"
    request_timeout: int = 30


settings = Settings()
