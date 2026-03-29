from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Optional, Dict
from dotenv import load_dotenv
import os

# 1. Discover absolute path to .env
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
env_path = os.path.join(project_root, ".env")

# 2. Pre-load into environment
load_dotenv(dotenv_path=env_path, override=True)

class Settings(BaseSettings):
    """
    Phase 0: System-wide configuration.
    Three-Tier Intelligence: Scout, Strategist, and Critic.
    """
    # Use the absolute path for Pydantic's internal loader as well
    model_config = SettingsConfigDict(
        env_file=env_path, 
        env_file_encoding="utf-8", 
        extra="ignore"
    )

    # API Keys
    GROQ_API_KEY: Optional[str] = Field(None)
    OPENAI_API_KEY: Optional[str] = Field(None)
    ANTHROPIC_API_KEY: Optional[str] = Field(None)
    TAVILY_API_KEY: Optional[str] = Field(None)
    FRED_API_KEY: Optional[str] = Field(None)
    ALPHA_VANTAGE_API_KEY: Optional[str] = Field(None)
    SEC_EDGAR_USER_AGENT: Optional[str] = Field(None)
    
    # Tier 1: Scout (Reading/Extraction/Coding)
    # Using Llama 3 8B for fast extraction
    SCOUT_MODEL: str = Field("llama-3.1-8b-instant", description="Fast model for data processing")
    
    # Tier 2: Strategist (Deep Economic Reasoning)
    # Using the latest Llama 3.3 70B
    STRATEGIST_MODEL: str = Field("llama-3.3-70b-versatile", description="Heavy reasoning model")
    
    # Tier 3: Critic (Asymmetric Red-Teaming)
    # Using Qwen 3 for its distinct logic bias
    CRITIC_MODEL: str = Field("qwen/qwen3-32b", description="Alternative model for red-teaming")
    
    # Global DSPy Settings
    DSPY_MAX_TOKENS: int = Field(8192, description="Max tokens for response generation")
    LOG_LEVEL: str = Field("INFO")
    EXPORT_DIR: str = Field("exports")

# Global Settings Instance
settings = Settings()
