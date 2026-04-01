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
    GEMINI_API_KEY: Optional[str] = Field(None)
    TAVILY_API_KEY: Optional[str] = Field(None)
    FRED_API_KEY: Optional[str] = Field(None)
    ALPHA_VANTAGE_API_KEY: Optional[str] = Field(None)
    SEC_EDGAR_USER_AGENT: Optional[str] = Field(None)
    
    # Tier 1: Scout (Reading/Extraction/Coding)
    # Using Llama 17B Scout for balanced extraction
    SCOUT_MODEL: str = Field("groq/llama-3.1-8b-instant", description="Fast model for data processing (e.g., groq/llama-3.1-8b-instant)")
    
    # Tier 2: Strategist (Deep Economic Reasoning)
    # Using the latest Llama 3.3 70B for better rate limits
    STRATEGIST_MODEL: str = Field("groq/llama-3.3-70b-versatile", description="Heavy reasoning model (e.g., openai/gpt-4o)")
    
    # Tier 3: Critic (Asymmetric Red-Teaming)
    # Using Llama 3.3 70B for its higher rate limits
    CRITIC_MODEL: str = Field("groq/llama-3.3-70b-versatile", description="Alternative model for red-teaming (e.g., anthropic/claude-3-5-sonnet)")
    
    # Phase 5: Compiler Teacher
    COMPILER_MODEL: str = Field("groq/llama-3.1-70b-versatile", description="Teacher model for prompt optimization")
    
    # Groq API Limits Mapping (Tokens Per Minute & Tokens Per Day)
    # Based on user-provided table
    MODEL_LIMITS: Dict[str, Dict[str, int]] = Field(
        default_factory=lambda: {
            "openai/gpt-oss-120b": {"tpm": 8000, "tpd": 200000, "rpd": 1000},
            "meta-llama/llama-4-scout-17b-16e-instruct": {"tpm": 30000, "tpd": 500000, "rpd": 1000},
            "llama-3.1-8b-instant": {"tpm": 6000, "tpd": 500000, "rpd": 14400},
            "llama-3.1-70b-versatile": {"tpm": 6000, "tpd": 500000, "rpd": 14400},
            "llama-3.3-70b-versatile": {"tpm": 12000, "tpd": 100000, "rpd": 1000},
            "qwen/qwen3-32b": {"tpm": 6000, "tpd": 500000, "rpd": 1000}
        }
    )

    def get_tpm_limit(self, model_name: str) -> int:
        """Returns the TPM limit for a given model, defaulting to 6000."""
        # Try full name, then stripped name
        clean_name = model_name.split("/")[-1] if "/" in model_name else model_name
        limit_data = self.MODEL_LIMITS.get(model_name) or self.MODEL_LIMITS.get(clean_name) or {}
        return limit_data.get("tpm", 6000)
    
    def get_tpd_limit(self, model_name: str) -> int:
        """Returns the TPD limit for a given model, defaulting to 100,000."""
        clean_name = model_name.split("/")[-1] if "/" in model_name else model_name
        limit_data = self.MODEL_LIMITS.get(model_name) or self.MODEL_LIMITS.get(clean_name) or {}
        return limit_data.get("tpd", 100000)

    # Global DSPy Settings
    DSPY_MAX_TOKENS: int = Field(8192, description="Max tokens for response generation")
    LOG_LEVEL: str = Field("INFO")
    EXPORT_DIR: str = Field("exports")

# Global Settings Instance
settings = Settings()
