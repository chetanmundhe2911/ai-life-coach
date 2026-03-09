"""
config.py — Central Configuration
==================================
WHY THIS EXISTS:
  Instead of scattering settings across files, we keep ALL config here.
  This follows the "12-factor app" principle: config comes from environment.
  
HOW IT WORKS:
  We use pydantic-settings which automatically reads from .env file.
  Every variable has a default so the app works even without .env.

USAGE in other files:
  from config import settings
  print(settings.MODEL_NAME)
"""

from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # ── OpenAI ────────────────────────────────────────────────────
    OPENAI_API_KEY: str = ""

    # ── Model Settings ────────────────────────────────────────────
    MODEL_NAME: str = "gpt-4"                        # LLM for chat
    EMBEDDING_MODEL: str = "text-embedding-3-small"  # Model for vectors
    MAX_TOKENS: int = 1000                            # Max reply length
    TEMPERATURE: float = 0.7                          # 0=precise, 1=creative

    # ── RAG (Retrieval-Augmented Generation) Settings ─────────────
    TOP_K_RESULTS: int = 3        # How many doc chunks to retrieve
    CHUNK_SIZE: int = 500         # Characters per chunk when splitting docs
    CHUNK_OVERLAP: int = 50       # Overlap between chunks (preserves context)

    # ── Paths ─────────────────────────────────────────────────────
    DOCS_DIR: str = "./docs"            # Where your personal docs live
    RAG_CACHE_DIR: str = "./.rag_cache" # Where FAISS index is saved

    # ── Agent Settings ────────────────────────────────────────────
    MAX_CONVERSATION_HISTORY: int = 10  # How many past messages to remember
    AGENT_NAME: str = "Aria"            # Your life coach's name

    # ── Computed Properties ───────────────────────────────────────
    @property
    def docs_path(self) -> Path:
        """Returns docs directory as a Path object"""
        return Path(self.DOCS_DIR)

    @property
    def cache_path(self) -> Path:
        """Returns cache directory as a Path object"""
        return Path(self.RAG_CACHE_DIR)

    @property
    def faiss_index_path(self) -> Path:
        """Where the FAISS vector index file is saved"""
        return self.cache_path / "faiss_index"

    class Config:
        # Tell pydantic to read from .env file
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Ignore extra vars in .env


# Create a single global instance
# Import this everywhere: `from config import settings`
settings = Settings()


# ── Validation ────────────────────────────────────────────────────
def validate_settings():
    """
    Call this at startup to catch config errors early.
    Better to fail fast with a clear message than fail mysteriously later.
    """
    errors = []
    
    if not settings.OPENAI_API_KEY:
        errors.append("❌ OPENAI_API_KEY is not set in .env")
    
    if not settings.OPENAI_API_KEY.startswith("sk-"):
        errors.append("❌ OPENAI_API_KEY looks invalid (should start with 'sk-')")
    
    if settings.TEMPERATURE < 0 or settings.TEMPERATURE > 1:
        errors.append("❌ TEMPERATURE must be between 0 and 1")
    
    if errors:
        print("\n".join(errors))
        print("\n💡 Copy .env.example to .env and fill in your values")
        raise ValueError("Invalid configuration")
    
    # Create directories if they don't exist
    settings.docs_path.mkdir(parents=True, exist_ok=True)
    settings.cache_path.mkdir(parents=True, exist_ok=True)
    
    print(f"✅ Config loaded | Model: {settings.MODEL_NAME} | Agent: {settings.AGENT_NAME}")
