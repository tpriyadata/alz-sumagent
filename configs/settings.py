
"""
configs/settings.py

"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator, model_validator
from pathlib import Path
from loguru import logger
from functools import lru_cache


# ── Root path of the project ─────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """
    Central settings for Alz-Bot.
    Reads from .env file automatically.
    All fields are validated at startup — fail fast, not late.
    """

    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",           # ignore unknown .env keys silently
    )

    # ── Anthropic / Claude ────────────────────────────────────────
    anthropic_api_key: str = Field(
        description="Anthropic API key  get from console.anthropic.com"
    )
    claude_model: str = Field(
        default="claude-sonnet-4-6",
        description="Claude model to use for generation"
    )
    max_tokens: int = Field(
        default=1500,
        ge=100,
        le=4096,
        description="Max tokens for Claude response"
    )
    #── Pinecone ──────────────────────────────────────────────────
    pinecone_api_key: str = Field(
        ...,
        description="Pinecone API key — get from app.pinecone.io"
    )
    pinecone_index_name: str = Field(
        default="alzdetect",
        description="Pinecone index name"
    )
    pinecone_environment: str = Field(
        default="us-east-1",
        description="Pinecone cloud region"
    )

    # ── PubMed ────────────────────────────────────────────────────
    pubmed_email: str = Field(
        ...,
        description="Your email — required by NCBI PubMed API"
    )
    target_papers: int = Field(
        default=25000,
        ge=100,
        le=50000,
        description="Target number of papers to fetch"
    )
    papers_per_query: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Papers fetched per query"
    )
    batch_size: int = Field(
        default=50,
        ge=10,
        le=500,
        description="API call batch size"
    )

    # ── Embeddings ────────────────────────────────────────────────
    embedding_model: str = Field(
        default="pritamdeka/S-PubMedBert-MS-MARCO",
        description="Sentence transformer model for embeddings"
    )
    embedding_dim: int = Field(
        default=768,
        description="Embedding dimension — must match Pinecone index"
    )
    embedding_batch_size: int = Field(
        default=128,
        description="Batch size for embedding — increase for GPU"
    )

    # ── Chunking ──────────────────────────────────────────────────
    chunk_size: int = Field(
        default=512,
        ge=64,
        le=2048,
        description="Max tokens per chunk"
    )
    chunk_overlap: int = Field(
        default=50,
        ge=0,
        le=200,
        description="Overlap between chunks in tokens"
    )
   
    # ── RAG retrieval ────────────────────────────────────────────
    top_k_results: int = Field(
        default=20,          # retrieve more, rerank to best
        ge=1,
        le=50,
        description="Number of chunks to retrieve from Pinecone"
    )
    rerank_top_n: int = Field(
        default=5,           # after reranking, pass top 5 to Claude
        ge=1,
        le=20,
        description="Final chunks passed to Claude after reranking"
    )
    rerank_enabled: bool = Field(
        default=True,
        description="Enable cross-encoder reranking"
    )
    rerank_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Cross-encoder model for reranking"
    )
    similarity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score to include chunk"
    )

    # ── Summarizer — Architecture 1 (PubMed) ─────────────────────
    summarizer_model: str = Field(
        default="facebook/bart-large-cnn",
        description="HuggingFace summarizer for PubMed papers"
    )
    summarizer_max_length: int = Field(
        default=256,
        ge=64,
        le=512,
        description="Max tokens in summarized paper output"
    )
    summarizer_min_length: int = Field(
        default=64,
        ge=32,
        le=256,
        description="Min tokens in summarized paper output"
    )

    # ── Vector DB (configurable) ─────────────────────────────────
    vector_db: str = Field(
        default="pinecone",
        description="Vector DB: pinecone | qdrant | weaviate | pgvector"
    )

    # ── HuggingFace Hub deployment ────────────────────────────────
    hf_token: str = Field(
        ...,
        description="HuggingFace API token — get from huggingface.co/settings/tokens"
    )
    hf_model_repo: str = Field(
        default="oviya08/Alz-SumAgent",
        description="HuggingFace model repo — oviya08/Alz-SumAgent"
    )
    hf_space_name: str = Field(
        default="oviya08/alz-bot",
        description="HuggingFace Space name for deployed app"
    )

    # ── Paths ─────────────────────────────────────────────────────
    raw_data_path: Path = Field(
        default=ROOT_DIR / "data" / "raw" / "alzheimer_papers.json",
        description="Path to raw PubMed papers JSON"
    )
    processed_data_path: Path = Field(
        default=ROOT_DIR / "data" / "processed" / "chunks.json",
        description="Path to processed chunks JSON"
    )

    # ── Paths — Shared ────────────────────────────────────────────
    embeddings_cache_path: Path = Field(
        default=ROOT_DIR / "data" / "cache" / "embeddings.npy",
        description="Cached embeddings — avoids recomputing 25k papers"
    )
    logs_path: Path = Field(
        default=ROOT_DIR / "logs",
        description="Loguru log file directory"
    )
    eval_results_path: Path = Field(
        default=ROOT_DIR / "data" / "eval" / "results.json",
        description="RAG evaluation results storage"
    )
    models_cache_path: Path = Field(
        default=ROOT_DIR / ".cache" / "models",
        description="HuggingFace model download cache"
    )

    # ── Logging ───────────────────────────────────────────────────
    log_level: str = Field(
        default="INFO",
        description="Logging level: DEBUG, INFO, WARNING, ERROR"
    )
    environment: str = Field(
        default="development",
        description="Environment: development or production"
    )

    # ── Validators ────────────────────────────────────────────────
    @field_validator("anthropic_api_key")
    @classmethod
    def validate_anthropic_key(cls, v: str) -> str:
        """Catch wrong key format before any API call is made."""
        if not v.startswith("sk-ant-"):
            raise ValueError(
                "ANTHROPIC_API_KEY must start with 'sk-ant-'. "
                "Get your key from console.anthropic.com"
            )
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR"}
        v = v.upper()
        if v not in allowed:
            raise ValueError(f"log_level must be one of {allowed}")
        return v

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        allowed = {"development", "production"}
        v = v.lower()
        if v not in allowed:
            raise ValueError(f"environment must be one of {allowed}")
        return v

    @field_validator("pinecone_api_key")
    @classmethod
    def validate_pinecone_key(cls, v: str) -> str:
        if not v.startswith("pcsk_"):
            raise ValueError(
                "PINECONE_API_KEY must start with 'pcsk_'. "
                "Get your key from app.pinecone.io"
            )
        return v

    @field_validator("pubmed_email")
    @classmethod
    def validate_pubmed_email(cls, v: str) -> str:
        if "@" not in v or "." not in v:
            raise ValueError(
                "PUBMED_EMAIL must be a valid email — required by NCBI"
            )
        return v

    @model_validator(mode="after")
    def validate_chunk_config(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be "
                f"less than chunk_size ({self.chunk_size})"
            )
        return self

    @model_validator(mode="after")
    def validate_rerank_config(self) -> "Settings":
        if self.rerank_top_n >= self.top_k_results:
            raise ValueError(
                f"rerank_top_n ({self.rerank_top_n}) must be "
                f"less than top_k_results ({self.top_k_results})"
            )
        return self

    @model_validator(mode="after")
    def create_directories(self) -> "Settings":
        for path in [
            self.embeddings_cache_path.parent,
            self.processed_data_path.parent,
        ]:
            path.mkdir(parents=True, exist_ok=True)
        return self

    @model_validator(mode="after")
    def validate_summarizer_config(self) -> "Settings":
        if self.summarizer_min_length >= self.summarizer_max_length:
            raise ValueError(
                f"summarizer_min_length ({self.summarizer_min_length}) must be "
                f"less than summarizer_max_length ({self.summarizer_max_length})"
            )
        return self

    @field_validator("vector_db")
    @classmethod
    def validate_vector_db(cls, v: str) -> str:
        allowed = {"pinecone", "qdrant", "weaviate", "pgvector"}
        v = v.lower()
        if v not in allowed:
            raise ValueError(f"vector_db must be one of {allowed}")
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Factory function — always use this to get settings.
    Never instantiate Settings() directly in other modules.

    Usage in any module:
        from configs.settings import get_settings
        settings = get_settings()
        print(settings.claude_model)
    """
    try:
        settings = Settings()
        logger.info(
            "[SETTINGS] Configuration loaded | "
            f"env={settings.environment} | "
            f"model={settings.claude_model} | "
            f"target_papers={settings.target_papers} | "
            f"vector_db={settings.vector_db} | "
            f"hf_space={settings.hf_space_name}"
        )
        return settings
    except Exception as e:
        logger.error(f"[SETTINGS] FATAL — configuration invalid: {e}")
        raise


if __name__ == "__main__":
    s = get_settings()
    print("\n" + "=" * 55)
    print("  ALZDETECT-AI — Settings Check")
    print("=" * 55)
    print(f"  Environment      : {s.environment}")
    print(f"  Claude model     : {s.claude_model}")
    print(f"  Max tokens       : {s.max_tokens}")
    print("─" * 55)
    print(f"  Vector DB        : {s.vector_db}")
    print(f"  Pinecone index   : {s.pinecone_index_name}")
    print(f"  Pinecone region  : {s.pinecone_environment}")
    print("─" * 55)
    print(f"  Target papers    : {s.target_papers:,}")
    print(f"  Papers per query : {s.papers_per_query:,}")
    print(f"  Batch size       : {s.batch_size}")
    print("─" * 55)
    print(f"  Embedding model  : {s.embedding_model}")
    print(f"  Embedding dim    : {s.embedding_dim}")
    print(f"  Embedding batch  : {s.embedding_batch_size}")
    print("─" * 55)
    print(f"  Chunk size       : {s.chunk_size}")
    print(f"  Chunk overlap    : {s.chunk_overlap}")
    print("─" * 55)
    print(f"  Top K results    : {s.top_k_results}")
    print(f"  Rerank enabled   : {s.rerank_enabled}")
    print(f"  Rerank top N     : {s.rerank_top_n}")
    print(f"  Rerank model     : {s.rerank_model}")
    print(f"  Similarity thresh: {s.similarity_threshold}")
    print("─" * 55)
    print(f"  Summarizer       : {s.summarizer_model}")
    print(f"  Summarizer min   : {s.summarizer_min_length}")
    print(f"  Summarizer max   : {s.summarizer_max_length}")
    print("─" * 55)
    print(f"  HF Model repo    : {s.hf_model_repo}")
    print(f"  HF Space         : {s.hf_space_name}")
    print("─" * 55)
    print(f"  Log level        : {s.log_level}")
    print("─" * 55)
    print("  API Keys (last 6 chars only — never print full keys)")
    print(f"  Anthropic        : sk-ant-...{s.anthropic_api_key[-6:]}")
    print(f"  Pinecone         : pcsk_...{s.pinecone_api_key[-6:]}")
    print(f"  HF token         : hf_...{s.hf_token[-6:]}")
    print("=" * 55)
    print("  All settings valid. Ready to run.")
    print("=" * 55 + "\n")
