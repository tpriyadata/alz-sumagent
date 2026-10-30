"""
tests/test_settings.py
=======================
ALZDETECT-AI — Unit tests for configs/settings.py

WHAT:   Tests Settings validation, defaults, and field constraints.
WHY:    settings.py is the foundation — if config breaks, everything breaks.
        These tests catch bad .env values before they reach the pipeline.
WHO:    Run after any change to settings.py or .env structure
WHEN:   Before every commit that touches configs/

Run:
    pytest tests/test_settings.py -v
"""

import pytest
from pydantic import ValidationError
from unittest.mock import patch


# ── Helpers ───────────────────────────────────────────────────────

VALID_ENV = {
    "ANTHROPIC_API_KEY": "sk-ant-test123456789",
    "PINECONE_API_KEY":  "pcsk_test123456789",
    "PUBMED_EMAIL":      "test@example.com",
    "HF_TOKEN":          "hf_test123456789",
}


def make_settings(**overrides):
    """Build Settings with valid base env, applying overrides."""
    from configs.settings import Settings
    env = {**VALID_ENV, **overrides}
    with patch.dict("os.environ", env, clear=True):
        return Settings()


# ── API key validators ────────────────────────────────────────────

def test_anthropic_key_valid():
    """Valid anthropic key passes."""
    s = make_settings()
    assert s.anthropic_api_key.startswith("sk-ant-")


def test_anthropic_key_wrong_format_rejected():
    """Wrong key format raises ValidationError."""
    with pytest.raises(ValidationError, match="sk-ant-"):
        make_settings(ANTHROPIC_API_KEY="wrong-key-format")


def test_pinecone_key_valid():
    """Valid pinecone key passes."""
    s = make_settings()
    assert s.pinecone_api_key.startswith("pcsk_")


def test_pinecone_key_wrong_format_rejected():
    """Wrong pinecone key format raises ValidationError."""
    with pytest.raises(ValidationError, match="pcsk_"):
        make_settings(PINECONE_API_KEY="wrong-key-format")


def test_pubmed_email_valid():
    """Valid email passes."""
    s = make_settings()
    assert "@" in s.pubmed_email


def test_pubmed_email_invalid_rejected():
    """Invalid email raises ValidationError."""
    with pytest.raises(ValidationError, match="valid email"):
        make_settings(PUBMED_EMAIL="notanemail")


# ── Defaults ──────────────────────────────────────────────────────

def test_claude_model_default():
    """Claude model default is correct."""
    s = make_settings()
    assert s.claude_model == "claude-sonnet-4-6"


def test_max_tokens_default():
    """Max tokens default is 1500."""
    s = make_settings()
    assert s.max_tokens == 1500


def test_target_papers_default():
    """Target papers default is 25000."""
    s = make_settings()
    assert s.target_papers == 25000


def test_pinecone_index_default():
    """Pinecone index default is alzdetect."""
    s = make_settings()
    assert s.pinecone_index_name == "alzdetect"


def test_chunk_size_default():
    """Chunk size default is 512."""
    s = make_settings()
    assert s.chunk_size == 512


def test_top_k_default():
    """Top K default is 20."""
    s = make_settings()
    assert s.top_k_results == 20


def test_vector_db_default():
    """Vector DB default is pinecone."""
    s = make_settings()
    assert s.vector_db == "pinecone"


def test_rerank_enabled_default():
    """Reranking is enabled by default."""
    s = make_settings()
    assert s.rerank_enabled is True


def test_environment_default():
    """Environment default is development."""
    s = make_settings()
    assert s.environment == "development"


# ── Range validators ──────────────────────────────────────────────

def test_max_tokens_too_low_rejected():
    """max_tokens below 100 is rejected."""
    with pytest.raises(ValidationError):
        make_settings(MAX_TOKENS="50")


def test_max_tokens_too_high_rejected():
    """max_tokens above 4096 is rejected."""
    with pytest.raises(ValidationError):
        make_settings(MAX_TOKENS="9999")


def test_chunk_size_too_low_rejected():
    """chunk_size below 64 is rejected."""
    with pytest.raises(ValidationError):
        make_settings(CHUNK_SIZE="10")


def test_top_k_zero_rejected():
    """top_k_results of 0 is rejected."""
    with pytest.raises(ValidationError):
        make_settings(TOP_K_RESULTS="0")


def test_similarity_threshold_out_of_range_rejected():
    """similarity_threshold above 1.0 is rejected."""
    with pytest.raises(ValidationError):
        make_settings(SIMILARITY_THRESHOLD="1.5")


# ── Cross-field validators ────────────────────────────────────────

def test_chunk_overlap_less_than_chunk_size():
    """chunk_overlap must be less than chunk_size."""
    s = make_settings(CHUNK_SIZE="512", CHUNK_OVERLAP="50")
    assert s.chunk_overlap < s.chunk_size


def test_chunk_overlap_equal_chunk_size_rejected():
    """chunk_overlap equal to chunk_size raises ValidationError."""
    with pytest.raises(ValidationError, match="chunk_overlap"):
        make_settings(CHUNK_SIZE="512", CHUNK_OVERLAP="512")


def test_rerank_top_n_less_than_top_k():
    """rerank_top_n must be less than top_k_results."""
    s = make_settings(TOP_K_RESULTS="20", RERANK_TOP_N="5")
    assert s.rerank_top_n < s.top_k_results


def test_rerank_top_n_equal_top_k_rejected():
    """rerank_top_n equal to top_k_results raises ValidationError."""
    with pytest.raises(ValidationError, match="rerank_top_n"):
        make_settings(TOP_K_RESULTS="5", RERANK_TOP_N="5")


def test_summarizer_min_less_than_max():
    """summarizer_min_length must be less than summarizer_max_length."""
    s = make_settings()
    assert s.summarizer_min_length < s.summarizer_max_length


def test_summarizer_min_equal_max_rejected():
    """summarizer_min_length equal to max raises ValidationError."""
    with pytest.raises(ValidationError, match="summarizer_min_length"):
        make_settings(SUMMARIZER_MIN_LENGTH="256", SUMMARIZER_MAX_LENGTH="256")


# ── String normalisation ──────────────────────────────────────────

def test_log_level_normalised_to_uppercase():
    """'debug' is normalised to 'DEBUG'."""
    s = make_settings(LOG_LEVEL="debug")
    assert s.log_level == "DEBUG"


def test_environment_normalised_to_lowercase():
    """'DEVELOPMENT' is normalised to 'development'."""
    s = make_settings(ENVIRONMENT="DEVELOPMENT")
    assert s.environment == "development"


def test_vector_db_normalised_to_lowercase():
    """'Pinecone' is normalised to 'pinecone'."""
    s = make_settings(VECTOR_DB="Pinecone")
    assert s.vector_db == "pinecone"


# ── Invalid string values ─────────────────────────────────────────

def test_invalid_log_level_rejected():
    """Unknown log level raises ValidationError."""
    with pytest.raises(ValidationError):
        make_settings(LOG_LEVEL="VERBOSE")


def test_invalid_environment_rejected():
    """'staging' environment raises ValidationError."""
    with pytest.raises(ValidationError):
        make_settings(ENVIRONMENT="staging")


def test_invalid_vector_db_rejected():
    """Unknown vector DB raises ValidationError."""
    with pytest.raises(ValidationError):
        make_settings(VECTOR_DB="elasticsearch")


# ── get_settings caching ──────────────────────────────────────────

def test_get_settings_returns_settings_object():
    """get_settings() returns a Settings instance."""
    from configs.settings import Settings, get_settings
    with patch.dict("os.environ", VALID_ENV, clear=True):
        get_settings.cache_clear()
        s = get_settings()
        assert isinstance(s, Settings)


def test_get_settings_is_cached():
    """get_settings() returns same object on repeated calls."""
    from configs.settings import get_settings
    with patch.dict("os.environ", VALID_ENV, clear=True):
        get_settings.cache_clear()
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2
