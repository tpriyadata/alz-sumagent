"""
tests/test_vector_store.py
===========================
ALZDETECT-AI — Unit tests for vector_store/pinecone_store.py

WHAT:   Tests QueryInput and RetrievedChunk Pydantic models.
WHY:    Every query entering Pinecone must be validated.
        Every chunk coming back must be validated.
WHO:    Run after any change to pinecone_store.py
WHEN:   Before every commit that touches vector_store/

Run:
    pytest tests/test_vector_store.py -v
"""

import pytest
from pydantic import ValidationError
from vector_store.pinecone_store import QueryInput, RetrievedChunk, QueryResult


# ── QueryInput tests ──────────────────────────────────────────────

def test_query_input_valid():
    """Valid question passes."""
    q = QueryInput(question="What biomarkers predict Alzheimer's disease?")
    assert q.question == "What biomarkers predict Alzheimer's disease?"
    assert q.top_k == 20       # default
    assert q.min_score == 0.3  # default


def test_query_input_too_short():
    """Question under 3 chars is rejected."""
    with pytest.raises(ValidationError):
        QueryInput(question="AD")


def test_query_input_empty_rejected():
    """Empty question is rejected."""
    with pytest.raises(ValidationError):
        QueryInput(question="")


def test_query_input_whitespace_rejected():
    """Whitespace-only question is rejected."""
    with pytest.raises(ValidationError):
        QueryInput(question="   ")


def test_query_input_top_k_valid():
    """top_k within range passes."""
    q = QueryInput(question="Alzheimer biomarkers", top_k=5)
    assert q.top_k == 5


def test_query_input_top_k_too_high():
    """top_k over 50 is rejected."""
    with pytest.raises(ValidationError):
        QueryInput(question="Alzheimer biomarkers", top_k=100)


def test_query_input_top_k_zero_rejected():
    """top_k of 0 is rejected."""
    with pytest.raises(ValidationError):
        QueryInput(question="Alzheimer biomarkers", top_k=0)


def test_query_input_source_pubmed():
    """source='pubmed' is valid."""
    q = QueryInput(question="Alzheimer biomarkers", source="pubmed")
    assert q.source == "pubmed"


def test_query_input_source_invalid():
    """Invalid source raises ValidationError."""
    with pytest.raises(ValidationError):
        QueryInput(question="Alzheimer biomarkers", source="twitter")


def test_query_input_source_none():
    """source=None means no filter — valid."""
    q = QueryInput(question="Alzheimer biomarkers", source=None)
    assert q.source is None


def test_query_input_year_from_valid():
    """Valid year_from passes."""
    q = QueryInput(question="Alzheimer biomarkers", year_from=2020)
    assert q.year_from == 2020


def test_query_input_year_from_too_old():
    """year_from before 2000 is rejected."""
    with pytest.raises(ValidationError):
        QueryInput(question="Alzheimer biomarkers", year_from=1990)


def test_query_input_min_score_valid():
    """Valid min_score passes."""
    q = QueryInput(question="Alzheimer biomarkers", min_score=0.5)
    assert q.min_score == 0.5


def test_query_input_min_score_negative():
    """Negative min_score is rejected."""
    with pytest.raises(ValidationError):
        QueryInput(question="Alzheimer biomarkers", min_score=-0.1)


# ── RetrievedChunk tests ──────────────────────────────────────────

@pytest.fixture
def valid_chunk() -> dict:
    """Valid RetrievedChunk dict — baseline."""
    return {
        "chunk_id": "37123456_chunk_0",
        "pmid":     "37123456",
        "text":     "Plasma pTau217 shows 96% sensitivity for early "
                    "Alzheimer detection in cognitively normal subjects.",
        "title":    "Blood biomarkers for Alzheimer detection",
        "score":    0.87,
        "year":     2024,
        "keywords": ["Alzheimer", "pTau217"],
        "source":   "pubmed",
    }


def test_retrieved_chunk_valid(valid_chunk):
    """Valid chunk passes."""
    chunk = RetrievedChunk(**valid_chunk)
    assert chunk.pmid == "37123456"
    assert chunk.score == 0.87


def test_retrieved_chunk_short_text_rejected(valid_chunk):
    """Text under 5 words is rejected."""
    valid_chunk["text"] = "Too short"
    with pytest.raises(ValidationError):
        RetrievedChunk(**valid_chunk)


def test_retrieved_chunk_score_valid(valid_chunk):
    """Score between 0 and 1 passes."""
    valid_chunk["score"] = 0.95
    chunk = RetrievedChunk(**valid_chunk)
    assert chunk.score == 0.95


def test_retrieved_chunk_to_context_string(valid_chunk):
    """to_context_string includes PMID and text."""
    chunk  = RetrievedChunk(**valid_chunk)
    ctx    = chunk.to_context_string()
    assert "37123456" in ctx
    assert "pTau217" in ctx
    assert "Score" in ctx


# ── QueryResult tests ─────────────────────────────────────────────

@pytest.fixture
def valid_result(valid_chunk) -> QueryResult:
    """Valid QueryResult with one chunk."""
    chunk = RetrievedChunk(**valid_chunk)
    return QueryResult(
        question       = "What biomarkers predict Alzheimer's?",
        chunks         = [chunk],
        total_returned = 1,
        query_time_ms  = 150.0,
        model_used     = "pritamdeka/S-PubMedBert-MS-MARCO",
    )


def test_query_result_has_results(valid_result):
    """has_results is True when chunks present."""
    assert valid_result.has_results is True


def test_query_result_empty():
    """has_results is False when no chunks."""
    result = QueryResult(
        question       = "test question here",
        chunks         = [],
        total_returned = 0,
        query_time_ms  = 50.0,
        model_used     = "test-model",
    )
    assert result.has_results is False


def test_query_result_top_score(valid_result):
    """top_score returns score of first chunk."""
    assert valid_result.top_score == 0.87


def test_query_result_top_score_empty():
    """top_score returns 0.0 when no chunks."""
    result = QueryResult(
        question       = "test question here",
        chunks         = [],
        total_returned = 0,
        query_time_ms  = 50.0,
        model_used     = "test-model",
    )
    assert result.top_score == 0.0


def test_query_result_context_block(valid_result):
    """to_context_block includes chunk text."""
    ctx = valid_result.to_context_block()
    assert "pTau217" in ctx
    assert "Source 1" in ctx


def test_query_result_no_results_context():
    """Empty result returns no results message."""
    result = QueryResult(
        question       = "test question here",
        chunks         = [],
        total_returned = 0,
        query_time_ms  = 50.0,
        model_used     = "test-model",
    )
    ctx = result.to_context_block()
    assert "No relevant" in ctx
    