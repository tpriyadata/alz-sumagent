"""
tests/test_generation.py
=========================
ALZDETECT-AI — Unit tests for generation/rag_pipeline.py

WHAT:   Tests AlzheimerAnswer Pydantic model/ validation
        and answer parsing logic.
WHY:    Claude output is non-deterministic — the parser
        must handle good JSON, bad JSON, and missing fields.
WHO:    Run after any change to rag_pipeline.py
WHEN:   Before every commit that touches generation/

Run:
    pytest tests/test_generation.py -v
"""

import pytest
from pydantic import ValidationError
from generation.rag_pipeline import AlzheimerAnswer, RAGPipeline


# ── AlzheimerAnswer validation tests ─────────────────────────────

@pytest.fixture
def valid_answer() -> dict:
    """Valid AlzheimerAnswer dict — baseline."""
    return {
        "summary":      "Plasma pTau217 is a strong early biomarker "
                        "for Alzheimer's disease with 96% sensitivity.",
        "key_findings": [
            "pTau217 detects AD 10 years before symptoms (PMID: 37123456)",
            "GFAP correlates with amyloid burden (PMID: 37234567)",
        ],
        "pmids_cited":  ["37123456", "37234567"],
        "confidence":   "high",
        "limitations":  "Studies limited to older adults over 65.",
    }


def test_answer_valid(valid_answer):
    """Valid answer passes."""
    answer = AlzheimerAnswer(**valid_answer)
    assert answer.confidence == "high"
    assert len(answer.key_findings) == 2


def test_answer_empty_summary_rejected(valid_answer):
    """Empty summary is rejected."""
    valid_answer["summary"] = ""
    with pytest.raises(ValidationError):
        AlzheimerAnswer(**valid_answer)


def test_answer_short_summary_rejected(valid_answer):
    """Summary under 10 chars is rejected."""
    valid_answer["summary"] = "Too short"
    with pytest.raises(ValidationError):
        AlzheimerAnswer(**valid_answer)


def test_answer_confidence_high(valid_answer):
    """confidence='high' is valid."""
    valid_answer["confidence"] = "high"
    answer = AlzheimerAnswer(**valid_answer)
    assert answer.confidence == "high"


def test_answer_confidence_medium(valid_answer):
    """confidence='medium' is valid."""
    valid_answer["confidence"] = "medium"
    answer = AlzheimerAnswer(**valid_answer)
    assert answer.confidence == "medium"


def test_answer_confidence_low(valid_answer):
    """confidence='low' is valid."""
    valid_answer["confidence"] = "low"
    answer = AlzheimerAnswer(**valid_answer)
    assert answer.confidence == "low"


def test_answer_confidence_invalid_defaults_medium(valid_answer):
    """
    Invalid confidence defaults to 'medium' — not a crash.
    Claude might return 'moderate' or 'uncertain'.
    """
    valid_answer["confidence"] = "maybe"
    answer = AlzheimerAnswer(**valid_answer)
    assert answer.confidence == "medium"


def test_answer_confidence_uppercase_normalized(valid_answer):
    """'HIGH' is normalized to 'high'."""
    valid_answer["confidence"] = "HIGH"
    answer = AlzheimerAnswer(**valid_answer)
    assert answer.confidence == "high"


def test_answer_empty_key_findings(valid_answer):
    """Empty key_findings list is allowed."""
    valid_answer["key_findings"] = []
    answer = AlzheimerAnswer(**valid_answer)
    assert answer.key_findings == []


def test_answer_empty_pmids(valid_answer):
    """Empty pmids_cited is allowed."""
    valid_answer["pmids_cited"] = []
    answer = AlzheimerAnswer(**valid_answer)
    assert answer.pmids_cited == []


def test_answer_no_limitations(valid_answer):
    """limitations=None is allowed."""
    valid_answer["limitations"] = None
    answer = AlzheimerAnswer(**valid_answer)
    assert answer.limitations is None


def test_answer_has_disclaimer(valid_answer):
    """Disclaimer is always present."""
    answer = AlzheimerAnswer(**valid_answer)
    assert "research purposes only" in answer.disclaimer


def test_answer_to_display(valid_answer):
    """to_display returns formatted string."""
    answer  = AlzheimerAnswer(**valid_answer)
    display = answer.to_display()
    assert "Summary" in display
    assert "Key Findings" in display
    assert "37123456" in display
    assert "high" in display


# ── _parse_answer tests ───────────────────────────────────────────

@pytest.fixture
def pipeline():
    """
    RAGPipeline instance for testing _parse_answer.
    Built with __new__ so no API keys, Pinecone or embedding model are needed:
    _parse_answer uses no instance state.
    """
    return RAGPipeline.__new__(RAGPipeline)


def test_parse_valid_json(pipeline):
    """Valid Claude JSON → AlzheimerAnswer."""
    raw = '''{
        "summary": "pTau217 is a strong biomarker for early Alzheimer detection.",
        "key_findings": ["pTau217 shows 96% sensitivity (PMID: 37123456)"],
        "pmids_cited": ["37123456"],
        "confidence": "high",
        "limitations": null
    }'''
    answer = pipeline._parse_answer(raw, retrieved_pmids=["37123456"])
    assert answer.confidence == "high"
    assert "37123456" in answer.pmids_cited


def test_parse_bad_json_fallback(pipeline):
    """
    Malformed JSON → fallback answer, no crash.
    Worst-case: Claude returns text instead of JSON.
    """
    raw    = "I cannot answer this question based on the provided papers."
    answer = pipeline._parse_answer(raw, retrieved_pmids=[])
    assert answer.confidence == "low"
    assert answer.summary != ""


def test_parse_empty_string_fallback(pipeline):
    """Empty string → fallback answer, no crash."""
    answer = pipeline._parse_answer("", retrieved_pmids=[])
    assert isinstance(answer, AlzheimerAnswer)


def test_parse_strips_markdown(pipeline):
    """Claude sometimes wraps JSON in markdown code blocks."""
    raw = '''```json
    {
        "summary": "pTau217 predicts Alzheimer disease progression reliably.",
        "key_findings": ["Finding 1 with citation"],
        "pmids_cited": ["37123456"],
        "confidence": "medium",
        "limitations": null
    }
```'''
    answer = pipeline._parse_answer(raw, retrieved_pmids=["37123456"])
    assert answer.confidence == "medium"


def test_parse_pmid_validation(pipeline):
    """
    PMIDs not in retrieved list are filtered out.
    Worst-case: Claude hallucinates a PMID.
    """
    raw = '''{
        "summary": "This is a valid summary about Alzheimer disease research.",
        "key_findings": ["Real finding (PMID: 37123456)"],
        "pmids_cited": ["37123456", "99999999"],
        "confidence": "high",
        "limitations": null
    }'''
    # Only 37123456 was actually retrieved — 99999999 is hallucinated
    answer = pipeline._parse_answer(
        raw,
        retrieved_pmids=["37123456"]
    )
    assert "37123456" in answer.pmids_cited
    assert "99999999" not in answer.pmids_cited


def test_parse_json_with_extra_fields(pipeline):
    """Extra fields in Claude JSON are ignored gracefully."""
    raw = '''{
        "summary": "Valid summary about Alzheimer biomarker research findings.",
        "key_findings": ["Finding with evidence"],
        "pmids_cited": ["37123456"],
        "confidence": "high",
        "limitations": null,
        "extra_field": "this should be ignored"
    }'''
    answer = pipeline._parse_answer(raw, retrieved_pmids=["37123456"])
    assert answer.confidence == "high"
