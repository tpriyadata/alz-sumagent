"""
tests/test_ingestion.py
========================
ALZDETECT-AI — Unit tests for ingestion/pubmed_fetch.py

WHAT:   Tests PubMedPaper Pydantic model validation.
WHY:    Pydantic is our main defence layer — every validator
        must be tested to confirm it catches what it should.
WHO:    Run by developer after any change to pubmed_fetch.py
WHEN:   Before every commit that touches ingestion/

Run:
    pytest tests/test_ingestion.py -v
"""

import pytest
from pydantic import ValidationError
from ingestion.pubmed_fetch import PubMedPaper


# ── Fixtures — reusable valid paper dict ──────────────────────────

@pytest.fixture
def valid_paper() -> dict:
    """A complete valid paper — baseline for all tests."""
    return {
        "pmid":         "37123456",
        "title":        "Blood biomarkers for Alzheimer's disease detection",
        "abstract":     "This study examines plasma pTau217 as an early "
                        "biomarker for Alzheimer's disease in 500 subjects. "
                        "Results show 96% sensitivity and 94% specificity "
                        "for detecting preclinical Alzheimer's pathology.",
        "authors":      ["Smith, John", "Jones, Mary"],
        "year":         2024,
        "keywords":     ["Alzheimer", "biomarker", "pTau217"],
        "journal":      "Nature Medicine",
        "source_query": "plasma_ptau217_diagnosis",
    }


# ── PMID validation tests ─────────────────────────────────────────

def test_pmid_valid(valid_paper):
    """Valid numeric PMID passes."""
    paper = PubMedPaper(**valid_paper)
    assert paper.pmid == "37123456"


def test_pmid_string_numeric(valid_paper):
    """PMID as string integer passes."""
    valid_paper["pmid"] = "12345678"
    paper = PubMedPaper(**valid_paper)
    assert paper.pmid == "12345678"


def test_pmid_non_numeric_rejected(valid_paper):
    """Non-numeric PMID raises ValidationError."""
    valid_paper["pmid"] = "ABC123"
    with pytest.raises(ValidationError) as exc:
        PubMedPaper(**valid_paper)
    assert "not numeric" in str(exc.value)


def test_pmid_empty_rejected(valid_paper):
    """Empty PMID raises ValidationError."""
    valid_paper["pmid"] = ""
    with pytest.raises(ValidationError):
        PubMedPaper(**valid_paper)


# ── Year coercion tests ───────────────────────────────────────────

def test_year_as_int(valid_paper):
    """Year as int passes directly."""
    valid_paper["year"] = 2024
    paper = PubMedPaper(**valid_paper)
    assert paper.year == 2024


def test_year_as_string_coerced(valid_paper):
    """Year as string '2024' is coerced to int 2024."""
    valid_paper["year"] = "2024"
    paper = PubMedPaper(**valid_paper)
    assert paper.year == 2024


def test_year_none_allowed(valid_paper):
    """Missing year is allowed — 49 papers had this in Plan 1."""
    valid_paper["year"] = None
    paper = PubMedPaper(**valid_paper)
    assert paper.year is None


def test_year_out_of_range(valid_paper):
    """Year 1800 is suspicious — set to None."""
    valid_paper["year"] = 1800
    paper = PubMedPaper(**valid_paper)
    assert paper.year is None


def test_year_future_rejected(valid_paper):
    """Year 9999 is suspicious — set to None."""
    valid_paper["year"] = 9999
    paper = PubMedPaper(**valid_paper)
    assert paper.year is None


# ── Authors coercion tests ────────────────────────────────────────

def test_authors_as_list(valid_paper):
    """Authors as list passes directly."""
    valid_paper["authors"] = ["Smith, John", "Jones, Mary"]
    paper = PubMedPaper(**valid_paper)
    assert len(paper.authors) == 2


def test_authors_as_string_coerced(valid_paper):
    """
    Authors as comma-separated string coerced to list.
    This was the format in old project JSON.
    """
    valid_paper["authors"] = "Smith, John, Jones, Mary"
    paper = PubMedPaper(**valid_paper)
    assert isinstance(paper.authors, list)
    assert len(paper.authors) > 0


def test_authors_empty_list(valid_paper):
    """Empty authors list is allowed."""
    valid_paper["authors"] = []
    paper = PubMedPaper(**valid_paper)
    assert paper.authors == []


# ── Keywords coercion tests ───────────────────────────────────────

def test_keywords_as_list(valid_paper):
    """Keywords as list passes directly."""
    paper = PubMedPaper(**valid_paper)
    assert "Alzheimer" in paper.keywords


def test_keywords_as_string_coerced(valid_paper):
    """
    Keywords as comma-separated string coerced to list.
    Old project format was 'Alzheimer, biomarker, pTau217'.
    """
    valid_paper["keywords"] = "Alzheimer, biomarker, pTau217"
    paper = PubMedPaper(**valid_paper)
    assert isinstance(paper.keywords, list)
    assert len(paper.keywords) == 3


def test_keywords_empty_allowed(valid_paper):
    """Empty keywords allowed — 3,390 papers had none in Plan 1."""
    valid_paper["keywords"] = []
    paper = PubMedPaper(**valid_paper)
    assert paper.keywords == []


def test_keywords_deduplicated(valid_paper):
    """Duplicate keywords are removed."""
    valid_paper["keywords"] = ["Alzheimer", "Alzheimer", "biomarker"]
    paper = PubMedPaper(**valid_paper)
    assert len(paper.keywords) == 2


# ── Abstract validation tests ─────────────────────────────────────

def test_abstract_valid(valid_paper):
    """Valid abstract passes."""
    paper = PubMedPaper(**valid_paper)
    assert len(paper.abstract) > 0


def test_abstract_placeholder_rejected(valid_paper):
    """'[No abstract available]' is rejected."""
    valid_paper["abstract"] = "[No abstract available]"
    with pytest.raises(ValidationError) as exc:
        PubMedPaper(**valid_paper)
    assert "placeholder" in str(exc.value)


def test_abstract_na_rejected(valid_paper):
    """'N/A' abstract is rejected."""
    valid_paper["abstract"] = "N/A"
    with pytest.raises(ValidationError):
        PubMedPaper(**valid_paper)


def test_abstract_empty_rejected(valid_paper):
    """Empty abstract is rejected."""
    valid_paper["abstract"] = ""
    with pytest.raises(ValidationError):
        PubMedPaper(**valid_paper)


# ── source_query coercion tests ───────────────────────────────────

def test_source_query_valid(valid_paper):
    """Valid source_query passes."""
    paper = PubMedPaper(**valid_paper)
    assert paper.source_query == "plasma_ptau217_diagnosis"


def test_source_query_missing_defaults_unknown(valid_paper):
    """Missing source_query defaults to 'unknown'."""
    valid_paper["source_query"] = ""
    paper = PubMedPaper(**valid_paper)
    assert paper.source_query == "unknown"


def test_source_query_old_format(valid_paper):
    """
    Old project used 'source':'PubMed' not 'source_query'.
    Missing source_query defaults to 'unknown' — not a crash.
    """
    del valid_paper["source_query"]
    paper = PubMedPaper(**valid_paper)
    assert paper.source_query == "unknown"


# ── word_count property test ──────────────────────────────────────

def test_word_count(valid_paper):
    """word_count property returns correct count."""
    paper = PubMedPaper(**valid_paper)
    assert paper.word_count == len(valid_paper["abstract"].split())


# ── to_dict serialization test ────────────────────────────────────

def test_to_dict(valid_paper):
    """to_dict returns serializable dict with all fields."""
    paper = PubMedPaper(**valid_paper)
    d = paper.to_dict()
    assert isinstance(d, dict)
    assert d["pmid"] == "37123456"
    assert d["year"] == 2024
    assert isinstance(d["authors"], list)
    