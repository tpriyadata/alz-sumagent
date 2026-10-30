"""
tests/test_preprocessing.py
=============================
ALZDETECT-AI — Unit tests for preprocessing/cleaner.py + chunker.py

WHAT:   Tests cleaning functions and chunking logic.
WHY:    Cleaner fixes 3 known problems — each fix must be tested.
        Chunker splits abstracts — overlap and edge cases must work.
WHO:    Run after any change to cleaner.py or chunker.py
WHEN:   Before every commit that touches preprocessing/

Run:
    pytest tests/test_preprocessing.py -v
"""

import pytest
from pydantic import ValidationError
from preprocessing.cleaner import (
    strip_html,
    infer_year,
    extract_keywords,
    CleanedPaper,
)
from preprocessing.chunker import (
    split_into_chunks,
    PaperChunk,
)


# ── strip_html tests ──────────────────────────────────────────────

def test_strip_html_italic():
    """<i>APOE</i> → 'APOE', was_html=True"""
    result, was_html = strip_html("<i>APOE</i> gene")
    assert "APOE" in result
    assert "<i>" not in result
    assert was_html is True


def test_strip_html_bold():
    """<b>results</b> → 'results', was_html=True"""
    result, was_html = strip_html("<b>results</b> show significance")
    assert "results" in result
    assert "<b>" not in result
    assert was_html is True


def test_strip_html_clean_text():
    """Plain text — no HTML — unchanged, was_html=False"""
    result, was_html = strip_html("normal text without tags")
    assert result == "normal text without tags"
    assert was_html is False


def test_strip_html_multiple_tags():
    """Multiple tags all stripped."""
    result, was_html = strip_html("<i>tau</i> and <b>amyloid</b> pathology")
    assert "tau" in result
    assert "amyloid" in result
    assert "<" not in result
    assert was_html is True


def test_strip_html_whitespace_collapsed():
    """Multiple spaces after stripping are collapsed to one."""
    result, _ = strip_html("word1  <i>tag</i>  word2")
    assert "  " not in result


# ── infer_year tests ──────────────────────────────────────────────

def test_infer_year_valid():
    """Valid ISO timestamp → correct year extracted."""
    year, inferred = infer_year("2026-04-30T10:00:00")
    assert year == 2026
    assert inferred is True


def test_infer_year_empty():
    """Empty string → None, not inferred."""
    year, inferred = infer_year("")
    assert year is None
    assert inferred is False


def test_infer_year_invalid():
    """Garbage string → None, not inferred."""
    year, inferred = infer_year("not-a-date")
    assert year is None
    assert inferred is False


def test_infer_year_2025():
    """2025 timestamp → year 2025."""
    year, inferred = infer_year("2025-01-15T08:30:00")
    assert year == 2025
    assert inferred is True


# ── extract_keywords tests ────────────────────────────────────────

def test_extract_keywords_returns_list():
    """Returns a list of strings."""
    abstract = (
        "Alzheimer disease biomarkers pTau217 blood plasma "
        "early detection clinical study results show sensitivity"
    )
    keywords, extracted = extract_keywords(abstract, "Alzheimer biomarkers")
    assert isinstance(keywords, list)
    assert extracted is True


def test_extract_keywords_max_limit():
    """Returns at most max_kw keywords."""
    abstract = " ".join([f"word{i}" for i in range(100)])
    keywords, _ = extract_keywords(abstract, "title", max_kw=5)
    assert len(keywords) <= 5


def test_extract_keywords_not_empty():
    """Non-empty abstract returns at least one keyword."""
    abstract = "Alzheimer disease biomarker detection study plasma tau"
    keywords, _ = extract_keywords(abstract, "title")
    assert len(keywords) > 0


def test_extract_keywords_excludes_stopwords():
    """Common stop words are excluded."""
    abstract = "the study was conducted with the patients in the hospital"
    keywords, _ = extract_keywords(abstract, "title")
    assert "the" not in keywords
    assert "was" not in keywords


# ── CleanedPaper validation tests ────────────────────────────────

@pytest.fixture
def valid_cleaned() -> dict:
    """Valid CleanedPaper dict — baseline."""
    return {
        "pmid":              "37123456",
        "title":             "Blood biomarkers for Alzheimer detection",
        "abstract":          "This study examines plasma pTau217 levels "
                             "in 500 subjects showing 96% sensitivity for "
                             "detecting preclinical Alzheimer pathology in "
                             "cognitively normal older adults.",
        "authors":           ["Smith, John"],
        "year":              2024,
        "keywords":          ["Alzheimer", "pTau217"],
        "source_query":      "plasma_ptau217_diagnosis",
        "year_inferred":     False,
        "keywords_extracted":False,
        "html_stripped":     False,
    }


def test_cleaned_paper_valid(valid_cleaned):
    """Valid CleanedPaper passes."""
    paper = CleanedPaper(**valid_cleaned)
    assert paper.pmid == "37123456"
    assert paper.year == 2024


def test_cleaned_paper_short_abstract_rejected(valid_cleaned):
    """Abstract under 20 words is rejected."""
    valid_cleaned["abstract"] = "Too short abstract here."
    with pytest.raises(ValidationError):
        CleanedPaper(**valid_cleaned)


def test_cleaned_paper_audit_flags(valid_cleaned):
    """Audit flags are stored correctly."""
    valid_cleaned["year_inferred"]      = True
    valid_cleaned["keywords_extracted"] = True
    valid_cleaned["html_stripped"]      = True
    paper = CleanedPaper(**valid_cleaned)
    assert paper.year_inferred is True
    assert paper.keywords_extracted is True
    assert paper.html_stripped is True


# ── split_into_chunks tests ───────────────────────────────────────

def test_split_short_abstract():
    """Abstract shorter than chunk_size → single chunk."""
    text   = "Short abstract with just a few words about Alzheimer."
    chunks = split_into_chunks(text, chunk_size=100, overlap=10)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_split_long_abstract():
    """Abstract longer than chunk_size → multiple chunks."""
    words  = ["word"] * 200
    text   = " ".join(words)
    chunks = split_into_chunks(text, chunk_size=50, overlap=10)
    assert len(chunks) > 1


def test_split_overlap():
    """Chunks share overlapping words."""
    words  = [f"word{i}" for i in range(100)]
    text   = " ".join(words)
    chunks = split_into_chunks(text, chunk_size=20, overlap=5)
    # Last words of chunk 0 should appear in start of chunk 1
    last_words_chunk0  = set(chunks[0].split()[-5:])
    first_words_chunk1 = set(chunks[1].split()[:5])
    assert len(last_words_chunk0 & first_words_chunk1) > 0


def test_split_no_empty_chunks():
    """No empty chunks produced."""
    text   = " ".join([f"word{i}" for i in range(50)])
    chunks = split_into_chunks(text, chunk_size=10, overlap=2)
    assert all(len(c.strip()) > 0 for c in chunks)


# ── PaperChunk validation tests ───────────────────────────────────

@pytest.fixture
def valid_chunk() -> dict:
    """Valid PaperChunk dict — baseline."""
    return {
        "chunk_id":    "37123456_chunk_0",
        "pmid":        "37123456",
        "chunk_idx":   0,
        "text":        "This study examines plasma pTau217 as an early "
                       "biomarker for Alzheimer disease detection.",
        "title":       "Blood biomarkers for Alzheimer detection",
        "year":        2024,
        "keywords":    ["Alzheimer", "pTau217"],
        "source":      "pubmed",
        "source_query":"plasma_ptau217_diagnosis",
    }


def test_paper_chunk_valid(valid_chunk):
    """Valid PaperChunk passes."""
    chunk = PaperChunk(**valid_chunk)
    assert chunk.chunk_id == "37123456_chunk_0"
    assert chunk.pmid == "37123456"


def test_paper_chunk_bad_id_rejected(valid_chunk):
    """chunk_id without '_chunk_' is rejected."""
    valid_chunk["chunk_id"] = "bad-format-id"
    with pytest.raises(ValidationError) as exc:
        PaperChunk(**valid_chunk)
    assert "_chunk_" in str(exc.value)


def test_paper_chunk_short_text_rejected(valid_chunk):
    """Text under 5 words is rejected."""
    valid_chunk["text"] = "Too short"
    with pytest.raises(ValidationError):
        PaperChunk(**valid_chunk)


def test_paper_chunk_word_count(valid_chunk):
    """word_count is auto-computed correctly."""
    chunk = PaperChunk(**valid_chunk)
    assert chunk.word_count == len(valid_chunk["text"].split())


def test_paper_chunk_to_pinecone_record(valid_chunk):
    """to_pinecone_record returns correct structure."""
    chunk  = PaperChunk(**valid_chunk)
    record = chunk.to_pinecone_record()
    assert record["id"] == "37123456_chunk_0"
    assert "metadata" in record
    assert record["metadata"]["pmid"] == "37123456"
    assert record["metadata"]["source"] == "pubmed"