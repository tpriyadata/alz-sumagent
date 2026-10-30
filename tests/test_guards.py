"""
tests/test_guards.py
Deterministic entry gate + citation helpers. No network, no API keys.
"""

import pytest

from evaluation.adversarial_cases import ADVERSARIAL_CASES
from evaluation.ragas_eval import GOLD_STANDARD
from generation.guards import check_entry, extract_pmids, find_unsupported_pmids

REFUSE_CASES = [c for c in ADVERSARIAL_CASES if c["expect"] == "refuse"]
GUARD_CASES  = [c for c in ADVERSARIAL_CASES if c["category"] == "gate_guard"]


@pytest.mark.parametrize("case", REFUSE_CASES, ids=lambda c: c["question"][:40])
def test_entry_gate_blocks_adversarial(case):
    decision = check_entry(case["question"])
    assert decision.allowed is False
    assert decision.category == case["gate"]
    assert decision.message


@pytest.mark.parametrize("case", GUARD_CASES, ids=lambda c: c["question"][:40])
def test_entry_gate_allows_legitimate_research_questions(case):
    """False-positive guard: look-alike wording must still pass."""
    assert check_entry(case["question"]).allowed is True


@pytest.mark.parametrize("case", GOLD_STANDARD, ids=lambda c: c["question"][:40])
def test_entry_gate_allows_every_gold_question(case):
    assert check_entry(case["question"]).allowed is True


def test_entry_gate_handles_empty_input():
    assert check_entry("").allowed is True   # length is QueryInput's job, not the gate's


def test_extract_pmids():
    text = "Found in PMID: 37123456 and (PMID 37234567); also PMID:12345678."
    assert extract_pmids(text) == ["37123456", "37234567", "12345678"]


def test_extract_pmids_none():
    assert extract_pmids("no citations here") == []


def test_find_unsupported_pmids():
    text = "See PMID: 11111111 and PMID: 99999999."
    assert find_unsupported_pmids(text, ["11111111"]) == ["99999999"]
    assert find_unsupported_pmids(text, ["11111111", "99999999"]) == []
