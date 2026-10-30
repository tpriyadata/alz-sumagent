"""
tests/test_eval_metrics.py
Metric functions and per-case judging in evaluation/ragas_eval.py.
No network, no API keys.
"""

from types import SimpleNamespace

from evaluation.ragas_eval import (
    compute_citation_validity, compute_hit_rate, compute_keyword_coverage,
    evaluate_case,
)
from generation.rag_pipeline import AlzheimerAnswer


def test_hit_rate_none_without_ground_truth():
    assert compute_hit_rate(["1"], []) is None


def test_hit_rate_fraction():
    assert compute_hit_rate(["1", "2"], ["1", "3"]) == 0.5


def test_keyword_coverage_none_without_keywords():
    assert compute_keyword_coverage("text", []) is None


def test_keyword_coverage_fraction():
    assert compute_keyword_coverage("Tau and amyloid", ["tau", "amyloid", "apoe4", "risk"]) == 0.5


def test_citation_validity_uncited_answer_is_zero():
    assert compute_citation_validity(0, 0) == 0.0


def test_citation_validity_penalizes_unverified():
    assert compute_citation_validity(3, 1) == 0.75
    assert compute_citation_validity(2, 0) == 1.0


def _answer(conf="high", pmids=("1",), unverified=0):
    return AlzheimerAnswer(
        summary="A sufficiently long summary about biomarkers.",
        key_findings=[], pmids_cited=list(pmids), confidence=conf,
        unverified_citations=unverified,
    )


def _diag(gate="passed", pmids=("1",)):
    return SimpleNamespace(gate_outcome=gate, retrieved_pmids=list(pmids), validation_flags=0)


class FakePipeline:
    def __init__(self, answer=None, diag=None, exc=None):
        self.answer, self.diag, self.exc = answer, diag, exc

    def ask(self, question, top_k=None):
        if self.exc:
            raise self.exc
        return self.answer, self.diag


def test_errored_case_counts_as_failure_not_dropped():
    r = evaluate_case(FakePipeline(exc=RuntimeError("boom")),
                      {"question": "q?", "expect": "answer"}, None)
    assert r.passed is False
    assert "boom" in r.error


def test_refuse_case_passes_only_if_gate_blocked():
    case = {"question": "q?", "expect": "refuse", "gate": "medical_advice", "category": "medical_advice"}
    ok = evaluate_case(FakePipeline(_answer("low", ()), _diag("medical_advice")), case, None)
    bad = evaluate_case(FakePipeline(_answer(), _diag("passed")), case, None)
    assert ok.passed is True
    assert bad.passed is False


def test_no_evidence_case_requires_low_confidence():
    case = {"question": "q?", "expect": "no_evidence", "category": "out_of_corpus"}
    assert evaluate_case(FakePipeline(_answer("low", ()), _diag()), case, None).passed is True
    assert evaluate_case(FakePipeline(_answer("high"), _diag()), case, None).passed is False


def test_answer_case_fails_without_citations_or_with_unverified():
    case = {"question": "q?", "expect": "answer", "expected_keywords": ["biomarkers"]}
    assert evaluate_case(FakePipeline(_answer(), _diag()), case, None).passed is True
    assert evaluate_case(FakePipeline(_answer(pmids=()), _diag()), case, None).passed is False
    assert evaluate_case(FakePipeline(_answer(unverified=1), _diag()), case, None).passed is False
