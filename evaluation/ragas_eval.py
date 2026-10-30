"""
evaluation/ragas_eval.py

Live evaluation against the real pipeline (needs API keys + Pinecone).
    python -c "from evaluation.ragas_eval import run_evaluation; run_evaluation()"

IMPORTANT - what these metrics are and are not:
    * hit_rate          retrieval recall against hand-labelled PMIDs
    * keyword_coverage  crude answer-relevancy proxy
    * citation_validity STRUCTURAL check: of the PMIDs the model cited, what
                        fraction were in the retrieved set. It does NOT verify
                        that a claim is supported by the source text, so it is
                        not RAG faithfulness. Real faithfulness needs a
                        claim-level check (e.g. an LLM judge calibrated against
                        human labels).
    * pass rates        per category, including adversarial cases

Errors are counted as failures, never silently dropped.
"""

import json
import time
from pathlib import Path
from typing import Optional

from loguru import logger
from pydantic import BaseModel, Field

from configs.settings import get_settings
from evaluation.adversarial_cases import ADVERSARIAL_CASES
from generation.rag_pipeline import RAGPipeline


# ── Gold standard test set (happy path) ───────────────────────────

GOLD_STANDARD = [
    {
        "question": "What blood biomarkers can detect Alzheimer's disease early?",
        "expected_pmids": ["35446894", "35858611", "41622474", "33019230"],
        "expected_keywords": ["ptau", "amyloid", "biomarker", "blood", "plasma"],
    },
    {
        "question": "What is the role of APOE4 in Alzheimer's disease risk?",
        "expected_pmids": ["37227619", "36614219", "39251089", "41596649"],
        "expected_keywords": ["apoe4", "risk", "genetic", "amyloid"],
    },
    {
        "question": "How does pTau217 predict Alzheimer's disease progression?",
        "expected_pmids": ["40768095", "39437657", "41728301", "39575854"],
        "expected_keywords": ["ptau217", "progression", "biomarker", "tau"],
    },
    {
        "question": "What is the relationship between amyloid beta and cognitive decline?",
        "expected_pmids": ["40347524", "33444630", "25661087", "34617016"],
        "expected_keywords": ["amyloid", "cognitive", "decline", "beta"],
    },
    {
        "question": "What treatments are currently in clinical trials for Alzheimer's disease?",
        "expected_pmids": ["38364913", "26721364", "35932841", "39316339"],
        "expected_keywords": ["lecanemab", "donanemab", "trial", "treatment"],
    },
    {
        "question": "How does neuroinflammation contribute to Alzheimer's disease?",
        "expected_pmids": ["37227619", "41596649"],
        "expected_keywords": ["neuroinflammation", "microglia", "inflammation"],
    },
    {
        "question": "What MMSE scores do MCI patients show in clinical studies?",
        "expected_pmids": [],
        "expected_keywords": ["mmse", "mci", "cognitive", "score"],
    },
    {
        "question": "How does APOE4 affect amyloid beta accumulation in the brain?",
        "expected_pmids": ["37227619", "36614219", "39251089"],
        "expected_keywords": ["apoe4", "amyloid", "accumulation", "brain"],
    },
    {
        "question": "What are the differences between CSF and blood biomarkers for AD diagnosis?",
        "expected_pmids": ["35858611", "40768095", "33019230"],
        "expected_keywords": ["csf", "blood", "biomarker", "diagnosis"],
    },
    {
        "question": "How does tau pathology progress in Alzheimer's disease?",
        "expected_pmids": ["40768095", "39437657"],
        "expected_keywords": ["tau", "pathology", "progression", "neurofibrillary"],
    },
]

for _case in GOLD_STANDARD:
    _case.setdefault("category", "answer")
    _case.setdefault("expect", "answer")


def all_cases() -> list[dict]:
    """Happy-path gold set plus adversarial cases."""
    return GOLD_STANDARD + ADVERSARIAL_CASES


# ── Result models ─────────────────────────────────────────────────

class QuestionResult(BaseModel):
    """Result for a single evaluation question."""
    question:          str
    category:          str
    expect:            str
    passed:            bool
    answer_summary:    str       = ""
    retrieved_pmids:   list[str] = []
    expected_pmids:    list[str] = []
    hit_rate:          Optional[float] = Field(default=None, ge=0.0, le=1.0)
    keyword_coverage:  Optional[float] = Field(default=None, ge=0.0, le=1.0)
    citation_validity: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    confidence:        str       = ""
    gate_outcome:      str       = ""
    validation_flags:  int       = 0
    response_time_ms:  float     = 0.0
    error:             Optional[str] = None


class EvaluationReport(BaseModel):
    """Full evaluation report."""
    prompt_version:           str
    total_questions:          int
    errored_questions:        int
    overall_pass_rate:        float
    pass_rate_by_category:    dict[str, float]
    avg_hit_rate:             Optional[float]
    avg_keyword_coverage:     Optional[float]
    avg_citation_validity:    Optional[float]
    avg_response_time_ms:     float
    results:                  list[QuestionResult]

    def log_summary(self) -> None:
        def f(v: Optional[float]) -> str:
            return "n/a" if v is None else f"{v:.2f}"

        logger.info("=" * 60)
        logger.info("[EVAL] Evaluation complete")
        logger.info(f"  Prompt version        : {self.prompt_version}")
        logger.info(f"  Questions             : {self.total_questions} "
                    f"({self.errored_questions} errored, counted as failures)")
        logger.info(f"  Overall pass rate     : {self.overall_pass_rate:.2f}")
        for cat, rate in sorted(self.pass_rate_by_category.items()):
            logger.info(f"    {cat:<18}: {rate:.2f}")
        logger.info(f"  Avg hit rate          : {f(self.avg_hit_rate)}")
        logger.info(f"  Avg keyword coverage  : {f(self.avg_keyword_coverage)}")
        logger.info(f"  Avg citation validity : {f(self.avg_citation_validity)}  (structural, not faithfulness)")
        logger.info(f"  Avg response time     : {self.avg_response_time_ms:.0f}ms")
        logger.info("=" * 60)

    def to_json(self, path: str = "evaluation/report.json") -> None:
        Path(path).write_text(
            json.dumps(self.model_dump(), indent=2),
            encoding="utf-8"
        )
        logger.info(f"[EVAL] Report saved to {path}")


# ── Metric functions ──────────────────────────────────────────────

def compute_hit_rate(
    retrieved_pmids: list[str],
    expected_pmids:  list[str],
) -> Optional[float]:
    """
    Fraction of expected PMIDs that were retrieved.
    Returns None when there is no ground truth (not a free 1.0).
    """
    if not expected_pmids:
        return None
    hits = sum(1 for p in expected_pmids if p in retrieved_pmids)
    return round(hits / len(expected_pmids), 3)


def compute_keyword_coverage(
    answer_text:       str,
    expected_keywords: list[str],
) -> Optional[float]:
    """Fraction of expected keywords present in the answer. None if no keywords."""
    if not expected_keywords:
        return None
    answer_lower = answer_text.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return round(hits / len(expected_keywords), 3)


def compute_citation_validity(
    verified_citations:   int,
    unverified_citations: int,
) -> float:
    """
    verified / (verified + unverified) citations.

    An answer with NO citations scores 0.0 - an uncited answer must not look
    perfect. Structural check only; see the module docstring.
    """
    total = verified_citations + unverified_citations
    if total == 0:
        return 0.0
    return round(verified_citations / total, 3)


# ── Per-case evaluation ───────────────────────────────────────────

def evaluate_case(pipeline: RAGPipeline, case: dict, top_k: Optional[int]) -> QuestionResult:
    """Run one case and judge it against its `expect` behaviour."""
    question = case["question"]
    expect   = case.get("expect", "answer")
    category = case.get("category", "answer")
    base = dict(question=question, category=category, expect=expect,
                expected_pmids=case.get("expected_pmids", []))

    start = time.time()
    try:
        answer, diagnostic = pipeline.ask(question=question, top_k=top_k)
    except Exception as e:
        logger.exception(f"[EVAL] '{question[:50]}' errored")
        return QuestionResult(passed=False, error=f"{type(e).__name__}: {e}", **base)

    response_ms = round((time.time() - start) * 1000, 1)
    cited       = answer.pmids_cited
    common = dict(
        answer_summary   = answer.summary[:200],
        retrieved_pmids  = diagnostic.retrieved_pmids,
        confidence       = answer.confidence,
        gate_outcome     = diagnostic.gate_outcome,
        validation_flags = diagnostic.validation_flags,
        response_time_ms = response_ms,
    )

    if expect == "refuse":
        passed = diagnostic.gate_outcome == case.get("gate", diagnostic.gate_outcome) \
                 and diagnostic.gate_outcome != "passed"
        return QuestionResult(passed=passed, **common, **base)

    if expect == "no_evidence":
        return QuestionResult(passed=(answer.confidence == "low"), **common, **base)

    # expect == "answer"
    hit = compute_hit_rate(diagnostic.retrieved_pmids, case.get("expected_pmids", []))
    kw  = compute_keyword_coverage(
        answer.summary + " " + " ".join(answer.key_findings),
        case.get("expected_keywords", []),
    )
    validity = compute_citation_validity(len(cited), answer.unverified_citations)
    passed = (
        diagnostic.gate_outcome == "passed"
        and len(cited) > 0
        and answer.unverified_citations == 0
    )
    return QuestionResult(
        passed=passed, hit_rate=hit, keyword_coverage=kw,
        citation_validity=validity, **common, **base,
    )


def _mean(values: list[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


# ── Core evaluation runner ────────────────────────────────────────

def run_evaluation(
    questions:   Optional[list[dict]] = None,
    top_k:       Optional[int] = None,
    save_report: bool = True,
) -> EvaluationReport:
    """
    Run the full evaluation (gold + adversarial) against the live pipeline.

    Usage:
        python -c "from evaluation.ragas_eval import run_evaluation; run_evaluation()"
    """
    from generation.prompts import PROMPT_VERSION

    logger.info("[EVAL] Starting evaluation")
    settings = get_settings()
    top_k    = top_k or settings.top_k_results
    pipeline = RAGPipeline()
    test_set = questions or all_cases()

    results: list[QuestionResult] = []
    for i, case in enumerate(test_set, start=1):
        logger.info(f"[EVAL] Q{i}/{len(test_set)} [{case.get('category', 'answer')}]: "
                    f"{case['question'][:60]}...")
        result = evaluate_case(pipeline, case, top_k)
        results.append(result)
        logger.info(f"  passed={result.passed} | conf={result.confidence} | "
                    f"gate={result.gate_outcome} | error={result.error}")
        time.sleep(1)  # rate limit protection

    by_cat: dict[str, list[bool]] = {}
    for r in results:
        by_cat.setdefault(r.category, []).append(r.passed)

    report = EvaluationReport(
        prompt_version        = PROMPT_VERSION,
        total_questions       = len(results),
        errored_questions     = sum(1 for r in results if r.error),
        overall_pass_rate     = round(sum(r.passed for r in results) / len(results), 3),
        pass_rate_by_category = {c: round(sum(v) / len(v), 3) for c, v in by_cat.items()},
        avg_hit_rate          = _mean([r.hit_rate for r in results]),
        avg_keyword_coverage  = _mean([r.keyword_coverage for r in results]),
        avg_citation_validity = _mean([r.citation_validity for r in results]),
        avg_response_time_ms  = round(
            sum(r.response_time_ms for r in results) / len(results), 1
        ),
        results               = results,
    )
    report.log_summary()
    if save_report:
        report.to_json()
    return report


# ── RE probe ──────────────────────────────────────────────────────

def diagnose_eval() -> None:
    """
    Quick probe - 2 gold questions + 2 adversarial cases to verify setup.

    Usage:
        python -c "from evaluation.ragas_eval import diagnose_eval; diagnose_eval()"
    """
    logger.info("[RE-EVAL] Running quick evaluation probe")
    report = run_evaluation(
        questions   = GOLD_STANDARD[:2] + ADVERSARIAL_CASES[:2],
        save_report = False,
    )
    logger.info(f"[RE-EVAL] Overall pass rate: {report.overall_pass_rate:.2f}")
    logger.info("[RE-EVAL] Evaluation working correctly")
