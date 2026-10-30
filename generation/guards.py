"""
generation/guards.py

Deterministic gates that run BEFORE and AFTER the LLM.

WHY: Rules are fast, free, explainable and cannot be talked around.
     They run first. An LLM judge belongs only where a rule cannot
     express the criterion (e.g. "is this claim supported by the text?").

ENTRY GATE  -> check_entry():  blocks requests this system must not answer
               (personal medical advice, prompt-injection attempts).
EXIT HELPERS -> extract_pmids(), find_unsupported_pmids(): find citations
               in free text that do not belong to the retrieved sources.

No third-party imports other than pydantic, so these are trivially unit-testable.
"""

import re

from pydantic import BaseModel


# ── Entry gate ────────────────────────────────────────────────────

class GateDecision(BaseModel):
    allowed:  bool
    category: str = "passed"   # passed | prompt_injection | medical_advice
    message:  str = ""         # user-facing text when blocked


_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+|any\s+|the\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|rules?)",
    r"disregard\s+(all\s+|the\s+)?(system|previous|prior|above)\b",
    r"(reveal|show|print|repeat)\s+(me\s+)?(your|the)\s+(system\s+)?(prompt|instructions)",
    r"\byou\s+are\s+now\b",
    r"\bnew\s+instructions?\s*:",
    r"\bjailbreak\b",
]

_MEDICAL_ADVICE_PATTERNS = [
    r"\b(should|can|could|may)\s+i\s+(take|stop|start|increase|decrease|combine|use|skip)\b",
    r"\bwhat\s+(dose|dosage|amount)\b",
    r"\bhow\s+(much|many)\b.{0,40}\b(mg|milligrams?|tablets?|pills?)\b",
    r"\bdo\s+i\s+have\s+(alzheimer'?s?|dementia|ad)\b",
    r"\bdiagnose\s+(me|him|her|my)\b",
    r"\bam\s+i\s+(going\s+to|at\s+risk|likely\s+to)\b",
    r"\b(my|his|her)\s+(mother|mom|father|dad|wife|husband|spouse|parent|"
    r"grandmother|grandfather|brother|sister)\b.{0,80}\b(has|have|is|was|diagnosed|"
    r"prescribed|taking)\b",
]

_INJECTION_RE = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]
_MEDICAL_RE   = [re.compile(p, re.IGNORECASE) for p in _MEDICAL_ADVICE_PATTERNS]

INJECTION_MESSAGE = (
    "I can only summarize published Alzheimer's research. "
    "That request looks like an attempt to change my instructions, so I did not process it."
)
MEDICAL_ADVICE_MESSAGE = (
    "I summarize published research and can't give personal medical advice, "
    "diagnosis or dosing. Please talk to a clinician. If you'd like, ask what the "
    "research says about the topic in general terms."
)


def check_entry(question: str) -> GateDecision:
    """Deterministic entry gate. Fail closed on the patterns above."""
    text = question or ""
    for rx in _INJECTION_RE:
        if rx.search(text):
            return GateDecision(
                allowed=False, category="prompt_injection", message=INJECTION_MESSAGE
            )
    for rx in _MEDICAL_RE:
        if rx.search(text):
            return GateDecision(
                allowed=False, category="medical_advice", message=MEDICAL_ADVICE_MESSAGE
            )
    return GateDecision(allowed=True)


# ── Exit helpers ──────────────────────────────────────────────────

_PMID_RE = re.compile(r"PMID\s*[:#]?\s*(\d{5,9})", re.IGNORECASE)


def extract_pmids(text: str) -> list[str]:
    """PMIDs written as 'PMID: 12345678' (or 'PMID 12345678') in free text."""
    return _PMID_RE.findall(text or "")


def find_unsupported_pmids(text: str, retrieved_pmids) -> list[str]:
    """PMIDs mentioned in `text` that were NOT in the retrieved sources."""
    valid = set(retrieved_pmids)
    return sorted({p for p in extract_pmids(text) if p not in valid})
