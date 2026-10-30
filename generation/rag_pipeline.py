"""
WORST-CASE DESIGN:
    - No chunks retrieved    → honest "no evidence" response
    - Claude API timeout     → retry 3 times with backoff
    - Claude bad JSON        → Pydantic catches, fallback response
    - Question too short     → rejected by QueryInput validator
    - API key missing        → clear error at startup
"""

import time
from typing import Optional

import anthropic
from pydantic import BaseModel, Field, field_validator
from loguru import logger

from configs.settings import get_settings
from configs.eval_store import save_eval_results
from vector_store.pinecone_store import PineconeStore, QueryInput, QueryResult
from generation.prompts import PROMPT_VERSION, build_rag_user_prompt, get_rag_system
from generation.guards import check_entry, find_unsupported_pmids


# ── Rolling conversation memory ───────────────────────────────────

class ConversationMemory:
    """
    Rolling summary of the current chat session.

    Keeps the last 2 Q&A turns verbatim so Claude can answer
    follow-up questions ("what about its side effects?", "compare
    with the previous finding"). Older turns are compressed into a
    one-line-per-turn summary to avoid unbounded context growth.

    Usage:
        memory = ConversationMemory()
        # after each pipeline.ask():
        memory.add_turn(question, answer)
        # before the next call:
        ctx = memory.to_context_str()   # inject into Claude prompt
    """

    _MAX_VERBATIM: int = 2  # keep this many recent turns in full

    def __init__(self) -> None:
        self._summary: str = ""          # compressed older turns
        self._recent: list[dict] = []    # verbatim recent turns

    def add_turn(self, question: str, answer: "AlzheimerAnswer") -> None:
        """Record a completed Q&A turn; compress oldest if needed."""
        self._recent.append({
            "q": question,
            "a": answer.summary[:300],
            "confidence": answer.confidence,
        })
        if len(self._recent) > self._MAX_VERBATIM:
            oldest = self._recent.pop(0)
            snippet = (
                f'- "{oldest["q"][:80]}" -> '
                f'{oldest["a"][:150]} ({oldest["confidence"]} confidence)'
            )
            self._summary = (
                (self._summary + "\n" + snippet).strip()
                if self._summary else snippet
            )

    def to_context_str(self) -> str:
        """Format memory for injection into Claude's prompt."""
        if not self._summary and not self._recent:
            return ""
        parts: list[str] = []
        if self._summary:
            parts.append(f"EARLIER CONVERSATION SUMMARY:\n{self._summary}")
        if self._recent:
            parts.append("RECENT EXCHANGES:")
            for t in self._recent:
                parts.append(f"  User: {t['q']}")
                parts.append(
                    f"  Assistant ({t['confidence']} confidence): {t['a']}"
                )
        return "\n".join(parts)

    def has_context(self) -> bool:
        return bool(self._summary or self._recent)

    def clear(self) -> None:
        self._summary = ""
        self._recent = []


# ── Answer model — what Claude must return ────────────────────────

class AlzheimerAnswer(BaseModel):
    """
    Validated answer from Claude.

    Analogy: The senior doctor's written diagnosis.
    Must have a summary, key findings, and cited sources.
    A diagnosis without citations is not acceptable
    in this hospital.

    WORST-CASE:
        - Claude returns empty summary → rejected
        - Claude invents PMIDs         → we only use PMIDs from retrieved chunks
        - Claude is uncertain          → confidence field captures this
    """
    summary:        str         = Field(..., min_length=10,
                                        description="Main answer to the question")
    key_findings:   list[str]   = Field(default_factory=list,
                                        description="Bullet points of key findings")
    pmids_cited:    list[str]   = Field(default_factory=list,
                                        description="PMIDs from retrieved papers")
    confidence:     str         = Field(default="medium",
                                        description="high / medium / low")
    limitations:    Optional[str] = Field(default=None,
                                          description="What this answer cannot address")
    disclaimer:     str         = Field(
                                    default="This is for research purposes only. "
                                            "Consult a medical professional for clinical decisions.",
                                    description="Medical disclaimer"
                                  )
    validation_notes: list[str] = Field(default_factory=list,
                                        description="What the exit gate changed or flagged")
    unverified_citations: int   = Field(default=0, ge=0,
                                        description="Distinct cited PMIDs removed because "
                                                    "they were not in the retrieved sources")
    prompt_version:   str       = Field(default=PROMPT_VERSION,
                                        description="Prompt version that produced this answer")

    @field_validator("summary")
    @classmethod
    def summary_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Summary cannot be empty")
        return v

    @field_validator("confidence")
    @classmethod
    def valid_confidence(cls, v: str) -> str:
        v = v.lower().strip()
        if v not in {"high", "medium", "low"}:
            return "medium"   # safe default
        return v

    def to_display(self) -> str:
        """Format answer for Streamlit display."""
        lines = [
            f"**Summary**\n{self.summary}",
        ]
        if self.key_findings:
            lines.append("\n**Key Findings**")
            for finding in self.key_findings:
                lines.append(f"- {finding}")
        if self.pmids_cited:
            links = [
                f"[{pmid}](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)"
                for pmid in self.pmids_cited
            ]
            lines.append(f"\n**Sources cited:** {' | '.join(links)}")
        lines.append(f"\n*Confidence: {self.confidence}*")
        if self.limitations:
            lines.append(f"\n*Limitations: {self.limitations}*")
        lines.append(f"\n---\n*{self.disclaimer}*")
        return "\n".join(lines)


# ── Exit gate (deterministic, fails closed) ───────────────────────

def fail_closed_answer(raw_text: str, reason: str) -> AlzheimerAnswer:
    """
    Safe answer used when model output cannot be trusted.
    The raw model text is logged (truncated) but NEVER shown to the user.
    """
    logger.warning(
        f"[RAG] Exit gate failed closed: {reason} | "
        f"raw output (not shown to user): {(raw_text or '')[:500]!r}"
    )
    return AlzheimerAnswer(
        summary      = "I could not produce a validated answer from the retrieved "
                       "sources. Please rephrase the question or try again.",
        key_findings = [],
        pmids_cited  = [],
        confidence   = "low",
        limitations  = "The model output failed validation and was withheld.",
        validation_notes = [reason],
    )


def validate_citations(
    answer: AlzheimerAnswer,
    retrieved_pmids: list[str],
) -> AlzheimerAnswer:
    """
    Enforce that every citation belongs to the retrieved sources.

    - summary cites an unretrieved PMID  -> fail closed (whole answer rejected)
    - key_finding cites an unretrieved PMID -> that finding is removed
    - pmids_cited contains unretrieved IDs  -> removed
    - nothing verifiable left to cite       -> confidence forced to "low"
    Every change is recorded in answer.validation_notes and logged.
    """
    valid = set(retrieved_pmids)

    bad_in_summary = find_unsupported_pmids(answer.summary, valid)
    if bad_in_summary:
        return fail_closed_answer(
            answer.summary, f"summary cites unretrieved PMIDs {bad_in_summary}"
        )

    notes: list[str] = []
    unverified: set[str] = set()

    removed = [p for p in answer.pmids_cited if p not in valid]
    if removed:
        unverified.update(removed)
        notes.append(f"removed unretrieved PMIDs from pmids_cited: {sorted(set(removed))}")
    answer.pmids_cited = [p for p in answer.pmids_cited if p in valid]

    kept: list[str] = []
    for finding in answer.key_findings:
        bad = find_unsupported_pmids(finding, valid)
        if bad:
            unverified.update(bad)
            notes.append(f"removed finding citing unretrieved PMIDs {bad}")
        else:
            kept.append(finding)
    answer.key_findings = kept
    answer.unverified_citations = len(unverified)

    if not answer.pmids_cited and answer.confidence != "low":
        notes.append("no verifiable citations remain; confidence set to low")
        answer.confidence = "low"

    if notes:
        logger.warning(f"[RAG] Exit gate modified the answer: {notes}")
        answer.validation_notes = notes
        flag = f"{len(notes)} item(s) were removed or adjusted because they could not be verified against the retrieved sources."
        answer.limitations = f"{answer.limitations} {flag}" if answer.limitations else flag
    return answer


# ── RAG pipeline diagnostic ───────────────────────────────────────

class RAGDiagnostic(BaseModel):
    """RE inspector for the RAG pipeline."""
    question:         str
    chunks_retrieved: int
    top_score:        float
    answer_length:    int
    confidence:       str
    total_time_ms:    float
    retrieval_ms:     float
    generation_ms:    float
    retrieved_pmids:  list[str] = []
    gate_outcome:     str       = "passed"   # passed | prompt_injection | medical_advice
    prompt_version:   str       = PROMPT_VERSION
    validation_flags: int       = 0          # items the exit gate removed or adjusted

    def log_summary(self) -> None:
        logger.info("=" * 60)
        logger.info("[RAG-DIAGNOSTIC] Run complete")
        logger.info(f"  Question         : {self.question[:60]}...")
        logger.info(f"  Chunks retrieved : {self.chunks_retrieved}")
        logger.info(f"  Top score        : {self.top_score:.3f}")
        logger.info(f"  Answer length    : {self.answer_length} chars")
        logger.info(f"  Confidence       : {self.confidence}")
        logger.info(f"  Retrieval time   : {self.retrieval_ms:.0f}ms")
        logger.info(f"  Generation time  : {self.generation_ms:.0f}ms")
        logger.info(f"  Total time       : {self.total_time_ms:.0f}ms")
        logger.info("=" * 60)


# ── Core RAG pipeline ─────────────────────────────────────────────

class RAGPipeline:
    """
    Enterprise RAG pipeline.

    Analogy: The senior doctor.
    1. Asks librarian for relevant papers (PineconeStore)
    2. Reads the papers (context building)
    3. Writes diagnosis with citations (Claude generation)
    4. Returns validated answer (Pydantic)

    Usage:
        pipeline = RAGPipeline()
        answer   = pipeline.ask("What biomarkers predict early Alzheimer's?")
        print(answer.to_display())
    """

    _MAX_RETRIES:   int   = 3
    _RETRY_BACKOFF: float = 2.0

    def __init__(self) -> None:
        self.settings = get_settings()
        self.store    = PineconeStore()
        self.client   = anthropic.Anthropic(
            api_key=self.settings.anthropic_api_key
        )
        logger.info("[RAG] Pipeline initialised")

    def _build_context(self, result: QueryResult) -> str:
        """
        Build context block from retrieved chunks.
        This goes into Claude's prompt.
        """
        if not result.has_results:
            return "No relevant papers found in the database."
        return result.to_context_block()

    def _call_claude(
        self,
        question: str,
        context: str,
        conversation_context: str = "",
    ) -> str:
        """
        Call Claude with retrieved context and question.
        Retry on failure.
        Returns raw response text.
        """
        prompt        = build_rag_user_prompt(question, context, conversation_context)
        system_prompt = get_rag_system()

        for attempt in range(1, self._MAX_RETRIES + 1):
            try:
                response = self.client.messages.create(
                    model      = self.settings.claude_model,
                    max_tokens = self.settings.max_tokens,
                    system     = system_prompt,
                    messages   = [{"role": "user", "content": prompt}]
                )
                return response.content[0].text

            except Exception as e:
                logger.warning(
                    f"[RAG] Claude attempt {attempt}/{self._MAX_RETRIES} "
                    f"failed: {e}"
                )
                if attempt < self._MAX_RETRIES:
                    time.sleep(self._RETRY_BACKOFF * attempt)
                else:
                    logger.error("[RAG] Claude failed after max retries")
                    raise

    def _parse_answer(
        self,
        raw_text: str,
        retrieved_pmids: list[str]
    ) -> AlzheimerAnswer:
        """
        Parse Claude's JSON response into a VALIDATED AlzheimerAnswer.

        Fails closed: malformed or untrustworthy output is withheld
        (logged, never shown to the user). Citations are checked by
        validate_citations().
        """
        import json
        import re

        clean = re.sub(r"```(?:json)?", "", raw_text or "").strip()

        try:
            data = json.loads(clean)
            if not isinstance(data, dict):
                raise ValueError("model output is not a JSON object")
            answer = AlzheimerAnswer(**data)
        except Exception as e:
            return fail_closed_answer(
                raw_text, f"output failed parsing/validation ({type(e).__name__})"
            )

        return validate_citations(answer, retrieved_pmids)

    def ask(
        self,
        question:  str,
        top_k:     Optional[int] = None,
        source:    Optional[str] = None,
        year_from: Optional[int] = None,
        memory:    Optional[ConversationMemory] = None,
    ) -> tuple[AlzheimerAnswer, RAGDiagnostic]:
        """
        Main entry point — ask a research question.

        Flow:
            1. Validate question
            2. Retrieve relevant chunks from Pinecone
            3. Build context
            4. Call Claude
            5. Parse and validate answer
            6. Return answer + diagnostic

        Returns:
            (AlzheimerAnswer, RAGDiagnostic)
        """
        total_start = time.time()
        top_k       = top_k or self.settings.top_k_results
        logger.info(f"[RAG] Question: '{question}' | top_k={top_k}")

        # Step 1 — validate question
        query = QueryInput(
            question  = question,
            top_k     = top_k,
            source    = source,
            year_from = year_from,
        )

        # Step 1b — deterministic entry gate (no retrieval or LLM call if blocked)
        gate = check_entry(query.question)
        if not gate.allowed:
            logger.warning(f"[RAG] Entry gate blocked request: {gate.category}")
            blocked = AlzheimerAnswer(
                summary      = gate.message,
                key_findings = [],
                pmids_cited  = [],
                confidence   = "low",
                limitations  = f"Request declined by the entry gate ({gate.category}).",
                validation_notes = [f"entry gate: {gate.category}"],
            )
            return blocked, RAGDiagnostic(
                question         = question,
                chunks_retrieved = 0,
                top_score        = 0.0,
                answer_length    = len(blocked.summary),
                confidence       = "low",
                total_time_ms    = round((time.time() - total_start) * 1000, 1),
                retrieval_ms     = 0.0,
                generation_ms    = 0.0,
                gate_outcome     = gate.category,
            )

        # Step 2 — retrieve chunks
        retrieval_start = time.time()
        result          = self.store.query(query)
        retrieval_ms    = (time.time() - retrieval_start) * 1000

        if not result.has_results:
            logger.warning(f"[RAG] No chunks retrieved for: '{question}'")

        # Step 3 — build context
        context = self._build_context(result)

        # Step 4 — call Claude (inject rolling conversation context if present)
        conv_ctx         = memory.to_context_str() if memory and memory.has_context() else ""
        generation_start = time.time()
        raw_text         = self._call_claude(question, context, conv_ctx)
        generation_ms    = (time.time() - generation_start) * 1000

        # Step 5 — parse answer
        retrieved_pmids = [c.pmid for c in result.chunks]
        answer          = self._parse_answer(raw_text, retrieved_pmids)

        # Update rolling memory with this completed turn
        if memory is not None:
            memory.add_turn(question, answer)

        # Step 6 — build diagnostic
        total_ms    = (time.time() - total_start) * 1000
        diagnostic  = RAGDiagnostic(
            question         = question,
            chunks_retrieved = result.total_returned,
            top_score        = result.top_score,
            answer_length    = len(answer.summary),
            confidence       = answer.confidence,
            total_time_ms    = round(total_ms, 1),
            retrieval_ms     = round(retrieval_ms, 1),
            generation_ms    = round(generation_ms, 1),
            retrieved_pmids  = [c.pmid for c in result.chunks],
            validation_flags = len(answer.validation_notes),
        )
        diagnostic.log_summary()
        save_eval_results({
            "top_score":        result.top_score,   # similarity score, NOT a hit rate
            "confidence":       answer.confidence,
            "chunks_used":      result.total_returned,
            "latency_ms":       round(total_ms, 1),
            "validation_flags": len(answer.validation_notes),
            "prompt_version":   PROMPT_VERSION,
        })

        return answer, diagnostic


# ── RE probe ──────────────────────────────────────────────────────

def diagnose_rag() -> None:
    """
    RE probe — test full RAG pipeline end to end.

    WHY: When Streamlit answers feel wrong, run this first.
    Tests complete pipeline: retrieve → generate → validate.

    Usage:
        python -c "from generation.rag_pipeline import diagnose_rag; diagnose_rag()"
    """
    logger.info("[RE-RAG] Running full RAG pipeline test")

    pipeline = RAGPipeline()
    answer, diagnostic = pipeline.ask(
        question = "What blood biomarkers can detect Alzheimer's disease early?",
        top_k    = 5,
    )

    logger.info("[RE-RAG] Answer received:")
    logger.info(f"  Summary    : {answer.summary[:100]}...")
    logger.info(f"  Findings   : {len(answer.key_findings)}")
    logger.info(f"  PMIDs cited: {answer.pmids_cited}")
    logger.info(f"  Confidence : {answer.confidence}")
    logger.info("[RE-RAG] RAG pipeline working correctly")
    print("\n" + answer.to_display())
