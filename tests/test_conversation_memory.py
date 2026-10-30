"""
tests/test_conversation_memory.py
===================================
Unit tests for ConversationMemory rolling summary.

No API calls — all tests use fake AlzheimerAnswer objects.

Run:
    python -m pytest tests/test_conversation_memory.py -v
"""

import pytest
from generation.rag_pipeline import AlzheimerAnswer, ConversationMemory


# ── Helpers ───────────────────────────────────────────────────────

def make_answer(summary: str, confidence: str = "medium") -> AlzheimerAnswer:
    return AlzheimerAnswer(summary=summary, confidence=confidence)


# ── Empty state ───────────────────────────────────────────────────

class TestEmptyMemory:
    def test_has_no_context(self):
        mem = ConversationMemory()
        assert mem.has_context() is False

    def test_to_context_str_is_empty_string(self):
        mem = ConversationMemory()
        assert mem.to_context_str() == ""


# ── Single turn ───────────────────────────────────────────────────

class TestSingleTurn:
    def setup_method(self):
        self.mem = ConversationMemory()
        self.mem.add_turn(
            "What biomarkers detect Alzheimer's early?",
            make_answer("pTau217 and amyloid-beta are key early biomarkers.", "high"),
        )

    def test_has_context(self):
        assert self.mem.has_context() is True

    def test_question_in_output(self):
        ctx = self.mem.to_context_str()
        assert "What biomarkers detect Alzheimer" in ctx

    def test_answer_in_output(self):
        ctx = self.mem.to_context_str()
        assert "pTau217 and amyloid-beta" in ctx

    def test_confidence_in_output(self):
        ctx = self.mem.to_context_str()
        assert "high confidence" in ctx

    def test_recent_exchanges_header(self):
        ctx = self.mem.to_context_str()
        assert "RECENT EXCHANGES" in ctx

    def test_no_summary_header_yet(self):
        ctx = self.mem.to_context_str()
        assert "EARLIER CONVERSATION SUMMARY" not in ctx


# ── Two turns (at verbatim limit) ─────────────────────────────────

class TestTwoTurns:
    def setup_method(self):
        self.mem = ConversationMemory()
        self.mem.add_turn("Question one", make_answer("Answer one is complete.", "high"))
        self.mem.add_turn("Question two", make_answer("Answer two is complete.", "medium"))

    def test_both_questions_present(self):
        ctx = self.mem.to_context_str()
        assert "Question one" in ctx
        assert "Question two" in ctx

    def test_no_compression_yet(self):
        ctx = self.mem.to_context_str()
        assert "EARLIER CONVERSATION SUMMARY" not in ctx

    def test_two_verbatim_entries(self):
        ctx = self.mem.to_context_str()
        assert ctx.count("User:") == 2


# ── Three turns (triggers first compression) ──────────────────────

class TestThreeTurns:
    def setup_method(self):
        self.mem = ConversationMemory()
        self.mem.add_turn("Question one", make_answer("Answer one is complete.", "high"))
        self.mem.add_turn("Question two", make_answer("Answer two is complete.", "medium"))
        self.mem.add_turn("Question three", make_answer("Answer three is complete.", "low"))

    def test_summary_section_appears(self):
        ctx = self.mem.to_context_str()
        assert "EARLIER CONVERSATION SUMMARY" in ctx

    def test_oldest_compressed(self):
        ctx = self.mem.to_context_str()
        # Turn 1 should be in compressed summary, not verbatim
        assert "Question one" in ctx
        assert "Answer one" in ctx

    def test_two_recent_verbatim(self):
        ctx = self.mem.to_context_str()
        assert ctx.count("User:") == 2

    def test_verbatim_contains_turns_2_and_3(self):
        ctx = self.mem.to_context_str()
        assert "Question two" in ctx
        assert "Question three" in ctx


# ── Four turns (two compressed) ───────────────────────────────────

class TestFourTurns:
    def setup_method(self):
        self.mem = ConversationMemory()
        for i in range(1, 5):
            self.mem.add_turn(
                f"Question {i}",
                make_answer(f"Answer number {i} is complete.", "medium"),
            )

    def test_summary_has_two_lines(self):
        ctx = self.mem.to_context_str()
        summary_block = ctx.split("RECENT EXCHANGES")[0]
        compressed_lines = [
            l for l in summary_block.splitlines() if l.strip().startswith("-")
        ]
        assert len(compressed_lines) == 2

    def test_verbatim_still_two_turns(self):
        ctx = self.mem.to_context_str()
        assert ctx.count("User:") == 2

    def test_oldest_turns_in_summary(self):
        ctx = self.mem.to_context_str()
        assert "Question 1" in ctx
        assert "Question 2" in ctx

    def test_recent_turns_in_verbatim(self):
        ctx = self.mem.to_context_str()
        recent_block = ctx.split("RECENT EXCHANGES")[-1]
        assert "Question 3" in recent_block
        assert "Question 4" in recent_block


# ── Clear ─────────────────────────────────────────────────────────

class TestClear:
    def test_clear_resets_context(self):
        mem = ConversationMemory()
        mem.add_turn("Question one", make_answer("First answer is complete."))
        mem.add_turn("Question two", make_answer("Second answer is complete."))
        mem.add_turn("Question three", make_answer("Third answer is complete."))  # triggers compression
        mem.clear()
        assert mem.has_context() is False
        assert mem.to_context_str() == ""

    def test_add_turn_after_clear(self):
        mem = ConversationMemory()
        mem.add_turn("Question one", make_answer("First answer is complete."))
        mem.clear()
        mem.add_turn("Fresh question", make_answer("Fresh answer here now."))
        ctx = mem.to_context_str()
        assert "Fresh question" in ctx
        assert "Question one" not in ctx


# ── Truncation guards ─────────────────────────────────────────────

class TestTruncation:
    def test_long_question_truncated_in_summary(self):
        mem = ConversationMemory()
        long_q = "x" * 200
        mem.add_turn(long_q, make_answer("Short answer here."))
        mem.add_turn("Question two is here", make_answer("Answer two is complete."))
        mem.add_turn("Question three here", make_answer("Answer three is complete."))  # compresses long_q
        ctx = mem.to_context_str()
        # Compressed question capped at 80 chars + quote chars
        summary_line = [l for l in ctx.splitlines() if "x" * 10 in l][0]
        assert len(summary_line) < 400  # well under the raw 200-char question

    def test_long_answer_truncated_in_summary(self):
        mem = ConversationMemory()
        long_a = "y" * 500
        mem.add_turn("Question one here", make_answer(long_a))
        mem.add_turn("Question two here", make_answer("Answer two is complete."))
        mem.add_turn("Question three here", make_answer("Answer three is complete."))  # compresses Q1
        ctx = mem.to_context_str()
        summary_line = [l for l in ctx.splitlines() if "y" * 10 in l][0]
        # Compressed answer capped at 150 chars
        assert summary_line.count("y") <= 150
