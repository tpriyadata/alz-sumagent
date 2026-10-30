"""
preprocessing/chunker.py
=========================
Alz SumAgent — Enterprise Semantic Chunker.

WHAT:   Splits 19,637 cleaned papers into ~40,000 searchable chunks.
WHY:    Pinecone retrieves chunks, not full papers.
        A question about pTau217 should return the exact sentence
        mentioning pTau217 — not the entire abstract.
WHO:    Called by scripts/run_pipeline.py after cleaner.
        Output consumed only by vector_store/embedder.py
WHERE:  Reads  → data/processed/cleaned_papers.json
        Writes → data/processed/chunks.json
WHEN:   Once per plan, after cleaning, before embedding.

WORST-CASE DESIGN:
    - Abstract too short to chunk  → single chunk, logged
    - Chunk text empty after split → chunk skipped
    - Missing metadata on paper    → defaults used, never crashes
    - Output dir missing           → created automatically
    - Duplicate chunk IDs          → caught by validator
"""

import json
import re
from pathlib import Path
from typing import Optional
from datetime import datetime

from pydantic import BaseModel, Field, field_validator, model_validator
from loguru import logger ## 
from tqdm import tqdm 

from configs.settings import get_settings 
from preprocessing.cleaner import CleanedPaper


# ── Single chunk model ────────────────────────────────────────────

class PaperChunk(BaseModel):
    """
    One chunk from one paper — the atomic unit of the RAG system.

    Analogy: One index card from the book editor.
    Contains: the content (text), the reference (pmid, chunk_idx),
    and enough metadata to filter in Pinecone (year, keywords, source).

    This is what gets embedded and stored in Pinecone.
    Every field here becomes either a vector or a metadata filter.
    """
    chunk_id:    str         = Field(..., description="Unique ID: pmid_chunk_N")
    pmid:        str         = Field(..., description="Parent paper PMID")
    chunk_idx:   int         = Field(..., ge=0, description="Position in paper")
    text:        str         = Field(..., min_length=20,
                                     description="Chunk text — what gets embedded")
    title:       str         = Field(..., description="Parent paper title")
    year:        Optional[int] = Field(default=None)
    keywords:    list[str]   = Field(default_factory=list)
    journal:     Optional[str] = Field(default=None)
    source_query: str        = Field(default="unknown")
    source:      str         = Field(default="pubmed",
                                     description="pubmed")
    word_count:  int         = Field(default=0, description="Words in this chunk")

    @field_validator("chunk_id")
    @classmethod
    def chunk_id_format(cls, v: str) -> str:
        """
        Enforce chunk ID format: pmid_chunk_N
        Wrong format = Pinecone upsert will have inconsistent IDs.
        """
        if "_chunk_" not in v:
            raise ValueError(
                f"chunk_id '{v}' must contain '_chunk_'. "
                f"Format: 'pmid_chunk_N'"
            )
        return v

    @field_validator("text")
    @classmethod
    def text_has_content(cls, v: str) -> str:
        """Chunk must have real content — not just whitespace."""
        v = v.strip()
        if len(v.split()) < 5:
            raise ValueError(
                f"Chunk text too short ({len(v.split())} words) — skipped"
            )
        return v

    @model_validator(mode="after")
    def compute_word_count(self) -> "PaperChunk":
        """Auto-compute word count after all fields validated."""
        self.word_count = len(self.text.split())
        return self

    def to_dict(self) -> dict:
        return self.model_dump()

    def to_pinecone_record(self) -> dict:
        """
        Format for Pinecone upsert.
        Pinecone expects: {id, values (embedding), metadata}
        This method builds the metadata part.
        Embedding is added by embedder.py.
        """
        return {
            "id": self.chunk_id,
            "metadata": {
                "pmid":         self.pmid,
                "chunk_idx":    self.chunk_idx,
                "text":         self.text[:1000],  # Pinecone metadata limit
                "title":        self.title[:200],
                "year":         self.year,
                "keywords":     self.keywords[:10],  # limit metadata size
                "journal":      self.journal,
                "source_query": self.source_query,
                "source":       self.source,
                "word_count":   self.word_count,
            }
        }


# ── Chunk diagnostic model ────────────────────────────────────────

class ChunkDiagnostic(BaseModel):
    """
    RE inspector for chunking stage.

    Analogy: The editor's production report.
    How many index cards were created, what was skipped,
    what is the average card size.
    """
    total_papers:      int
    total_chunks:      int
    rejected_chunks:   int
    avg_chunks_per_paper: float
    avg_words_per_chunk:  float
    min_words:         int
    max_words:         int
    chunk_duration_secs: float
    output_path:       str

    def log_summary(self) -> None:
        logger.info("=" * 60)
        logger.info("[CHUNK-DIAGNOSTIC] Run complete")
        logger.info(f"  Papers processed  : {self.total_papers:,}")
        logger.info(f"  Total chunks      : {self.total_chunks:,}")
        logger.info(f"  Rejected chunks   : {self.rejected_chunks:,}")
        logger.info(f"  Avg chunks/paper  : {self.avg_chunks_per_paper:.1f}")
        logger.info(f"  Avg words/chunk   : {self.avg_words_per_chunk:.1f}")
        logger.info(f"  Min words         : {self.min_words}")
        logger.info(f"  Max words         : {self.max_words}")
        logger.info(f"  Duration          : {self.chunk_duration_secs:.1f}s")
        logger.info(f"  Saved to          : {self.output_path}")
        logger.info("=" * 60)


# ── Chunking functions ────────────────────────────────────────────

def split_into_chunks(
    text: str,
    chunk_size: int,
    overlap: int,
) -> list[str]:
    """
    Split text into overlapping word-based chunks.

    WHY word-based not character-based:
        Words are the natural unit for biomedical text.
        Character splits can cut mid-word — "Alzheimer" becomes
        "Alzhe" + "imer" — meaningless to the embedding model.

    WHY overlap:
        If a key sentence falls at the boundary between two chunks,
        overlap ensures it appears in both — so retrieval finds it
        regardless of which chunk is retrieved.

    Analogy: Photocopying pages of a book with 1 page overlap.
    You never miss a sentence that falls between two photocopies.

    Args:
        text       : full abstract text
        chunk_size : max words per chunk (from settings)
        overlap    : words shared between adjacent chunks

    Returns:
        list of chunk strings — at least 1, even for short abstracts
    """
    words = text.split()

    # Short abstracts — return as single chunk
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    start  = 0
    step   = max(1, chunk_size - overlap)  # prevent infinite loop

    while start < len(words):
        end   = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start += step

    return chunks


def chunk_paper(
    paper: CleanedPaper,
    chunk_size: int,
    overlap: int,
) -> list[PaperChunk]:
    """
    Chunk one paper into validated PaperChunk objects.

    Worst-case handled:
        - Abstract too short → single chunk returned
        - Chunk text invalid → Pydantic rejects, chunk skipped
        - No chunks produced → empty list, caller logs warning
    """
    raw_chunks = split_into_chunks(paper.abstract, chunk_size, overlap)
    validated  = []

    for idx, chunk_text in enumerate(raw_chunks):
        try:
            chunk = PaperChunk(
                chunk_id     = f"{paper.pmid}_chunk_{idx}",
                pmid         = paper.pmid,
                chunk_idx    = idx,
                text         = chunk_text,
                title        = paper.title,
                year         = paper.year,
                keywords     = paper.keywords,
                journal      = paper.journal,
                source_query = paper.source_query,
                source       = "pubmed",
            )
            validated.append(chunk)
        except Exception as e:
            logger.debug(
                f"[CHUNKER] Skipped chunk {idx} of PMID {paper.pmid}: {e}"
            )

    return validated


# ── Core chunker class ────────────────────────────────────────────

class PaperChunker:
    """
    Enterprise paper chunker.

    Analogy: The book editor team.
    Receives prepared patients (CleanedPaper objects),
    cuts each paper into index cards (PaperChunk objects),
    saves the full card catalogue (chunks.json).

    Usage:
        chunker    = PaperChunker()
        diagnostic = chunker.run()
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self._setup_paths()

    def _setup_paths(self) -> None:
        self.input_path  = (
            self.settings.processed_data_path.parent / "cleaned_papers.json"
        )
        self.output_path = self.settings.processed_data_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"[CHUNKER] Input : {self.input_path}")
        logger.info(f"[CHUNKER] Output: {self.output_path}")
        logger.info(
            f"[CHUNKER] chunk_size={self.settings.chunk_size} | "
            f"overlap={self.settings.chunk_overlap}"
        )

    def _load_cleaned_papers(self) -> list[dict]:
        """Load cleaned papers — fail fast if file missing."""
        if not self.input_path.exists():
            logger.error(f"[CHUNKER] Input not found: {self.input_path}")
            raise FileNotFoundError(
                f"Run cleaner first. No file at: {self.input_path}"
            )
        with open(self.input_path, encoding="utf-8") as f:
            papers = json.load(f)
        logger.info(f"[CHUNKER] Loaded {len(papers):,} cleaned papers")
        return papers

    def run(self) -> ChunkDiagnostic:
        """
        Main entry point — chunks all cleaned papers.

        Flow:
            1. Load cleaned papers
            2. For each paper → split into chunks
            3. Validate each chunk through PaperChunk
            4. Save all chunks to JSON
            5. Return ChunkDiagnostic
        """
        import time
        start_time = time.time()

        logger.info("[CHUNKER] Starting enterprise chunking")

        raw_papers     = self._load_cleaned_papers()
        all_chunks     = []
        rejected_total = 0
        word_counts    = []

        for raw in tqdm(raw_papers, desc="Chunking", unit="paper"):
            try:
                paper  = CleanedPaper(**raw)
                chunks = chunk_paper(
                    paper,
                    self.settings.chunk_size,
                    self.settings.chunk_overlap,
                )

                if not chunks:
                    logger.warning(
                        f"[CHUNKER] No chunks produced for PMID {paper.pmid}"
                    )
                    rejected_total += 1
                    continue

                for chunk in chunks:
                    all_chunks.append(chunk.to_dict())
                    word_counts.append(chunk.word_count)

            except Exception as e:
                logger.warning(f"[CHUNKER] Failed paper: {e}")
                rejected_total += 1

        # Save output
        try:
            with open(self.output_path, "w", encoding="utf-8") as f:
                json.dump(all_chunks, f, ensure_ascii=False, indent=2)
            logger.info(
                f"[CHUNKER] Saved {len(all_chunks):,} chunks → {self.output_path}"
            )
        except Exception as e:
            logger.error(f"[CHUNKER] FATAL — could not save: {e}")
            raise

        duration = round(time.time() - start_time, 1)

        diagnostic = ChunkDiagnostic(
            total_papers         = len(raw_papers),
            total_chunks         = len(all_chunks),
            rejected_chunks      = rejected_total,
            avg_chunks_per_paper = round(len(all_chunks) / max(len(raw_papers), 1), 1),
            avg_words_per_chunk  = round(sum(word_counts) / max(len(word_counts), 1), 1),
            min_words            = min(word_counts) if word_counts else 0,
            max_words            = max(word_counts) if word_counts else 0,
            chunk_duration_secs  = duration,
            output_path          = str(self.output_path),
        )
        diagnostic.log_summary()
        return diagnostic


# ── RE probe ──────────────────────────────────────────────────────

def diagnose_chunks(filepath: Optional[str] = None) -> ChunkDiagnostic:
    """
    Reverse engineering probe for chunking stage.

    WHY: Run this when Pinecone retrieval feels wrong.
    Inspect chunks.json without re-running chunking.

    Usage:
        python -c "from preprocessing.chunker import diagnose_chunks; diagnose_chunks()"
    """
    import time
    settings = get_settings()
    path     = Path(filepath) if filepath else settings.processed_data_path

    if not path.exists():
        logger.error(f"[RE-CHUNKER] File not found: {path}")
        raise FileNotFoundError(f"No chunks at {path}. Run chunker first.")

    logger.info(f"[RE-CHUNKER] Inspecting: {path}")

    with open(path, encoding="utf-8") as f:
        chunks = json.load(f)

    word_counts = [len(c.get("text", "").split()) for c in chunks]

    diagnostic = ChunkDiagnostic(
        total_papers         = len(set(c.get("pmid") for c in chunks)),
        total_chunks         = len(chunks),
        rejected_chunks      = 0,
        avg_chunks_per_paper = round(
            len(chunks) / max(len(set(c.get("pmid") for c in chunks)), 1), 1
        ),
        avg_words_per_chunk  = round(
            sum(word_counts) / max(len(word_counts), 1), 1
        ),
        min_words            = min(word_counts) if word_counts else 0,
        max_words            = max(word_counts) if word_counts else 0,
        chunk_duration_secs  = 0.0,
        output_path          = str(path),
    )
    diagnostic.log_summary()
    return diagnostic