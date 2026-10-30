"""
vector_store/pinecone_store.py

"""

import time
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field, field_validator
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from loguru import logger

from configs.settings import get_settings


# ── Query input model ─────────────────────────────────────────────

class QueryInput(BaseModel):
   
    question:    str         = Field(..., min_length=3,
                                     description="User research question")
    top_k:       int         = Field(default=20, ge=1, le=50,
                                     description="Number of chunks to retrieve")
    source:      Optional[str] = Field(default=None,
                                       description="Filter: 'pubmed' or None")
    year_from:   Optional[int] = Field(default=None, ge=2000, le=2030,
                                       description="Filter papers from this year")
    min_score:   float       = Field(default=0.3, ge=0.0, le=1.0,
                                     description="Minimum similarity score")

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Question cannot be empty")
        return v

    @field_validator("source")
    @classmethod
    def valid_source(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v != "pubmed":
            raise ValueError(f"source must be 'pubmed' or None, got: {v}")
        return v


# ── Retrieved chunk model ─────────────────────────────────────────

class RetrievedChunk(BaseModel):
   
    chunk_id:    str   = Field(..., description="Unique chunk ID")
    pmid:        str   = Field(..., description="Parent paper PMID")
    text:        str   = Field(..., min_length=10,
                               description="Chunk text for context")
    title:       str   = Field(default="",
                               description="Parent paper title")
    score:       float = Field(..., ge=0.0, le=1.0,
                               description="Cosine similarity score")
    year:        Optional[int]  = Field(default=None)
    keywords:    list[str]      = Field(default_factory=list)
    journal:     Optional[str]  = Field(default=None)
    source:      str            = Field(default="pubmed")
    source_query: str           = Field(default="unknown")

    @field_validator("text")
    @classmethod
    def text_has_content(cls, v: str) -> str:
        if len(v.strip().split()) < 5:
            raise ValueError("Retrieved chunk text too short — filtering out")
        return v.strip()

    def to_context_string(self) -> str:
        """
        Format chunk as context for Claude prompt.
        Includes citation info so Claude can reference sources.
        """
        year_str = str(self.year) if self.year else "unknown year"
        return (
            f"[PMID: {self.pmid} | {year_str} | Score: {self.score:.3f}]\n"
            f"Title: {self.title}\n"
            f"Content: {self.text}"
        )


# ── Query result model ────────────────────────────────────────────

class QueryResult(BaseModel):
    """
    Full result of one Pinecone query — returned to RAG pipeline.

    Contains all retrieved cards plus metadata about the search.
    """
    question:       str
    chunks:         list[RetrievedChunk]
    total_returned: int
    query_time_ms:  float
    model_used:     str

    @property
    def has_results(self) -> bool:
        return len(self.chunks) > 0

    @property
    def top_score(self) -> float:
        if not self.chunks:
            return 0.0
        return self.chunks[0].score

    def to_context_block(self) -> str:
        """
        Build the full context block for Claude prompt.
        All retrieved chunks formatted and joined.
        """
        if not self.chunks:
            return "No relevant papers found for this query."
        parts = [
            f"--- Source {i+1} ---\n{chunk.to_context_string()}"
            for i, chunk in enumerate(self.chunks)
        ]
        return "\n\n".join(parts)

    def log_summary(self) -> None:
        logger.info(
            f"[QUERY] '{self.question[:50]}...' | "
            f"returned: {self.total_returned} | "
            f"top_score: {self.top_score:.3f} | "
            f"time: {self.query_time_ms:.0f}ms"
        )


# ── Core query store ──────────────────────────────────────────────

class PineconeStore:
    """
    Enterprise Pinecone query layer.

   
    Usage:
        store  = PineconeStore()
        result = store.query(QueryInput(question="pTau217 biomarkers"))
        print(result.to_context_block())
    """

    _MAX_RETRIES:   int   = 3
    _RETRY_BACKOFF: float = 2.0

    def __init__(self) -> None:
        self.settings = get_settings()
        self._setup_model()
        self._setup_pinecone()

    def _setup_model(self) -> None:
        """Load embedding model — same model used at index time."""
        logger.info(f"[STORE] Loading model: {self.settings.embedding_model}")
        self.model = SentenceTransformer(self.settings.embedding_model)
        logger.info("[STORE] Model loaded")

    def _setup_pinecone(self) -> None:
        """Connect to Pinecone — verify index is healthy."""
        pc         = Pinecone(api_key=self.settings.pinecone_api_key)
        self.index = pc.Index(self.settings.pinecone_index_name)

        try:
            stats = self.index.describe_index_stats()
            logger.info(
                f"[STORE] Pinecone connected | "
                f"index: {self.settings.pinecone_index_name} | "
                f"vectors: {stats.total_vector_count:,}"
            )
            if stats.total_vector_count == 0:
                logger.warning(
                    "[STORE] Index is empty — run embedder first. "
                    "No results will be returned."
                )
        except Exception as e:
            logger.error(
                f"[STORE] Pinecone connection failed: {e}. "
                f"Check PINECONE_API_KEY and index name in .env"
            )
            raise

    def _embed_query(self, question: str) -> list[float]:
        """
        Convert question text to embedding vector.
        Must use SAME model as used at index time.
        Different model = wrong vectors = bad retrieval.
        """
        embedding = self.model.encode(
            question,
            convert_to_numpy=True,
        )
        return embedding.tolist()

    def _build_filter(self, query: QueryInput) -> Optional[dict]:
        """
        Build Pinecone metadata filter from query options.

        Analogy: Telling the librarian to only search
        cards from 2023 onwards, or only PubMed cards.
        """
        filters = {}

        if query.source:
            filters["source"] = {"$eq": query.source}

        if query.year_from:
            filters["year"] = {"$gte": query.year_from}

        return filters if filters else None

    def query(self, query: QueryInput) -> QueryResult:
        """
        Main query method — search Pinecone, return validated results.

        Flow:
            1. Validate QueryInput (Pydantic)
            2. Embed question using PubMedBERT
            3. Search Pinecone with optional metadata filters
            4. Parse and validate each result as RetrievedChunk
            5. Filter by min_score
            6. Return QueryResult

        Worst-case handled:
            - Pinecone timeout    → retry 3 times
            - Empty results       → return QueryResult with empty list
            - Bad chunk metadata  → Pydantic rejects, chunk skipped
        """
        start_time = time.time()
        logger.info(
            f"[STORE] Querying: '{query.question[:60]}' | "
            f"top_k={query.top_k} | source={query.source}"
        )

        # Step 1 — embed question
        query_vector = self._embed_query(query.question)

        # Step 2 — build metadata filter
        metadata_filter = self._build_filter(query)

        # Step 3 — search Pinecone with retry
        raw_results = None
        for attempt in range(1, self._MAX_RETRIES + 1):
            try:
                raw_results = self.index.query(
                    vector          = query_vector,
                    top_k           = query.top_k,
                    include_metadata= True,
                    filter          = metadata_filter,
                )
                break
            except Exception as e:
                logger.warning(
                    f"[STORE] Query attempt {attempt}/{self._MAX_RETRIES} "
                    f"failed: {e}"
                )
                if attempt < self._MAX_RETRIES:
                    time.sleep(self._RETRY_BACKOFF * attempt)
                else:
                    logger.error("[STORE] Query failed after max retries")
                    raw_results = None

        # Step 4 — parse and validate results
        chunks = []
        if raw_results and raw_results.matches:
            for match in raw_results.matches:
                # Filter by minimum score
                if match.score < query.min_score:
                    logger.debug(
                        f"[STORE] Filtered low score: "
                        f"{match.score:.3f} < {query.min_score}"
                    )
                    continue

                try:
                    meta  = match.metadata or {}
                    chunk = RetrievedChunk(
                        chunk_id     = match.id,
                        pmid         = meta.get("pmid", "unknown"),
                        text         = meta.get("text", ""),
                        title        = meta.get("title", ""),
                        score        = match.score,
                        year         = meta.get("year"),
                        keywords     = meta.get("keywords", []),
                        journal      = meta.get("journal"),
                        source       = meta.get("source", "pubmed"),
                        source_query = meta.get("source_query", "unknown"),
                    )
                    chunks.append(chunk)
                except Exception as e:
                    logger.debug(f"[STORE] Chunk validation failed: {e}")
                    continue

        query_time = round((time.time() - start_time) * 1000, 1)

        result = QueryResult(
            question       = query.question,
            chunks         = chunks,
            total_returned = len(chunks),
            query_time_ms  = query_time,
            model_used     = self.settings.embedding_model,
        )
        result.log_summary()
        return result


# ── RE probe ──────────────────────────────────────────────────────

def diagnose_pinecone_store() -> None:
    """
    RE probe — test full query pipeline end to end.

    WHY: When RAG answers feel irrelevant, run this FIRST.
    Tests the complete retrieval chain with a known good query.

    Usage:
        python -c "from vector_store.pinecone_store import diagnose_pinecone_store; diagnose_pinecone_store()"
    """
    logger.info("[RE-STORE] Running end-to-end retrieval test")

    store = PineconeStore()

    # Test with a known Alzheimer's query
    test_query = QueryInput(
        question = "What blood biomarkers predict early Alzheimer's disease?",
        top_k    = 5,
        min_score= 0.3,
    )

    result = store.query(test_query)

    logger.info("[RE-STORE] Results:")
    logger.info(f"  Returned  : {result.total_returned}")
    logger.info(f"  Top score : {result.top_score:.3f}")
    logger.info(f"  Query time: {result.query_time_ms:.0f}ms")

    if result.has_results:
        logger.info("[RE-STORE] Top result:")
        logger.info(f"  PMID  : {result.chunks[0].pmid}")
        logger.info(f"  Score : {result.chunks[0].score:.3f}")
        logger.info(f"  Text  : {result.chunks[0].text[:100]}...")
        logger.info("[RE-STORE] Retrieval working correctly")
    else:
        logger.warning(
            "[RE-STORE] No results returned. "
            "Check: 1) Pinecone has vectors 2) min_score not too high "
            "3) same embedding model used at index and query time"
        )