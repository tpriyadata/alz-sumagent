"""
preprocessing/cleaner.py
=========================


WORST-CASE DESIGN:
    - Missing year         → infer from fetched_at timestamp
    - HTML in abstract     → strip all tags, collapse whitespace
    - Empty abstract       → paper rejected, logged, counted
    - Abstract too short   → paper rejected (< 20 words)
    - Corrupt JSON input   → caught, logged, pipeline stops cleanly
    - Output dir missing   → created automatically
"""

import json
import re
from pathlib import Path
from typing import Optional
from datetime import datetime , UTC

from pydantic import BaseModel, Field, field_validator, ValidationError
from loguru import logger
from tqdm import tqdm

from configs.settings import get_settings
from ingestion.pubmed_fetch import PubMedPaper


# ── Cleaned paper model ───────────────────────────────────────────

class CleanedPaper(BaseModel):
    """
    A fully cleaned PubMed paper — ready for chunking.

    Analogy: A patient who has passed triage, had bloodwork done,
    wounds cleaned, and is now ready for the operating room (chunker).
    No surgery happens on uncleaned patients.

    Differences from PubMedPaper:
        - abstract    : guaranteed HTML-free, whitespace normalized
        - year        : guaranteed int or None (never missing silently)
        - keywords    : guaranteed non-empty list (fallback extracted)
        - clean_flag  : records which fixes were applied
    """
    pmid:          str          = Field(..., description="PubMed unique ID")
    title:         str          = Field(..., description="Paper title — HTML stripped")
    abstract:      str          = Field(..., min_length=10,
                                        description="Abstract — HTML stripped, normalized")
    authors:       list[str]    = Field(default_factory=list)
    year:          Optional[int]= Field(default=None,
                                        description="Year — inferred if originally missing")
    keywords:      list[str]    = Field(default_factory=list,
                                        description="Keywords — extracted if originally empty")
    journal:       Optional[str]= Field(default=None)
    source_query:  str          = Field(default="unknown")
    fetched_at:    str          = Field(default="")

    # Audit trail — which fixes were applied to this paper
    year_inferred:     bool = Field(default=False,
                                    description="True if year was inferred from fetched_at")
    keywords_extracted:bool = Field(default=False,
                                    description="True if keywords were extracted from abstract")
    html_stripped:     bool = Field(default=False,
                                    description="True if HTML tags were found and removed")

    @field_validator("abstract")
    @classmethod
    def abstract_has_content(cls, v: str) -> str:
        """Final gate — abstract must have real content after cleaning."""
        if len(v.split()) < 20:
            raise ValueError(
                f"Abstract too short after cleaning ({len(v.split())} words) "
                f"— paper rejected"
            )
        return v

    def to_dict(self) -> dict:
        return self.model_dump()

    @property
    def word_count(self) -> int:
        return len(self.abstract.split())


# ── Clean diagnostic model ────────────────────────────────────────

class CleanDiagnostic(BaseModel):
    """
    RE inspector for the cleaning stage.

    Analogy: Lab report after patient preparation.
    Shows exactly how many patients needed treatment
    and what treatment they received.
    """
    total_input:          int
    total_output:         int
    total_rejected:       int
    years_inferred:       int
    keywords_extracted:   int
    html_stripped:        int
    rejected_too_short:   int
    clean_duration_secs:  float
    output_path:          str

    @property
    def retention_rate(self) -> float:
        """What % of papers survived cleaning."""
        if self.total_input == 0:
            return 0.0
        return round((self.total_output / self.total_input) * 100, 2)

    def log_summary(self) -> None:
        logger.info("=" * 60)
        logger.info("[CLEAN-DIAGNOSTIC] Run complete")
        logger.info(f"  Input papers      : {self.total_input:,}")
        logger.info(f"  Output papers     : {self.total_output:,}")
        logger.info(f"  Rejected          : {self.total_rejected:,}")
        logger.info(f"  Years inferred    : {self.years_inferred:,}")
        logger.info(f"  Keywords extracted: {self.keywords_extracted:,}")
        logger.info(f"  HTML stripped     : {self.html_stripped:,}")
        logger.info(f"  Too short         : {self.rejected_too_short:,}")
        logger.info(f"  Retention rate    : {self.retention_rate}%")
        logger.info(f"  Duration          : {self.clean_duration_secs:.1f}s")
        logger.info(f"  Saved to          : {self.output_path}")
        logger.info("=" * 60)


# ── Cleaning functions ────────────────────────────────────────────

def strip_html(text: str) -> tuple[str, bool]:
    """
    Remove HTML tags from text.
    Returns (cleaned_text, was_html_found).

    WHY: PubMed XML contains tags like <i>APOE</i>, <b>results</b>,
    <sub>2</sub>. These appear as literal characters to the embedding
    model — adding noise to vectors.

    Analogy: Removing bandage packaging before storing in the med kit.
    The bandage (content) is what matters, not the wrapper (HTML).

    Examples:
        "<i>APOE</i> gene"  → "APOE gene", True
        "normal text"       → "normal text", False
    """
    html_pattern = re.compile(r"<[^>]+>")
    was_html = bool(html_pattern.search(text))
    cleaned  = html_pattern.sub(" ", text)
    # Collapse multiple spaces into one
    cleaned  = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned, was_html


def infer_year(fetched_at: str) -> tuple[Optional[int], bool]:
    """
    Infer publication year from fetch timestamp as last resort.

    WHY: 49 papers have no year. We cannot leave year=None because
    the RAG system uses year for filtering. Inferring from fetch
    timestamp is imprecise but better than nothing — and we flag it.

    Analogy: A patient with no ID card. We estimate age from
    appearance and flag the record as 'estimated'. Better than
    rejecting the patient entirely.

    Returns:
        (year, was_inferred) — year as int or None, bool flag
    """
    if not fetched_at:
        return None, False
    try:
        dt   = datetime.fromisoformat(fetched_at)
        year = dt.year
        logger.debug(f"[CLEANER] Year inferred from fetched_at: {year}")
        return year, True
    except (ValueError, TypeError):
        return None, False


def extract_keywords(abstract: str, title: str, max_kw: int = 10) -> tuple[list[str], bool]:
    """
    Extract simple keywords from abstract + title when MeSH is missing.

    WHY: 3,390 papers in Plan 1 had no keywords. Without keywords,
    Pinecone metadata filtering is degraded. Simple extraction is
    better than empty.

    HOW: Split abstract + title into words, filter by length and
    medical relevance. Not NLP — simple but fast and dependency-free.

    Analogy: A patient with no medical history. We ask them basic
    questions to fill the minimum required fields.

    Returns:
        (keywords, was_extracted) — list of strings, bool flag
    """
    # Common English stop words to exclude
    stop_words = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "by", "from", "as", "is", "was", "are",
        "were", "be", "been", "have", "has", "had", "do", "does", "did",
        "will", "would", "could", "should", "may", "might", "this", "that",
        "these", "those", "it", "its", "we", "our", "they", "their",
        "study", "results", "conclusion", "background", "methods", "patients",
        "showed", "found", "used", "using", "based", "compared", "between",
    }

    combined = f"{title} {abstract}".lower()
    words    = re.findall(r"\b[a-z][a-z\-]{3,}\b", combined)
    filtered = [
        w for w in words
        if w not in stop_words
        and len(w) >= 4
    ]

    # Count frequency — most common words are likely keywords
    from collections import Counter
    counts   = Counter(filtered)
    keywords = [word for word, _ in counts.most_common(max_kw)]

    return keywords, True


# ── Core cleaner class ────────────────────────────────────────────

class PaperCleaner:
    """
    Enterprise paper cleaner.

    Analogy: The hospital preparation lab.
    Receives admitted patients (PubMedPaper objects),
    runs standardised preparation procedures,
    releases cleaned patients (CleanedPaper objects)
    ready for the operating room (chunker).

    Usage:
        cleaner    = PaperCleaner()
        diagnostic = cleaner.run()
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self._setup_paths()

    def _setup_paths(self) -> None:
        """Ensure output directory exists — worst-case: it doesn't."""
        self.input_path  = self.settings.raw_data_path
        self.output_path = self.settings.processed_data_path.parent / "cleaned_papers.json"
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"[CLEANER] Input : {self.input_path}")
        logger.info(f"[CLEANER] Output: {self.output_path}")

    def _load_papers(self) -> list[dict]:
        """
        Load raw papers from JSON.
        Worst-case: file missing, file corrupt, encoding error.
        """
        if not self.input_path.exists():
            logger.error(f"[CLEANER] Input file not found: {self.input_path}")
            raise FileNotFoundError(
                f"Run pubmed_fetch first. No file at: {self.input_path}"
            )
        try:
            with open(self.input_path, encoding="utf-8") as f:
                papers = json.load(f)
            logger.info(f"[CLEANER] Loaded {len(papers):,} raw papers")
            return papers
        except json.JSONDecodeError as e:
            logger.error(f"[CLEANER] FATAL — JSON corrupt: {e}")
            raise

    def _clean_one(self, raw: dict) -> Optional[CleanedPaper]:
        """
        Clean a single paper dict → CleanedPaper.

        Steps applied in order:
            1. Validate raw dict through PubMedPaper first
            2. Strip HTML from abstract and title
            3. Infer year if missing
            4. Extract keywords if empty
            5. Validate result through CleanedPaper

        Returns None if paper cannot be cleaned to minimum standard.
        """
        # Step 1 — validate raw input
        try:
            paper = PubMedPaper(**raw)
        except ValidationError as e:
            logger.debug(f"[CLEANER] PubMedPaper rejected: {e}")
            return None

        # Step 2 — strip HTML from abstract and title
        abstract, html_in_abstract = strip_html(paper.abstract)
        title,    html_in_title    = strip_html(paper.title)
        html_stripped = html_in_abstract or html_in_title

        # Step 3 — infer year if missing
        year           = paper.year
        year_inferred  = False
        if year is None:
            year, year_inferred = infer_year(paper.fetched_at)

        # Step 4 — extract keywords if empty
        keywords            = paper.keywords
        keywords_extracted  = False
        if not keywords:
            keywords, keywords_extracted = extract_keywords(abstract, title)

        # Step 5 — build and validate CleanedPaper
        try:
            cleaned = CleanedPaper(
                pmid               = paper.pmid,
                title              = title,
                abstract           = abstract,
                authors            = paper.authors,
                year               = year,
                keywords           = keywords,
                journal            = paper.journal,
                source_query       = paper.source_query,
                fetched_at         = paper.fetched_at,
                year_inferred      = year_inferred,
                keywords_extracted = keywords_extracted,
                html_stripped      = html_stripped,
            )
            return cleaned
        except ValidationError as e:
            logger.debug(f"[CLEANER] CleanedPaper rejected PMID {paper.pmid}: {e}")
            return None

    def run(self) -> CleanDiagnostic:
        """
        Main entry point — cleans all papers.

        Flow:
            1. Load raw papers
            2. Clean each paper
            3. Count fixes applied
            4. Save cleaned papers
            5. Return CleanDiagnostic

        Returns:
            CleanDiagnostic with full statistics
        """
        import time
        start_time = time.time()

        logger.info("[CLEANER] Starting enterprise paper cleaning")

        raw_papers = self._load_papers()

        # Counters
        cleaned_papers      = []
        total_rejected      = 0
        years_inferred      = 0
        keywords_extracted  = 0
        html_stripped       = 0
        rejected_too_short  = 0

        for raw in tqdm(raw_papers, desc="Cleaning", unit="paper"):
            cleaned = self._clean_one(raw)

            if cleaned is None:
                total_rejected     += 1
                rejected_too_short += 1
                continue

            # Count fixes applied
            if cleaned.year_inferred:
                years_inferred += 1
            if cleaned.keywords_extracted:
                keywords_extracted += 1
            if cleaned.html_stripped:
                html_stripped += 1

            cleaned_papers.append(cleaned.to_dict())

        # Save output
        try:
            with open(self.output_path, "w", encoding="utf-8") as f:
                json.dump(cleaned_papers, f, ensure_ascii=False, indent=2)
            logger.info(
                f"[CLEANER] Saved {len(cleaned_papers):,} "
                f"cleaned papers → {self.output_path}"
            )
        except Exception as e:
            logger.error(f"[CLEANER] FATAL — could not save output: {e}")
            raise

        duration = round(time.time() - start_time, 1)

        diagnostic = CleanDiagnostic(
            total_input         = len(raw_papers),
            total_output        = len(cleaned_papers),
            total_rejected      = total_rejected,
            years_inferred      = years_inferred,
            keywords_extracted  = keywords_extracted,
            html_stripped       = html_stripped,
            rejected_too_short  = rejected_too_short,
            clean_duration_secs = duration,
            output_path         = str(self.output_path),
        )
        diagnostic.log_summary()
        return diagnostic


# ── RE probe ──────────────────────────────────────────────────────

def diagnose_cleaned(filepath: Optional[str] = None) -> CleanDiagnostic:
    """
    Reverse engineering probe for the cleaning stage.

    WHY: Run this when chunker produces bad results.
    Inspects cleaned_papers.json without re-running cleaning.

    Usage:
        python -c "from preprocessing.cleaner import diagnose_cleaned; diagnose_cleaned()"
    """
    import time
    settings    = get_settings()
    output_path = settings.processed_data_path.parent / "cleaned_papers.json"
    path        = Path(filepath) if filepath else output_path

    if not path.exists():
        logger.error(f"[RE-CLEANER] File not found: {path}")
        raise FileNotFoundError(f"No cleaned JSON at {path}. Run cleaner first.")

    logger.info(f"[RE-CLEANER] Inspecting: {path}")

    with open(path, encoding="utf-8") as f:
        papers = json.load(f)

    years_inferred     = sum(1 for p in papers if p.get("year_inferred"))
    keywords_extracted = sum(1 for p in papers if p.get("keywords_extracted"))
    html_stripped      = sum(1 for p in papers if p.get("html_stripped"))

    diagnostic = CleanDiagnostic(
        total_input         = len(papers),
        total_output        = len(papers),
        total_rejected      = 0,
        years_inferred      = years_inferred,
        keywords_extracted  = keywords_extracted,
        html_stripped       = html_stripped,
        rejected_too_short  = 0,
        clean_duration_secs = 0.0,
        output_path         = str(path),
    )
    diagnostic.log_summary()
    return diagnostic