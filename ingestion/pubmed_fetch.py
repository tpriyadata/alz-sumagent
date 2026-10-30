"""
ingestion/pubmed_fetch.py
==========================

"""

import json 
import time ## for rate limiting and diagnostics
import hashlib ##  content-based deduplication
from pathlib import Path ## robust path handling
from typing import Optional ## for type hints
from datetime import datetime, UTC ## for timestamps


from Bio import Entrez ## Biopython for PubMed API
from pydantic import BaseModel, Field, field_validator, ValidationError
from loguru import logger  ## structured logging with levels and formatting
from tqdm import tqdm   ## progress bars for long-running fetches

from configs.settings import get_settings #
from ingestion.query_config import get_query_config, PubMedQuery



# ── Pydantic model — one validated PubMed paper ───────────────────
class PubMedPaper(BaseModel):
    """
    Contract for a single PubMed paper.

    Handles BOTH formats:
        - Old project format (authors/keywords as strings, year as string)
        - New fetch format  (authors/keywords as lists, year as int)

    Analogy: Patient admission form that accepts handwritten
    OR typed entries — both formats validated into clean standard form.

    WORST-CASE FIELDS:
        year         → Optional — may arrive as string "2025" or int 2025
        keywords     → may arrive as comma-separated string or list
        authors      → may arrive as comma-separated string or list
        source_query → old data has 'source' field, not 'source_query'
    """
    pmid:         str           = Field(..., min_length=1,
                                        description="PubMed unique ID") # UNIQUE identifier 
    title:        str           = Field(..., min_length=3,
                                        description="Paper title") ## for RAG, not just metadata
    abstract:     str           = Field(..., min_length=10,
                                        description="Paper abstract — required for RAG") ##
    authors:      list[str]     = Field(default_factory=list,
                                        description="Author names")
    year:         Optional[int] = Field(default=None,
                                        description="Publication year — may be missing")
    keywords:     list[str]     = Field(default_factory=list,
                                        description="MeSH/author keywords")
    journal:      Optional[str] = Field(default=None,
                                        description="Journal name")
    source_query: str           = Field(default="unknown",
                                        description="Which query retrieved this paper")
    fetched_at:   str           = Field(
                                    default_factory=lambda: datetime.now(UTC).isoformat(),
                                    description="UTC timestamp of fetch"
                                  )

    model_config = {"populate_by_name": True}

    @field_validator("pmid")
    @classmethod
    def pmid_must_be_numeric(cls, v: str) -> str:
        """PubMed IDs are always numeric strings."""
        v = str(v).strip()
        if not v.isdigit():
            raise ValueError(f"PMID '{v}' is not numeric — invalid PubMed ID")
        return v

    @field_validator("year", mode="before")
    @classmethod
    def coerce_year(cls, v) -> Optional[int]:
        """
        Worst-case: year comes as string "2025" from old project.
        Coerce to int safely — return None if unparseable.
        """
        if v is None:
            return None
        try:
            year = int(str(v).strip()[:4])
            if not (1900 <= year <= 2030):
                logger.warning(f"[PAPER] Suspicious year: {year} — setting None")
                return None
            return year
        except (ValueError, TypeError):
            return None

    @field_validator("authors", mode="before")
    @classmethod
    def coerce_authors(cls, v) -> list[str]:
        """
        Worst-case: authors arrives as comma-separated string
        from old project format — split into list cleanly.
        """
        if isinstance(v, str):
            return [a.strip() for a in v.split(",") if a.strip()]
        if isinstance(v, list):
            return [str(a).strip() for a in v if str(a).strip()]
        return []

    @field_validator("keywords", mode="before")
    @classmethod
    def coerce_keywords(cls, v) -> list[str]:
        """
        Worst-case: keywords arrives as comma-separated string
        from old project format — split into list cleanly.
        """
        if isinstance(v, str):
            return [k.strip() for k in v.split(",") if k.strip()]
        if isinstance(v, list):
            return [str(k).strip() for k in v if str(k).strip()]
        return []

    @field_validator("abstract")
    @classmethod
    def abstract_not_placeholder(cls, v: str) -> str:
        """
        Worst-case: PubMed returns '[No abstract available]'.
        These are useless for RAG — reject them at the door.
        """
        v = v.strip()
        if v.lower() in {"[no abstract available]", "n/a", "none", ""}:
            raise ValueError("Abstract is a placeholder — paper skipped")
        return v

    @field_validator("source_query", mode="before")
    @classmethod
    def coerce_source_query(cls, v) -> str:
        """
        Old project used 'source':'PubMed' not 'source_query'.
        Default to 'unknown' if missing or empty.
        """
        if not v or str(v).strip() == "":
            return "unknown"
        return str(v).strip()

    @field_validator("keywords")
    @classmethod
    def clean_keywords(cls, v: list[str]) -> list[str]:
        """Remove empty strings, strip whitespace, deduplicate."""
        return list(set(k.strip() for k in v if k.strip()))

    def to_dict(self) -> dict:
        """Serialize to dict for JSON storage."""
        return self.model_dump()

    @property
    def word_count(self) -> int:
        """Quick quality check — abstracts under 20 words are useless."""
        return len(self.abstract.split())

# ── Fetch diagnostics model ───────────────────────────────────────

class FetchDiagnostic(BaseModel):
    """
    RE inspector for the fetch stage.
    Generated after every run — shows exactly what happened.

    Analogy: The ambulance service daily report.
    How many patients picked up, rejected, or lost in transit.
    """
    total_fetched:       int
    total_valid:         int
    total_invalid:       int
    total_duplicates:    int
    missing_abstract:    int
    missing_year:        int
    missing_keywords:    int
    queries_completed:   int
    queries_failed:      int
    fetch_duration_secs: float
    output_path:         str

    @property
    def success_rate(self) -> float:
        if self.total_fetched == 0:
            return 0.0
        return round((self.total_valid / self.total_fetched) * 100, 2)

    def log_summary(self) -> None:
        """Print a clean summary to logs — readable by anyone."""
        logger.info("=" * 60)
        logger.info("[FETCH-DIAGNOSTIC] Run complete")
        logger.info(f"  Total fetched    : {self.total_fetched:,}")
        logger.info(f"  Valid papers     : {self.total_valid:,}")
        logger.info(f"  Invalid / skipped: {self.total_invalid:,}")
        logger.info(f"  Duplicates removed: {self.total_duplicates:,}")
        logger.info(f"  Missing abstract : {self.missing_abstract:,}")
        logger.info(f"  Missing year     : {self.missing_year:,}")
        logger.info(f"  Missing keywords : {self.missing_keywords:,}")
        logger.info(f"  Success rate     : {self.success_rate}%")
        logger.info(f"  Duration         : {self.fetch_duration_secs:.1f}s")
        logger.info(f"  Saved to         : {self.output_path}")
        logger.info("=" * 60)


# ── Core fetcher class ────────────────────────────────────────────

class PubMedFetcher:
    """
    Enterprise PubMed fetcher.

    Analogy: The ambulance service dispatcher.
    Knows all 40 pickup locations (queries), sends vehicles
    (API calls), validates patients (Pydantic), deduplicates,
    checkpoints, and reports.

    Usage:
        fetcher = PubMedFetcher()
        diagnostic = fetcher.run()
    """

    # NCBI requires max 3 requests/second for API key users
    # We use 0.34s delay = ~3 req/sec safely
    _REQUEST_DELAY_SECS: float = 0.34
    _MAX_RETRIES: int = 3
    _RETRY_BACKOFF_SECS: float = 2.0
    _CHECKPOINT_EVERY: int = 1000

    def __init__(self) -> None:
        self.settings   = get_settings()
        self.config     = get_query_config()
        self._setup_entrez()
        self._setup_paths()

    def _setup_entrez(self) -> None:
        """Configure Biopython Entrez — required before any API call."""
        Entrez.email = self.settings.pubmed_email
        # Use API key if available for higher rate limits
        api_key = getattr(self.settings, "entrez_api_key", None)
        if api_key:
            Entrez.api_key = api_key
            logger.info("[ENTREZ] API key configured — higher rate limits active")
        else:
            logger.warning(
                "[ENTREZ] No API key — limited to 3 req/sec. "
                "Add ENTREZ_API_KEY to .env for 10 req/sec."
            )

    def _setup_paths(self) -> None:
        """Ensure output directories exist — worst-case: they don't."""
        self.output_path     = self.settings.raw_data_path
        self.checkpoint_path = self.output_path.parent / "checkpoint_papers.json"
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"[FETCHER] Output path: {self.output_path}")

    def _fetch_pmids(self, query: PubMedQuery) -> list[str]:
        """
        Step 1: Get list of PMIDs matching a query.

        Worst-case handled:
            - Network timeout → retry with backoff
            - Empty results   → return [] and log warning
            - API error       → retry, then skip query
        """
        for attempt in range(1, self._MAX_RETRIES + 1):
            try:
                handle = Entrez.esearch(
                    db="pubmed",
                    term=query.search_term,
                    retmax=query.max_results,
                    sort="relevance",
                )
                record = Entrez.read(handle)
                handle.close()
                pmids = record.get("IdList", [])

                if not pmids:
                    logger.warning(
                        f"[FETCHER] Query '{query.name}' returned 0 PMIDs. "
                        f"Check search term: {query.search_term}"
                    )
                    return []

                logger.debug(
                    f"[FETCHER] Query '{query.name}' → {len(pmids)} PMIDs found"
                )
                time.sleep(self._REQUEST_DELAY_SECS)
                return pmids

            except Exception as e:
                logger.warning(
                    f"[FETCHER] Query '{query.name}' attempt {attempt}/{self._MAX_RETRIES} "
                    f"failed: {e}"
                )
                if attempt < self._MAX_RETRIES:
                    backoff = self._RETRY_BACKOFF_SECS * attempt
                    logger.info(f"[FETCHER] Retrying in {backoff}s...")
                    time.sleep(backoff)
                else:
                    logger.error(
                        f"[FETCHER] Query '{query.name}' failed after "
                        f"{self._MAX_RETRIES} attempts — skipping."
                    )
                    return []

    def _fetch_paper_details(self, pmids: list[str]) -> list[dict]:
        """
        Step 2: Fetch full paper details for a batch of PMIDs.

        Worst-case handled:
            - Batch fetch failure → retry with backoff
            - Partial results     → process what we got
        """
        for attempt in range(1, self._MAX_RETRIES + 1):
            try:
                handle = Entrez.efetch(
                    db="pubmed",
                    id=",".join(pmids),
                    rettype="xml",
                    retmode="xml",
                )
                records = Entrez.read(handle)
                handle.close()
                time.sleep(self._REQUEST_DELAY_SECS)
                return records.get("PubmedArticle", [])

            except Exception as e:
                logger.warning(
                    f"[FETCHER] Batch fetch attempt {attempt}/{self._MAX_RETRIES} "
                    f"failed: {e}"
                )
                if attempt < self._MAX_RETRIES:
                    backoff = self._RETRY_BACKOFF_SECS * attempt
                    time.sleep(backoff)
                else:
                    logger.error("[FETCHER] Batch fetch failed — skipping batch")
                    return []

    def _parse_paper(
        self,
        raw: dict,
        source_query: str
    ) -> Optional[PubMedPaper]:
        """
        Step 3: Parse one raw PubMed XML record → PubMedPaper.

        Worst-case handled:
            - Missing any field → use safe default
            - Pydantic rejects  → return None, caller skips
            - Abstract too short → Pydantic catches it
        """
        try:
            article    = raw["MedlineCitation"]["Article"]
            pmid       = str(raw["MedlineCitation"]["PMID"])
            title      = str(article.get("ArticleTitle", "")).strip()

            # Abstract — may be structured (list) or plain string
            abstract_raw = article.get("Abstract", {}).get("AbstractText", "")
            if isinstance(abstract_raw, list):
                abstract = " ".join(str(a) for a in abstract_raw).strip()
            else:
                abstract = str(abstract_raw).strip()

            # Authors
            authors = []
            for author in article.get("AuthorList", []):
                last  = author.get("LastName", "")
                first = author.get("ForeName", "")
                name  = f"{last}, {first}".strip(", ")
                if name:
                    authors.append(name)

            # Year — nested path in XML
            pub_date = (
                article
                .get("Journal", {})
                .get("JournalIssue", {})
                .get("PubDate", {})
            )
            year_str = pub_date.get("Year", pub_date.get("MedlineDate", ""))[:4]
            year     = int(year_str) if year_str.isdigit() else None

            # Keywords — MeSH + author keywords
            keywords = []
            mesh_list = raw["MedlineCitation"].get("MeshHeadingList", [])
            for mesh in mesh_list:
                descriptor = mesh.get("DescriptorName", "")
                if descriptor:
                    keywords.append(str(descriptor))

            kw_list = raw["MedlineCitation"].get("KeywordList", [])
            for kw_group in kw_list:
                for kw in kw_group:
                    if str(kw).strip():
                        keywords.append(str(kw).strip())

            # Journal
            journal = str(
                article.get("Journal", {}).get("Title", "")
            ).strip() or None

            # Validate through Pydantic — fails fast if invalid
            paper = PubMedPaper(
                pmid         = pmid,
                title        = title,
                abstract     = abstract,
                authors      = authors,
                year         = year,
                keywords     = list(set(keywords)),   # dedupe keywords
                journal      = journal,
                source_query = source_query,
            )

            # Extra quality check — skip very short abstracts
            if paper.word_count < 20:
                logger.debug(
                    f"[PARSER] PMID {pmid} abstract too short "
                    f"({paper.word_count} words) — skipping"
                )
                return None

            return paper

        except ValidationError as e:
            logger.debug(f"[PARSER] Pydantic rejected PMID {pmid}: {e}")
            return None
        except Exception as e:
            logger.warning(f"[PARSER] Unexpected parse error: {e}")
            return None

    def _load_checkpoint(self) -> tuple[list[dict], set[str]]:
        """
        Load existing checkpoint if it exists.
        Allows resuming a crashed fetch run.

        Returns:
            papers    : list of already-fetched paper dicts
            seen_pmids: set of PMIDs already collected
        """
        if self.checkpoint_path.exists():
            try:
                with open(self.checkpoint_path, encoding="utf-8") as f:
                    papers = json.load(f)
                seen = {p["pmid"] for p in papers}
                logger.info(
                    f"[FETCHER] Checkpoint found — resuming from "
                    f"{len(papers):,} papers already fetched"
                )
                return papers, seen
            except Exception as e:
                logger.warning(f"[FETCHER] Checkpoint corrupt — starting fresh: {e}")

        logger.info("[FETCHER] No checkpoint found — starting fresh")
        return [], set()

    def _save_checkpoint(self, papers: list[dict]) -> None:
        """Save progress checkpoint — called every 1,000 papers."""
        try:
            with open(self.checkpoint_path, "w", encoding="utf-8") as f:
                json.dump(papers, f, ensure_ascii=False, indent=2)
            logger.debug(f"[FETCHER] Checkpoint saved: {len(papers):,} papers")
        except Exception as e:
            logger.error(f"[FETCHER] Checkpoint save failed: {e}")

    def run(self) -> FetchDiagnostic:
        """
        Main entry point — runs the full fetch pipeline.

        Flow:
            1. Load checkpoint (resume if crashed)
            2. For each query → get PMIDs
            3. Batch fetch paper details
            4. Parse + validate each paper
            5. Deduplicate by PMID
            6. Checkpoint every 1,000 papers
            7. Save final JSON
            8. Return FetchDiagnostic

        Returns:
            FetchDiagnostic with full run statistics
        """
        start_time = time.time()
        logger.info("[FETCHER] Starting enterprise PubMed fetch")
        logger.info(
            f"[FETCHER] Target: {self.settings.target_papers:,} papers | "
            f"Queries: {self.config.total_queries}"
        )

        # Counters
        total_fetched    = 0
        total_invalid    = 0
        total_duplicates = 0
        missing_abstract = 0
        queries_failed   = 0

        # Load checkpoint
        papers, seen_pmids = self._load_checkpoint()

        # Process each query
        for query in tqdm(self.config.queries, desc="Queries", unit="query"):

            # Check if target already reached
            if len(papers) >= self.settings.target_papers:
                logger.info(
                    f"[FETCHER] Target {self.settings.target_papers:,} reached "
                    f"— stopping early"
                )
                break

            logger.info(f"[FETCHER] Processing query: {query.name}")

            # Step 1 — get PMIDs
            pmids = self._fetch_pmids(query)
            if not pmids:
                queries_failed += 1
                continue

            # Filter already-seen PMIDs
            new_pmids = [p for p in pmids if p not in seen_pmids]
            total_duplicates += len(pmids) - len(new_pmids)

            if not new_pmids:
                logger.info(
                    f"[FETCHER] Query '{query.name}' — all {len(pmids)} "
                    f"PMIDs already fetched"
                )
                continue

            # Step 2 — fetch in batches of 100
            batch_size = 100
            for i in range(0, len(new_pmids), batch_size):
                batch = new_pmids[i : i + batch_size]
                raw_records = self._fetch_paper_details(batch)
                total_fetched += len(raw_records)

                for raw in raw_records:
                    paper = self._parse_paper(raw, source_query=query.name)

                    if paper is None:
                        total_invalid += 1
                        missing_abstract += 1
                        continue

                    if paper.pmid in seen_pmids:
                        total_duplicates += 1
                        continue

                    # Valid new paper
                    seen_pmids.add(paper.pmid)
                    papers.append(paper.to_dict())

                    # Checkpoint every N papers
                    if len(papers) % self._CHECKPOINT_EVERY == 0:
                        self._save_checkpoint(papers)
                        logger.info(
                            f"[FETCHER] Progress: {len(papers):,} / "
                            f"{self.settings.target_papers:,} papers"
                        )

        # Save final output
        try:
            with open(self.output_path, "w", encoding="utf-8") as f:
                json.dump(papers, f, ensure_ascii=False, indent=2)
            logger.info(
                f"[FETCHER] Final save: {len(papers):,} papers → {self.output_path}"
            )
        except Exception as e:
            logger.error(f"[FETCHER] FATAL — could not save output: {e}")
            raise

        # Build diagnostic
        duration = round(time.time() - start_time, 1)
        diagnostic = FetchDiagnostic(
            total_fetched       = total_fetched,
            total_valid         = len(papers),
            total_invalid       = total_invalid,
            total_duplicates    = total_duplicates,
            missing_abstract    = missing_abstract,
            missing_year        = sum(1 for p in papers if not p.get("year")),
            missing_keywords    = sum(1 for p in papers if not p.get("keywords")),
            queries_completed   = self.config.total_queries - queries_failed,
            queries_failed      = queries_failed,
            fetch_duration_secs = duration,
            output_path         = str(self.output_path),
        )
        diagnostic.log_summary()
        return diagnostic


# ── RE probe — diagnose existing JSON without re-fetching ─────────

def diagnose_raw_json(filepath: Optional[str] = None) -> FetchDiagnostic:
    """
    Reverse engineering probe for the fetch stage.

    WHY: When answers feel wrong, run this FIRST.
    It inspects what actually entered the pipeline
    without touching PubMed API at all.

    Usage:
        python -c "from ingestion.pubmed_fetch import diagnose_raw_json; diagnose_raw_json()"
    """
    settings = get_settings()
    path     = Path(filepath) if filepath else settings.raw_data_path

    if not path.exists():
        logger.error(f"[RE-PUBMED] File not found: {path}")
        raise FileNotFoundError(f"No raw JSON at {path}. Run fetcher first.")

    logger.info(f"[RE-PUBMED] Inspecting: {path}")

    with open(path, encoding="utf-8") as f:
        raw_papers = json.load(f)

    valid, invalid = 0, 0
    missing_abstract = missing_year = missing_keywords = 0
    pmids = []

    for raw in tqdm(raw_papers, desc="Validating", unit="paper"):
        try:
            paper = PubMedPaper(**raw)
            valid += 1
            pmids.append(paper.pmid)
            if not paper.year:
                missing_year += 1
            if not paper.keywords:
                missing_keywords += 1
        except ValidationError:
            invalid += 1
            missing_abstract += 1

    duplicate_pmids = len(pmids) - len(set(pmids))

    diagnostic = FetchDiagnostic(
        total_fetched       = len(raw_papers),
        total_valid         = valid,
        total_invalid       = invalid,
        total_duplicates    = duplicate_pmids,
        missing_abstract    = missing_abstract,
        missing_year        = missing_year,
        missing_keywords    = missing_keywords,
        queries_completed   = 0,
        queries_failed      = 0,
        fetch_duration_secs = 0.0,
        output_path         = str(path),
    )
    diagnostic.log_summary()
    return diagnostic