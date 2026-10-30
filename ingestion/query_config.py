"""
ingestion/query_config.py
==========================

WORST-CASE DESIGN:
    - Duplicate query names     → caught at startup by validator
    - Empty search term         → caught by Pydantic field constraint
    - Zero max_results          → caught by Pydantic field constraint
    - Wrong type                → caught by Pydantic at import time

QUERY CATEGORIES:
    - Core Alzheimer's          : 8 queries
    - Additional Alzheimer's    : 7 queries
    - Diabetes Connection       : 5 queries
    - Treatment                 : 5 queries
    - Risk Factors              : 3 queries
    - Related Diseases          : 3 queries
    - Technology / AI           : 2 queries
    - Additional / Emerging     : 7 queries
    ─────────────────────────────────────
    TOTAL                       : 40 queries
    MAX POSSIBLE PAPERS         : 40,000 (before deduplication)
"""

from pydantic import BaseModel, Field, field_validator
from loguru import logger


# ── Single query model ────────────────────────────────────────────

class PubMedQuery(BaseModel):
    """
    One PubMed search query — validated at definition time.

    Analogy: One row in a typed purchase order.
    Has an item code (name), description (search_term),
    and quantity limit (max_results).
    """
    name:        str = Field(..., min_length=3,
                             description="Unique query identifier — no spaces")
    search_term: str = Field(..., min_length=5,
                             description="PubMed search string")
    max_results: int = Field(default=1000, ge=100, le=10000,
                             description="Max papers to fetch for this query")

    @field_validator("name")
    @classmethod
    def name_no_spaces(cls, v: str) -> str:
        """Query names must use underscores — makes logging readable."""
        v = v.strip()
        if " " in v:
            raise ValueError(
                f"Query name '{v}' contains spaces. Use underscores. "
                f"Example: 'alzheimer_early_detection'"
            )
        return v

    @field_validator("search_term")
    @classmethod
    def search_term_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("search_term cannot be empty or whitespace")
        return v


# ── Full query collection model ───────────────────────────────────

class QueryConfig(BaseModel):
    """
    Full validated collection of all PubMed queries.

    Analogy: The complete typed purchase order for the entire
    data collection run. All 40 items validated together —
    no duplicates, no empty rows allowed.
    """
    queries: list[PubMedQuery] = Field(..., min_length=1)

    @field_validator("queries")
    @classmethod
    def no_duplicate_names(cls, queries: list[PubMedQuery]) -> list[PubMedQuery]:
        """
        Worst-case: someone adds same query name twice.
        Caught here at startup — not discovered after 2 hours of fetching.
        """
        names = [q.name for q in queries]
        duplicates = [n for n in names if names.count(n) > 1]
        if duplicates:
            raise ValueError(
                f"Duplicate query names found: {set(duplicates)}. "
                f"Every query must have a unique name."
            )
        return queries

    @field_validator("queries")
    @classmethod
    def no_duplicate_search_terms(cls, queries: list[PubMedQuery]) -> list[PubMedQuery]:
        """
        Worst-case: same search term with different names.
        Would waste API calls fetching identical papers twice.
        """
        terms = [q.search_term.lower() for q in queries]
        duplicates = [t for t in terms if terms.count(t) > 1]
        if duplicates:
            raise ValueError(
                f"Duplicate search terms found: {set(duplicates)}. "
                f"Remove or rephrase duplicate queries."
            )
        return queries

    @property
    def total_queries(self) -> int:
        """How many queries are configured."""
        return len(self.queries)

    @property
    def max_possible_papers(self) -> int:
        """
        Theoretical maximum before deduplication.
        Real yield will be lower due to overlapping papers.
        """
        return sum(q.max_results for q in self.queries)

    def get_by_name(self, name: str) -> PubMedQuery | None:
        """
        Fetch a specific query by name.
        Useful for RE probes — test one query without running all 40.
        """
        for q in self.queries:
            if q.name == name:
                return q
        return None

    def get_category(self, prefix: str) -> list[PubMedQuery]:
        """
        Fetch all queries in a category by name prefix.
        Example: get_category('diabetes') → all diabetes queries.
        """
        return [q for q in self.queries if q.name.startswith(prefix)]


# ── 40 Enterprise Queries — your original + enterprise structure ──

ALZHEIMER_QUERIES = QueryConfig(
    queries=[

        # ── Core Alzheimer's (8) ──────────────────────────────────
        PubMedQuery(
            name="alzheimer_early_detection_biomarkers",
            search_term="Alzheimer's early detection biomarkers[Title/Abstract]",
        ),
        PubMedQuery(
            name="plasma_ptau217_diagnosis",
            search_term="plasma p-tau217 Alzheimer's diagnosis[Title/Abstract]",
        ),
        PubMedQuery(
            name="diabetes_alzheimer_connection",
            search_term="diabetes Alzheimer's connection mechanisms[Title/Abstract]",
        ),
        PubMedQuery(
            name="blood_brain_barrier_drug",
            search_term="blood-brain barrier Alzheimer's drug delivery[Title/Abstract]",
        ),
        PubMedQuery(
            name="amyloid_beta_ratio",
            search_term="amyloid beta 42 40 ratio Alzheimer's[Title/Abstract]",
        ),
        PubMedQuery(
            name="plasma_gfap_early",
            search_term="plasma GFAP Alzheimer's early detection[Title/Abstract]",
        ),
        PubMedQuery(
            name="insulin_resistance_neurodegeneration",
            search_term="insulin resistance Alzheimer's neurodegeneration[Title/Abstract]",
        ),
        PubMedQuery(
            name="alzheimer_biomarker_sensitivity",
            search_term="Alzheimer's biomarker sensitivity specificity[Title/Abstract]",
        ),

        # ── Additional Alzheimer's (7) ────────────────────────────
        PubMedQuery(
            name="tau_protein_biomarker",
            search_term="tau protein Alzheimer's biomarker[Title/Abstract]",
        ),
        PubMedQuery(
            name="gfap_neurofilament_blood",
            search_term="GFAP neurofilament Alzheimer's blood[Title/Abstract]",
        ),
        PubMedQuery(
            name="mri_neuroimaging_biomarker",
            search_term="Alzheimer's MRI neuroimaging biomarker[Title/Abstract]",
        ),
        PubMedQuery(
            name="cognitive_decline_early",
            search_term="cognitive decline early detection[Title/Abstract]",
        ),
        PubMedQuery(
            name="neurofilament_light_dementia",
            search_term="neurofilament light chain NfL dementia[Title/Abstract]",
        ),
        PubMedQuery(
            name="csf_biomarker_alzheimer",
            search_term="Alzheimer's cerebrospinal fluid biomarker[Title/Abstract]",
        ),
        PubMedQuery(
            name="pet_amyloid_detection",
            search_term="PET scan amyloid Alzheimer's detection[Title/Abstract]",
        ),

        # ── Diabetes Connection (5) ───────────────────────────────
        PubMedQuery(
            name="diabetes_cognitive_decline",
            search_term="type 2 diabetes cognitive decline[Title/Abstract]",
        ),
        PubMedQuery(
            name="metabolic_syndrome_alzheimer",
            search_term="metabolic syndrome Alzheimer's risk[Title/Abstract]",
        ),
        PubMedQuery(
            name="glucose_metabolism_brain",
            search_term="glucose metabolism brain Alzheimer's[Title/Abstract]",
        ),
        PubMedQuery(
            name="glp1_receptor_brain",
            search_term="GLP1 receptor brain Alzheimer's[Title/Abstract]",
        ),
        PubMedQuery(
            name="metformin_prevention",
            search_term="metformin Alzheimer's prevention[Title/Abstract]",
        ),

        # ── Treatment (5) ─────────────────────────────────────────
        PubMedQuery(
            name="lecanemab_treatment",
            search_term="Lecanemab Alzheimer's treatment[Title/Abstract]",
        ),
        PubMedQuery(
            name="donanemab_amyloid",
            search_term="Donanemab amyloid clearance[Title/Abstract]",
        ),
        PubMedQuery(
            name="drug_therapy_trial",
            search_term="Alzheimer's drug therapy clinical trial[Title/Abstract]",
        ),
        PubMedQuery(
            name="anti_amyloid_side_effects",
            search_term="anti-amyloid therapy side effects ARIA[Title/Abstract]",
        ),
        PubMedQuery(
            name="immunotherapy_aria",
            search_term="Alzheimer's immunotherapy ARIA treatment[Title/Abstract]",
        ),

        # ── Risk Factors (3) ──────────────────────────────────────
        PubMedQuery(
            name="apoe4_genetic_risk",
            search_term="APOE4 Alzheimer's genetic risk[Title/Abstract]",
        ),
        PubMedQuery(
            name="sleep_deprivation_risk",
            search_term="sleep deprivation Alzheimer's risk[Title/Abstract]",
        ),
        PubMedQuery(
            name="neuroinflammation_alzheimer",
            search_term="neuroinflammation Alzheimer's pathology[Title/Abstract]",
        ),

        # ── Related Diseases (3) ──────────────────────────────────
        PubMedQuery(
            name="vascular_dementia_biomarkers",
            search_term="vascular dementia biomarkers diagnosis[Title/Abstract]",
        ),
        PubMedQuery(
            name="mci_progression",
            search_term="mild cognitive impairment progression Alzheimer[Title/Abstract]",
        ),
        PubMedQuery(
            name="parkinsons_biomarkers",
            search_term="Parkinson's disease biomarkers detection[Title/Abstract]",
        ),

        # ── Technology / AI (2) ───────────────────────────────────
        PubMedQuery(
            name="ai_ml_detection",
            search_term="AI machine learning Alzheimer's detection[Title/Abstract]",
        ),
        PubMedQuery(
            name="deep_learning_brain_mri",
            search_term="deep learning brain MRI Alzheimer's[Title/Abstract]",
        ),

        # ── Emerging / Additional (7) ─────────────────────────────
        PubMedQuery(
            name="apoe2_apoe3_protective",
            search_term="APOE2 APOE3 Alzheimer's protective allele[Title/Abstract]",
        ),
        PubMedQuery(
            name="gut_microbiome_dementia",
            search_term="gut microbiome Alzheimer's dementia[Title/Abstract]",
        ),
        PubMedQuery(
            name="exercise_dementia_prevention",
            search_term="exercise physical activity dementia prevention[Title/Abstract]",
        ),
        PubMedQuery(
            name="mediterranean_diet_cognition",
            search_term="Mediterranean diet cognitive decline prevention[Title/Abstract]",
        ),
        PubMedQuery(
            name="early_onset_familial",
            search_term="early onset familial Alzheimer's PSEN1 PSEN2[Title/Abstract]",
        ),
        PubMedQuery(
            name="social_isolation_dementia",
            search_term="loneliness social isolation dementia risk[Title/Abstract]",
        ),
        PubMedQuery(
            name="air_pollution_neurodegeneration",
            search_term="air pollution neurodegeneration brain Alzheimer[Title/Abstract]",
        ),
    ]
)


# ── Factory function ──────────────────────────────────────────────

def get_query_config() -> QueryConfig:
    """
    Factory function — always use this to get queries.
    Never reference ALZHEIMER_QUERIES directly in other modules.

    Usage:
        from ingestion.query_config import get_query_config
        config = get_query_config()
        for query in config.queries:
            fetch(query.search_term, query.max_results)
    """
    try:
        logger.info(
            f"[QUERY-CONFIG] Loaded {ALZHEIMER_QUERIES.total_queries} queries | "
            f"max possible papers before dedup: "
            f"{ALZHEIMER_QUERIES.max_possible_papers:,}"
        )
        return ALZHEIMER_QUERIES
    except Exception as e:
        logger.error(f"[QUERY-CONFIG] FATAL — query config invalid: {e}")
        raise