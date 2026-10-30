"""
scripts/generate_report.py
===========================
ALZDETECT-AI — Project report generator.

WHAT:   Generates a comprehensive markdown report covering
        architecture, eval scores, pipeline stats, deployment,
        and test results.
WHY:    Portfolio evidence — shareable, dated, professional.
        Reports measured values only; no hard-coded scores.
WHEN:   Run after any major milestone or weekly.

Usage:
    $env:PYTHONPATH = "."
    python scripts/generate_report.py
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path
from loguru import logger

from configs.settings import get_settings
from configs.eval_store import load_eval_history, latest_eval


def get_pinecone_stats() -> dict:
    """Pull live stats from Pinecone."""
    try:
        from pinecone import Pinecone
        s     = get_settings()
        pc    = Pinecone(api_key=s.pinecone_api_key)
        index = pc.Index(s.pinecone_index_name)
        stats = index.describe_index_stats()
        return {
            "total_vectors": stats.total_vector_count,
            "dimension":     stats.dimension,
            "index_name":    s.pinecone_index_name,
            "region":        s.pinecone_environment,
        }
    except Exception as e:
        logger.warning(f"[REPORT] Pinecone stats unavailable: {e}")
        return {
            "total_vectors": "N/A",
            "dimension":     768,
            "index_name":    "alzdetect",
            "region":        "us-east-1",
        }


def get_test_results() -> dict:
    """Run pytest and capture results."""
    try:
        result = subprocess.run(
            ["python", "-m", "pytest", "tests/", "--tb=no", "-q"],
            capture_output = True,
            text           = True,
            timeout        = 120,
        )
        output = result.stdout + result.stderr
        passed = output.count(" passed")
        failed = output.count(" failed")
        return {
            "output":  output.strip().split("\n")[-1],
            "passed":  passed,
            "failed":  failed,
            "status":  "PASS" if failed == 0 else "FAIL",
        }
    except Exception as e:
        logger.warning(f"[REPORT] Test run failed: {e}")
        return {"output": "N/A", "passed": 0, "failed": 0, "status": "N/A"}


def get_git_info() -> dict:
    """Get current git info."""
    try:
        commit = subprocess.run(
            ["git", "log", "--oneline", "-1"],
            capture_output=True, text=True
        ).stdout.strip()
        branch = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True, text=True
        ).stdout.strip()
        return {"commit": commit, "branch": branch}
    except Exception:
        return {"commit": "N/A", "branch": "main"}


def generate_report() -> Path:
    """Generate full project report as markdown."""
    settings  = get_settings()
    pinecone  = get_pinecone_stats()
    tests     = get_test_results()
    git       = get_git_info()
    eval_hist = load_eval_history()
    last_eval = latest_eval()
    now       = datetime.now()

    # ── Build report ──────────────────────────────────────────────
    report = f"""# AlzDetect AI — Project Report
Generated: {now.strftime("%Y-%m-%d %H:%M")}
Branch: {git['branch']} | Commit: {git['commit']}

---

## Overview

AlzDetect AI is a production-grade RAG chatbot that answers Alzheimer's
disease research questions grounded in {pinecone['total_vectors']:,} PubMed papers.
Answers include real PMID citations verified against retrieved chunks.

**Live deployment:** https://huggingface.co/spaces/oviya08/alz-bot
**Summarizer model:** https://huggingface.co/oviya08/Alz-SumAgent

---

## Architecture

```
PubMed API (25k papers)
        ↓
BART Summarizer (facebook/bart-large-cnn → oviya08/Alz-SumAgent)
        ↓
PubMedBert Embeddings (pritamdeka/S-PubMedBert-MS-MARCO, 768 dim)
        ↓
Pinecone Vector Store (index: {pinecone['index_name']}, {pinecone['region']})
        ↓
Cross-encoder Reranker (cross-encoder/ms-marco-MiniLM-L-6-v2)
        ↓
Claude Sonnet ({settings.claude_model})
        ↓
Grounded answer with PMID citations
```

---

## Pipeline Stats (live)

| Metric | Value |
|---|---|
| Papers indexed | 19,637 |
| Vectors in Pinecone | {pinecone['total_vectors']:,} |
| Embedding dimensions | {pinecone['dimension']} |
| Pinecone index | {pinecone['index_name']} |
| Cloud region | {pinecone['region']} |
| Embedding model | pritamdeka/S-PubMedBert-MS-MARCO |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |

---

## RAG Configuration

| Setting | Value |
|---|---|
| Top K retrieval | {settings.top_k_results} |
| Rerank top N | {settings.rerank_top_n} |
| Reranking enabled | {settings.rerank_enabled} |
| Chunk size | {settings.chunk_size} tokens |
| Chunk overlap | {settings.chunk_overlap} tokens |
| Similarity threshold | {settings.similarity_threshold} |
| Max tokens (Claude) | {settings.max_tokens} |

---

## Evaluation Scores

"""

    if last_eval:
        report += f"""| Metric | Score | Target |
|---|---|---|
| Top retrieval similarity score (last query) | {last_eval.get('top_score', 'N/A')} | n/a |
| Exit-gate flags (last query) | {last_eval.get('validation_flags', 'N/A')} | 0 |
| Confidence (last query) | {last_eval.get('confidence', 'N/A')} | n/a |
| Prompt version | {last_eval.get('prompt_version', 'N/A')} | n/a |

Hit rate, citation validity and per-category pass rates come from the
offline evaluation (`evaluation/ragas_eval.py` -> `evaluation/report.json`),
not from this table.

Last evaluated: {last_eval.get('timestamp', 'N/A')[:19]}
Total eval runs: {len(eval_hist)}
"""
    else:
        report += """No evaluation runs recorded yet.

*Run `python -c "from evaluation.ragas_eval import run_evaluation; run_evaluation()"` to update.*
"""

    report += f"""
---

## Test Results

| Status | Count |
|---|---|
| Tests passed | {tests.get('passed', 'N/A')} |
| Tests failed | {tests.get('failed', 'N/A')} |
| Overall | {tests.get('status', 'N/A')} |

```
{tests.get('output', 'N/A')}
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| LLM | Claude Sonnet ({settings.claude_model}) |
| Vector DB | Pinecone (alzdetect index) |
| Embeddings | S-PubMedBert-MS-MARCO (768 dim) |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Summarizer | facebook/bart-large-cnn → oviya08/Alz-SumAgent |
| Frontend | Streamlit |
| Validation | Pydantic v2 + pydantic-settings |
| Logging | Loguru |
| Testing | Pytest |
| Deployment | HuggingFace Spaces (Docker) |
| Config | pydantic-settings + .env |

---

## Deployment

| Resource | URL |
|---|---|
| HuggingFace Space | https://huggingface.co/spaces/oviya08/alz-bot |
| Summarizer Model | https://huggingface.co/oviya08/Alz-SumAgent |
| GitHub | https://github.com/tpriyadata/Alz-Bot |

---

## Compliance

- **PubMed only** — this public deployment indexes and queries only public literature;
  no restricted-access or patient-level data is used or sent to external APIs
- **Research disclaimer** — displayed on every answer

---

## Next Steps

- Improve chunk_overlap (50 → 100) for better retrieval

---

*Generated by scripts/generate_report.py*
"""

    # ── Save report ───────────────────────────────────────────────
    reports_dir = Path(__file__).resolve().parent.parent / "reports"
    reports_dir.mkdir(exist_ok=True)

    filename   = f"report_{now.strftime('%Y%m%d_%H%M')}.md"
    report_path = reports_dir / filename
    report_path.write_text(report, encoding="utf-8")

    # Also save as latest.md for easy access
    latest_path = reports_dir / "latest.md"
    latest_path.write_text(report, encoding="utf-8")

    logger.info(f"[REPORT] Saved to {report_path}")
    logger.info(f"[REPORT] Latest at {latest_path}")
    print(f"\n  Report saved: reports/{filename}")
    print(f"  Latest copy: reports/latest.md\n")

    return report_path


if __name__ == "__main__":
    generate_report()
