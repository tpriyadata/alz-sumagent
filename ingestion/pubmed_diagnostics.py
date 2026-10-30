"""
ingestion/pubmed_diagnostics.py
================================
RE probe for pubmed_fetch stage.
diagnose_raw_json() lives in pubmed_fetch.py directly.
This file is kept for backwards compatibility.
"""

from ingestion.pubmed_fetch import diagnose_raw_json

__all__ = ["diagnose_raw_json"]