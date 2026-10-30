"""
vector_store/embedder.py
=========================

WORST-CASE DESIGN:
    - Upsert timeout mid-run  → checkpoint every 1,000 vectors
    - Pinecone upsert fail    → retry 3 times with backoff
    - Wrong embedding dim     → caught before first upsert
    - Chunk text empty        → skipped, logged, counted
"""

import json
import time
import os
from pathlib import Path
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from loguru import logger
from tqdm import tqdm

from configs.settings import get_settings


# ── Embed diagnostic model ────────────────────────────────────────

class EmbedDiagnostic(BaseModel):
    """
    RE inspector for embedding stage.

    
    """
    total_chunks:        int
    total_embedded:      int
    total_upserted:      int
    total_failed:        int
    embedding_model:     str
    embedding_dim:       int
    pinecone_index:      str
    duration_secs:       float
    vectors_in_index:    int

    def log_summary(self) -> None:
        logger.info("=" * 60)
        logger.info("[EMBED-DIAGNOSTIC] Run complete")
        logger.info(f"  Total chunks      : {self.total_chunks:,}")
        logger.info(f"  Embedded          : {self.total_embedded:,}")
        logger.info(f"  Upserted          : {self.total_upserted:,}")
        logger.info(f"  Failed            : {self.total_failed:,}")
        logger.info(f"  Model             : {self.embedding_model}")
        logger.info(f"  Dimensions        : {self.embedding_dim}")
        logger.info(f"  Pinecone index    : {self.pinecone_index}")
        logger.info(f"  Vectors in index  : {self.vectors_in_index:,}")
        logger.info(f"  Duration          : {self.duration_secs:.1f}s")