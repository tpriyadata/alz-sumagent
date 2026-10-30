"""
scripts/deploy_space.py
========================
ALZDETECT-AI — Deploy Streamlit app to HuggingFace Space.

WHAT:   Pushes the project to oviya08/alz-bot HuggingFace Space.
WHY:    Makes the app publicly accessible — no local setup needed.
WHEN:   Run after any major update to app.py or streamlit_app.py.

Usage:
    $env:PYTHONPATH = "."
    python scripts/deploy_space.py

IMPORTANT:
    Set these secrets in HuggingFace Space Settings UI first:
        ANTHROPIC_API_KEY
        PINECONE_API_KEY
        PUBMED_EMAIL
        HF_TOKEN
"""

from loguru import logger
from huggingface_hub import HfApi
from pathlib import Path

from configs.settings import get_settings


# Files to upload to the Space
SPACE_FILES = [
    "app.py",
    "Dockerfile",
    "requirements-space.txt",
    "README.md",
    "generation/rag_pipeline.py",
    "generation/summarizer_agent.py",
    "generation/prompts.py",
    "generation/__init__.py",
    "vector_store/pinecone_store.py",
    "vector_store/embedder.py",
    "vector_store/__init__.py",
    "configs/settings.py",
    "configs/logging_setup.py",
    "configs/eval_store.py",
    "configs/__init__.py",
    "preprocessing/chunker.py",
    "preprocessing/cleaner.py",
    "preprocessing/__init__.py",
]


def deploy_space() -> None:
    settings = get_settings()
    api      = HfApi()
    repo_id  = settings.hf_space_name

    logger.info(f"[HF-SPACE] Deploying to {repo_id}")

    # ── Create Space if it doesn't exist ─────────────────────────
    try:
        api.create_repo(
            repo_id   = repo_id,
            repo_type = "space",
            space_sdk = "docker",
            token     = settings.hf_token,
            exist_ok  = True,
        )
        logger.info(f"[HF-SPACE] Space ready: {repo_id}")
    except Exception as e:
        logger.error(f"[HF-SPACE] Could not create space: {e}")
        raise

    # ── Upload files ──────────────────────────────────────────────
    root = Path(__file__).resolve().parent.parent

    for file_path in SPACE_FILES:
        full_path = root / file_path
        if not full_path.exists():
            logger.warning(f"[HF-SPACE] Skipping missing file: {file_path}")
            continue

        api.upload_file(
            path_or_fileobj = str(full_path),
            path_in_repo    = file_path,
            repo_id         = repo_id,
            repo_type       = "space",
            token           = settings.hf_token,
        )
        logger.info(f"[HF-SPACE] Uploaded: {file_path}")

    logger.info("[HF-SPACE] All files uploaded ✓")
    logger.info(f"[HF-SPACE] Space live at: https://huggingface.co/spaces/{repo_id}")
    logger.info("[HF-SPACE] Set secrets in Space Settings UI:")
    logger.info("  ANTHROPIC_API_KEY")
    logger.info("  PINECONE_API_KEY")
    logger.info("  PUBMED_EMAIL")
    logger.info("  HF_TOKEN")

    print(f"\n  Space live at: https://huggingface.co/spaces/{repo_id}")
    print("  Add secrets in: Space Settings → Variables and secrets\n")


if __name__ == "__main__":
    deploy_space()
