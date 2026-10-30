"""
scripts/push_summarizer.py
===========================
ALZDETECT-AI — Push BART summarizer to HuggingFace Hub.

WHAT:   Wraps facebook/bart-large-cnn in a pipeline and pushes
        to oviya08/Alz-SumAgent on HuggingFace Hub.
WHY:    Makes the summarizer reusable, versioned, and publicly
        accessible — anyone can pull it with one line.
WHEN:   Run once to publish. Re-run after fine-tuning.

Usage:
    $env:PYTHONPATH = "."
    python scripts/push_summarizer.py
"""

import sys
from pathlib import Path
from loguru import logger
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from huggingface_hub import HfApi

from configs.settings import get_settings


def push_summarizer() -> None:
    settings = get_settings()

    logger.info("[HF-PUSH] Loading BART summarizer")
    logger.info(f"[HF-PUSH] Source model : {settings.summarizer_model}")
    logger.info(f"[HF-PUSH] Target repo  : {settings.hf_model_repo}")

    # ── Load model + tokenizer ────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(settings.summarizer_model)
    model     = AutoModelForSeq2SeqLM.from_pretrained(
        settings.summarizer_model
    )

    logger.info("[HF-PUSH] Model loaded — running test inference")

    # ── Test inference before pushing ────────────────────────────
    test_text = (
        "Plasma phosphorylated tau217 (pTau217) is a strong early biomarker "
        "for Alzheimer's disease. Studies show 96% sensitivity and 97% "
        "specificity for detecting amyloid pathology up to 10 years before "
        "symptom onset. Blood-based pTau217 assays outperform CSF biomarkers "
        "in ease of collection and patient acceptance."
    )
    inputs  = tokenizer(test_text, return_tensors="pt",
                        max_length=1024, truncation=True)
    outputs = model.generate(
        **inputs,
        max_length = settings.summarizer_max_length,
        min_length = settings.summarizer_min_length,
        do_sample  = False,
    )
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.info(f"[HF-PUSH] Test summary : {summary}")
    logger.info("[HF-PUSH] Test inference passed ✓")

    # ── Push to HuggingFace Hub ───────────────────────────────────
    logger.info(f"[HF-PUSH] Pushing to {settings.hf_model_repo}")

    model.push_to_hub(
        settings.hf_model_repo,
        token = settings.hf_token,
    )
    tokenizer.push_to_hub(
        settings.hf_model_repo,
        token = settings.hf_token,
    )

    # ── Verify push ───────────────────────────────────────────────
    api  = HfApi()
    info = api.model_info(settings.hf_model_repo, token=settings.hf_token)
    logger.info(f"[HF-PUSH] Verified on Hub: {info.id}")
    logger.info(f"[HF-PUSH] Model URL: https://huggingface.co/{settings.hf_model_repo}")
    logger.info("[HF-PUSH] Summarizer deployment complete ✓")

    print(f"\n  Model live at: https://huggingface.co/{settings.hf_model_repo}\n")


if __name__ == "__main__":
    push_summarizer()
