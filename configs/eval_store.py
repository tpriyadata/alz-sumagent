""" Evalstore"""

import json
from datetime import datetime
from loguru import logger
from configs.settings import get_settings


def save_eval_results(metrics: dict) -> None:
    """
    Appends eval metrics to the results JSON file.
    Creates the file if it does not exist.
    """
    settings = get_settings()
    settings.eval_results_path.parent.mkdir(parents=True, exist_ok=True)

    history = load_eval_history()

    entry = {
        "timestamp":      datetime.now().isoformat(),
        "vector_db":      settings.vector_db,
        "top_k_results":  settings.top_k_results,
        "rerank_top_n":   settings.rerank_top_n,
        "chunk_size":     settings.chunk_size,
        "chunk_overlap":  settings.chunk_overlap,
        **metrics,
    }

    history.append(entry)
    settings.eval_results_path.write_text(
        json.dumps(history, indent=2)
    )

    logger.info(
        "[EVAL] Results saved | "
        f"hit_rate={metrics.get('hit_rate', 'n/a')} | "
        f"faithfulness={metrics.get('faithfulness', 'n/a')} | "
        f"coverage={metrics.get('coverage', 'n/a')} | "
        f"total_runs={len(history)}"
    )


def load_eval_history() -> list[dict]:
    """
    Returns all saved eval runs, oldest first.
    Returns empty list if no results saved yet.
    """
    settings = get_settings()

    if not settings.eval_results_path.exists():
        return []

    try:
        return json.loads(settings.eval_results_path.read_text())
    except json.JSONDecodeError:
        logger.warning("[EVAL] Results file corrupted — starting fresh")
        return []


def latest_eval() -> dict | None:
    """
    Returns the most recent eval run, or None if no history.
    """
    history = load_eval_history()
    return history[-1] if history else None


def print_eval_summary() -> None:
    """
    Prints a summary table of all eval runs to console.
    Useful for tracking hit_rate improvement over time.
    """
    history = load_eval_history()

    if not history:
        logger.info("[EVAL] No eval history found")
        return

    logger.info("[EVAL] ── Evaluation History ──────────────────────")
    logger.info(f"{'Timestamp':<25} {'Mode':<12} {'hit_rate':<10} {'faithfulness':<14} {'coverage'}")
    logger.info("─" * 75)

    for entry in history:
        logger.info(
            f"{entry['timestamp'][:19]:<25} "
            f"{entry.get('vector_db', 'n/a'):<12} "
            f"{entry.get('hit_rate', 'n/a'):<10} "
            f"{entry.get('faithfulness', 'n/a'):<14} "
            f"{entry.get('coverage', 'n/a')}"
        )
