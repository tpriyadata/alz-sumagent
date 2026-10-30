"""
configs/logging_setup.py

"""

from loguru import logger
from configs.settings import get_settings


def setup_logging() -> None:
    """
    Adds a rotating file handler to loguru.
    Safe to call multiple times — loguru deduplicates handlers.
    """
    settings = get_settings()
    settings.logs_path.mkdir(parents=True, exist_ok=True)

    logger.add(
        settings.logs_path / "alz_bot.log",
        level=settings.log_level,
        rotation="10 MB",        # new file after 10MB
        retention="7 days",      # keep last 7 days only
        compression="zip",       # compress rotated files
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {module}:{line} | {message}",
        enqueue=True,            # thread-safe writes
    )

    logger.info(
        "[LOGGING] File logging active | "
        f"path={settings.logs_path / 'alz_bot.log'} | "
        f"level={settings.log_level}"
    )
