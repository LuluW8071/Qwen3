import logging
import sys

from loguru import logger as loguru_logger


def setup_logging():
    loguru_logger.remove()

    loguru_logger.add(
        sys.stdout,
        level="DEBUG",
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan> | "
            "<blue>{function}</blue>:<cyan>{line}</cyan> - "
            "{message}"
        ),
        colorize=True,
        backtrace=False,
        diagnose=False,
    )

    logging.basicConfig(
        level=logging.INFO,
        handlers=[],
        force=True,
    )

    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"):
        third_party_logger = logging.getLogger(name)
        third_party_logger.handlers.clear()
        third_party_logger.propagate = True

    return loguru_logger


logger = setup_logging()
