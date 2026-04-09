"""
Centralized logger for the Kessler Env project.

Log level is controlled by the LOG_LEVEL environment variable (default: WARNING).
Set LOG_LEVEL=DEBUG or LOG_LEVEL=INFO in your .env file for verbose output.

Usage:
    from logger import get_logger
    logger = get_logger(__name__)
    logger.info("Something happened")
    logger.debug("Detailed trace: %s", data)
"""
import logging
import os
import sys


def _build_formatter() -> logging.Formatter:
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%dT%H:%M:%S"
    return logging.Formatter(fmt, datefmt=datefmt)


def _configure_root_logger() -> None:
    raw_level = os.getenv("LOG_LEVEL", "WARNING").upper()
    level = getattr(logging, raw_level, logging.WARNING)

    root = logging.getLogger("kessler")
    if root.handlers:
        # Already configured (e.g. imported twice)
        return

    root.setLevel(level)

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)
    handler.setFormatter(_build_formatter())
    root.addHandler(handler)

    # Silence noisy third-party loggers unless we're in DEBUG
    if level > logging.DEBUG:
        for noisy in ("websockets", "asyncio", "httpcore", "httpx", "openai"):
            logging.getLogger(noisy).setLevel(logging.ERROR)


def get_logger(name: str) -> logging.Logger:
    """
    Return a child logger under the 'kessler' hierarchy.

    Args:
        name: Typically __name__ of the calling module.

    Returns:
        A configured Logger instance.
    """
    _configure_root_logger()
    # Strip package prefixes so log names stay short, e.g. "kessler.environment"
    short = name.split(".")[-1] if "." in name else name
    return logging.getLogger(f"kessler.{short}")