"""Shared helpers for the pipeline phase scripts.

Kept intentionally small: logging setup and a couple of convenience utilities.
Phase scripts import this alongside `config`.
"""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager


def get_logger(name: str) -> logging.Logger:
    """Return a module logger with a consistent, timestamped format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


@contextmanager
def timed(logger: logging.Logger, label: str):
    """Context manager that logs how long a block took."""
    logger.info("START %s", label)
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info("DONE  %s (%.1fs)", label, elapsed)


def not_implemented(script_path: str) -> None:
    """Uniform placeholder for phase stubs that are not yet implemented."""
    raise SystemExit(
        f"[stub] {script_path} is scaffolded but not implemented yet. "
        "It will be filled in phase by phase."
    )
