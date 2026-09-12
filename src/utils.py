"""Shared helpers for the pipeline phase scripts.

Kept intentionally small: logging setup and a couple of convenience utilities.
Phase scripts import this alongside `config`.
"""
from __future__ import annotations

import logging
import re
import time
from contextlib import contextmanager
from pathlib import Path


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


def update_manifest_row(manifest_path: Path, row_marker: str, retrieved: str, version_notes: str) -> None:
    """Fill the "Retrieved" / "Version / notes" cells of one DATA_MANIFEST.md table row.

    `row_marker` is a unique substring identifying the row (e.g. the dataset name).
    Only replaces literal "_to fill_" placeholders, so re-running is idempotent
    once a row has already been filled in.
    """
    text = manifest_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if row_marker in line and "_to fill_" in line:
            parts = line.split("|")
            fill_idx = [j for j, p in enumerate(parts) if p.strip() == "_to fill_"]
            if len(fill_idx) >= 2:
                parts[fill_idx[0]] = f" {retrieved} "
                parts[fill_idx[1]] = f" {version_notes} "
                lines[i] = "|".join(parts)
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def mark_barangay_source_used(manifest_path: Path, source_name: str, feature_count: int) -> None:
    """Check off the resolved ladder item and fill the summary line in DATA_MANIFEST.md."""
    text = manifest_path.read_text(encoding="utf-8")
    text = re.sub(
        rf"- \[ \] (\d+\. .*{re.escape(source_name)}.*)",
        r"- [x] \1",
        text,
    )
    text = text.replace(
        "**Source used:** _to fill_   **Feature count:** _to fill_ (expected ~142)",
        f"**Source used:** {source_name}   **Feature count:** {feature_count} (expected ~142)",
    )
    manifest_path.write_text(text, encoding="utf-8")
