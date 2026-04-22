"""
pipeline/logger.py — Centralised logging for the SRHI pipeline.

Creates a logger that writes to BOTH the terminal (console) and a
timestamped log file under logs/<run_id>/<step>.log.

Usage in every step script:
    from pipeline.logger import get_logger
    log = get_logger(__name__)
    log.info("Starting step 01 for video %s", video_id)
    log.warning("Prosody frame count is zero for turn at %.2f s", start)
    log.error("ffmpeg failed: %s", str(e))

The run_id is set once by run_all.py at startup via set_run_id().
Individual step scripts may also be run directly — in that case a
new run_id is generated automatically.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

# ── Module-level state ────────────────────────────────────────────────────────
_run_id: str | None = None
_log_dir: Path | None = None
_handlers_added: set[str] = set()   # Track which loggers already have file handlers


def set_run_id(run_id: str, log_dir: Path) -> None:
    """Called once by run_all.py to establish the shared run identifier."""
    global _run_id, _log_dir
    _run_id = run_id
    _log_dir = log_dir
    _log_dir.mkdir(parents=True, exist_ok=True)


def _get_run_id() -> str:
    global _run_id
    if _run_id is None:
        _run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    return _run_id


def _get_log_dir() -> Path:
    global _log_dir
    if _log_dir is None:
        from config import LOGS_DIR
        _log_dir = LOGS_DIR / _get_run_id()
        _log_dir.mkdir(parents=True, exist_ok=True)
    return _log_dir


def get_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    """
    Return a named logger with console + file output.

    The file handler writes to logs/<run_id>/<name>.log.
    The console handler writes INFO and above to stderr.
    Both handlers use a detailed format including timestamp, level, and name.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if the logger is already configured
    if name in _handlers_added:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Console handler (INFO+) ───────────────────────────────────────────────
    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # ── File handler (DEBUG+) — captures everything ───────────────────────────
    log_dir = _get_log_dir()
    # Sanitise name: replace dots/slashes so it's a safe filename
    safe_name = name.replace(".", "_").replace("/", "_")
    log_file = log_dir / f"{safe_name}.log"

    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    _handlers_added.add(name)

    logger.debug("Logger initialised — writing to %s", log_file)
    return logger


def get_run_log_path() -> Path:
    """Return the path to the shared pipeline run log file."""
    return _get_log_dir() / "pipeline_run.log"
