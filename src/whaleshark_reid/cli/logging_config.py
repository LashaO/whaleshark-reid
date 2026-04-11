"""Plain text logging for CLI runs.

One logger per run_id. Writes to both stdout (rich-colorized in TTY) and a
per-run text file at cache_dir/logs/<run_id>.log. Idempotent — calling
get_logger() twice for the same run_id returns the same logger and does not
double-attach handlers.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path


_LOGGER_CACHE: dict[str, logging.Logger] = {}


def get_logger(run_id: str, log_path: Path, stage: str) -> logging.Logger:
    """Return (or create) a logger for the given run_id.

    Format: [HH:MM:SS] [stage] [run_id] message
    """
    if run_id in _LOGGER_CACHE:
        return _LOGGER_CACHE[run_id]

    logger = logging.getLogger(f"catalog.{run_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False  # don't bubble to root logger

    fmt = logging.Formatter(
        fmt=f"[%(asctime)s] [{stage}] [{run_id}] %(message)s",
        datefmt="%H:%M:%S",
    )

    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    # stdout handler — rich if TTY, plain otherwise
    if sys.stdout.isatty():
        try:
            from rich.logging import RichHandler
            stdout_handler: logging.Handler = RichHandler(
                show_time=True,
                show_level=False,
                show_path=False,
                rich_tracebacks=True,
            )
            stdout_handler.setFormatter(logging.Formatter(f"[{stage}] [{run_id}] %(message)s"))
        except ImportError:
            stdout_handler = logging.StreamHandler(sys.stdout)
            stdout_handler.setFormatter(fmt)
    else:
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(fmt)
    logger.addHandler(stdout_handler)

    _LOGGER_CACHE[run_id] = logger
    return logger
