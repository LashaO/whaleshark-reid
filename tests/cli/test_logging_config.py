"""Tests for cli.logging_config — get_logger() builds a per-run logger."""
from __future__ import annotations

import logging
from pathlib import Path

from whaleshark_reid.cli.logging_config import get_logger


def test_get_logger_writes_to_file(tmp_path: Path):
    log_path = tmp_path / "run_test.log"
    logger = get_logger(run_id="run_test", log_path=log_path, stage="ingest")

    logger.info("hello world")

    # Flush all handlers so the file is written
    for h in logger.handlers:
        h.flush()

    assert log_path.exists()
    contents = log_path.read_text()
    assert "hello world" in contents
    assert "ingest" in contents
    assert "run_test" in contents


def test_get_logger_format_has_timestamp(tmp_path: Path):
    log_path = tmp_path / "run_fmt.log"
    logger = get_logger(run_id="run_fmt", log_path=log_path, stage="embed")

    logger.info("a message")
    for h in logger.handlers:
        h.flush()

    contents = log_path.read_text()
    # Format: [HH:MM:SS] [stage] [run_id] message
    assert "[embed]" in contents
    assert "[run_fmt]" in contents
    # Timestamp prefix
    import re
    assert re.search(r"\[\d{2}:\d{2}:\d{2}\]", contents)


def test_get_logger_is_idempotent_for_same_run_id(tmp_path: Path):
    """Calling get_logger twice with the same run_id should not double-add handlers."""
    log_path = tmp_path / "run_idem.log"
    logger1 = get_logger(run_id="run_idem", log_path=log_path, stage="ingest")
    logger2 = get_logger(run_id="run_idem", log_path=log_path, stage="ingest")

    assert logger1 is logger2
    # Should have the file handler exactly once (plus optional stdout)
    file_handlers = [h for h in logger1.handlers if isinstance(h, logging.FileHandler)]
    assert len(file_handlers) == 1
