"""RunContext: per-CLI-invocation state, plus the run_context() context manager
that wraps every command in a transactional run row + try/except/finally.
"""
from __future__ import annotations

import secrets
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from logging import Logger
from pathlib import Path
from typing import Iterator, Optional

from whaleshark_reid.cli.logging_config import get_logger
from whaleshark_reid.storage.db import Storage


def detect_git_sha() -> Optional[str]:
    """Return the current git HEAD sha, or None if not in a repo / git missing."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=2,
        )
        sha = result.stdout.strip()
        return sha if len(sha) == 40 else None
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None


@dataclass
class RunContext:
    run_id: str
    stage: str
    storage: Storage
    cache_dir: Path
    config: dict
    logger: Logger
    git_sha: Optional[str] = None
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def new(cls, stage: str, storage: Storage, cache_dir: Path, config: dict) -> "RunContext":
        run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(2)}"
        git_sha = detect_git_sha()
        storage.begin_run(run_id=run_id, stage=stage, config=config, git_sha=git_sha)
        log_path = cache_dir / "logs" / f"{run_id}.log"
        logger = get_logger(run_id=run_id, log_path=log_path, stage=stage)
        return cls(
            run_id=run_id,
            stage=stage,
            storage=storage,
            cache_dir=cache_dir,
            config=config,
            logger=logger,
            git_sha=git_sha,
        )

    def finish(
        self,
        status: str = "ok",
        metrics: Optional[dict] = None,
        error: Optional[str] = None,
        notes: str = "",
    ) -> None:
        self.storage.finish_run(
            run_id=self.run_id,
            status=status,
            metrics=metrics or {},
            error=error,
            notes=notes,
        )


@contextmanager
def run_context(
    stage: str,
    storage: Storage,
    cache_dir: Path,
    config: dict,
) -> Iterator[RunContext]:
    """Standardized try/except/finally wrapper around RunContext.new.

    On exception: calls ctx.finish('failed', error=str(e)) and re-raises.
    On clean exit: if the command didn't explicitly call ctx.finish, the
    safety net calls ctx.finish('ok', metrics={}).
    """
    ctx = RunContext.new(stage=stage, storage=storage, cache_dir=cache_dir, config=config)
    try:
        yield ctx
    except Exception as e:
        ctx.finish(status="failed", error=str(e))
        raise
    else:
        if ctx.storage.get_run_status(ctx.run_id) == "running":
            ctx.finish(status="ok")
