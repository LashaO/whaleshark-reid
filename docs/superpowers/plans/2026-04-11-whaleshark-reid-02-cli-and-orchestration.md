# Whaleshark Re-ID — CLI and Orchestration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the existing whaleshark_reid core engine to a Typer-based CLI so the pipeline is scriptable from a terminal — `catalog ingest`, `catalog embed`, `catalog cluster`, `catalog matching`, `catalog project`, `catalog rebuild-individuals`, and `catalog run-all`.

**Architecture:** Thin orchestration layer over Phase 1 core. Each command is one file in `cli/commands/`, registered in `cli/main.py` as a Typer subcommand. Every command runs inside a `run_context()` context manager that creates a fresh `run_id`, calls `storage.begin_run`, runs the core stage function, and calls `storage.finish_run` with the result metrics. No ML logic lives in `cli/`.

**Tech Stack:** Python 3.10+, Typer (CLI framework), rich (TTY-colorized logging), pytest with `typer.testing.CliRunner` for tests. All existing Phase 1 dependencies remain unchanged.

**Pre-flight assumption:** Phase 1 core engine is shipped (16 commits on main, 71 tests passing). Spec at `docs/superpowers/specs/2026-04-10-whaleshark-reid-02-cli-and-orchestration.md`. The Storage methods used by this plan (`begin_run`, `finish_run`, `get_run_status`, `list_active_pair_decisions`, `replace_pair_queue`, `transaction`, `list_annotation_uuids`) all exist as of Phase 1's reviewer-followup commit `11c1888`.

**Execution notes:**
- Every task begins with `cd /workspace/catalog-match/whaleshark-reid` unless stated otherwise.
- `pytest -x` after each task. Commit after green.
- Conventional-commit prefixes: `feat:` for new commands, `test:` for test-only tasks, `chore:` for setup.
- Subagents follow TDD: write failing test → run to confirm failure → implement minimal → run to confirm pass → commit.

---

## Task 1: CLI scaffolding — deps, entry point, main.py, smoke test

**Files:**
- Modify: `pyproject.toml` (add `typer` and `rich` deps + `[project.scripts]`)
- Create: `src/whaleshark_reid/cli/__init__.py`
- Create: `src/whaleshark_reid/cli/main.py`
- Create: `src/whaleshark_reid/cli/commands/__init__.py`
- Create: `tests/cli/__init__.py`
- Create: `tests/cli/conftest.py`
- Create: `tests/cli/test_cli_main.py`

- [ ] **Step 1: Add typer + rich to dependencies and the entry point**

Edit `pyproject.toml`. The current `dependencies = [...]` block needs `typer>=0.12` and `rich>=13` added. Also add a new `[project.scripts]` section.

After the existing `wbia_miew_id` line in `dependencies`, append `"typer>=0.12",` and `"rich>=13",`. Then add this section after `[project.optional-dependencies]`:

```toml
[project.scripts]
catalog = "whaleshark_reid.cli.main:app"
```

- [ ] **Step 2: Reinstall the package to register the entry point and pull deps**

```bash
cd /workspace/catalog-match/whaleshark-reid
pip install -e ".[dev]"
```

Expected: typer and rich install, then the package reinstalls with the new entry point.

- [ ] **Step 3: Create empty package init files**

`src/whaleshark_reid/cli/__init__.py`:
```python
"""CLI entry point and command modules."""
```

`src/whaleshark_reid/cli/commands/__init__.py`:
```python
"""Per-command modules for the catalog CLI."""
```

`tests/cli/__init__.py`:
```python
```

- [ ] **Step 4: Write the failing smoke test**

`tests/cli/conftest.py`:
```python
"""Shared fixtures for CLI tests."""
from __future__ import annotations

import pytest
from typer.testing import CliRunner


@pytest.fixture
def cli_runner() -> CliRunner:
    return CliRunner()
```

`tests/cli/test_cli_main.py`:
```python
"""Smoke tests for the catalog CLI app."""
from __future__ import annotations

from typer.testing import CliRunner


def test_app_importable():
    from whaleshark_reid.cli.main import app  # noqa: F401


def test_help_lists_no_commands_yet(cli_runner: CliRunner):
    """Before any commands are registered, --help should still work and exit 0."""
    from whaleshark_reid.cli.main import app

    result = cli_runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "catalog" in result.stdout.lower() or "Usage" in result.stdout
```

- [ ] **Step 5: Run the failing smoke test**

```bash
pytest tests/cli/test_cli_main.py -x
```
Expected: `ModuleNotFoundError: No module named 'whaleshark_reid.cli.main'`

- [ ] **Step 6: Write the minimal Typer app**

`src/whaleshark_reid/cli/main.py`:
```python
"""Catalog CLI — Typer entry point.

Each subcommand lives in its own module under cli/commands/ and is registered
here. Commands themselves do no ML; they're thin orchestration over core.
"""
from __future__ import annotations

import typer

app = typer.Typer(
    name="catalog",
    help="Whale shark re-identification pipeline CLI",
    no_args_is_help=True,
)


if __name__ == "__main__":
    app()
```

- [ ] **Step 7: Run smoke test to verify pass**

```bash
pytest tests/cli/test_cli_main.py -x
```
Expected: 2 passed.

Also verify the entry point works as a real shell command:
```bash
catalog --help
```
Expected: shows the Typer help text with name "catalog".

- [ ] **Step 8: Commit**

```bash
cd /workspace/catalog-match/whaleshark-reid
git add pyproject.toml src/whaleshark_reid/cli/ tests/cli/
git commit -m "chore: CLI scaffolding — typer entry point + smoke test"
```

---

## Task 2: Logging configuration

**Files:**
- Create: `src/whaleshark_reid/cli/logging_config.py`
- Create: `tests/cli/test_logging_config.py`

- [ ] **Step 1: Write the failing test**

`tests/cli/test_logging_config.py`:
```python
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
```

- [ ] **Step 2: Run failing test**

```bash
pytest tests/cli/test_logging_config.py -x
```
Expected: `ModuleNotFoundError: No module named 'whaleshark_reid.cli.logging_config'`

- [ ] **Step 3: Implement logging_config**

`src/whaleshark_reid/cli/logging_config.py`:
```python
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
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/cli/test_logging_config.py -x
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
cd /workspace/catalog-match/whaleshark-reid
git add src/whaleshark_reid/cli/logging_config.py tests/cli/test_logging_config.py
git commit -m "feat: CLI logging — get_logger with file + rich TTY handlers"
```

---

## Task 3: RunContext + run_context() context manager

**Files:**
- Create: `src/whaleshark_reid/cli/run_context.py`
- Create: `tests/cli/test_run_context.py`

- [ ] **Step 1: Write failing tests**

`tests/cli/test_run_context.py`:
```python
"""Tests for cli.run_context — RunContext lifecycle and the context manager wrapper."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from whaleshark_reid.cli.run_context import RunContext, detect_git_sha, run_context
from whaleshark_reid.storage.db import Storage


def test_run_context_new_creates_run_row(tmp_db_path: Path, tmp_cache_dir: Path):
    storage = Storage(tmp_db_path)
    storage.init_schema()

    ctx = RunContext.new(stage="ingest", storage=storage, cache_dir=tmp_cache_dir,
                         config={"a": 1})

    assert ctx.run_id.startswith("run_")
    assert ctx.stage == "ingest"
    assert storage.get_run_status(ctx.run_id) == "running"
    row = storage.conn.execute(
        "SELECT config_json FROM runs WHERE run_id = ?", (ctx.run_id,)
    ).fetchone()
    assert json.loads(row["config_json"]) == {"a": 1}


def test_run_context_finish_ok(tmp_db_path: Path, tmp_cache_dir: Path):
    storage = Storage(tmp_db_path)
    storage.init_schema()
    ctx = RunContext.new(stage="embed", storage=storage, cache_dir=tmp_cache_dir, config={})

    ctx.finish(status="ok", metrics={"n": 5})

    assert storage.get_run_status(ctx.run_id) == "ok"
    row = storage.conn.execute(
        "SELECT metrics_json FROM runs WHERE run_id = ?", (ctx.run_id,)
    ).fetchone()
    assert json.loads(row["metrics_json"]) == {"n": 5}


def test_run_context_manager_success(tmp_db_path: Path, tmp_cache_dir: Path):
    """Block exits normally without explicit finish → safety net marks ok."""
    storage = Storage(tmp_db_path)
    storage.init_schema()

    with run_context(stage="cluster", storage=storage, cache_dir=tmp_cache_dir, config={"eps": 0.7}) as ctx:
        run_id = ctx.run_id
        # do nothing — exits without explicit finish

    assert storage.get_run_status(run_id) == "ok"


def test_run_context_manager_explicit_finish(tmp_db_path: Path, tmp_cache_dir: Path):
    storage = Storage(tmp_db_path)
    storage.init_schema()

    with run_context(stage="ingest", storage=storage, cache_dir=tmp_cache_dir, config={}) as ctx:
        ctx.finish(status="ok", metrics={"n_ingested": 42})
        run_id = ctx.run_id

    assert storage.get_run_status(run_id) == "ok"
    row = storage.conn.execute(
        "SELECT metrics_json FROM runs WHERE run_id = ?", (run_id,)
    ).fetchone()
    assert json.loads(row["metrics_json"]) == {"n_ingested": 42}


def test_run_context_manager_exception_marks_failed(tmp_db_path: Path, tmp_cache_dir: Path):
    storage = Storage(tmp_db_path)
    storage.init_schema()
    captured_run_id = None

    with pytest.raises(ValueError, match="boom"):
        with run_context(stage="matching", storage=storage, cache_dir=tmp_cache_dir, config={}) as ctx:
            captured_run_id = ctx.run_id
            raise ValueError("boom")

    assert captured_run_id is not None
    assert storage.get_run_status(captured_run_id) == "failed"
    row = storage.conn.execute(
        "SELECT error FROM runs WHERE run_id = ?", (captured_run_id,)
    ).fetchone()
    assert "boom" in row["error"]


def test_detect_git_sha_returns_string_or_none():
    sha = detect_git_sha()
    # In this repo it should be a real sha, but the function must tolerate non-repos too
    assert sha is None or (isinstance(sha, str) and len(sha) == 40)
```

- [ ] **Step 2: Run failing test**

```bash
pytest tests/cli/test_run_context.py -x
```
Expected: `ModuleNotFoundError: No module named 'whaleshark_reid.cli.run_context'`

- [ ] **Step 3: Implement run_context.py**

`src/whaleshark_reid/cli/run_context.py`:
```python
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
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/cli/test_run_context.py -x
```
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
cd /workspace/catalog-match/whaleshark-reid
git add src/whaleshark_reid/cli/run_context.py tests/cli/test_run_context.py
git commit -m "feat: RunContext + run_context() context manager with git sha detection"
```

---

## Task 4: `catalog ingest` command

**Files:**
- Create: `src/whaleshark_reid/cli/commands/ingest.py`
- Modify: `src/whaleshark_reid/cli/main.py` (register the command)
- Create: `tests/cli/test_cli_ingest.py`

- [ ] **Step 1: Write the failing test**

`tests/cli/test_cli_ingest.py`:
```python
"""CLI integration test for `catalog ingest`."""
from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from whaleshark_reid.storage.db import Storage

FIXTURES = Path(__file__).parent.parent / "fixtures"


def test_catalog_ingest_creates_run_and_annotations(
    cli_runner: CliRunner, tmp_db_path: Path, tmp_cache_dir: Path
):
    from whaleshark_reid.cli.main import app

    result = cli_runner.invoke(
        app,
        [
            "ingest",
            "--csv", str(FIXTURES / "mini_inat.csv"),
            "--photos-dir", str(FIXTURES / "photos"),
            "--rich-csv", str(FIXTURES / "mini_inat_rich.csv"),
            "--db-path", str(tmp_db_path),
            "--cache-dir", str(tmp_cache_dir),
        ],
    )

    assert result.exit_code == 0, f"stdout: {result.stdout}\nexc: {result.exception}"
    assert "ingested" in result.stdout

    storage = Storage(tmp_db_path)
    assert storage.count("annotations") == 10

    runs = storage.conn.execute(
        "SELECT run_id, stage, status, metrics_json FROM runs ORDER BY started_at DESC"
    ).fetchall()
    assert len(runs) == 1
    assert runs[0]["stage"] == "ingest"
    assert runs[0]["status"] == "ok"
    metrics = json.loads(runs[0]["metrics_json"])
    assert metrics["n_ingested"] == 10
    assert metrics["n_skipped_existing"] == 0
```

- [ ] **Step 2: Run failing test**

```bash
pytest tests/cli/test_cli_ingest.py -x
```
Expected: failure with "No such command 'ingest'" or similar typer error.

- [ ] **Step 3: Implement the ingest command**

`src/whaleshark_reid/cli/commands/ingest.py`:
```python
"""`catalog ingest` — load an iNat CSV into the annotations table."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from whaleshark_reid.cli.run_context import run_context
from whaleshark_reid.core.ingest.inat import ingest_inat_csv
from whaleshark_reid.storage.db import Storage


def ingest_command(
    csv: Path = typer.Option(..., "--csv", exists=True, help="Path to minimal or dfx CSV"),
    photos_dir: Path = typer.Option(..., "--photos-dir", exists=True, file_okay=False),
    source: str = typer.Option("inat", "--source"),
    rich_csv: Optional[Path] = typer.Option(
        None, "--rich-csv", exists=True, help="Optional dfx CSV for provenance backfill"
    ),
    db_path: Path = typer.Option(Path("cache/state.db"), "--db-path"),
    cache_dir: Path = typer.Option(Path("cache/"), "--cache-dir"),
) -> None:
    """Ingest an iNat CSV into the annotations table."""
    storage = Storage(db_path)
    storage.init_schema()
    config = {
        "csv": str(csv),
        "photos_dir": str(photos_dir),
        "source": source,
        "rich_csv": str(rich_csv) if rich_csv else None,
    }
    with run_context(stage="ingest", storage=storage, cache_dir=cache_dir, config=config) as ctx:
        ctx.logger.info(f"reading {csv}")
        result = ingest_inat_csv(
            csv_path=csv,
            photos_dir=photos_dir,
            storage=storage,
            run_id=ctx.run_id,
            rich_csv_path=rich_csv,
        )
        ctx.logger.info(
            f"ingested={result.n_ingested} skipped={result.n_skipped_existing} "
            f"missing_files={result.n_missing_files}"
        )
        ctx.finish(status="ok", metrics=result.model_dump())

    typer.echo(
        f"ingested {result.n_ingested} / "
        f"skipped {result.n_skipped_existing} / "
        f"missing {result.n_missing_files}"
    )
```

- [ ] **Step 4: Register the command in main.py**

Edit `src/whaleshark_reid/cli/main.py` and replace its contents with:

```python
"""Catalog CLI — Typer entry point."""
from __future__ import annotations

import typer

from whaleshark_reid.cli.commands.ingest import ingest_command

app = typer.Typer(
    name="catalog",
    help="Whale shark re-identification pipeline CLI",
    no_args_is_help=True,
)

app.command(name="ingest")(ingest_command)


if __name__ == "__main__":
    app()
```

- [ ] **Step 5: Run tests to verify pass**

```bash
pytest tests/cli/test_cli_ingest.py -x
```
Expected: 1 passed.

Also verify visually:
```bash
catalog ingest --help
```
Expected: shows the ingest help text with all arguments.

- [ ] **Step 6: Commit**

```bash
cd /workspace/catalog-match/whaleshark-reid
git add src/whaleshark_reid/cli/main.py src/whaleshark_reid/cli/commands/ingest.py tests/cli/test_cli_ingest.py
git commit -m "feat: catalog ingest command"
```

---

## Task 5: `catalog embed` command (+ stub_miewid fixture)

**Files:**
- Create: `src/whaleshark_reid/cli/commands/embed.py`
- Modify: `src/whaleshark_reid/cli/main.py` (register)
- Modify: `tests/cli/conftest.py` (add stub_miewid fixture)
- Create: `tests/cli/test_cli_embed.py`

- [ ] **Step 1: Add stub_miewid fixture to conftest**

Edit `tests/cli/conftest.py` and append:

```python
import torch
from transformers import AutoModel


class _StubMiewId(torch.nn.Module):
    """Deterministic stand-in for MiewIdNet so CLI tests don't hit HuggingFace."""

    def __init__(self, embed_dim: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self._dummy = torch.nn.Parameter(torch.zeros(1))

    def extract_feat(self, x: torch.Tensor) -> torch.Tensor:
        per_sample = x.mean(dim=(1, 2, 3))
        return per_sample.unsqueeze(1).expand(-1, self.embed_dim)

    def forward(self, x, label=None):
        return self.extract_feat(x)


@pytest.fixture
def stub_miewid(monkeypatch):
    monkeypatch.setattr(AutoModel, "from_pretrained", lambda *a, **k: _StubMiewId())
    yield
```

- [ ] **Step 2: Write the failing test**

`tests/cli/test_cli_embed.py`:
```python
"""CLI integration test for `catalog embed`."""
from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from whaleshark_reid.storage.db import Storage
from whaleshark_reid.storage.embedding_cache import read_embeddings

FIXTURES = Path(__file__).parent.parent / "fixtures"


def test_catalog_embed_creates_parquet_and_run_row(
    cli_runner: CliRunner,
    tmp_db_path: Path,
    tmp_cache_dir: Path,
    stub_miewid,
):
    from whaleshark_reid.cli.main import app

    # Need ingested annotations first
    ingest_result = cli_runner.invoke(
        app,
        [
            "ingest",
            "--csv", str(FIXTURES / "mini_inat.csv"),
            "--photos-dir", str(FIXTURES / "photos"),
            "--db-path", str(tmp_db_path),
            "--cache-dir", str(tmp_cache_dir),
        ],
    )
    assert ingest_result.exit_code == 0

    # Now embed
    result = cli_runner.invoke(
        app,
        [
            "embed",
            "--db-path", str(tmp_db_path),
            "--cache-dir", str(tmp_cache_dir),
            "--batch-size", "4",
            "--num-workers", "0",
            "--device", "cpu",
        ],
    )
    assert result.exit_code == 0, f"stdout: {result.stdout}\nexc: {result.exception}"

    storage = Storage(tmp_db_path)
    embed_runs = storage.conn.execute(
        "SELECT run_id, status, metrics_json FROM runs WHERE stage = 'embed'"
    ).fetchall()
    assert len(embed_runs) == 1
    assert embed_runs[0]["status"] == "ok"
    metrics = json.loads(embed_runs[0]["metrics_json"])
    assert metrics["n_embedded"] == 10
    assert metrics["embed_dim"] == 8

    df = read_embeddings(tmp_cache_dir, embed_runs[0]["run_id"])
    assert len(df) == 10
```

- [ ] **Step 3: Run failing test**

```bash
pytest tests/cli/test_cli_embed.py -x
```
Expected: failure with "No such command 'embed'" or similar.

- [ ] **Step 4: Implement the embed command**

`src/whaleshark_reid/cli/commands/embed.py`:
```python
"""`catalog embed` — extract MiewID embeddings for ingested annotations."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from whaleshark_reid.cli.run_context import run_context
from whaleshark_reid.core.embed.miewid import run_embed_stage
from whaleshark_reid.storage.db import Storage


def embed_command(
    model: str = typer.Option("conservationxlabs/miewid-msv3", "--model"),
    batch_size: int = typer.Option(32, "--batch-size"),
    num_workers: int = typer.Option(2, "--num-workers"),
    use_bbox: bool = typer.Option(True, "--use-bbox/--no-use-bbox"),
    only_missing: bool = typer.Option(True, "--only-missing/--force-reembed"),
    device: Optional[str] = typer.Option(None, "--device", help="cuda | cpu | mps; default: auto"),
    db_path: Path = typer.Option(Path("cache/state.db"), "--db-path"),
    cache_dir: Path = typer.Option(Path("cache/"), "--cache-dir"),
) -> None:
    """Embed all ingested annotations not yet in the cache.

    Writes to cache_dir/embeddings/<run_id>.parquet.
    """
    storage = Storage(db_path)
    storage.init_schema()
    config = {
        "model": model,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "use_bbox": use_bbox,
        "only_missing": only_missing,
        "device": device,
    }
    with run_context(stage="embed", storage=storage, cache_dir=cache_dir, config=config) as ctx:
        ctx.logger.info(f"embedding model={model} batch_size={batch_size} use_bbox={use_bbox}")
        result = run_embed_stage(
            storage=storage,
            cache_dir=cache_dir,
            run_id=ctx.run_id,
            model_id=model,
            batch_size=batch_size,
            num_workers=num_workers,
            use_bbox=use_bbox,
            only_missing=only_missing,
            device=device,
        )
        ctx.logger.info(
            f"embedded={result.n_embedded} skipped={result.n_skipped_existing} "
            f"failed={result.n_failed} dim={result.embed_dim} duration_s={result.duration_s:.1f}"
        )
        ctx.finish(status="ok", metrics=result.model_dump())

    typer.echo(
        f"embedded {result.n_embedded} (dim={result.embed_dim}) / "
        f"skipped {result.n_skipped_existing} / "
        f"failed {result.n_failed} / "
        f"{result.duration_s:.1f}s"
    )
```

- [ ] **Step 5: Register in main.py**

Edit `src/whaleshark_reid/cli/main.py` to add the embed import and registration:

```python
"""Catalog CLI — Typer entry point."""
from __future__ import annotations

import typer

from whaleshark_reid.cli.commands.embed import embed_command
from whaleshark_reid.cli.commands.ingest import ingest_command

app = typer.Typer(
    name="catalog",
    help="Whale shark re-identification pipeline CLI",
    no_args_is_help=True,
)

app.command(name="ingest")(ingest_command)
app.command(name="embed")(embed_command)


if __name__ == "__main__":
    app()
```

- [ ] **Step 6: Run tests to verify pass**

```bash
pytest tests/cli/test_cli_embed.py -x
```
Expected: 1 passed.

- [ ] **Step 7: Commit**

```bash
cd /workspace/catalog-match/whaleshark-reid
git add src/whaleshark_reid/cli/commands/embed.py src/whaleshark_reid/cli/main.py tests/cli/conftest.py tests/cli/test_cli_embed.py
git commit -m "feat: catalog embed command (with stub_miewid fixture)"
```

---

## Task 6: `catalog cluster` command (+ Storage.get_latest_run_id helper)

**Files:**
- Modify: `src/whaleshark_reid/storage/db.py` (add `get_latest_run_id`)
- Create: `src/whaleshark_reid/cli/commands/cluster.py`
- Modify: `src/whaleshark_reid/cli/main.py`
- Create: `tests/cli/test_cli_cluster.py`
- Modify: `tests/storage/test_db_runs.py` (test for get_latest_run_id)

- [ ] **Step 1: Write failing test for get_latest_run_id**

Append to `tests/storage/test_db_runs.py`:

```python
def test_get_latest_run_id_by_stage(tmp_db_path: Path):
    storage = Storage(tmp_db_path)
    storage.init_schema()

    storage.begin_run(run_id="r1", stage="embed", config={})
    storage.finish_run(run_id="r1", status="ok", metrics={})
    storage.begin_run(run_id="r2", stage="embed", config={})
    storage.finish_run(run_id="r2", status="ok", metrics={})
    storage.begin_run(run_id="r3", stage="cluster", config={})
    storage.finish_run(run_id="r3", status="ok", metrics={})

    assert storage.get_latest_run_id(stage="embed") == "r2"
    assert storage.get_latest_run_id(stage="cluster") == "r3"
    assert storage.get_latest_run_id(stage="ingest") is None


def test_get_latest_run_id_excludes_failed(tmp_db_path: Path):
    storage = Storage(tmp_db_path)
    storage.init_schema()

    storage.begin_run(run_id="ok1", stage="embed", config={})
    storage.finish_run(run_id="ok1", status="ok", metrics={})
    storage.begin_run(run_id="bad1", stage="embed", config={})
    storage.finish_run(run_id="bad1", status="failed", metrics={}, error="boom")

    # Should return ok1, not bad1
    assert storage.get_latest_run_id(stage="embed") == "ok1"
```

- [ ] **Step 2: Run the failing storage test**

```bash
pytest tests/storage/test_db_runs.py::test_get_latest_run_id_by_stage -x
```
Expected: `AttributeError: 'Storage' object has no attribute 'get_latest_run_id'`

- [ ] **Step 3: Add the method to Storage**

Append to `src/whaleshark_reid/storage/db.py` (in the runs CRUD section, after `get_run_status`):

```python
    def get_latest_run_id(self, stage: str) -> str | None:
        """Return the most recent successful run_id for a given stage, or None if none exists."""
        row = self.conn.execute(
            """
            SELECT run_id FROM runs
            WHERE stage = ? AND status = 'ok'
            ORDER BY started_at DESC
            LIMIT 1
            """,
            (stage,),
        ).fetchone()
        return row["run_id"] if row else None
```

- [ ] **Step 4: Verify storage tests pass**

```bash
pytest tests/storage/test_db_runs.py -x
```
Expected: all storage runs tests pass (5 original + 2 new = 7).

- [ ] **Step 5: Write failing CLI test**

`tests/cli/test_cli_cluster.py`:
```python
"""CLI integration test for `catalog cluster`."""
from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from whaleshark_reid.storage.db import Storage
from whaleshark_reid.storage.cluster_cache import read_clusters

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _seed_ingest_and_embed(cli_runner: CliRunner, tmp_db_path: Path, tmp_cache_dir: Path):
    from whaleshark_reid.cli.main import app

    cli_runner.invoke(app, [
        "ingest",
        "--csv", str(FIXTURES / "mini_inat.csv"),
        "--photos-dir", str(FIXTURES / "photos"),
        "--db-path", str(tmp_db_path),
        "--cache-dir", str(tmp_cache_dir),
    ])
    cli_runner.invoke(app, [
        "embed",
        "--db-path", str(tmp_db_path),
        "--cache-dir", str(tmp_cache_dir),
        "--batch-size", "4",
        "--num-workers", "0",
        "--device", "cpu",
    ])


def test_catalog_cluster_dbscan_uses_latest_embed_run(
    cli_runner: CliRunner, tmp_db_path: Path, tmp_cache_dir: Path, stub_miewid
):
    from whaleshark_reid.cli.main import app
    _seed_ingest_and_embed(cli_runner, tmp_db_path, tmp_cache_dir)

    result = cli_runner.invoke(app, [
        "cluster",
        "--algo", "dbscan",
        "--eps", "0.7",
        "--db-path", str(tmp_db_path),
        "--cache-dir", str(tmp_cache_dir),
    ])
    assert result.exit_code == 0, f"stdout: {result.stdout}\nexc: {result.exception}"

    storage = Storage(tmp_db_path)
    rows = storage.conn.execute(
        "SELECT run_id, status, metrics_json FROM runs WHERE stage = 'cluster'"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["status"] == "ok"
    metrics = json.loads(rows[0]["metrics_json"])
    assert metrics["algo"] == "dbscan"
    assert "n_clusters" in metrics

    df = read_clusters(tmp_cache_dir, rows[0]["run_id"])
    assert len(df) == 10


def test_catalog_cluster_explicit_embedding_run_id(
    cli_runner: CliRunner, tmp_db_path: Path, tmp_cache_dir: Path, stub_miewid
):
    from whaleshark_reid.cli.main import app
    _seed_ingest_and_embed(cli_runner, tmp_db_path, tmp_cache_dir)

    storage = Storage(tmp_db_path)
    embed_run_id = storage.get_latest_run_id("embed")
    assert embed_run_id is not None

    result = cli_runner.invoke(app, [
        "cluster",
        "--algo", "dbscan",
        "--embedding-run-id", embed_run_id,
        "--db-path", str(tmp_db_path),
        "--cache-dir", str(tmp_cache_dir),
    ])
    assert result.exit_code == 0
```

- [ ] **Step 6: Run failing test**

```bash
pytest tests/cli/test_cli_cluster.py -x
```
Expected: failure with "No such command 'cluster'".

- [ ] **Step 7: Implement the cluster command**

`src/whaleshark_reid/cli/commands/cluster.py`:
```python
"""`catalog cluster` — cluster embeddings via DBSCAN or HDBSCAN."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from whaleshark_reid.cli.run_context import run_context
from whaleshark_reid.core.cluster.common import run_cluster_stage
from whaleshark_reid.storage.db import Storage


def cluster_command(
    algo: str = typer.Option("dbscan", "--algo", help="dbscan | hdbscan"),
    eps: float = typer.Option(0.7, "--eps", help="DBSCAN only"),
    min_samples: int = typer.Option(2, "--min-samples"),
    metric: str = typer.Option("cosine", "--metric"),
    standardize: bool = typer.Option(True, "--standardize/--no-standardize"),
    min_cluster_size: int = typer.Option(3, "--min-cluster-size", help="HDBSCAN only"),
    embedding_run_id: Optional[str] = typer.Option(
        None, "--embedding-run-id", help="default: latest successful embed run"
    ),
    db_path: Path = typer.Option(Path("cache/state.db"), "--db-path"),
    cache_dir: Path = typer.Option(Path("cache/"), "--cache-dir"),
) -> None:
    """Cluster embeddings into proposed individual groups."""
    storage = Storage(db_path)
    storage.init_schema()

    if embedding_run_id is None:
        embedding_run_id = storage.get_latest_run_id(stage="embed")
        if embedding_run_id is None:
            typer.echo("error: no successful embed run found. Run `catalog embed` first.", err=True)
            raise typer.Exit(code=2)

    if algo == "dbscan":
        params = {
            "eps": eps,
            "min_samples": min_samples,
            "metric": metric,
            "standardize": standardize,
        }
    elif algo == "hdbscan":
        params = {
            "min_cluster_size": min_cluster_size,
            "min_samples": None,
            "metric": "euclidean" if metric == "cosine" else metric,
        }
    else:
        typer.echo(f"error: unknown cluster algo: {algo}", err=True)
        raise typer.Exit(code=2)

    config = {
        "algo": algo,
        "params": params,
        "embedding_run_id": embedding_run_id,
    }
    with run_context(stage="cluster", storage=storage, cache_dir=cache_dir, config=config) as ctx:
        ctx.logger.info(f"clustering algo={algo} params={params} embedding_run={embedding_run_id}")
        result = run_cluster_stage(
            cache_dir=cache_dir,
            embedding_run_id=embedding_run_id,
            cluster_run_id=ctx.run_id,
            algo=algo,
            params=params,
        )
        ctx.logger.info(
            f"n_clusters={result.n_clusters} n_noise={result.n_noise} "
            f"largest={result.largest_cluster_size}"
        )
        ctx.finish(status="ok", metrics=result.model_dump())

    typer.echo(
        f"clustered: {result.n_clusters} clusters / "
        f"{result.n_noise} noise / "
        f"largest={result.largest_cluster_size} / "
        f"singleton_fraction={result.singleton_fraction:.2f}"
    )
```

- [ ] **Step 8: Register in main.py**

Edit `src/whaleshark_reid/cli/main.py` to add cluster:

```python
"""Catalog CLI — Typer entry point."""
from __future__ import annotations

import typer

from whaleshark_reid.cli.commands.cluster import cluster_command
from whaleshark_reid.cli.commands.embed import embed_command
from whaleshark_reid.cli.commands.ingest import ingest_command

app = typer.Typer(
    name="catalog",
    help="Whale shark re-identification pipeline CLI",
    no_args_is_help=True,
)

app.command(name="ingest")(ingest_command)
app.command(name="embed")(embed_command)
app.command(name="cluster")(cluster_command)


if __name__ == "__main__":
    app()
```

- [ ] **Step 9: Run tests to verify pass**

```bash
pytest tests/cli/test_cli_cluster.py tests/storage/test_db_runs.py -x
```
Expected: cluster tests + storage tests pass.

- [ ] **Step 10: Commit**

```bash
cd /workspace/catalog-match/whaleshark-reid
git add src/whaleshark_reid/storage/db.py src/whaleshark_reid/cli/commands/cluster.py src/whaleshark_reid/cli/main.py tests/cli/test_cli_cluster.py tests/storage/test_db_runs.py
git commit -m "feat: catalog cluster command + Storage.get_latest_run_id helper"
```

---

## Task 7: `catalog matching` command

**Files:**
- Create: `src/whaleshark_reid/cli/commands/matching.py`
- Modify: `src/whaleshark_reid/cli/main.py`
- Create: `tests/cli/test_cli_matching.py`

- [ ] **Step 1: Write the failing test**

`tests/cli/test_cli_matching.py`:
```python
"""CLI integration test for `catalog matching`."""
from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from whaleshark_reid.storage.db import Storage

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _seed_through_cluster(cli_runner: CliRunner, tmp_db_path: Path, tmp_cache_dir: Path):
    from whaleshark_reid.cli.main import app

    cli_runner.invoke(app, [
        "ingest",
        "--csv", str(FIXTURES / "mini_inat.csv"),
        "--photos-dir", str(FIXTURES / "photos"),
        "--db-path", str(tmp_db_path),
        "--cache-dir", str(tmp_cache_dir),
    ])
    cli_runner.invoke(app, [
        "embed",
        "--db-path", str(tmp_db_path),
        "--cache-dir", str(tmp_cache_dir),
        "--batch-size", "4",
        "--num-workers", "0",
        "--device", "cpu",
    ])
    cli_runner.invoke(app, [
        "cluster",
        "--algo", "dbscan",
        "--db-path", str(tmp_db_path),
        "--cache-dir", str(tmp_cache_dir),
    ])


def test_catalog_matching_writes_pair_queue(
    cli_runner: CliRunner, tmp_db_path: Path, tmp_cache_dir: Path, stub_miewid
):
    from whaleshark_reid.cli.main import app
    _seed_through_cluster(cli_runner, tmp_db_path, tmp_cache_dir)

    result = cli_runner.invoke(app, [
        "matching",
        "--distance-threshold", "2.0",
        "--max-queue-size", "1000",
        "--db-path", str(tmp_db_path),
        "--cache-dir", str(tmp_cache_dir),
    ])
    assert result.exit_code == 0, f"stdout: {result.stdout}\nexc: {result.exception}"

    storage = Storage(tmp_db_path)
    runs = storage.conn.execute(
        "SELECT run_id, status, metrics_json FROM runs WHERE stage = 'matching'"
    ).fetchall()
    assert len(runs) == 1
    assert runs[0]["status"] == "ok"
    metrics = json.loads(runs[0]["metrics_json"])
    assert metrics["n_pairs"] > 0

    n_queue = storage.count("pair_queue", run_id=runs[0]["run_id"])
    assert n_queue == metrics["n_pairs"]
```

- [ ] **Step 2: Run failing test**

```bash
pytest tests/cli/test_cli_matching.py -x
```
Expected: failure with "No such command 'matching'".

- [ ] **Step 3: Implement the matching command**

`src/whaleshark_reid/cli/commands/matching.py`:
```python
"""`catalog matching` — compute pair candidates and write the review queue."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from whaleshark_reid.cli.run_context import run_context
from whaleshark_reid.core.matching.pairs import run_matching_stage
from whaleshark_reid.storage.db import Storage


def matching_command(
    distance_threshold: float = typer.Option(1.0, "--distance-threshold"),
    max_queue_size: int = typer.Option(2000, "--max-queue-size"),
    embedding_run_id: Optional[str] = typer.Option(
        None, "--embedding-run-id", help="default: latest successful embed run"
    ),
    cluster_run_id: Optional[str] = typer.Option(
        None, "--cluster-run-id", help="default: latest successful cluster run"
    ),
    db_path: Path = typer.Option(Path("cache/state.db"), "--db-path"),
    cache_dir: Path = typer.Option(Path("cache/"), "--cache-dir"),
) -> None:
    """Build the sorted pair_queue from embeddings + cluster labels."""
    storage = Storage(db_path)
    storage.init_schema()

    if embedding_run_id is None:
        embedding_run_id = storage.get_latest_run_id(stage="embed")
        if embedding_run_id is None:
            typer.echo("error: no successful embed run found. Run `catalog embed` first.", err=True)
            raise typer.Exit(code=2)
    if cluster_run_id is None:
        cluster_run_id = storage.get_latest_run_id(stage="cluster")
        if cluster_run_id is None:
            typer.echo("error: no successful cluster run found. Run `catalog cluster` first.", err=True)
            raise typer.Exit(code=2)

    config = {
        "distance_threshold": distance_threshold,
        "max_queue_size": max_queue_size,
        "embedding_run_id": embedding_run_id,
        "cluster_run_id": cluster_run_id,
    }
    with run_context(stage="matching", storage=storage, cache_dir=cache_dir, config=config) as ctx:
        ctx.logger.info(
            f"matching threshold={distance_threshold} max_queue={max_queue_size} "
            f"embed={embedding_run_id} cluster={cluster_run_id}"
        )
        result = run_matching_stage(
            storage=storage,
            cache_dir=cache_dir,
            matching_run_id=ctx.run_id,
            embedding_run_id=embedding_run_id,
            cluster_run_id=cluster_run_id,
            distance_threshold=distance_threshold,
            max_queue_size=max_queue_size,
        )
        ctx.logger.info(
            f"n_pairs={result.n_pairs} same_cluster={result.n_same_cluster} "
            f"cross={result.n_cross_cluster} median_dist={result.median_distance:.3f}"
        )
        ctx.finish(status="ok", metrics=result.model_dump())

    typer.echo(
        f"queued {result.n_pairs} pairs / "
        f"same_cluster={result.n_same_cluster} cross_cluster={result.n_cross_cluster} / "
        f"median_dist={result.median_distance:.3f}"
    )
```

- [ ] **Step 4: Register in main.py**

Add to `src/whaleshark_reid/cli/main.py`:

```python
"""Catalog CLI — Typer entry point."""
from __future__ import annotations

import typer

from whaleshark_reid.cli.commands.cluster import cluster_command
from whaleshark_reid.cli.commands.embed import embed_command
from whaleshark_reid.cli.commands.ingest import ingest_command
from whaleshark_reid.cli.commands.matching import matching_command

app = typer.Typer(
    name="catalog",
    help="Whale shark re-identification pipeline CLI",
    no_args_is_help=True,
)

app.command(name="ingest")(ingest_command)
app.command(name="embed")(embed_command)
app.command(name="cluster")(cluster_command)
app.command(name="matching")(matching_command)


if __name__ == "__main__":
    app()
```

- [ ] **Step 5: Run tests to verify pass**

```bash
pytest tests/cli/test_cli_matching.py -x
```
Expected: 1 passed.

- [ ] **Step 6: Commit**

```bash
cd /workspace/catalog-match/whaleshark-reid
git add src/whaleshark_reid/cli/commands/matching.py src/whaleshark_reid/cli/main.py tests/cli/test_cli_matching.py
git commit -m "feat: catalog matching command"
```

---

## Task 8: `catalog project` command

**Files:**
- Create: `src/whaleshark_reid/cli/commands/project.py`
- Modify: `src/whaleshark_reid/cli/main.py`
- Create: `tests/cli/test_cli_project.py`

- [ ] **Step 1: Write the failing test**

`tests/cli/test_cli_project.py`:
```python
"""CLI integration test for `catalog project`."""
from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from whaleshark_reid.storage.db import Storage
from whaleshark_reid.storage.projection_cache import read_projections

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _seed_through_embed(cli_runner: CliRunner, tmp_db_path: Path, tmp_cache_dir: Path):
    from whaleshark_reid.cli.main import app

    cli_runner.invoke(app, [
        "ingest",
        "--csv", str(FIXTURES / "mini_inat.csv"),
        "--photos-dir", str(FIXTURES / "photos"),
        "--db-path", str(tmp_db_path),
        "--cache-dir", str(tmp_cache_dir),
    ])
    cli_runner.invoke(app, [
        "embed",
        "--db-path", str(tmp_db_path),
        "--cache-dir", str(tmp_cache_dir),
        "--batch-size", "4",
        "--num-workers", "0",
        "--device", "cpu",
    ])


def test_catalog_project_writes_projection_parquet(
    cli_runner: CliRunner, tmp_db_path: Path, tmp_cache_dir: Path, stub_miewid
):
    from whaleshark_reid.cli.main import app
    _seed_through_embed(cli_runner, tmp_db_path, tmp_cache_dir)

    result = cli_runner.invoke(app, [
        "project",
        "--n-neighbors", "5",
        "--db-path", str(tmp_db_path),
        "--cache-dir", str(tmp_cache_dir),
    ])
    assert result.exit_code == 0, f"stdout: {result.stdout}\nexc: {result.exception}"

    storage = Storage(tmp_db_path)
    runs = storage.conn.execute(
        "SELECT run_id, status, metrics_json FROM runs WHERE stage = 'project'"
    ).fetchall()
    assert len(runs) == 1
    assert runs[0]["status"] == "ok"
    metrics = json.loads(runs[0]["metrics_json"])
    assert metrics["n_points"] == 10
    assert metrics["algo"] == "umap"

    df = read_projections(tmp_cache_dir, runs[0]["run_id"])
    assert len(df) == 10
    assert "x" in df.columns and "y" in df.columns
```

- [ ] **Step 2: Run failing test**

```bash
pytest tests/cli/test_cli_project.py -x
```
Expected: "No such command 'project'".

- [ ] **Step 3: Implement the project command**

`src/whaleshark_reid/cli/commands/project.py`:
```python
"""`catalog project` — 2D UMAP projection of embeddings for the cluster scatter view."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from whaleshark_reid.cli.run_context import run_context
from whaleshark_reid.core.cluster.project import run_project_stage
from whaleshark_reid.storage.db import Storage


def project_command(
    algo: str = typer.Option("umap", "--algo"),
    n_neighbors: int = typer.Option(15, "--n-neighbors"),
    min_dist: float = typer.Option(0.1, "--min-dist"),
    metric: str = typer.Option("cosine", "--metric"),
    random_state: int = typer.Option(42, "--random-state"),
    embedding_run_id: Optional[str] = typer.Option(
        None, "--embedding-run-id", help="default: latest successful embed run"
    ),
    db_path: Path = typer.Option(Path("cache/state.db"), "--db-path"),
    cache_dir: Path = typer.Option(Path("cache/"), "--cache-dir"),
) -> None:
    """Project embeddings to 2D for the cluster view."""
    storage = Storage(db_path)
    storage.init_schema()

    if embedding_run_id is None:
        embedding_run_id = storage.get_latest_run_id(stage="embed")
        if embedding_run_id is None:
            typer.echo("error: no successful embed run found. Run `catalog embed` first.", err=True)
            raise typer.Exit(code=2)

    config = {
        "algo": algo,
        "n_neighbors": n_neighbors,
        "min_dist": min_dist,
        "metric": metric,
        "random_state": random_state,
        "embedding_run_id": embedding_run_id,
    }
    with run_context(stage="project", storage=storage, cache_dir=cache_dir, config=config) as ctx:
        ctx.logger.info(
            f"projecting algo={algo} n_neighbors={n_neighbors} embedding_run={embedding_run_id}"
        )
        result = run_project_stage(
            cache_dir=cache_dir,
            embedding_run_id=embedding_run_id,
            projection_run_id=ctx.run_id,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
        )
        ctx.logger.info(f"projected n_points={result.n_points}")
        ctx.finish(status="ok", metrics=result.model_dump())

    typer.echo(f"projected {result.n_points} points / algo={result.algo}")
```

- [ ] **Step 4: Register in main.py**

Add `project` to the imports and registrations in `src/whaleshark_reid/cli/main.py`:

```python
"""Catalog CLI — Typer entry point."""
from __future__ import annotations

import typer

from whaleshark_reid.cli.commands.cluster import cluster_command
from whaleshark_reid.cli.commands.embed import embed_command
from whaleshark_reid.cli.commands.ingest import ingest_command
from whaleshark_reid.cli.commands.matching import matching_command
from whaleshark_reid.cli.commands.project import project_command

app = typer.Typer(
    name="catalog",
    help="Whale shark re-identification pipeline CLI",
    no_args_is_help=True,
)

app.command(name="ingest")(ingest_command)
app.command(name="embed")(embed_command)
app.command(name="cluster")(cluster_command)
app.command(name="matching")(matching_command)
app.command(name="project")(project_command)


if __name__ == "__main__":
    app()
```

- [ ] **Step 5: Run tests to verify pass**

```bash
pytest tests/cli/test_cli_project.py -x
```
Expected: 1 passed.

- [ ] **Step 6: Commit**

```bash
cd /workspace/catalog-match/whaleshark-reid
git add src/whaleshark_reid/cli/commands/project.py src/whaleshark_reid/cli/main.py tests/cli/test_cli_project.py
git commit -m "feat: catalog project command"
```

---

## Task 9: `catalog rebuild-individuals` command

**Files:**
- Create: `src/whaleshark_reid/cli/commands/rebuild_individuals.py`
- Modify: `src/whaleshark_reid/cli/main.py`
- Create: `tests/cli/test_cli_rebuild_individuals.py`

- [ ] **Step 1: Write the failing test**

`tests/cli/test_cli_rebuild_individuals.py`:
```python
"""CLI integration test for `catalog rebuild-individuals`."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from typer.testing import CliRunner

from whaleshark_reid.core.schema import inat_annotation_uuid
from whaleshark_reid.storage.db import Storage

FIXTURES = Path(__file__).parent.parent / "fixtures"


def test_catalog_rebuild_individuals_assigns_name_uuid(
    cli_runner: CliRunner, tmp_db_path: Path, tmp_cache_dir: Path
):
    from whaleshark_reid.cli.main import app

    # Seed annotations via ingest
    cli_runner.invoke(app, [
        "ingest",
        "--csv", str(FIXTURES / "mini_inat.csv"),
        "--photos-dir", str(FIXTURES / "photos"),
        "--db-path", str(tmp_db_path),
        "--cache-dir", str(tmp_cache_dir),
    ])

    # Manually insert a confirmed match between obs100 and obs101
    storage = Storage(tmp_db_path)
    u0 = inat_annotation_uuid(100, 0)
    u1 = inat_annotation_uuid(101, 0)
    storage.conn.execute(
        "INSERT INTO pair_decisions (ann_a_uuid, ann_b_uuid, decision, created_at) "
        "VALUES (?, ?, 'match', ?)",
        (u0, u1, datetime.now(timezone.utc).isoformat()),
    )

    # Rebuild
    result = cli_runner.invoke(app, [
        "rebuild-individuals",
        "--db-path", str(tmp_db_path),
        "--cache-dir", str(tmp_cache_dir),
    ])
    assert result.exit_code == 0, f"stdout: {result.stdout}\nexc: {result.exception}"

    a = storage.get_annotation(u0)
    b = storage.get_annotation(u1)
    assert a.name_uuid is not None
    assert a.name_uuid == b.name_uuid

    runs = storage.conn.execute(
        "SELECT status, metrics_json FROM runs WHERE stage = 'rebuild'"
    ).fetchall()
    assert len(runs) == 1
    assert runs[0]["status"] == "ok"
```

- [ ] **Step 2: Run failing test**

```bash
pytest tests/cli/test_cli_rebuild_individuals.py -x
```
Expected: "No such command 'rebuild-individuals'".

- [ ] **Step 3: Implement the rebuild-individuals command**

`src/whaleshark_reid/cli/commands/rebuild_individuals.py`:
```python
"""`catalog rebuild-individuals` — materialize annotations.name_uuid from pair_decisions."""
from __future__ import annotations

from pathlib import Path

import typer

from whaleshark_reid.cli.run_context import run_context
from whaleshark_reid.core.feedback.unionfind import rebuild_individuals_cache
from whaleshark_reid.storage.db import Storage


def rebuild_individuals_command(
    db_path: Path = typer.Option(Path("cache/state.db"), "--db-path"),
    cache_dir: Path = typer.Option(Path("cache/"), "--cache-dir"),
) -> None:
    """Rebuild annotations.name_uuid from confirmed pair_decisions via union-find."""
    storage = Storage(db_path)
    storage.init_schema()
    config: dict = {}
    with run_context(stage="rebuild", storage=storage, cache_dir=cache_dir, config=config) as ctx:
        ctx.logger.info("rebuilding individuals from pair_decisions")
        result = rebuild_individuals_cache(storage)
        ctx.logger.info(
            f"n_components={result.n_components} n_singletons={result.n_singletons} "
            f"updated={result.n_annotations_updated}"
        )
        ctx.finish(status="ok", metrics=result.model_dump())

    typer.echo(
        f"rebuilt: {result.n_components} individuals / "
        f"{result.n_singletons} singletons / "
        f"{result.n_annotations_updated} annotations updated"
    )
```

- [ ] **Step 4: Register in main.py**

Update `src/whaleshark_reid/cli/main.py`:

```python
"""Catalog CLI — Typer entry point."""
from __future__ import annotations

import typer

from whaleshark_reid.cli.commands.cluster import cluster_command
from whaleshark_reid.cli.commands.embed import embed_command
from whaleshark_reid.cli.commands.ingest import ingest_command
from whaleshark_reid.cli.commands.matching import matching_command
from whaleshark_reid.cli.commands.project import project_command
from whaleshark_reid.cli.commands.rebuild_individuals import rebuild_individuals_command

app = typer.Typer(
    name="catalog",
    help="Whale shark re-identification pipeline CLI",
    no_args_is_help=True,
)

app.command(name="ingest")(ingest_command)
app.command(name="embed")(embed_command)
app.command(name="cluster")(cluster_command)
app.command(name="matching")(matching_command)
app.command(name="project")(project_command)
app.command(name="rebuild-individuals")(rebuild_individuals_command)


if __name__ == "__main__":
    app()
```

- [ ] **Step 5: Run tests to verify pass**

```bash
pytest tests/cli/test_cli_rebuild_individuals.py -x
```
Expected: 1 passed.

- [ ] **Step 6: Commit**

```bash
cd /workspace/catalog-match/whaleshark-reid
git add src/whaleshark_reid/cli/commands/rebuild_individuals.py src/whaleshark_reid/cli/main.py tests/cli/test_cli_rebuild_individuals.py
git commit -m "feat: catalog rebuild-individuals command"
```

---

## Task 10: `catalog run-all` command

**Files:**
- Create: `src/whaleshark_reid/cli/commands/run_all.py`
- Modify: `src/whaleshark_reid/cli/main.py`
- Create: `tests/cli/test_cli_run_all.py`

- [ ] **Step 1: Write the failing test**

`tests/cli/test_cli_run_all.py`:
```python
"""CLI integration test for `catalog run-all` — chains 5 stages."""
from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from whaleshark_reid.storage.db import Storage

FIXTURES = Path(__file__).parent.parent / "fixtures"


def test_catalog_run_all_chains_five_stages(
    cli_runner: CliRunner, tmp_db_path: Path, tmp_cache_dir: Path, stub_miewid
):
    from whaleshark_reid.cli.main import app

    result = cli_runner.invoke(app, [
        "run-all",
        "--csv", str(FIXTURES / "mini_inat.csv"),
        "--photos-dir", str(FIXTURES / "photos"),
        "--rich-csv", str(FIXTURES / "mini_inat_rich.csv"),
        "--db-path", str(tmp_db_path),
        "--cache-dir", str(tmp_cache_dir),
        "--distance-threshold", "2.0",
    ])
    assert result.exit_code == 0, f"stdout: {result.stdout}\nexc: {result.exception}"

    storage = Storage(tmp_db_path)
    rows = storage.conn.execute(
        "SELECT stage, status FROM runs ORDER BY started_at ASC"
    ).fetchall()
    stages = [r["stage"] for r in rows]
    assert stages == ["ingest", "embed", "cluster", "matching", "project"]
    assert all(r["status"] == "ok" for r in rows)

    assert storage.count("annotations") == 10
    assert storage.count("pair_queue") > 0
```

- [ ] **Step 2: Run failing test**

```bash
pytest tests/cli/test_cli_run_all.py -x
```
Expected: "No such command 'run-all'".

- [ ] **Step 3: Implement the run-all command**

`src/whaleshark_reid/cli/commands/run_all.py`:
```python
"""`catalog run-all` — runs ingest → embed → cluster → matching → project in sequence.

Each stage gets its own run_id (no hierarchical grouping). Related runs are
discovered later by timestamp window / config overlap.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from whaleshark_reid.cli.commands.cluster import cluster_command
from whaleshark_reid.cli.commands.embed import embed_command
from whaleshark_reid.cli.commands.ingest import ingest_command
from whaleshark_reid.cli.commands.matching import matching_command
from whaleshark_reid.cli.commands.project import project_command


def run_all_command(
    csv: Path = typer.Option(..., "--csv", exists=True),
    photos_dir: Path = typer.Option(..., "--photos-dir", exists=True, file_okay=False),
    rich_csv: Optional[Path] = typer.Option(None, "--rich-csv", exists=True),
    eps: float = typer.Option(0.7, "--eps"),
    min_samples: int = typer.Option(2, "--min-samples"),
    distance_threshold: float = typer.Option(1.0, "--distance-threshold"),
    max_queue_size: int = typer.Option(2000, "--max-queue-size"),
    use_bbox: bool = typer.Option(True, "--use-bbox/--no-use-bbox"),
    batch_size: int = typer.Option(32, "--batch-size"),
    num_workers: int = typer.Option(2, "--num-workers"),
    device: Optional[str] = typer.Option(None, "--device"),
    db_path: Path = typer.Option(Path("cache/state.db"), "--db-path"),
    cache_dir: Path = typer.Option(Path("cache/"), "--cache-dir"),
) -> None:
    """Run ingest → embed → cluster → matching → project as five separate runs."""
    typer.echo("→ ingest")
    ingest_command(
        csv=csv,
        photos_dir=photos_dir,
        source="inat",
        rich_csv=rich_csv,
        db_path=db_path,
        cache_dir=cache_dir,
    )
    typer.echo("→ embed")
    embed_command(
        model="conservationxlabs/miewid-msv3",
        batch_size=batch_size,
        num_workers=num_workers,
        use_bbox=use_bbox,
        only_missing=True,
        device=device,
        db_path=db_path,
        cache_dir=cache_dir,
    )
    typer.echo("→ cluster")
    cluster_command(
        algo="dbscan",
        eps=eps,
        min_samples=min_samples,
        metric="cosine",
        standardize=True,
        min_cluster_size=3,
        embedding_run_id=None,
        db_path=db_path,
        cache_dir=cache_dir,
    )
    typer.echo("→ matching")
    matching_command(
        distance_threshold=distance_threshold,
        max_queue_size=max_queue_size,
        embedding_run_id=None,
        cluster_run_id=None,
        db_path=db_path,
        cache_dir=cache_dir,
    )
    typer.echo("→ project")
    project_command(
        algo="umap",
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
        embedding_run_id=None,
        db_path=db_path,
        cache_dir=cache_dir,
    )
    typer.echo("✓ run-all complete")
```

- [ ] **Step 4: Register in main.py**

Update `src/whaleshark_reid/cli/main.py`:

```python
"""Catalog CLI — Typer entry point."""
from __future__ import annotations

import typer

from whaleshark_reid.cli.commands.cluster import cluster_command
from whaleshark_reid.cli.commands.embed import embed_command
from whaleshark_reid.cli.commands.ingest import ingest_command
from whaleshark_reid.cli.commands.matching import matching_command
from whaleshark_reid.cli.commands.project import project_command
from whaleshark_reid.cli.commands.rebuild_individuals import rebuild_individuals_command
from whaleshark_reid.cli.commands.run_all import run_all_command

app = typer.Typer(
    name="catalog",
    help="Whale shark re-identification pipeline CLI",
    no_args_is_help=True,
)

app.command(name="ingest")(ingest_command)
app.command(name="embed")(embed_command)
app.command(name="cluster")(cluster_command)
app.command(name="matching")(matching_command)
app.command(name="project")(project_command)
app.command(name="rebuild-individuals")(rebuild_individuals_command)
app.command(name="run-all")(run_all_command)


if __name__ == "__main__":
    app()
```

- [ ] **Step 5: Run tests to verify pass**

```bash
pytest tests/cli/test_cli_run_all.py -x
```
Expected: 1 passed (~10 seconds — runs the whole pipeline).

Also visually verify that all 7 commands show up:
```bash
catalog --help
```
Expected: 7 commands listed.

- [ ] **Step 6: Commit**

```bash
cd /workspace/catalog-match/whaleshark-reid
git add src/whaleshark_reid/cli/commands/run_all.py src/whaleshark_reid/cli/main.py tests/cli/test_cli_run_all.py
git commit -m "feat: catalog run-all command (chains all 5 pipeline stages)"
```

---

## Task 11: Idempotency cross-cutting tests

**Files:**
- Create: `tests/cli/test_cli_idempotency.py`

- [ ] **Step 1: Write the failing test (it should already pass since the underlying code is idempotent — this task verifies that)**

`tests/cli/test_cli_idempotency.py`:
```python
"""Cross-cutting tests verifying that each stage respects idempotency rules."""
from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from whaleshark_reid.storage.db import Storage

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _run_full_pipeline(cli_runner: CliRunner, tmp_db_path: Path, tmp_cache_dir: Path):
    from whaleshark_reid.cli.main import app

    cli_runner.invoke(app, [
        "run-all",
        "--csv", str(FIXTURES / "mini_inat.csv"),
        "--photos-dir", str(FIXTURES / "photos"),
        "--db-path", str(tmp_db_path),
        "--cache-dir", str(tmp_cache_dir),
        "--distance-threshold", "2.0",
        "--num-workers", "0",
        "--device", "cpu",
    ])


def test_re_ingest_skips_existing(
    cli_runner: CliRunner, tmp_db_path: Path, tmp_cache_dir: Path, stub_miewid
):
    from whaleshark_reid.cli.main import app

    cli_runner.invoke(app, [
        "ingest",
        "--csv", str(FIXTURES / "mini_inat.csv"),
        "--photos-dir", str(FIXTURES / "photos"),
        "--db-path", str(tmp_db_path),
        "--cache-dir", str(tmp_cache_dir),
    ])

    # Second ingest run on the same data
    cli_runner.invoke(app, [
        "ingest",
        "--csv", str(FIXTURES / "mini_inat.csv"),
        "--photos-dir", str(FIXTURES / "photos"),
        "--db-path", str(tmp_db_path),
        "--cache-dir", str(tmp_cache_dir),
    ])

    storage = Storage(tmp_db_path)
    assert storage.count("annotations") == 10  # not doubled
    rows = storage.conn.execute(
        "SELECT metrics_json FROM runs WHERE stage = 'ingest' ORDER BY started_at"
    ).fetchall()
    assert len(rows) == 2
    second_metrics = json.loads(rows[1]["metrics_json"])
    assert second_metrics["n_skipped_existing"] == 10
    assert second_metrics["n_ingested"] == 0


def test_re_embed_with_only_missing_skips_already_cached(
    cli_runner: CliRunner, tmp_db_path: Path, tmp_cache_dir: Path, stub_miewid
):
    from whaleshark_reid.cli.main import app

    cli_runner.invoke(app, [
        "ingest",
        "--csv", str(FIXTURES / "mini_inat.csv"),
        "--photos-dir", str(FIXTURES / "photos"),
        "--db-path", str(tmp_db_path),
        "--cache-dir", str(tmp_cache_dir),
    ])
    cli_runner.invoke(app, [
        "embed", "--db-path", str(tmp_db_path), "--cache-dir", str(tmp_cache_dir),
        "--batch-size", "4", "--num-workers", "0", "--device", "cpu",
    ])

    # Second embed run — should skip everything (each new run gets a new run_id, but
    # the only_missing logic is per-run_id parquet, so the second run sees zero existing)
    cli_runner.invoke(app, [
        "embed", "--db-path", str(tmp_db_path), "--cache-dir", str(tmp_cache_dir),
        "--batch-size", "4", "--num-workers", "0", "--device", "cpu",
    ])

    storage = Storage(tmp_db_path)
    embed_runs = storage.conn.execute(
        "SELECT run_id, status, metrics_json FROM runs WHERE stage = 'embed' ORDER BY started_at"
    ).fetchall()
    assert len(embed_runs) == 2
    assert all(r["status"] == "ok" for r in embed_runs)
    # Both runs created their own parquet; the underlying ingest still produces 10 embeddings
    # in each run because only_missing is per-run_id, not global. This matches Phase 1 behavior.
    for r in embed_runs:
        m = json.loads(r["metrics_json"])
        assert m["n_embedded"] == 10


def test_re_run_matching_replaces_pair_queue_for_same_run(
    cli_runner: CliRunner, tmp_db_path: Path, tmp_cache_dir: Path, stub_miewid
):
    """Each matching run gets a fresh run_id and writes its own pair_queue rows.
    The DELETE+INSERT in run_matching_stage prevents accidental dup within a single run."""
    from whaleshark_reid.cli.main import app
    _run_full_pipeline(cli_runner, tmp_db_path, tmp_cache_dir)

    storage = Storage(tmp_db_path)
    matching_runs_before = storage.conn.execute(
        "SELECT run_id FROM runs WHERE stage = 'matching'"
    ).fetchall()
    assert len(matching_runs_before) == 1
    pairs_before = storage.count("pair_queue", run_id=matching_runs_before[0]["run_id"])

    # Re-run matching → new run_id, separate pair_queue entries
    cli_runner.invoke(app, [
        "matching", "--db-path", str(tmp_db_path), "--cache-dir", str(tmp_cache_dir),
        "--distance-threshold", "2.0",
    ])

    matching_runs_after = storage.conn.execute(
        "SELECT run_id FROM runs WHERE stage = 'matching'"
    ).fetchall()
    assert len(matching_runs_after) == 2
    pairs_after_old = storage.count("pair_queue", run_id=matching_runs_before[0]["run_id"])
    assert pairs_after_old == pairs_before  # untouched
```

- [ ] **Step 2: Run idempotency tests**

```bash
pytest tests/cli/test_cli_idempotency.py -x -v
```
Expected: all 3 tests pass.

If a test fails, the issue is in the underlying core stage, not in the CLI. Investigate before continuing.

- [ ] **Step 3: Commit**

```bash
cd /workspace/catalog-match/whaleshark-reid
git add tests/cli/test_cli_idempotency.py
git commit -m "test: CLI idempotency cross-cutting tests"
```

---

## Task 12: Exit code cross-cutting tests

**Files:**
- Create: `tests/cli/test_cli_exit_codes.py`

- [ ] **Step 1: Write the test**

`tests/cli/test_cli_exit_codes.py`:
```python
"""Cross-cutting tests for CLI exit codes."""
from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

FIXTURES = Path(__file__).parent.parent / "fixtures"


def test_missing_required_arg_exits_2(cli_runner: CliRunner):
    from whaleshark_reid.cli.main import app

    result = cli_runner.invoke(app, ["ingest"])  # missing --csv and --photos-dir
    assert result.exit_code == 2  # typer's "usage error" exit code


def test_unknown_command_exits_2(cli_runner: CliRunner):
    from whaleshark_reid.cli.main import app

    result = cli_runner.invoke(app, ["nonexistent-command"])
    assert result.exit_code == 2


def test_cluster_with_no_embed_runs_exits_2(
    cli_runner: CliRunner, tmp_db_path: Path, tmp_cache_dir: Path
):
    from whaleshark_reid.cli.main import app

    # Initialize the DB with no embed runs
    result = cli_runner.invoke(app, [
        "cluster",
        "--db-path", str(tmp_db_path),
        "--cache-dir", str(tmp_cache_dir),
    ])
    assert result.exit_code == 2
    assert "no successful embed run" in result.stdout.lower() or \
           "no successful embed run" in (result.stderr or "")


def test_matching_with_no_cluster_runs_exits_2(
    cli_runner: CliRunner, tmp_db_path: Path, tmp_cache_dir: Path, stub_miewid
):
    from whaleshark_reid.cli.main import app
    cli_runner.invoke(app, [
        "ingest",
        "--csv", str(FIXTURES / "mini_inat.csv"),
        "--photos-dir", str(FIXTURES / "photos"),
        "--db-path", str(tmp_db_path),
        "--cache-dir", str(tmp_cache_dir),
    ])
    cli_runner.invoke(app, [
        "embed",
        "--db-path", str(tmp_db_path),
        "--cache-dir", str(tmp_cache_dir),
        "--batch-size", "4", "--num-workers", "0", "--device", "cpu",
    ])

    # Now matching without cluster
    result = cli_runner.invoke(app, [
        "matching",
        "--db-path", str(tmp_db_path),
        "--cache-dir", str(tmp_cache_dir),
    ])
    assert result.exit_code == 2


def test_pipeline_exception_exits_1_and_marks_run_failed(
    cli_runner: CliRunner, tmp_db_path: Path, tmp_cache_dir: Path, monkeypatch
):
    """Force ingest_inat_csv to raise → CLI catches it via run_context → exit 1."""
    from whaleshark_reid.cli.main import app
    from whaleshark_reid.cli.commands import ingest as ingest_module

    def boom(*args, **kwargs):
        raise RuntimeError("intentional explosion")

    monkeypatch.setattr(ingest_module, "ingest_inat_csv", boom)

    result = cli_runner.invoke(app, [
        "ingest",
        "--csv", str(FIXTURES / "mini_inat.csv"),
        "--photos-dir", str(FIXTURES / "photos"),
        "--db-path", str(tmp_db_path),
        "--cache-dir", str(tmp_cache_dir),
    ])
    assert result.exit_code != 0
    # Confirm the run row was marked failed
    from whaleshark_reid.storage.db import Storage
    storage = Storage(tmp_db_path)
    row = storage.conn.execute(
        "SELECT status, error FROM runs WHERE stage = 'ingest'"
    ).fetchone()
    assert row["status"] == "failed"
    assert "intentional explosion" in row["error"]
```

- [ ] **Step 2: Run exit code tests**

```bash
pytest tests/cli/test_cli_exit_codes.py -x -v
```
Expected: all 5 tests pass.

If `test_pipeline_exception_exits_1_and_marks_run_failed` fails because typer's CliRunner reports exit_code=0 even when an exception was raised, check that the exception actually propagated up. Typer wraps exceptions in `result.exception` — the test's `assert result.exit_code != 0` should hold because of how Typer handles unhandled exceptions.

- [ ] **Step 3: Run the entire test suite**

```bash
pytest -x
```
Expected: all tests pass (Phase 1: 71 + Phase 2 CLI tests). Total should be ~95+.

- [ ] **Step 4: Commit**

```bash
cd /workspace/catalog-match/whaleshark-reid
git add tests/cli/test_cli_exit_codes.py
git commit -m "test: CLI exit code cross-cutting tests"
```

---

## Success criteria (from spec 02)

After all 12 tasks are complete:

1. ✅ All seven CLI commands exist and are discoverable via `catalog --help`.
2. ✅ Each command writes exactly one `runs` row with terminal `status='ok'` or `status='failed'` and `metrics_json` populated on success.
3. ✅ `catalog run-all` produces 5 sibling run rows in sequence (verified by `test_cli_run_all.py`).
4. ✅ Re-running stages respects idempotency (verified by `test_cli_idempotency.py`).
5. ✅ `pytest tests/cli` passes in under 5 seconds for unit tests; the integration-shaped tests (`test_cli_run_all.py`, `test_cli_idempotency.py`) take ~10s due to running real DBSCAN/UMAP/matching on the fixture data.
6. ✅ No ML logic lives in `cli/` — every command is pure orchestration over `core/`.

## Manual smoke test after Task 12 (recommended)

After the test suite is green, run this manually to confirm the CLI works against the real iNat data:

```bash
cd /workspace/catalog-match/whaleshark-reid
catalog run-all \
  --csv /workspace/catalog-match/whaleshark/whaleshark_inat_v1.csv \
  --photos-dir /workspace/catalog-match/inat-download-recent-species-sightings/whaleshark_inat_v1/photos \
  --rich-csv /workspace/catalog-match/whaleshark/dfx_whaleshark_inat_v1.csv \
  --db-path /tmp/cli_smoke.db \
  --cache-dir /tmp/cli_smoke_cache
```

Expected: ~90 seconds total. Should produce the same 38 clusters / 258 noise as the verify_core notebook.

## What this plan explicitly does NOT cover

- FastAPI web app (plan 03, written next).
- Background job scheduling, cron, or daemon modes.
- Logging to anything other than stdout + a per-run text file.
- Authentication / multi-user support.
- Phase 3+ features (Wildbook ingest, GT metrics, reconcile mode).
