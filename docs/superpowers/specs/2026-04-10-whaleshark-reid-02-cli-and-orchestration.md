# Whale Shark Re-ID — Spec 02: CLI and Pipeline Orchestration

**Date:** 2026-04-10
**Depends on:** [01-core-engine.md](2026-04-10-whaleshark-reid-01-core-engine.md)
**Related specs:** [overview](2026-04-10-whaleshark-reid-overview.md), [03-web-ui](2026-04-10-whaleshark-reid-03-web-ui.md)

## Purpose

Wire the core engine up to a Typer-based CLI so the pipeline is scriptable, reproducible, and idempotent. The CLI is the **source of truth** for pipeline operations — the web UI's "run pipeline" button in spec 03 calls these same commands via subprocess.

This sub-spec does NOT reimplement any ML logic. It is a thin orchestration layer that parses arguments, constructs a `RunContext`, calls `core.*.run_*_stage()` entry points, and persists the `runs` + `experiments` rows.

## Module layout

```
src/whaleshark_reid/
└── cli/
    ├── __init__.py
    ├── main.py             # Typer app factory, `catalog` entry point
    ├── run_context.py      # RunContext dataclass
    ├── commands/
    │   ├── __init__.py
    │   ├── ingest.py
    │   ├── embed.py
    │   ├── cluster.py
    │   ├── matching.py
    │   ├── project.py
    │   ├── rebuild_individuals.py
    │   └── run_all.py
    └── logging_config.py   # Plain text logging to stdout and cache_dir/logs/<run_id>.log (rich-colorized in TTY)
```

Installed entry point in `pyproject.toml`:

```toml
[project.scripts]
catalog = "whaleshark_reid.cli.main:app"
```

Usage: `catalog ingest --csv ... --photos-dir ...`, `catalog run-all ...`, etc.

## 1. `RunContext`

Shared abstraction passed to every stage. Created once per CLI invocation, holds the DB handle, cache directory, run_id, logger, and git SHA. Metrics are attached at the end via `finish(status='ok', metrics={...})` — there is no separate experiments table, metrics land in `runs.metrics_json`.

```python
@dataclass
class RunContext:
    run_id: str                     # e.g. "run_20260410_143012_a3f8"
    stage: str                      # 'ingest' | 'embed' | 'cluster' | 'matching' | 'project' | 'rebuild' | 'all'
    storage: Storage
    cache_dir: Path                 # /workspace/catalog-match/whaleshark-reid/cache (default, configurable)
    config: dict                    # serialized to runs.config_json
    logger: Logger
    git_sha: Optional[str] = None
    started_at: datetime = field(default_factory=datetime.utcnow)

    @classmethod
    def new(cls, stage: str, storage: Storage, cache_dir: Path, config: dict) -> "RunContext":
        run_id = f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(2)}"
        storage.begin_run(run_id=run_id, stage=stage, config=config)
        return cls(run_id=run_id, stage=stage, storage=storage, cache_dir=cache_dir,
                   config=config, logger=get_logger(run_id), git_sha=detect_git_sha())

    def finish(self, status: str = "ok", metrics: Optional[dict] = None,
               error: Optional[str] = None, notes: str = "") -> None:
        self.storage.finish_run(self.run_id, status=status, metrics=metrics or {},
                                error=error, notes=notes)


@contextmanager
def run_context(stage: str, storage: Storage, cache_dir: Path, config: dict) -> Iterator[RunContext]:
    """Standardized try/except/finally wrapper around RunContext.new.
    On exception: calls ctx.finish('failed', error=str(e)) and re-raises.
    On success: each command is responsible for calling ctx.finish('ok', metrics=...)
    before the block exits (so metrics land in runs.metrics_json).
    If the command exits normally without calling finish, we call finish('ok', metrics={})
    as a safety net."""
    ctx = RunContext.new(stage=stage, storage=storage, cache_dir=cache_dir, config=config)
    try:
        yield ctx
    except Exception as e:
        ctx.finish("failed", error=str(e))
        raise
    else:
        # Safety net: if the command didn't explicitly call ctx.finish, do it now.
        if ctx.storage.get_run_status(ctx.run_id) == "running":
            ctx.finish("ok")
```

Every command uses this `run_context` wrapper so run rows always have a terminal status and metrics_json.

## 2. Typer commands

### `catalog ingest`

```python
@app.command()
def ingest(
    csv: Path = typer.Option(..., exists=True, help="Path to minimal or dfx CSV"),
    photos_dir: Path = typer.Option(..., exists=True, file_okay=False),
    source: str = typer.Option("inat"),
    rich_csv: Optional[Path] = typer.Option(None, exists=True, help="Optional dfx CSV for provenance backfill"),
    db_path: Path = typer.Option("cache/state.db"),
    cache_dir: Path = typer.Option("cache/"),
):
    storage = Storage(db_path)
    storage.init_schema()
    config = {"csv": str(csv), "photos_dir": str(photos_dir), "source": source, "rich_csv": str(rich_csv) if rich_csv else None}
    with run_context(stage="ingest", storage=storage, cache_dir=cache_dir, config=config) as ctx:
        result = ingest_inat_csv(csv, photos_dir, storage, ctx.run_id, rich_csv_path=rich_csv)
        ctx.finish(status="ok", metrics=result.model_dump())
    typer.echo(f"ingested {result.n_ingested} / skipped {result.n_skipped_existing} / missing {result.n_missing_files}")
```

### `catalog embed`

```python
@app.command()
def embed(
    model: str = "conservationxlabs/miewid-msv3",
    batch_size: int = 32,
    use_bbox: bool = True,
    db_path: Path = "cache/state.db",
    cache_dir: Path = "cache/",
    only_missing: bool = True,
    target_run_id: Optional[str] = None,  # optional: write embeddings under a specific run_id
):
    """Embed all annotations not yet in the cache. Writes to cache_dir/embeddings/<run_id>.parquet."""
```

### `catalog cluster`

```python
@app.command()
def cluster(
    algo: str = typer.Option("dbscan", help="dbscan | hdbscan"),
    eps: float = 0.7,
    min_samples: int = 2,
    metric: str = "cosine",
    standardize: bool = True,
    min_cluster_size: int = 3,   # HDBSCAN only
    embedding_run_id: Optional[str] = None,   # default: latest embed run
    db_path: Path = "cache/state.db",
    cache_dir: Path = "cache/",
):
    ...
```

### `catalog matching`

```python
@app.command()
def matching(
    distance_threshold: float = 1.0,
    max_queue_size: int = 2000,
    embedding_run_id: Optional[str] = None,
    cluster_run_id: Optional[str] = None,
    db_path: Path = "cache/state.db",
    cache_dir: Path = "cache/",
):
    ...
```

### `catalog project`

```python
@app.command()
def project(
    algo: str = "umap",
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    random_state: int = 42,
    embedding_run_id: Optional[str] = None,
    db_path: Path = "cache/state.db",
    cache_dir: Path = "cache/",
):
    ...
```

### `catalog rebuild-individuals`

```python
@app.command("rebuild-individuals")
def rebuild_individuals(
    db_path: Path = "cache/state.db",
):
    """Rebuild annotations.name_uuid from pair_decisions via union-find. Safe to run any time."""
```

### `catalog run-all`

```python
@app.command("run-all")
def run_all(
    csv: Path = typer.Option(..., exists=True),
    photos_dir: Path = typer.Option(..., exists=True, file_okay=False),
    rich_csv: Optional[Path] = None,
    eps: float = 0.7,
    min_samples: int = 2,
    distance_threshold: float = 1.0,
    max_queue_size: int = 2000,
    use_bbox: bool = True,
    db_path: Path = "cache/state.db",
    cache_dir: Path = "cache/",
):
    """Run ingest → embed → cluster → matching → project as five separate runs in sequence.

    Each stage gets its own run_id (no hierarchical grouping). The experiments view
    groups related runs by shared timestamp window / config overlap, not via a
    parent_run_id column.
    """
    storage = Storage(db_path)
    storage.init_schema()
    ingest_cmd(csv=csv, photos_dir=photos_dir, rich_csv=rich_csv, db_path=db_path, cache_dir=cache_dir)
    embed_cmd(batch_size=32, use_bbox=use_bbox, db_path=db_path, cache_dir=cache_dir)
    cluster_cmd(algo="dbscan", eps=eps, min_samples=min_samples, db_path=db_path, cache_dir=cache_dir)
    matching_cmd(distance_threshold=distance_threshold, max_queue_size=max_queue_size, db_path=db_path, cache_dir=cache_dir)
    project_cmd(algo="umap", db_path=db_path, cache_dir=cache_dir)
```

## 3. Logging

One logger per run, plain text only. Format: `[HH:MM:SS] [stage] [run_id] message`. Colorized via `rich` when running in a TTY. Written to both stdout (for CLI visibility) and `cache_dir/logs/<run_id>.log` (for the web UI's experiments page to tail).

No structured JSONL. If we later need structured queries over log lines, we add it then.

## 4. Idempotency rules

Every stage must be safely re-runnable:

- **ingest:** unique constraint on `(source, observation_id, photo_index)` in `annotations`. Re-ingest increments `n_skipped_existing`.
- **embed:** `only_missing=True` (default) filters out annotation_uuids already in the latest embedding parquet. Can be overridden to force re-embed.
- **cluster:** always re-runs on the latest embedding. Old cluster parquet stays on disk; cluster_run_id is stamped on the new run.
- **matching:** DELETE existing `pair_queue` rows for the current `run_id` before INSERT. `(run_id, ann_a_uuid, ann_b_uuid)` unique constraint prevents accidental dup.
- **project:** always re-runs; old projection parquet stays.
- **rebuild-individuals:** writes fresh uuid4 per component; idempotent in the sense that the same pair_decisions state always produces the same grouping structure (though the specific UUIDs differ across rebuilds, by design).
- **run-all:** each substage respects its own idempotency rules. There is no shared parent run.

## 5. Error handling & exit codes

- Missing image files at embed time: log warning, continue, exit 0.
- MiewID model load failure: exit 1 with clear message pointing at HuggingFace cache.
- SQLite locked (concurrent CLI + web): retry with backoff up to 10 seconds, then exit 2.
- Invalid cluster params (e.g. `eps <= 0`): pydantic validation fires before any work, exit 2.
- Pipeline stage exception: `ctx.finish('failed', error=...)`, print traceback, exit 1.

## 6. Experiment tracking contract

Every command writes ONE row to `runs`. The row is created with `status='running'` and `metrics_json=NULL` at the start, and updated at finish with `status='ok'|'failed'`, `metrics_json=<pydantic dump of the stage result>`, and `finished_at`. This single-row contract is what makes the experiments view in spec 03 work:

```
runs.config_json    = pydantic dump of the command's arguments
runs.metrics_json   = dict returned by the corresponding core.run_*_stage() function
                      (populated by ctx.finish(status='ok', metrics=...))
runs.git_sha        = output of `git rev-parse HEAD`, NULL if not in a repo
runs.status         = 'running' → 'ok' | 'failed'
```

No experiments table. No parent_run_id. Related runs are discovered by timestamp window / stage sequence, not by explicit parent linkage.

## 7. Tests (`tests/cli/`)

- `test_cli_ingest.py` — invoke `catalog ingest` via `typer.testing.CliRunner` on the fixture CSV + photos dir; assert exactly one `runs` row with `status='ok'` and `metrics_json` populated; exit 0.
- `test_cli_embed.py` — same pattern with monkeypatched MiewID stub.
- `test_cli_cluster.py`, `test_cli_matching.py`, `test_cli_project.py` — same pattern.
- `test_cli_run_all.py` — end-to-end on fixture data, verify 5 sibling `runs` rows with `status='ok'` (no parent linkage).
- `test_cli_idempotency.py` — run each stage twice, verify second invocation respects idempotency (embed skips cached, ingest skips existing, matching overwrites pair_queue).
- `test_cli_exit_codes.py` — bad args → exit 2, simulated pipeline exception → exit 1, success → exit 0.

## Success criteria

1. All seven CLI commands exist and are discoverable via `catalog --help`.
2. Each command writes exactly one `runs` row with terminal `status='ok'` or `status='failed'` and `metrics_json` populated on success.
3. `catalog run-all` produces 5 sibling run rows in sequence.
4. Re-running `run-all` on the same inputs produces another sequence where embed, cluster, matching short-circuit via idempotency (quick runs, correct cached outputs).
5. `pytest tests/cli` passes in under 5 seconds (uses the MiewID stub fixture for embed).
6. No ML logic lives in `cli/` — every command is pure orchestration over `core/`.

## What this spec explicitly does NOT cover

- FastAPI web app (spec 03).
- The actual matching/clustering/embedding code (spec 01).
- Web UI's "run pipeline" button wiring (spec 03 — it subprocesses these CLI commands).
- Background job scheduling, cron, or daemon modes (not needed; CLI is invoked on-demand).
