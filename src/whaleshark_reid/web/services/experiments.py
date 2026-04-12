"""Experiments service — list runs, run detail, diff two runs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from whaleshark_reid.storage.db import Storage


class RunSummary(BaseModel):
    run_id: str
    stage: str
    status: str
    config_preview: str
    metrics_preview: str
    started_at: str
    finished_at: Optional[str] = None
    duration_s: Optional[float] = None


class RunDetail(BaseModel):
    run: RunSummary
    config: dict
    metrics: dict
    reproduce_cmd: str
    log_tail: list[str]


class DiffLine(BaseModel):
    key: str
    val_a: str
    val_b: str
    changed: bool


class RunDiff(BaseModel):
    run_a: RunDetail
    run_b: RunDetail
    config_diff: list[DiffLine]
    metrics_diff: list[DiffLine]


def _row_to_summary(row) -> RunSummary:
    config_str = ""
    if row["config_json"]:
        try:
            c = json.loads(row["config_json"])
            config_str = ", ".join(f"{k}={v}" for k, v in list(c.items())[:3])
        except json.JSONDecodeError:
            pass

    metrics_str = ""
    if row["metrics_json"]:
        try:
            m = json.loads(row["metrics_json"])
            metrics_str = ", ".join(f"{k}={v}" for k, v in list(m.items())[:4])
        except json.JSONDecodeError:
            pass

    duration = None
    if row["started_at"] and row["finished_at"]:
        try:
            from datetime import datetime
            t0 = datetime.fromisoformat(row["started_at"])
            t1 = datetime.fromisoformat(row["finished_at"])
            duration = round((t1 - t0).total_seconds(), 1)
        except (ValueError, TypeError):
            pass

    return RunSummary(
        run_id=row["run_id"],
        stage=row["stage"],
        status=row["status"],
        config_preview=config_str,
        metrics_preview=metrics_str,
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        duration_s=duration,
    )


def list_runs(storage: Storage) -> list[RunSummary]:
    rows = storage.conn.execute(
        "SELECT * FROM runs WHERE stage != 'adhoc_queue' ORDER BY started_at DESC"
    ).fetchall()
    return [_row_to_summary(r) for r in rows]


def _build_reproduce_cmd(config: dict, stage: str) -> str:
    parts = [f"catalog {stage}"]
    for k, v in config.items():
        if v is not None and v != "" and k not in ("source",):
            flag = f"--{k.replace('_', '-')}"
            parts.append(f"{flag} {v}")
    return " ".join(parts)


def get_run_detail(storage: Storage, cache_dir: Path, run_id: str) -> Optional[RunDetail]:
    row = storage.conn.execute(
        "SELECT * FROM runs WHERE run_id = ?", (run_id,)
    ).fetchone()
    if row is None:
        return None

    config = json.loads(row["config_json"]) if row["config_json"] else {}
    metrics = json.loads(row["metrics_json"]) if row["metrics_json"] else {}

    log_path = cache_dir / "logs" / f"{run_id}.log"
    log_tail: list[str] = []
    if log_path.exists():
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        log_tail = lines[-20:]

    return RunDetail(
        run=_row_to_summary(row),
        config=config,
        metrics=metrics,
        reproduce_cmd=_build_reproduce_cmd(config, row["stage"]),
        log_tail=log_tail,
    )


def diff_runs(
    storage: Storage, cache_dir: Path, run_a: str, run_b: str
) -> Optional[RunDiff]:
    detail_a = get_run_detail(storage, cache_dir, run_a)
    detail_b = get_run_detail(storage, cache_dir, run_b)
    if detail_a is None or detail_b is None:
        return None

    all_config_keys = sorted(set(detail_a.config) | set(detail_b.config))
    config_diff = [
        DiffLine(
            key=k,
            val_a=str(detail_a.config.get(k, "—")),
            val_b=str(detail_b.config.get(k, "—")),
            changed=detail_a.config.get(k) != detail_b.config.get(k),
        )
        for k in all_config_keys
    ]

    all_metrics_keys = sorted(set(detail_a.metrics) | set(detail_b.metrics))
    metrics_diff = [
        DiffLine(
            key=k,
            val_a=str(detail_a.metrics.get(k, "—")),
            val_b=str(detail_b.metrics.get(k, "—")),
            changed=detail_a.metrics.get(k) != detail_b.metrics.get(k),
        )
        for k in all_metrics_keys
    ]

    return RunDiff(
        run_a=detail_a, run_b=detail_b,
        config_diff=config_diff, metrics_diff=metrics_diff,
    )
