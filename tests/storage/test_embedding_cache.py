"""Tests for parquet-backed embedding cache."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from whaleshark_reid.storage.embedding_cache import (
    existing_annotation_uuids,
    read_embeddings,
    read_embeddings_as_array,
    write_embeddings,
)


def test_write_and_read_embeddings_roundtrip(tmp_cache_dir: Path):
    rows = [
        {
            "annotation_uuid": f"uuid-{i}",
            "embedding": [0.1 * i, 0.2 * i, 0.3 * i],
            "model_id": "miewid-msv3",
            "model_version": "msv3",
            "created_at": "2026-04-10T00:00:00Z",
        }
        for i in range(5)
    ]

    out_path = write_embeddings(tmp_cache_dir, "run_abc", rows)
    assert out_path.exists()

    df = read_embeddings(tmp_cache_dir, "run_abc")
    assert len(df) == 5
    assert set(df.columns) == {"annotation_uuid", "embedding", "model_id", "model_version", "created_at"}


def test_read_embeddings_as_array(tmp_cache_dir: Path):
    rows = [
        {
            "annotation_uuid": "a",
            "embedding": [1.0, 2.0, 3.0],
            "model_id": "m",
            "model_version": "v",
            "created_at": "t",
        },
        {
            "annotation_uuid": "b",
            "embedding": [4.0, 5.0, 6.0],
            "model_id": "m",
            "model_version": "v",
            "created_at": "t",
        },
    ]
    write_embeddings(tmp_cache_dir, "run_abc", rows)

    uuids, mat = read_embeddings_as_array(tmp_cache_dir, "run_abc")
    assert uuids == ["a", "b"]
    assert mat.shape == (2, 3)
    assert np.allclose(mat[0], [1.0, 2.0, 3.0])


def test_existing_annotation_uuids(tmp_cache_dir: Path):
    rows = [
        {"annotation_uuid": f"u{i}", "embedding": [0.0], "model_id": "m",
         "model_version": "v", "created_at": "t"}
        for i in range(3)
    ]
    write_embeddings(tmp_cache_dir, "run_abc", rows)

    assert existing_annotation_uuids(tmp_cache_dir, "run_abc") == {"u0", "u1", "u2"}


def test_existing_annotation_uuids_empty_when_no_file(tmp_cache_dir: Path):
    assert existing_annotation_uuids(tmp_cache_dir, "run_missing") == set()
