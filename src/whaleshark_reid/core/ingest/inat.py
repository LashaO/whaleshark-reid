"""iNat CSV ingest.

Reads either the minimal 10-column schema (theta, viewpoint, name, file_name,
species, file_path, x, y, w, h) or the richer dfx schema (41 columns with
provenance). Auto-detects by presence of 'observation_id' column.

For Phase 1 cold-start, 'name' is always 'unknown' → stored as NULL. Viewpoint
stays as 'unknown' (it is categorical, not an identity).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import pandas as pd

from whaleshark_reid.core.schema import (
    Annotation,
    IngestResult,
    inat_annotation_uuid,
    inat_image_uuid,
)

# Filename patterns like "327286790_1.jpg" or "obs100_1.jpg" → observation_id, photo_index (1-based → 0-based)
_FILENAME_RE = re.compile(r"^.*?(\d+)_(\d+)\.(?:jpg|jpeg|png)$", re.IGNORECASE)


def _parse_obs_and_index_from_filename(filename: str) -> tuple[int, int]:
    m = _FILENAME_RE.match(filename)
    if not m:
        raise ValueError(f"Filename does not match <obs_id>_<n>.jpg pattern: {filename}")
    return int(m.group(1)), int(m.group(2)) - 1  # 1-based → 0-based


def _normalize_name(raw) -> Optional[str]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    s = str(raw).strip()
    return None if s.lower() in ("", "unknown", "nan") else s


def ingest_inat_csv(
    csv_path: Path,
    photos_dir: Path,
    storage,
    run_id: str,
    rich_csv_path: Optional[Path] = None,
) -> IngestResult:
    df = pd.read_csv(csv_path)

    rich_df: Optional[pd.DataFrame] = None
    if rich_csv_path is not None:
        rich_df = pd.read_csv(rich_csv_path).set_index("observation_id")

    n_read = len(df)
    n_ingested = 0
    n_skipped_existing = 0
    n_missing_files = 0

    for _, row in df.iterrows():
        file_name = str(row["file_name"])
        obs_id, photo_index = _parse_obs_and_index_from_filename(file_name)

        resolved_path = photos_dir / file_name
        if not resolved_path.exists():
            n_missing_files += 1

        # Base fields from the minimal CSV
        ann_kwargs = dict(
            annotation_uuid=inat_annotation_uuid(obs_id, photo_index),
            image_uuid=inat_image_uuid(obs_id, photo_index),
            source="inat",
            observation_id=obs_id,
            photo_index=photo_index,
            file_path=str(resolved_path),
            file_name=file_name,
            bbox=[float(row["x"]), float(row["y"]), float(row["w"]), float(row["h"])],
            theta=float(row.get("theta", 0.0) or 0.0),
            viewpoint=str(row.get("viewpoint", "unknown")) or "unknown",
            species=str(row.get("species", "whaleshark")) or "whaleshark",
            name=_normalize_name(row.get("name")),
        )

        # Backfill provenance from rich CSV if available
        if rich_df is not None and obs_id in rich_df.index:
            r = rich_df.loc[obs_id]
            for col, field in [
                ("photographer", "photographer"),
                ("license", "license"),
                ("date_captured", "date_captured"),
                ("gps_lat_captured", "gps_lat_captured"),
                ("gps_lon_captured", "gps_lon_captured"),
                ("coco_url", "coco_url"),
                ("flickr_url", "flickr_url"),
                ("height", "height"),
                ("width", "width"),
                ("conf", "conf"),
                ("quality_grade", "quality_grade"),
            ]:
                if col in r and not (isinstance(r[col], float) and pd.isna(r[col])):
                    val = r[col]
                    if field in ("height", "width"):
                        val = int(val)
                    elif field in ("gps_lat_captured", "gps_lon_captured", "conf"):
                        val = float(val)
                    ann_kwargs[field] = val

        # Derived composite keys (used later by Wildbook stratification; cheap to fill now)
        ann_kwargs["name_viewpoint"] = f"{ann_kwargs['name']}_{ann_kwargs['viewpoint']}"
        ann_kwargs["species_viewpoint"] = f"{ann_kwargs['species']}_{ann_kwargs['viewpoint']}"

        ann = Annotation(**ann_kwargs)

        before_changes = storage.conn.total_changes
        storage.upsert_annotation(ann, run_id=run_id)
        if storage.conn.total_changes > before_changes:
            n_ingested += 1
        else:
            n_skipped_existing += 1

    return IngestResult(
        n_read=n_read,
        n_ingested=n_ingested,
        n_skipped_existing=n_skipped_existing,
        n_missing_files=n_missing_files,
    )
