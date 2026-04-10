"""Core data schema — Annotation, UUID helpers, and stage result dataclasses.

All primary identifiers are UUIDs (strings). Integer IDs from source systems
(iNat observation_id, COCO id) are preserved as separate traceability columns
but never used as keys — this is what makes multi-catalog merging collision-safe.
"""
from __future__ import annotations

import uuid
from typing import Optional

from pydantic import BaseModel, Field

# Fixed namespace for deterministic uuid5() generation on iNat sources.
# Stable across ingest runs so re-running ingest is idempotent.
INAT_NAMESPACE = uuid.UUID("6f4e6a5e-7b7a-4f3b-9c1d-1f0a2c3d4e5f")


def inat_annotation_uuid(observation_id: int, photo_index: int) -> str:
    return str(uuid.uuid5(INAT_NAMESPACE, f"inat:annotation:{observation_id}:{photo_index}"))


def inat_image_uuid(observation_id: int, photo_index: int) -> str:
    return str(uuid.uuid5(INAT_NAMESPACE, f"inat:image:{observation_id}:{photo_index}"))


def new_name_uuid() -> str:
    """Fresh uuid4 for a derived individual. Called by feedback rebuild."""
    return str(uuid.uuid4())


class Annotation(BaseModel):
    # --- Canonical UUID identifiers (primary) ---
    annotation_uuid: str
    image_uuid: str
    name_uuid: Optional[str] = None

    # --- Source reference IDs (traceability only) ---
    source: str
    source_annotation_id: Optional[str] = None
    source_image_id: Optional[str] = None
    source_individual_id: Optional[str] = None
    observation_id: Optional[int] = None
    photo_index: Optional[int] = None

    # --- MiewID-required fields (names match MiewIdDataset.__getitem__) ---
    file_path: str
    file_name: str
    bbox: list[float] = Field(..., description="[x, y, w, h]")
    theta: float = 0.0
    viewpoint: str = "unknown"
    species: str = "whaleshark"
    name: Optional[str] = None

    # --- Provenance / dev-mode display ---
    photographer: Optional[str] = None
    license: Optional[str] = None
    date_captured: Optional[str] = None
    gps_lat_captured: Optional[float] = None
    gps_lon_captured: Optional[float] = None
    coco_url: Optional[str] = None
    flickr_url: Optional[str] = None
    height: Optional[int] = None
    width: Optional[int] = None
    conf: Optional[float] = None
    quality_grade: Optional[str] = None

    # --- Derived ---
    name_viewpoint: Optional[str] = None
    species_viewpoint: Optional[str] = None


class PairCandidate(BaseModel):
    ann_a_uuid: str
    ann_b_uuid: str
    distance: float
    cluster_a: Optional[int] = None
    cluster_b: Optional[int] = None
    same_cluster: bool = False


class ClusterResult(BaseModel):
    annotation_uuid: str
    cluster_label: int
    cluster_algo: str
    cluster_params: dict


class ProjectionPoint(BaseModel):
    annotation_uuid: str
    x: float
    y: float
    algo: str
    params: dict


# --- Stage result dataclasses (returned by core.*.run_*_stage, attached to runs.metrics_json) ---

class IngestResult(BaseModel):
    n_read: int
    n_ingested: int
    n_skipped_existing: int
    n_missing_files: int


class EmbedResult(BaseModel):
    n_embedded: int
    n_skipped_existing: int
    n_failed: int
    embed_dim: int
    model_id: str
    duration_s: float


class ClusterStageResult(BaseModel):
    algo: str
    n_clusters: int
    n_noise: int
    largest_cluster_size: int
    singleton_fraction: float
    median_cluster_size: float


class ProjectStageResult(BaseModel):
    algo: str
    n_points: int
    params: dict


class MatchingResult(BaseModel):
    n_pairs: int
    n_same_cluster: int
    n_cross_cluster: int
    n_filtered_out_by_decisions: int
    median_distance: float
    min_distance: float
    max_distance: float
    p10: float
    p25: float
    p50: float
    p75: float
    p90: float


class RebuildResult(BaseModel):
    n_components: int
    n_singletons: int
    n_annotations_updated: int
