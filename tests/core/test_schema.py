"""Tests for core.schema — Annotation, UUIDs, result dataclasses."""
from __future__ import annotations

import uuid

import pytest

from whaleshark_reid.core.schema import (
    Annotation,
    ClusterResult,
    EmbedResult,
    INAT_NAMESPACE,
    IngestResult,
    MatchingResult,
    PairCandidate,
    ProjectionPoint,
    RebuildResult,
    inat_annotation_uuid,
    inat_image_uuid,
    new_name_uuid,
)


def test_inat_annotation_uuid_is_deterministic():
    u1 = inat_annotation_uuid(327286790, 0)
    u2 = inat_annotation_uuid(327286790, 0)
    assert u1 == u2
    assert uuid.UUID(u1)  # parseable


def test_inat_annotation_uuid_differs_by_photo_index():
    assert inat_annotation_uuid(327286790, 0) != inat_annotation_uuid(327286790, 1)


def test_inat_image_uuid_is_deterministic_and_distinct_from_annotation_uuid():
    ann = inat_annotation_uuid(327286790, 0)
    img = inat_image_uuid(327286790, 0)
    assert ann != img
    assert img == inat_image_uuid(327286790, 0)


def test_new_name_uuid_is_unique_uuid4():
    u1 = new_name_uuid()
    u2 = new_name_uuid()
    assert u1 != u2
    parsed = uuid.UUID(u1)
    assert parsed.version == 4


def test_inat_namespace_is_stable():
    assert str(INAT_NAMESPACE) == "6f4e6a5e-7b7a-4f3b-9c1d-1f0a2c3d4e5f"


def test_annotation_round_trip():
    ann = Annotation(
        annotation_uuid=inat_annotation_uuid(327286790, 0),
        image_uuid=inat_image_uuid(327286790, 0),
        source="inat",
        observation_id=327286790,
        photo_index=0,
        file_path="/tmp/327286790_1.jpg",
        file_name="327286790_1.jpg",
        bbox=[201.68, 4.49, 1349.56, 708.53],
        theta=0.0,
        viewpoint="unknown",
        species="whaleshark",
    )
    dumped = ann.model_dump()
    re = Annotation(**dumped)
    assert re == ann


def test_annotation_name_defaults_none():
    ann = Annotation(
        annotation_uuid=inat_annotation_uuid(1, 0),
        image_uuid=inat_image_uuid(1, 0),
        source="inat",
        file_path="/x.jpg",
        file_name="x.jpg",
        bbox=[0, 0, 10, 10],
    )
    assert ann.name is None
    assert ann.name_uuid is None


def test_pair_candidate_defaults():
    p = PairCandidate(
        ann_a_uuid="a",
        ann_b_uuid="b",
        distance=0.3,
    )
    assert p.same_cluster is False
    assert p.cluster_a is None


def test_cluster_result_holds_noise_label():
    r = ClusterResult(
        annotation_uuid="a",
        cluster_label=-1,
        cluster_algo="dbscan",
        cluster_params={"eps": 0.7},
    )
    assert r.cluster_label == -1


def test_projection_point_shape():
    p = ProjectionPoint(
        annotation_uuid="a",
        x=1.2,
        y=-3.4,
        algo="umap",
        params={"n_neighbors": 15},
    )
    assert p.x == 1.2


def test_ingest_result_fields():
    r = IngestResult(n_read=10, n_ingested=8, n_skipped_existing=2, n_missing_files=1)
    assert r.n_read == 10


def test_embed_result_fields():
    r = EmbedResult(
        n_embedded=10,
        n_skipped_existing=0,
        n_failed=0,
        embed_dim=512,
        model_id="conservationxlabs/miewid-msv3",
        duration_s=1.5,
    )
    assert r.embed_dim == 512


def test_matching_result_fields():
    r = MatchingResult(
        n_pairs=100,
        n_same_cluster=20,
        n_cross_cluster=80,
        n_filtered_out_by_decisions=5,
        median_distance=0.4,
        min_distance=0.1,
        max_distance=0.9,
        p10=0.15, p25=0.25, p50=0.4, p75=0.55, p90=0.8,
    )
    assert r.n_pairs == 100


def test_rebuild_result_fields():
    r = RebuildResult(n_components=3, n_singletons=5, n_annotations_updated=9)
    assert r.n_components == 3
