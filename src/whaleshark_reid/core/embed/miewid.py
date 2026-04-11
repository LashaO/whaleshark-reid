"""MiewID embedding extraction via in-process wbia_miew_id reuse.

This module owns almost nothing: it just builds a DataFrame from a list of
Annotations, hands it to wbia_miew_id's MiewIdDataset, runs the canonical
extract_embeddings() loop from wbia_miew_id.engine.eval_fn, and returns a
numpy array of shape (N, embed_dim).

No custom inference loop. No custom transforms. No custom get_chip_from_img
call — MiewIdDataset handles all of that internally. This is critical: the
repo uses Albumentations transforms (not torchvision) and any drift here would
silently change embedding quality from the benchmark notebook's outputs.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from whaleshark_reid.core.schema import Annotation, EmbedResult


def embed_annotations(
    annotations: list[Annotation],
    model_id: str = "conservationxlabs/miewid-msv3",
    image_size: tuple[int, int] = (440, 440),
    batch_size: int = 32,
    num_workers: int = 2,
    use_bbox: bool = True,
    device: Optional[str] = None,
) -> np.ndarray:
    """Extract MiewID embeddings for a list of annotations.

    Returns np.ndarray of shape (N, embed_dim) float32. Preserves input order.
    """
    # Imported lazily so the test stub can monkeypatch AutoModel before import-time side effects.
    from transformers import AutoModel
    from transformers.modeling_utils import PreTrainedModel
    from wbia_miew_id.datasets import MiewIdDataset, get_test_transforms
    from wbia_miew_id.engine.eval_fn import extract_embeddings as _extract_embeddings

    # Compatibility shim: transformers >= 5.x expects every PreTrainedModel subclass to
    # define `all_tied_weights_keys`, but the conservationxlabs/miewid-msv3 custom code
    # on HuggingFace predates that API. Provide a class-level default of {} so the
    # tied-weights initialization step becomes a no-op. MiewID has no tied weights, so
    # this is correct. Idempotent — safe to call on every embed.
    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
        PreTrainedModel.all_tied_weights_keys = {}

    # Real-world iNat photo downloads occasionally arrive truncated (network hiccup,
    # interrupted download). PIL's default behavior is to raise OSError. Tell PIL to
    # pad missing bytes with zeros instead, so a single corrupt photo doesn't kill an
    # entire embed run. Affected photos will produce slightly degraded embeddings but
    # we don't lose the run.
    from PIL import ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.DataFrame(
        [
            {
                "file_path": ann.file_path,
                "bbox": ann.bbox,
                "theta": ann.theta,
                "name": i,            # dummy int label — MiewIdDataset casts this to a tensor
                "species": ann.species,
                "viewpoint": ann.viewpoint,
            }
            for i, ann in enumerate(annotations)
        ]
    )

    dataset = MiewIdDataset(
        csv=df,
        transforms=get_test_transforms(image_size),
        crop_bbox=use_bbox,
        fliplr=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
        drop_last=False,
    )

    model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device).eval()
    embeddings, _labels = _extract_embeddings(loader, model, device)
    return np.asarray(embeddings, dtype=np.float32)


def run_embed_stage(
    storage,
    cache_dir: Path,
    run_id: str,
    model_id: str = "conservationxlabs/miewid-msv3",
    batch_size: int = 32,
    num_workers: int = 2,
    use_bbox: bool = True,
    only_missing: bool = True,
    device: Optional[str] = None,
) -> EmbedResult:
    """End-to-end embed stage: load annotations → embed → write parquet → return metrics."""
    from datetime import datetime, timezone

    from whaleshark_reid.storage.embedding_cache import (
        existing_annotation_uuids,
        write_embeddings,
    )

    all_anns = storage.list_annotations()
    if only_missing:
        existing = existing_annotation_uuids(cache_dir, run_id)
        to_embed = [a for a in all_anns if a.annotation_uuid not in existing]
    else:
        to_embed = all_anns
    n_skipped = len(all_anns) - len(to_embed)

    if not to_embed:
        return EmbedResult(
            n_embedded=0,
            n_skipped_existing=n_skipped,
            n_failed=0,
            embed_dim=0,
            model_id=model_id,
            duration_s=0.0,
        )

    t0 = time.time()
    mat = embed_annotations(
        to_embed,
        model_id=model_id,
        batch_size=batch_size,
        num_workers=num_workers,
        use_bbox=use_bbox,
        device=device,
    )
    duration = time.time() - t0

    rows = [
        {
            "annotation_uuid": ann.annotation_uuid,
            "embedding": mat[i].tolist(),
            "model_id": model_id,
            "model_version": "msv3",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        for i, ann in enumerate(to_embed)
    ]
    write_embeddings(cache_dir, run_id, rows)

    return EmbedResult(
        n_embedded=len(to_embed),
        n_skipped_existing=n_skipped,
        n_failed=0,
        embed_dim=int(mat.shape[1]),
        model_id=model_id,
        duration_s=duration,
    )
