"""PairX (Pairwise mAtching of Intermediate Representations for eXplainability).

Backprops through MiewID itself on a chosen layer and renders matched-region
visualizations as PNG frames. Lightweight integration for A/B comparison
against the LightGlue overlay — no interactive controls, just rendered images.

Upstream: https://github.com/pairx-explains/pairx (no setup.py — vendored at
/workspace/pairx and added to sys.path on first call).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_PAIRX_PATH = Path("/workspace/pairx")


class PairXUnavailable(RuntimeError):
    """Raised when PairX or its deps can't be loaded."""


_MODEL_CACHE: dict[str, Any] = {}


def _ensure_pairx_on_path() -> None:
    p = str(DEFAULT_PAIRX_PATH)
    if p not in sys.path:
        sys.path.insert(0, p)


def _apply_miewid_compat_shim() -> None:
    """Same shim used in core.embed.miewid — newer transformers expect
    `all_tied_weights_keys` on PreTrainedModel subclasses; the msv3 custom code
    on HuggingFace predates that. MiewID has no tied weights so {} is correct."""
    from transformers.modeling_utils import PreTrainedModel
    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
        PreTrainedModel.all_tied_weights_keys = {}


def _get_model(model_id: str = "conservationxlabs/miewid-msv3"):
    """Load MiewID lazily, cached per model_id. PairX needs the actual model
    in eval mode on the chosen device — same model we use for embeddings."""
    if model_id in _MODEL_CACHE:
        return _MODEL_CACHE[model_id]

    import torch
    from transformers import AutoModel

    _apply_miewid_compat_shim()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device).eval()
    _MODEL_CACHE[model_id] = (model, device)
    return _MODEL_CACHE[model_id]


def _load_chip_arrays(file_path: str, bbox, theta: float, size: int = 440):
    """Returns (chip_uint8_resized, chip_tensor_normalized) — the two formats
    PairX wants. Reuses the same get_chip_from_img pipeline as /image?crop=true
    so PairX sees exactly what the reviewer sees.
    """
    import cv2
    from torchvision import transforms
    from wbia_miew_id.datasets.helpers import get_chip_from_img
    from PIL import Image

    img = Image.open(file_path).convert("RGB")
    arr = np.array(img)
    if bbox:
        chip = get_chip_from_img(arr, list(bbox), float(theta or 0.0))
    else:
        chip = arr
    chip_resized = cv2.resize(chip, (size, size))
    pil_chip = Image.fromarray(chip_resized)

    tfm = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = tfm(pil_chip).unsqueeze(0)  # (1, 3, size, size)
    return chip_resized, tensor


def explain_pair(
    file_path_a: str, bbox_a, theta_a: float,
    file_path_b: str, bbox_b, theta_b: float,
    layer_key: str = "backbone.blocks.3",
    k_lines: int = 20,
    k_colors: int = 5,
    model_id: str = "conservationxlabs/miewid-msv3",
) -> list[np.ndarray]:
    """Run PairX on a single pair. Returns list of uint8 RGB visualizations
    (one per layer; with one layer_key in we get one image)."""
    _ensure_pairx_on_path()
    try:
        import core as pairx_core
    except ImportError as e:
        raise PairXUnavailable(
            f"could not import PairX from {DEFAULT_PAIRX_PATH}: {e}"
        ) from e

    model, device = _get_model(model_id)

    chip_a, tensor_a = _load_chip_arrays(file_path_a, bbox_a, theta_a)
    chip_b, tensor_b = _load_chip_arrays(file_path_b, bbox_b, theta_b)
    tensor_a = tensor_a.to(device)
    tensor_b = tensor_b.to(device)

    # PairX's draw_matches_and_color_maps calls .shape on img_np_*, so pass
    # the ndarrays directly (not lists), even though the wbia ml-service router
    # appears to pass lists — that path may be batched differently.
    images = pairx_core.explain(
        tensor_a, tensor_b,
        chip_a, chip_b,
        model,
        [layer_key],
        k_lines=k_lines,
        k_colors=k_colors,
    )
    # Free grad memory (PairX backprops through the model)
    model.zero_grad(set_to_none=True)
    return images
