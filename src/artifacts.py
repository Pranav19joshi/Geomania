"""
src/artifacts.py
================
Centralised artifact resolver for GeoMania's two large binary files.

Resolution order (LOCAL FIRST):
    1. Check for the local file at its well-known project-relative path.
       If it exists → return that path immediately (no network activity).
    2. If the local file is absent → download / cache via hf_hub_download()
       from JoshiPranav19/GeoMania on Hugging Face and return the cached path.

Environment variables
---------------------
HF_TOKEN : optional
    When set, passed to hf_hub_download() so private HF repositories work.
    Do NOT hard-code any token here.

Public API
----------
get_model_path()               → Path
get_reference_embeddings_path() → Path
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Project-root resolution — portable regardless of launch directory
# ---------------------------------------------------------------------------
# This file lives at  <PROJECT_ROOT>/src/artifacts.py
# So the project root is two levels up from this file's own directory.
_THIS_FILE   = Path(__file__).resolve()
PROJECT_ROOT = _THIS_FILE.parent.parent   # …/Geomania

# ---------------------------------------------------------------------------
# Hugging Face repository coordinates
# ---------------------------------------------------------------------------
HF_REPO_ID             = "JoshiPranav19/GeoMania"
HF_MODEL_FILENAME      = "best_finetuned_geolocation.pt"
HF_EMBEDDINGS_FILENAME = "final_reference_embeddings.pt"

# ---------------------------------------------------------------------------
# Well-known local paths (relative to project root)
# These must match the paths the rest of the project already uses.
# ---------------------------------------------------------------------------
LOCAL_MODEL_PATH      = PROJECT_ROOT / "models" / HF_MODEL_FILENAME
LOCAL_EMBEDDINGS_PATH = PROJECT_ROOT / "data"   / HF_EMBEDDINGS_FILENAME

# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _hf_download(filename: str) -> Path:
    """Download *filename* from HF_REPO_ID and return the cached path."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required for remote artifact download. "
            "Install it with:  pip install huggingface_hub"
        ) from exc

    token: str | None = os.environ.get("HF_TOKEN") or None  # None → anonymous

    print(f"[GeoMania] Downloading '{filename}' from Hugging Face:")
    print(f"[GeoMania]   repo: {HF_REPO_ID}")
    print(f"[GeoMania]   (this may take a while for large files)")

    cached_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=filename,
        token=token,
        # Use HF's default cache (~/.cache/huggingface/hub); do NOT force a
        # local_dir so subsequent calls reuse the cache without re-downloading.
    )
    return Path(cached_path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_model_path() -> Path:
    """
    Return the path to ``best_finetuned_geolocation.pt``.

    LOCAL FIRST: if ``models/best_finetuned_geolocation.pt`` exists relative
    to the project root, it is returned immediately without contacting
    Hugging Face.  Only when the local file is absent is hf_hub_download()
    invoked.
    """
    if LOCAL_MODEL_PATH.is_file():
        print(f"[GeoMania] Model artifact   : LOCAL")
        print(f"[GeoMania] Path             : {LOCAL_MODEL_PATH}")
        return LOCAL_MODEL_PATH

    print(f"[GeoMania] Local model not found: {LOCAL_MODEL_PATH}")
    print(f"[GeoMania] Downloading model from Hugging Face: {HF_REPO_ID}")
    cached = _hf_download(HF_MODEL_FILENAME)
    print(f"[GeoMania] Model artifact   : HUGGING FACE CACHE")
    print(f"[GeoMania] Path             : {cached}")
    return cached


def get_reference_embeddings_path() -> Path:
    """
    Return the path to ``final_reference_embeddings.pt``.

    LOCAL FIRST: if ``data/final_reference_embeddings.pt`` exists relative
    to the project root, it is returned immediately without contacting
    Hugging Face.  Only when the local file is absent is hf_hub_download()
    invoked.
    """
    if LOCAL_EMBEDDINGS_PATH.is_file():
        print(f"[GeoMania] Reference embeddings : LOCAL")
        print(f"[GeoMania] Path                 : {LOCAL_EMBEDDINGS_PATH}")
        return LOCAL_EMBEDDINGS_PATH

    print(f"[GeoMania] Local embeddings not found: {LOCAL_EMBEDDINGS_PATH}")
    print(f"[GeoMania] Downloading embeddings from Hugging Face: {HF_REPO_ID}")
    cached = _hf_download(HF_EMBEDDINGS_FILENAME)
    print(f"[GeoMania] Reference embeddings : HUGGING FACE CACHE")
    print(f"[GeoMania] Path                 : {cached}")
    return cached
