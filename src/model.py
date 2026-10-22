"""
src/model.py
============
GeolocationModel wraps the fine-tuned OpenCLIP ViT-L/14 backbone and the
512-class geocell classifier.

Checkpoint structure (models/best_finetuned_geolocation.pt)
------------------------------------------------------------
  "vision"      -> state_dict of the COMPLETE OpenCLIP model (446 keys)
                   (includes visual tower, text encoder, etc.)
  "classifier"  -> state_dict of nn.Linear(768, 512)  [2 keys: weight, bias]
  "epoch"       -> int
  "val_loss"    -> float
  "val_top1"    -> float  (Top-1 accuracy %)
  "val_top5"    -> float  (Top-5 accuracy %)

Public API
----------
model = GeolocationModel(checkpoint_path, device=None)

embedding: torch.Tensor  = model.embed(pil_image)
    -> shape (768,), L2-normalised, returned on CPU

predictions: list[dict]  = model.predict(pil_image, top_k=10)
    -> list of {"geocell_id": int, "probability": float},
       sorted descending by probability
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from PIL import Image
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants — must match training configuration exactly
# ---------------------------------------------------------------------------
CLIP_MODEL_NAME = "ViT-L-14"
CLIP_PRETRAINED = "datacomp_xl_s13b_b90k"
EMBED_DIM       = 768
NUM_GEOCELLS    = 512


class GeolocationModel:
    """
    Loads and wraps the fine-tuned geolocation model.

    Parameters
    ----------
    checkpoint_path : str | Path
        Path to ``best_finetuned_geolocation.pt``.
    device : str | torch.device | None
        Target device. If *None* (default), CUDA is used when available,
        otherwise CPU.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | torch.device | None = None,
    ) -> None:

        # ── 1. Device selection ─────────────────────────────────────────────
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"[GeolocationModel] Device          : {self.device}")

        # ── 2. Create full OpenCLIP model + preprocessing transform ─────────
        print(
            f"[GeolocationModel] Building OpenCLIP {CLIP_MODEL_NAME} "
            f"(pretrained='{CLIP_PRETRAINED}') ..."
        )
        self._clip_model, _, self._preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL_NAME,
            pretrained=None,
        )
        self._clip_model.to(self.device)

        # ── 3. Build geocell classifier ──────────────────────────────────────
        self._classifier = nn.Linear(EMBED_DIM, NUM_GEOCELLS)
        self._classifier.to(self.device)

        # ── 4. Load fine-tuned weights from checkpoint ───────────────────────
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path.resolve()}"
            )

        print(f"[GeolocationModel] Loading checkpoint: {checkpoint_path} ...")
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Verify expected keys are present
        required_keys = {"vision", "classifier"}
        missing = required_keys - set(ckpt.keys())
        if missing:
            raise KeyError(
                f"Checkpoint is missing required keys: {missing}. "
                f"Found keys: {list(ckpt.keys())}"
            )

        # Load full OpenCLIP state_dict (446 keys — includes text encoder)
        self._clip_model.load_state_dict(ckpt["vision"], strict=True)

        # Load classifier weights
        self._classifier.load_state_dict(ckpt["classifier"], strict=True)

        # ── 5. Log checkpoint metadata ───────────────────────────────────────
        print(f"[GeolocationModel] Checkpoint epoch : {ckpt.get('epoch', 'N/A')}")
        print(f"[GeolocationModel] Val loss         : {ckpt.get('val_loss', float('nan')):.4f}")
        print(f"[GeolocationModel] Val Top-1        : {ckpt.get('val_top1', float('nan')):.2f}%")
        print(f"[GeolocationModel] Val Top-5        : {ckpt.get('val_top5', float('nan')):.2f}%")
        print(f"[GeolocationModel] Model loaded successfully.")

        # ── 6. Evaluation mode ───────────────────────────────────────────────
        self._clip_model.eval()
        self._classifier.eval()

    # -----------------------------------------------------------------------
    # Internal helper
    # -----------------------------------------------------------------------
    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Apply OpenCLIP preprocessing and return a (1, C, H, W) batch tensor."""
        if image.mode != "RGB":
            image = image.convert("RGB")
        tensor = self._preprocess(image)            # (C, H, W)
        return tensor.unsqueeze(0).to(self.device)  # (1, C, H, W)

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------
    @torch.no_grad()
    def embed(self, image: Image.Image) -> torch.Tensor:
        """
        Encode a PIL image and return its normalised 768-D embedding.

        Uses ``model.encode_image(..., normalize=True)`` which applies the
        visual projection head and L2-normalises the output in one step.

        Parameters
        ----------
        image : PIL.Image.Image

        Returns
        -------
        torch.Tensor
            Shape ``(768,)`` on CPU, L2-normalised (norm ~= 1.0).
        """
        pixel_values = self._preprocess_image(image)  # (1, C, H, W)
        # encode_image with normalize=True applies the visual projection
        # head + L2 normalisation internally.
        embedding = self._clip_model.encode_image(
            pixel_values, normalize=True
        )                                             # (1, 768)
        return embedding.squeeze(0).cpu()             # (768,)

    @torch.no_grad()
    def predict(
        self,
        image: Image.Image,
        top_k: int = 10,
    ) -> list[dict]:
        """
        Predict the top-*k* geocell IDs for a PIL image.

        Parameters
        ----------
        image : PIL.Image.Image
        top_k : int
            Number of top geocells to return (default 10).

        Returns
        -------
        list[dict]
            Each element is ``{"geocell_id": int, "probability": float}``,
            sorted descending by probability.
        """
        embedding = self.embed(image)                           # (768,) CPU
        logits    = self._classifier(embedding.to(self.device)) # (512,)
        probs     = F.softmax(logits, dim=-1)                   # (512,)

        top_k   = min(top_k, NUM_GEOCELLS)
        values, indices = torch.topk(probs, k=top_k)

        return [
            {
                "geocell_id":  int(idx.item()),
                "probability": float(val.item()),
            }
            for idx, val in zip(indices, values)
        ]
