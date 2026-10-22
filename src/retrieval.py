"""
src/retrieval.py
================
GeoCell-filtered reference retrieval using cosine similarity (dot product
on pre-normalized embeddings).

Design
------
- Reference embeddings and their geocell/image mappings are loaded ONCE at
  initialization and kept in memory.
- retrieve() narrows the search to only those reference images whose
  geocell_id is in ``top_geocells``, then performs an exact dot-product
  similarity search over that (typically 1 000–2 000 item) candidate set.
- No FAISS — PyTorch matmul over the filtered candidate set is fast enough.
- Metadata (latitude, longitude, country) is joined from final_metadata.csv
  using image_id.

Public API
----------
retriever = ReferenceRetriever(embeddings_path, metadata_path)

results: list[dict] = retriever.retrieve(
    query_embedding,   # torch.Tensor (768,) L2-normalised
    top_geocells,      # list[int]  e.g. from model.predict()
    k=30
)

Each result dict contains:
    image_id    int
    geocell_id  int
    similarity  float   cosine similarity in [-1, 1]
    latitude    float | None
    longitude   float | None
    country     str   | None
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
import pandas as pd


# ---------------------------------------------------------------------------
# Expected embedding dimension
# ---------------------------------------------------------------------------
EMBED_DIM = 768


class ReferenceRetriever:
    """
    Loads and indexes the reference embedding database for fast retrieval.

    Parameters
    ----------
    embeddings_path : str | Path
        Path to ``final_reference_embeddings.pt``.
    metadata_path : str | Path
        Path to ``final_metadata.csv``.
    device : str | torch.device | None
        Device to keep reference embeddings on.
        Defaults to CPU (embeddings are ~268 MB, always fits in RAM).
    """

    def __init__(
        self,
        embeddings_path: str | Path,
        metadata_path: str | Path,
        device: str | torch.device | None = None,
    ) -> None:
        self.device = torch.device(device) if device else torch.device("cpu")

        # ── 1. Load reference embeddings ─────────────────────────────────────
        embeddings_path = Path(embeddings_path)
        if not embeddings_path.is_file():
            raise FileNotFoundError(
                f"Reference embeddings not found: {embeddings_path.resolve()}"
            )

        print(f"[ReferenceRetriever] Loading embeddings: {embeddings_path} ...")
        t0 = time.perf_counter()
        db = torch.load(embeddings_path, map_location=self.device, weights_only=False)
        load_ms = (time.perf_counter() - t0) * 1000
        print(f"[ReferenceRetriever] Embeddings loaded in {load_ms:.1f} ms")

        # ── 2. Validate structure ────────────────────────────────────────────
        required_keys = {"embeddings", "geocell_ids", "image_ids"}
        missing = required_keys - set(db.keys())
        if missing:
            raise KeyError(
                f"Reference embeddings file is missing keys: {missing}. "
                f"Found: {list(db.keys())}"
            )

        self._embeddings: torch.Tensor = db["embeddings"].to(
            self.device, dtype=torch.float32
        )
        self._geocell_ids: torch.Tensor = db["geocell_ids"].to(self.device)
        self._image_ids:   torch.Tensor = db["image_ids"].to(self.device)

        N = self._embeddings.shape[0]

        # Shape checks
        if self._embeddings.ndim != 2 or self._embeddings.shape[1] != EMBED_DIM:
            raise ValueError(
                f"Expected embeddings shape [N, {EMBED_DIM}], "
                f"got {tuple(self._embeddings.shape)}"
            )
        if self._geocell_ids.shape[0] != N:
            raise ValueError(
                f"geocell_ids length {self._geocell_ids.shape[0]} != "
                f"embeddings N {N}"
            )
        if self._image_ids.shape[0] != N:
            raise ValueError(
                f"image_ids length {self._image_ids.shape[0]} != "
                f"embeddings N {N}"
            )

        print(
            f"[ReferenceRetriever] Validated: {N:,} embeddings, "
            f"dim={EMBED_DIM}, "
            f"unique geocells={int(self._geocell_ids.unique().numel())}"
        )

        # Log provenance if available
        if "model" in db:
            print(
                f"[ReferenceRetriever] Embeddings model   : {db['model']} "
                f"/ {db.get('pretrained', '?')} "
                f"(epoch {db.get('checkpoint_epoch', '?')})"
            )

        # ── 3. Build geocell -> indices mapping ──────────────────────────────
        print("[ReferenceRetriever] Building geocell index ...")
        geocell_ids_cpu = self._geocell_ids.cpu().tolist()
        self._geocell_index: dict[int, list[int]] = {}
        for idx, gc in enumerate(geocell_ids_cpu):
            self._geocell_index.setdefault(int(gc), []).append(idx)

        print(
            f"[ReferenceRetriever] Geocell index built: "
            f"{len(self._geocell_index)} cells"
        )

        # ── 4. Load and index metadata CSV ───────────────────────────────────
        metadata_path = Path(metadata_path)
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Metadata CSV not found: {metadata_path.resolve()}"
            )

        print(f"[ReferenceRetriever] Loading metadata: {metadata_path} ...")
        meta_df = pd.read_csv(
            metadata_path,
            usecols=["image_id", "latitude", "longitude", "country"],
            dtype={"image_id": int, "country": str},
        )
        # Index by image_id for O(1) lookup
        self._metadata: dict[int, dict] = (
            meta_df.set_index("image_id")
            .to_dict(orient="index")
        )
        print(
            f"[ReferenceRetriever] Metadata loaded: "
            f"{len(self._metadata):,} records"
        )

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------
    def retrieve(
        self,
        query_embedding: torch.Tensor,
        top_geocells: list[int],
        k: int = 30,
    ) -> list[dict]:
        """
        Retrieve the top-*k* most similar reference images from the geocells
        in ``top_geocells``.

        Parameters
        ----------
        query_embedding : torch.Tensor
            Shape ``(768,)`` or ``(1, 768)``, L2-normalised.
        top_geocells : list[int]
            GeoCell IDs to search within (typically the Top-10 classifier
            predictions from GeolocationModel.predict()).
        k : int
            Number of top matches to return (default 30).

        Returns
        -------
        list[dict]
            Up to *k* dicts, sorted by descending similarity:
            {
                "image_id":   int,
                "geocell_id": int,
                "similarity": float,
                "latitude":   float | None,
                "longitude":  float | None,
                "country":    str   | None,
            }
        """
        # Normalise query shape to (768,)
        q = query_embedding.squeeze().to(self.device, dtype=torch.float32)
        if q.ndim != 1 or q.shape[0] != EMBED_DIM:
            raise ValueError(
                f"query_embedding must be shape ({EMBED_DIM},) or "
                f"(1, {EMBED_DIM}), got {tuple(query_embedding.shape)}"
            )

        # ── Gather candidate indices from filtered geocells ──────────────────
        candidate_indices: list[int] = []
        matched_cells:     list[int] = []
        missing_cells:     list[int] = []

        for gc in top_geocells:
            idxs = self._geocell_index.get(int(gc))
            if idxs:
                candidate_indices.extend(idxs)
                matched_cells.append(int(gc))
            else:
                missing_cells.append(int(gc))

        if missing_cells:
            print(
                f"[ReferenceRetriever] Warning: {len(missing_cells)} geocell(s) "
                f"not found in reference DB: {missing_cells}"
            )

        n_candidates = len(candidate_indices)
        if n_candidates == 0:
            print("[ReferenceRetriever] Warning: no candidates after geocell filtering.")
            return []

        # ── Dot-product similarity (cosine, since both sides are normalised) ──
        idx_tensor  = torch.tensor(candidate_indices, device=self.device)
        cand_embs   = self._embeddings[idx_tensor]   # (C, 768)
        similarities = cand_embs @ q                  # (C,)  dot product

        # Select top-k within candidates
        actual_k    = min(k, n_candidates)
        top_values, top_local = torch.topk(similarities, k=actual_k)

        # Map local indices back to global DB indices
        global_indices = idx_tensor[top_local].cpu().tolist()
        sim_scores     = top_values.cpu().tolist()

        # ── Build result dicts ───────────────────────────────────────────────
        results: list[dict] = []
        for g_idx, sim in zip(global_indices, sim_scores):
            img_id  = int(self._image_ids[g_idx].item())
            cell_id = int(self._geocell_ids[g_idx].item())
            meta    = self._metadata.get(img_id, {})
            results.append(
                {
                    "image_id":   img_id,
                    "geocell_id": cell_id,
                    "similarity": float(sim),
                    "latitude":   meta.get("latitude"),
                    "longitude":  meta.get("longitude"),
                    "country":    meta.get("country"),
                }
            )

        return results

    # -----------------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------------
    def candidates_for_geocells(self, geocell_ids: list[int]) -> int:
        """Return the total number of reference images in the given geocells."""
        return sum(
            len(self._geocell_index.get(int(gc), []))
            for gc in geocell_ids
        )

    def retrieve_with_meta(
        self,
        query_embedding: torch.Tensor,
        top_geocells: list[int],
        k: int = 100,
    ) -> dict:
        """
        Like retrieve() but also returns candidate pool statistics.

        Supports retrieval_k values of 30, 50, 100, 150 (or any integer).
        All candidates from the selected GeoCells compete globally by cosine
        similarity — no per-cell quota is enforced.

        Parameters
        ----------
        query_embedding : torch.Tensor  (768,) L2-normalised
        top_geocells    : list[int]     GeoCell IDs to search within
        k               : int           Number of top results to return

        Returns
        -------
        dict with keys:
            results                  list[dict]  — same schema as retrieve()
            candidate_count_before_topk  int
            retrieval_count_after_topk   int
        """
        candidate_count = self.candidates_for_geocells(top_geocells)
        results = self.retrieve(query_embedding, top_geocells, k=k)
        return {
            "results":                   results,
            "candidate_count_before_topk": candidate_count,
            "retrieval_count_after_topk":  len(results),
        }

