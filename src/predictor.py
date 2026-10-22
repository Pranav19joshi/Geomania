"""
src/predictor.py
================
High-level pipeline combining all layers into a single predict(image) call.

Pipeline
--------
PIL Image
    -> GeolocationModel.embed()         ->  768-D normalised embedding
                                            (ViT forward pass runs ONCE)
    -> nn.Linear classifier             ->  all 512 GeoCell probabilities
    -> Top-K GeoCell IDs                ->  geocell_probs dict
    -> ReferenceRetriever.retrieve()    ->  Top-N reference matches with GPS
    -> clustering.predict_gps()         ->  Final (lat, lon) prediction

All hyperparameters are configurable at construction or call time.
V1 methods (nearest / weighted / dbscan) remain fully operational.

Public API
----------
predictor = GeoPredictor(
    checkpoint_path,
    embeddings_path,
    metadata_path,
    device=None,
    top_geocells=10,
    retrieval_k=30,
    method="dbscan_v2",
    temperature=0.05,
    eps_km=100.0,
    min_samples=3,
    use_classifier_prior=False,
    alpha=1.0,
    beta=1.0,
)

result: dict = predictor.predict(pil_image)

Returned dict schema
--------------------
    latitude                  float
    longitude                 float
    method                    str
    top_geocells              list[{geocell_id, probability}]
    candidate_count           int
    retrieval_k               int
    top_similarity            float
    winning_cluster_size      int | None
    winning_cluster_score     float | None
    mean_cluster_similarity   float | None
    median_cluster_similarity float | None
    num_clusters              int | None
    num_noise_points          int | None
    median_cluster_radius_km  float | None
    max_cluster_radius_km     float | None
    winning_cluster_geocells  list[int] | None
    fallback_used             str | None
    retrieval                 list[dict]
    timing                    dict[str, float]   (milliseconds)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image

from src.model      import GeolocationModel
from src.retrieval  import ReferenceRetriever
from src.clustering import predict_gps


class GeoPredictor:
    """
    End-to-end geolocation predictor.

    Loads all sub-components once at construction time; inference via predict().

    Parameters
    ----------
    checkpoint_path       : path to best_finetuned_geolocation.pt
    embeddings_path       : path to final_reference_embeddings.pt
    metadata_path         : path to final_metadata.csv
    device                : "cuda" | "cpu" | None (auto)
    top_geocells          : number of top classifier predictions to use
    retrieval_k           : reference images to retrieve per query
    method                : "nearest" | "weighted" | "dbscan" | "dbscan_v2"
    temperature           : softmax temperature for similarity weighting
    eps_km                : DBSCAN neighbourhood radius in kilometres
    min_samples           : DBSCAN minimum cluster size
    use_classifier_prior  : enable classifier prior in V2 cluster scoring
    alpha                 : classifier prior exponent (V2)
    beta                  : visual evidence exponent  (V2)
    """

    def __init__(
        self,
        checkpoint_path:      str | Path,
        embeddings_path:      str | Path,
        metadata_path:        str | Path,
        device:               str | None = None,
        top_geocells:         int   = 10,
        retrieval_k:          int   = 30,
        method:               str   = "dbscan_v2",
        temperature:          float = 0.05,
        eps_km:               float = 100.0,
        min_samples:          int   = 3,
        use_classifier_prior: bool  = False,
        alpha:                float = 1.0,
        beta:                 float = 1.0,
    ) -> None:
        self.top_geocells         = top_geocells
        self.retrieval_k          = retrieval_k
        self.method               = method
        self.temperature          = temperature
        self.eps_km               = eps_km
        self.min_samples          = min_samples
        self.use_classifier_prior = use_classifier_prior
        self.alpha                = alpha
        self.beta                 = beta

        self._model = GeolocationModel(
            checkpoint_path=checkpoint_path,
            device=device,
        )
        self._retriever = ReferenceRetriever(
            embeddings_path=embeddings_path,
            metadata_path=metadata_path,
        )

    # -----------------------------------------------------------------------
    def predict(self, image: Image.Image) -> dict[str, Any]:
        """
        Run the full geolocation pipeline for a single PIL image.

        The ViT forward pass runs exactly ONCE per call.

        Parameters
        ----------
        image : PIL.Image.Image

        Returns
        -------
        dict — see module docstring for full schema.
        """
        timing: dict[str, float] = {}

        # ── 1. Single inference pass: embedding + all 512 classifier probs ────
        t0 = time.perf_counter()
        embedding = self._model.embed(image)                        # (768,) CPU

        with torch.no_grad():
            logits    = self._model._classifier(embedding.to(self._model.device))
            all_probs = F.softmax(logits, dim=-1).cpu()             # (512,)

        timing["inference_ms"] = (time.perf_counter() - t0) * 1000

        # Build top-k geocell predictions from the full probability vector
        top_vals, top_idx = torch.topk(all_probs, k=self.top_geocells)
        geocell_preds = [
            {"geocell_id": int(top_idx[i]), "probability": float(top_vals[i])}
            for i in range(self.top_geocells)
        ]
        top_geocell_ids = [p["geocell_id"] for p in geocell_preds]
        geocell_probs   = {p["geocell_id"]: p["probability"] for p in geocell_preds}

        # ── 2. GeoCell-filtered reference retrieval ───────────────────────────
        t1 = time.perf_counter()
        candidate_count    = self._retriever.candidates_for_geocells(top_geocell_ids)
        retrieval_results  = self._retriever.retrieve(
            query_embedding=embedding,
            top_geocells=top_geocell_ids,
            k=self.retrieval_k,
        )
        timing["retrieval_ms"] = (time.perf_counter() - t1) * 1000

        top_similarity = (
            retrieval_results[0]["similarity"] if retrieval_results else 0.0
        )

        # ── 3. GPS prediction ─────────────────────────────────────────────────
        t2 = time.perf_counter()
        gps = predict_gps(
            retrieval_results,
            method=self.method,
            temperature=self.temperature,
            eps_km=self.eps_km,
            min_samples=self.min_samples,
            geocell_probs=geocell_probs,
            use_classifier_prior=self.use_classifier_prior,
            alpha=self.alpha,
            beta=self.beta,
        )
        timing["clustering_ms"] = (time.perf_counter() - t2) * 1000
        timing["total_ms"]      = (time.perf_counter() - t0) * 1000

        # ── 4. Assemble result ────────────────────────────────────────────────
        return {
            "latitude":                  gps["latitude"],
            "longitude":                 gps["longitude"],
            "method":                    gps["method"],
            "top_geocells":              geocell_preds,
            "candidate_count":           candidate_count,
            "retrieval_k":               self.retrieval_k,
            "top_similarity":            top_similarity,
            "retrieval":                 retrieval_results,
            # Cluster diagnostics (V1 dbscan or V2 dbscan_v2)
            "winning_cluster_size":      gps.get("winning_cluster_size")
                                         or gps.get("cluster_size"),
            "winning_cluster_score":     gps.get("winning_cluster_score")
                                         or gps.get("cluster_score"),
            "mean_cluster_similarity":   gps.get("mean_cluster_similarity"),
            "median_cluster_similarity": gps.get("median_cluster_similarity"),
            "num_clusters":              gps.get("num_clusters"),
            "num_noise_points":          gps.get("num_noise_points"),
            "median_cluster_radius_km":  gps.get("median_cluster_radius_km"),
            "max_cluster_radius_km":     gps.get("max_cluster_radius_km"),
            "winning_cluster_geocells":  gps.get("winning_cluster_geocells"),
            "fallback_used":             gps.get("fallback_used"),
            "timing":                    timing,
        }
