"""
src/clustering.py
=================
GPS coordinate clustering and spherical weighted centroid prediction.

Methods exposed via predict_gps():

  "nearest"    -- GPS of the single highest-similarity retrieved reference.

  "weighted"   -- Similarity-weighted spherical centroid over ALL Top-N
                  retrieved references. Temperature-scaled softmax weights.

  "dbscan"     -- V1: DBSCAN → highest-scoring (by summed weights) cluster
                  → similarity-weighted spherical centroid inside that cluster.
                  Falls back to "weighted" when no valid cluster is found.

  "dbscan_v2"  -- V2: same Haversine-DBSCAN but with richer per-cluster
                  diagnostics (mean/median/max similarity, cluster radius),
                  optional classifier-prior scoring, and structured fallbacks.

Key design choices
------------------
* DBSCAN uses metric="haversine" on (lat_rad, lon_rad) — correct for
  geographic data; NOT Euclidean distance on raw degrees.
* eps in kilometres converted to radians via Earth's mean radius (6 371.0088 km).
* Weights use temperature-scaled softmax so visually-similar images dominate.
* Centroids computed as spherical weighted means (3-D unit-vector average)
  to avoid longitude wraparound artefacts for worldwide data.
* V1 methods are preserved verbatim so comparisons remain valid.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.cluster import DBSCAN

# Earth's mean radius in kilometres (IAU 1976)
_EARTH_RADIUS_KM: float = 6371.0088


# ---------------------------------------------------------------------------
# Shared internal helpers
# ---------------------------------------------------------------------------

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Haversine distance in kilometres between two (lat, lon) points in degrees.
    """
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2.0 * _EARTH_RADIUS_KM * math.asin(math.sqrt(max(0.0, a)))


def _temperature_softmax(similarities: np.ndarray, temperature: float) -> np.ndarray:
    """
    Temperature-scaled softmax over an array of cosine similarities.

    weight_i = exp(sim_i / T) / sum_j exp(sim_j / T)
    Numerically stable: subtract max before exponentiation.
    """
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    scaled = similarities / temperature
    scaled -= scaled.max()          # numerical stability
    exps = np.exp(scaled)
    return exps / exps.sum()


def _spherical_weighted_centroid(
    lats_deg: np.ndarray,
    lons_deg: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float]:
    """
    Similarity-weighted geographic centroid on the unit sphere.

    Steps:
      1. (lat, lon) degrees -> radians
      2. Radians -> 3-D unit vectors (x, y, z)
      3. Weighted mean vector
      4. Normalise to unit length
      5. Convert back to (lat, lon) degrees

    Handles longitude wraparound correctly — safe for worldwide data.
    """
    lat_r = np.radians(lats_deg)
    lon_r = np.radians(lons_deg)

    x = np.cos(lat_r) * np.cos(lon_r)
    y = np.cos(lat_r) * np.sin(lon_r)
    z = np.sin(lat_r)

    wx = float(np.dot(weights, x))
    wy = float(np.dot(weights, y))
    wz = float(np.dot(weights, z))

    norm = math.sqrt(wx * wx + wy * wy + wz * wz)
    if norm < 1e-10:
        # Degenerate: fall back to arithmetic mean
        return (float(np.average(lats_deg, weights=weights)),
                float(np.average(lons_deg, weights=weights)))

    wx /= norm
    wy /= norm
    wz /= norm

    lat_out = math.degrees(math.asin(max(-1.0, min(1.0, wz))))
    lon_out = math.degrees(math.atan2(wy, wx))
    return lat_out, lon_out


def _extract_coords(results: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pull valid (lat, lon, similarity) triples — rows with None coords dropped."""
    lats, lons, sims = [], [], []
    for r in results:
        if r.get("latitude") is None or r.get("longitude") is None:
            continue
        lats.append(float(r["latitude"]))
        lons.append(float(r["longitude"]))
        sims.append(float(r["similarity"]))
    return np.array(lats), np.array(lons), np.array(sims)


def _extract_coords_v2(
    results: list[dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Like _extract_coords but also returns geocell_ids (for classifier prior)."""
    lats, lons, sims, gcids = [], [], [], []
    for r in results:
        if r.get("latitude") is None or r.get("longitude") is None:
            continue
        lats.append(float(r["latitude"]))
        lons.append(float(r["longitude"]))
        sims.append(float(r["similarity"]))
        gcids.append(int(r.get("geocell_id", -1)))
    return np.array(lats), np.array(lons), np.array(sims), np.array(gcids, dtype=int)


# ---------------------------------------------------------------------------
# Method A: nearest  (V1 — unchanged)
# ---------------------------------------------------------------------------

def predict_nearest(results: list[dict]) -> dict[str, Any]:
    """Return the GPS of the single highest-similarity retrieved reference."""
    for r in results:
        if r.get("latitude") is not None and r.get("longitude") is not None:
            return {
                "latitude":  float(r["latitude"]),
                "longitude": float(r["longitude"]),
                "method":    "nearest",
            }
    raise ValueError("No result with valid lat/lon found in retrieval results.")


# ---------------------------------------------------------------------------
# Method B: weighted  (V1 — unchanged)
# ---------------------------------------------------------------------------

def predict_weighted(
    results:     list[dict],
    temperature: float = 0.05,
) -> dict[str, Any]:
    """Similarity-weighted spherical centroid over ALL retrieved references."""
    lats, lons, sims = _extract_coords(results)
    if len(lats) == 0:
        raise ValueError("No valid coordinates in retrieval results.")
    weights = _temperature_softmax(sims, temperature)
    lat_out, lon_out = _spherical_weighted_centroid(lats, lons, weights)
    return {"latitude": lat_out, "longitude": lon_out, "method": "weighted"}


# ---------------------------------------------------------------------------
# Method C: dbscan  (V1 — unchanged)
# ---------------------------------------------------------------------------

def predict_dbscan(
    results:     list[dict],
    temperature: float = 0.05,
    eps_km:      float = 100.0,
    min_samples: int   = 3,
) -> dict[str, Any]:
    """
    V1 DBSCAN: haversine DBSCAN → highest-scoring cluster →
    similarity-weighted spherical centroid inside that cluster.
    Falls back to weighted when no valid cluster is found.
    """
    lats, lons, sims = _extract_coords(results)
    n = len(lats)
    if n == 0:
        raise ValueError("No valid coordinates in retrieval results.")

    eps_rad     = eps_km / _EARTH_RADIUS_KM
    coords_rad  = np.column_stack([np.radians(lats), np.radians(lons)])
    db          = DBSCAN(eps=eps_rad, min_samples=min_samples,
                         algorithm="ball_tree", metric="haversine").fit(coords_rad)

    labels         = db.labels_
    num_noise      = int(np.sum(labels == -1))
    cluster_labels = [lb for lb in set(labels) if lb != -1]
    num_clusters   = len(cluster_labels)
    all_weights    = _temperature_softmax(sims, temperature)

    if num_clusters == 0:
        lat_out, lon_out = _spherical_weighted_centroid(lats, lons, all_weights)
        return {
            "latitude": lat_out, "longitude": lon_out,
            "method": "dbscan_fallback_weighted",
            "num_clusters": 0, "num_noise_points": num_noise,
            "cluster_size": None, "cluster_score": None,
        }

    best_label, best_score = -1, -1.0
    for lb in cluster_labels:
        score = float(all_weights[labels == lb].sum())
        if score > best_score:
            best_score, best_label = score, lb

    mask            = labels == best_label
    cluster_weights = _temperature_softmax(sims[mask], temperature)
    lat_out, lon_out = _spherical_weighted_centroid(lats[mask], lons[mask], cluster_weights)

    return {
        "latitude": lat_out, "longitude": lon_out,
        "method": "dbscan",
        "num_clusters": num_clusters, "num_noise_points": num_noise,
        "cluster_size": int(mask.sum()), "cluster_score": float(best_score),
    }


# ---------------------------------------------------------------------------
# Method D: dbscan_v2  (NEW — full diagnostics + optional classifier prior)
# ---------------------------------------------------------------------------

def predict_dbscan_v2(
    results:               list[dict],
    geocell_probs:         dict[int, float] | None = None,
    temperature:           float = 0.05,
    eps_km:                float = 100.0,
    min_samples:           int   = 3,
    use_classifier_prior:  bool  = False,
    alpha:                 float = 1.0,
    beta:                  float = 1.0,
) -> dict[str, Any]:
    """
    V2 DBSCAN GPS prediction with full per-cluster diagnostics.

    Architecture
    ------------
    1. Compute temperature-scaled softmax weights over ALL retrieved references.
    2. Haversine DBSCAN on GPS coordinates.
    3. Per-cluster diagnostics:
         cluster_size
         visual_score        = sum of softmax weights in cluster
         mean/median/max similarity
         spherical centroid  (for diagnostics only)
         median/max Haversine radius from centroid
         unique GeoCells represented
         classifier_score    = sum of unique-geocell probabilities
         combined_score      = visual_score**beta * classifier_score**alpha
                               (only when use_classifier_prior=True)
    4. Select cluster with highest combined_score (defaults to visual_score).
    5. Similarity-weighted spherical centroid inside winning cluster.
    6. Two-level fallback if no valid cluster found.

    Parameters
    ----------
    results               : list[dict]  retrieval results (lat/lon/sim/geocell_id)
    geocell_probs         : dict {geocell_id: classifier_probability}
                            Required when use_classifier_prior=True.
    temperature           : softmax temperature (default 0.05)
    eps_km                : DBSCAN neighbourhood radius in km (default 100)
    min_samples           : DBSCAN minimum cluster size (default 3)
    use_classifier_prior  : combine visual evidence with classifier priors
    alpha                 : classifier prior exponent (default 1.0)
    beta                  : visual evidence exponent  (default 1.0)

    Returns
    -------
    dict with keys:
        latitude, longitude, method
        num_clusters, num_noise_points
        winning_cluster_size, winning_cluster_score
        mean_cluster_similarity, median_cluster_similarity
        median_cluster_radius_km, max_cluster_radius_km
        winning_cluster_geocells
        fallback_used
        all_cluster_diagnostics   (list of per-cluster dicts)
    """
    if geocell_probs is None:
        geocell_probs = {}

    lats, lons, sims, gcids = _extract_coords_v2(results)
    n = len(lats)

    if n == 0:
        raise ValueError("No valid coordinates in retrieval results.")

    # ── 1. Global softmax weights (over ALL retrieved references) ────────────
    all_weights = _temperature_softmax(sims, temperature)

    # ── 2. Haversine DBSCAN ─────────────────────────────────────────────────
    eps_rad    = eps_km / _EARTH_RADIUS_KM
    coords_rad = np.column_stack([np.radians(lats), np.radians(lons)])
    db         = DBSCAN(eps=eps_rad, min_samples=min_samples,
                        algorithm="ball_tree", metric="haversine").fit(coords_rad)

    labels         = db.labels_
    num_noise      = int(np.sum(labels == -1))
    cluster_labels = [lb for lb in set(labels) if lb != -1]
    num_clusters   = len(cluster_labels)

    # ── 3. Per-cluster diagnostics ───────────────────────────────────────────
    cluster_diagnostics: list[dict] = []

    for lb in cluster_labels:
        mask      = labels == lb
        c_lats    = lats[mask]
        c_lons    = lons[mask]
        c_sims    = sims[mask]
        c_weights = all_weights[mask]           # global weights, subset to cluster
        c_gcids   = gcids[mask]
        c_size    = int(mask.sum())

        # Visual evidence scores
        visual_score  = float(c_weights.sum())
        mean_sim      = float(c_sims.mean())
        max_sim       = float(c_sims.max())
        median_sim    = float(np.median(c_sims))

        # Spherical centroid of this cluster (for radius diagnostics)
        norm_weights = c_weights / c_weights.sum()
        c_lat_c, c_lon_c = _spherical_weighted_centroid(c_lats, c_lons, norm_weights)

        # Haversine radii from centroid
        radii = [
            haversine_km(c_lat_c, c_lon_c, float(la), float(lo))
            for la, lo in zip(c_lats, c_lons)
        ]
        median_radius = float(np.median(radii)) if radii else 0.0
        max_radius    = float(max(radii))        if radii else 0.0

        # Unique GeoCells and classifier score
        unique_geocells  = list({int(g) for g in c_gcids})
        classifier_score = sum(geocell_probs.get(g, 0.0) for g in unique_geocells)

        # Combined score
        if use_classifier_prior and classifier_score > 0:
            combined_score = (visual_score ** beta) * (classifier_score ** alpha)
        else:
            combined_score = visual_score

        cluster_diagnostics.append({
            "label":             lb,
            "cluster_size":      c_size,
            "visual_score":      visual_score,
            "combined_score":    combined_score,
            "mean_similarity":   mean_sim,
            "max_similarity":    max_sim,
            "median_similarity": median_sim,
            "centroid_lat":      c_lat_c,
            "centroid_lon":      c_lon_c,
            "median_radius_km":  median_radius,
            "max_radius_km":     max_radius,
            "unique_geocells":   unique_geocells,
            "classifier_score":  classifier_score,
        })

    # ── 4. Fallback: no valid clusters ───────────────────────────────────────
    if num_clusters == 0:
        # Fallback 1: weighted centroid over all references
        if n > 0:
            lat_out, lon_out = _spherical_weighted_centroid(lats, lons, all_weights)
            return {
                "latitude":                  lat_out,
                "longitude":                 lon_out,
                "method":                    "dbscan_v2_fallback_weighted",
                "num_clusters":              0,
                "num_noise_points":          num_noise,
                "fallback_used":             "weighted",
                "winning_cluster_size":      None,
                "winning_cluster_score":     None,
                "mean_cluster_similarity":   None,
                "median_cluster_similarity": None,
                "median_cluster_radius_km":  None,
                "max_cluster_radius_km":     None,
                "winning_cluster_geocells":  None,
                "all_cluster_diagnostics":   [],
            }
        # Fallback 2: nearest
        return {
            "latitude":                  float(lats[0]),
            "longitude":                 float(lons[0]),
            "method":                    "dbscan_v2_fallback_nearest",
            "num_clusters":              0,
            "num_noise_points":          num_noise,
            "fallback_used":             "nearest",
            "winning_cluster_size":      None,
            "winning_cluster_score":     None,
            "mean_cluster_similarity":   None,
            "median_cluster_similarity": None,
            "median_cluster_radius_km":  None,
            "max_cluster_radius_km":     None,
            "winning_cluster_geocells":  None,
            "all_cluster_diagnostics":   [],
        }

    # ── 5. Select winning cluster (highest combined_score) ───────────────────
    winner = max(cluster_diagnostics, key=lambda c: c["combined_score"])

    # ── 6. Similarity-weighted spherical centroid inside winning cluster ──────
    w_mask    = labels == winner["label"]
    w_sims    = sims[w_mask]
    w_weights = _temperature_softmax(w_sims, temperature)
    lat_out, lon_out = _spherical_weighted_centroid(
        lats[w_mask], lons[w_mask], w_weights
    )

    return {
        "latitude":                  lat_out,
        "longitude":                 lon_out,
        "method":                    "dbscan_v2",
        "num_clusters":              num_clusters,
        "num_noise_points":          num_noise,
        "fallback_used":             None,
        "winning_cluster_size":      winner["cluster_size"],
        "winning_cluster_score":     winner["combined_score"],
        "mean_cluster_similarity":   winner["mean_similarity"],
        "median_cluster_similarity": winner["median_similarity"],
        "median_cluster_radius_km":  winner["median_radius_km"],
        "max_cluster_radius_km":     winner["max_radius_km"],
        "winning_cluster_geocells":  winner["unique_geocells"],
        "all_cluster_diagnostics":   cluster_diagnostics,
    }


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def predict_gps(
    retrieval_results:     list[dict],
    method:                str   = "dbscan",
    temperature:           float = 0.05,
    eps_km:                float = 100.0,
    min_samples:           int   = 3,
    # V2-only parameters
    geocell_probs:         dict[int, float] | None = None,
    use_classifier_prior:  bool  = False,
    alpha:                 float = 1.0,
    beta:                  float = 1.0,
) -> dict[str, Any]:
    """
    Predict GPS from retrieval results.

    Parameters
    ----------
    retrieval_results     : list[dict]  output of ReferenceRetriever.retrieve()
    method                : "nearest" | "weighted" | "dbscan" | "dbscan_v2"
    temperature           : softmax temperature (default 0.05)
    eps_km                : DBSCAN neighbourhood radius in km (default 100)
    min_samples           : DBSCAN min cluster size (default 3)
    geocell_probs         : {geocell_id: probability} — V2 classifier prior
    use_classifier_prior  : enable classifier prior in V2 scoring (default False)
    alpha                 : classifier prior exponent for V2 (default 1.0)
    beta                  : visual evidence exponent for V2 (default 1.0)

    Returns
    -------
    dict with at least: latitude, longitude, method
    """
    if not retrieval_results:
        raise ValueError("retrieval_results is empty.")

    if method == "nearest":
        return predict_nearest(retrieval_results)
    elif method == "weighted":
        return predict_weighted(retrieval_results, temperature=temperature)
    elif method == "dbscan":
        return predict_dbscan(retrieval_results, temperature=temperature,
                              eps_km=eps_km, min_samples=min_samples)
    elif method == "dbscan_v2":
        return predict_dbscan_v2(
            retrieval_results,
            geocell_probs=geocell_probs,
            temperature=temperature,
            eps_km=eps_km,
            min_samples=min_samples,
            use_classifier_prior=use_classifier_prior,
            alpha=alpha,
            beta=beta,
        )
    else:
        raise ValueError(
            f"Unknown method '{method}'. "
            f"Choose from: 'nearest', 'weighted', 'dbscan', 'dbscan_v2'."
        )
