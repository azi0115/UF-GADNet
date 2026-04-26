#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract 100 core time-series traffic features from a phishing URL dataset.

Dataset format expected:
[
  {
    "url": "https://example.com/path",
    "traffic": [[timestamp_1, packet_size_1], [timestamp_2, packet_size_2], ...],
    "label": 0 or 1,
    "phish_type": ...,
    "risk_score": ...
  },
  ...
]

Only timestamp and packet size are used because the uploaded dataset does not
contain packet direction. Therefore, fwd/bwd direction features are not computed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.stats import mannwhitneyu, ks_2samp
except Exception:  # pragma: no cover
    mannwhitneyu = None
    ks_2samp = None

EPS = 1e-12


# =========================================================
# 1. Exact 100 feature names
# =========================================================

BASIC_FEATURES = [
    "packet_count",
    "duration",
    "start_time",
    "end_time",
    "total_bytes",
    "log_total_bytes",
    "packet_rate",
    "byte_rate",
    "mean_bytes_per_packet",
    "active_span_ratio",
]

PKT_LEN_STATS = [
    "pkt_len_mean",
    "pkt_len_std",
    "pkt_len_var",
    "pkt_len_min",
    "pkt_len_max",
    "pkt_len_range",
    "pkt_len_median",
    "pkt_len_mad",
    "pkt_len_q05",
    "pkt_len_q10",
    "pkt_len_q25",
    "pkt_len_q75",
    "pkt_len_q90",
    "pkt_len_q95",
    "pkt_len_iqr",
    "pkt_len_cv",
    "pkt_len_skew",
    "pkt_len_kurtosis",
    "pkt_len_rms",
    "pkt_len_gini",
]

PKT_LEN_DIST = [
    "pkt_le_64_ratio",
    "pkt_le_128_ratio",
    "pkt_le_256_ratio",
    "pkt_le_512_ratio",
    "pkt_gt_1024_ratio",
    "pkt_gt_1500_ratio",
    "top10_bytes_fraction",
    "largest_pkt_fraction",
]

IAT_STATS = [
    "iat_count",
    "iat_mean",
    "iat_std",
    "iat_var",
    "iat_min",
    "iat_max",
    "iat_range",
    "iat_median",
    "iat_mad",
    "iat_q05",
    "iat_q10",
    "iat_q25",
    "iat_q75",
    "iat_q90",
    "iat_q95",
    "iat_iqr",
    "iat_cv",
    "iat_skew",
    "iat_kurtosis",
    "iat_max_fraction",
]

IAT_THRESHOLDS = [
    "iat_lt_005_ratio",
    "iat_lt_010_ratio",
    "iat_lt_020_ratio",
    "iat_lt_050_ratio",
    "iat_gt_100_ratio",
    "iat_gt_200_ratio",
]

BURST_FEATURES = [
    "burst_count_100ms",
    "burst_count_ratio_100ms",
    "burst_size_mean_100ms",
    "burst_size_std_100ms",
    "burst_size_max_100ms",
    "burst_bytes_mean_100ms",
    "burst_bytes_std_100ms",
    "burst_bytes_max_100ms",
    "burst_duration_mean_100ms",
    "first_burst_byte_fraction_100ms",
]

EARLY_WINDOW_FEATURES = []
for n in (5, 10, 20):
    EARLY_WINDOW_FEATURES.extend(
        [
            f"first_{n}_pkt_len_mean",
            f"first_{n}_bytes_sum",
            f"first_{n}_duration",
            f"first_{n}_bytes_fraction",
        ]
    )

TIME_BIN_FEATURES = []
for i in range(1, 6):
    TIME_BIN_FEATURES.extend(
        [
            f"bin_{i}_pkt_count_ratio",
            f"bin_{i}_byte_fraction",
        ]
    )

CUMULATIVE_FEATURES = [
    "time_to_50pct_bytes_ratio",
    "time_to_80pct_bytes_ratio",
    "early_20pct_time_byte_fraction",
    "cumulative_bytes_auc_norm",
]

FEATURE_NAMES_100 = (
    BASIC_FEATURES
    + PKT_LEN_STATS
    + PKT_LEN_DIST
    + IAT_STATS
    + IAT_THRESHOLDS
    + BURST_FEATURES
    + EARLY_WINDOW_FEATURES
    + TIME_BIN_FEATURES
    + CUMULATIVE_FEATURES
)

assert len(FEATURE_NAMES_100) == 100, len(FEATURE_NAMES_100)

FEATURE_CATEGORY: Dict[str, str] = {}
for name in BASIC_FEATURES:
    FEATURE_CATEGORY[name] = "basic_scale_time"
for name in PKT_LEN_STATS:
    FEATURE_CATEGORY[name] = "packet_length_stats"
for name in PKT_LEN_DIST:
    FEATURE_CATEGORY[name] = "packet_length_distribution"
for name in IAT_STATS:
    FEATURE_CATEGORY[name] = "iat_stats"
for name in IAT_THRESHOLDS:
    FEATURE_CATEGORY[name] = "iat_thresholds"
for name in BURST_FEATURES:
    FEATURE_CATEGORY[name] = "burst_100ms"
for name in EARLY_WINDOW_FEATURES:
    FEATURE_CATEGORY[name] = "early_window"
for name in TIME_BIN_FEATURES:
    FEATURE_CATEGORY[name] = "time_bins"
for name in CUMULATIVE_FEATURES:
    FEATURE_CATEGORY[name] = "cumulative_shape"


# =========================================================
# 2. Numeric utilities
# =========================================================

def safe_div(a: float, b: float, default: float = 0.0) -> float:
    if b is None or abs(float(b)) < EPS:
        return float(default)
    return float(a) / float(b)


def finite_or_zero(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    if math.isnan(v) or math.isinf(v):
        return 0.0
    return v


def quantile(x: np.ndarray, q: float) -> float:
    if x.size == 0:
        return 0.0
    return float(np.quantile(x, q))


def mad(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def skewness(x: np.ndarray) -> float:
    if x.size < 3:
        return 0.0
    mu = np.mean(x)
    sigma = np.std(x)
    if sigma < EPS:
        return 0.0
    return float(np.mean(((x - mu) / sigma) ** 3))


def kurtosis_excess(x: np.ndarray) -> float:
    if x.size < 4:
        return 0.0
    mu = np.mean(x)
    sigma = np.std(x)
    if sigma < EPS:
        return 0.0
    return float(np.mean(((x - mu) / sigma) ** 4) - 3.0)


def gini_coefficient(x: np.ndarray) -> float:
    """Gini coefficient for non-negative values."""
    if x.size == 0:
        return 0.0
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    x = np.maximum(x, 0.0)
    total = np.sum(x)
    if total < EPS:
        return 0.0
    xs = np.sort(x)
    n = xs.size
    idx = np.arange(1, n + 1)
    return float(np.sum((2 * idx - n - 1) * xs) / (n * total))


def array_stats(prefix: str, x: np.ndarray) -> Dict[str, float]:
    """Return the 20 statistics used for packet length or IAT, with selected prefix."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if x.size == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_var": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_range": 0.0,
            f"{prefix}_median": 0.0,
            f"{prefix}_mad": 0.0,
            f"{prefix}_q05": 0.0,
            f"{prefix}_q10": 0.0,
            f"{prefix}_q25": 0.0,
            f"{prefix}_q75": 0.0,
            f"{prefix}_q90": 0.0,
            f"{prefix}_q95": 0.0,
            f"{prefix}_iqr": 0.0,
            f"{prefix}_cv": 0.0,
            f"{prefix}_skew": 0.0,
            f"{prefix}_kurtosis": 0.0,
            f"{prefix}_rms": 0.0,
            f"{prefix}_gini": 0.0,
        }

    mean_v = float(np.mean(x))
    std_v = float(np.std(x))
    min_v = float(np.min(x))
    max_v = float(np.max(x))
    q25 = quantile(x, 0.25)
    q75 = quantile(x, 0.75)

    return {
        f"{prefix}_mean": mean_v,
        f"{prefix}_std": std_v,
        f"{prefix}_var": float(np.var(x)),
        f"{prefix}_min": min_v,
        f"{prefix}_max": max_v,
        f"{prefix}_range": max_v - min_v,
        f"{prefix}_median": float(np.median(x)),
        f"{prefix}_mad": mad(x),
        f"{prefix}_q05": quantile(x, 0.05),
        f"{prefix}_q10": quantile(x, 0.10),
        f"{prefix}_q25": q25,
        f"{prefix}_q75": q75,
        f"{prefix}_q90": quantile(x, 0.90),
        f"{prefix}_q95": quantile(x, 0.95),
        f"{prefix}_iqr": q75 - q25,
        f"{prefix}_cv": safe_div(std_v, mean_v),
        f"{prefix}_skew": skewness(x),
        f"{prefix}_kurtosis": kurtosis_excess(x),
        f"{prefix}_rms": float(np.sqrt(np.mean(x ** 2))),
        f"{prefix}_gini": gini_coefficient(x),
    }


def iat_stats(iat: np.ndarray, duration: float) -> Dict[str, float]:
    """20 IAT statistics. Uses 19 generic stats + max/duration fraction."""
    base = array_stats("iat", iat)
    # Drop rms/gini to keep exact IAT feature list and add count/max_fraction.
    base.pop("iat_rms", None)
    base.pop("iat_gini", None)
    base["iat_count"] = float(len(iat))
    base["iat_max_fraction"] = safe_div(base.get("iat_max", 0.0), duration)
    return {name: float(base.get(name, 0.0)) for name in IAT_STATS}


def clean_traffic(traffic: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Convert raw traffic list into sorted timestamp and packet-size arrays."""
    if not isinstance(traffic, list):
        return np.array([], dtype=float), np.array([], dtype=float)

    pairs = []
    for item in traffic:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        try:
            t = float(item[0])
            size = float(item[1])
        except Exception:
            continue
        if math.isnan(t) or math.isnan(size) or math.isinf(t) or math.isinf(size):
            continue
        # Packet length should be non-negative. Abnormal negative values are clipped.
        pairs.append((t, max(size, 0.0)))

    if not pairs:
        return np.array([], dtype=float), np.array([], dtype=float)

    pairs.sort(key=lambda z: z[0])
    arr = np.asarray(pairs, dtype=float)
    return arr[:, 0], arr[:, 1]


# =========================================================
# 3. Feature extraction
# =========================================================

def compute_bursts(t: np.ndarray, lengths: np.ndarray, threshold: float = 0.1) -> List[Tuple[int, float, float]]:
    """
    Burst definition:
    Consecutive packets belong to the same burst if inter-arrival time <= threshold.

    Returns a list of tuples:
        (burst_packet_count, burst_bytes, burst_duration)
    """
    n = len(t)
    if n == 0:
        return []
    if n == 1:
        return [(1, float(lengths[0]), 0.0)]

    bursts: List[Tuple[int, float, float]] = []
    start = 0
    for i in range(1, n):
        if (t[i] - t[i - 1]) > threshold:
            count = i - start
            bytes_sum = float(np.sum(lengths[start:i]))
            dur = float(t[i - 1] - t[start]) if count > 1 else 0.0
            bursts.append((count, bytes_sum, dur))
            start = i

    count = n - start
    bytes_sum = float(np.sum(lengths[start:n]))
    dur = float(t[n - 1] - t[start]) if count > 1 else 0.0
    bursts.append((count, bytes_sum, dur))
    return bursts


def extract_traffic_features_100(
    traffic: Any,
    burst_threshold: float = 0.1,
    n_time_bins: int = 5,
) -> Dict[str, float]:
    """Extract exactly 100 traffic time-series features from one record."""
    t, lengths = clean_traffic(traffic)
    n = len(lengths)

    features: Dict[str, float] = {name: 0.0 for name in FEATURE_NAMES_100}

    if n == 0:
        return features

    start_time = float(t[0])
    end_time = float(t[-1])
    duration = max(end_time - start_time, 0.0)
    total_bytes = float(np.sum(lengths))
    iat = np.diff(t) if n >= 2 else np.array([], dtype=float)
    iat = np.maximum(iat, 0.0)

    # -------------------------
    # A. Basic scale/time: 10
    # -------------------------
    features.update(
        {
            "packet_count": float(n),
            "duration": duration,
            "start_time": start_time,
            "end_time": end_time,
            "total_bytes": total_bytes,
            "log_total_bytes": float(np.log1p(total_bytes)),
            "packet_rate": safe_div(n, duration),
            "byte_rate": safe_div(total_bytes, duration),
            "mean_bytes_per_packet": safe_div(total_bytes, n),
            "active_span_ratio": safe_div(duration, end_time),
        }
    )

    # -------------------------
    # B. Packet length stats: 20
    # -------------------------
    features.update(array_stats("pkt_len", lengths))

    # -------------------------
    # C. Packet length distribution: 8
    # -------------------------
    sorted_lengths_desc = np.sort(lengths)[::-1]
    top10_k = max(1, int(math.ceil(0.10 * n)))
    features.update(
        {
            "pkt_le_64_ratio": safe_div(np.sum(lengths <= 64), n),
            "pkt_le_128_ratio": safe_div(np.sum(lengths <= 128), n),
            "pkt_le_256_ratio": safe_div(np.sum(lengths <= 256), n),
            "pkt_le_512_ratio": safe_div(np.sum(lengths <= 512), n),
            "pkt_gt_1024_ratio": safe_div(np.sum(lengths > 1024), n),
            "pkt_gt_1500_ratio": safe_div(np.sum(lengths > 1500), n),
            "top10_bytes_fraction": safe_div(np.sum(sorted_lengths_desc[:top10_k]), total_bytes),
            "largest_pkt_fraction": safe_div(np.max(lengths), total_bytes),
        }
    )

    # -------------------------
    # D. IAT stats: 20
    # -------------------------
    features.update(iat_stats(iat, duration))

    # -------------------------
    # E. IAT thresholds: 6
    # Timestamp unit is assumed to be seconds.
    # 0.005 = 5 ms, 0.010 = 10 ms, etc.
    # -------------------------
    if len(iat) > 0:
        features.update(
            {
                "iat_lt_005_ratio": safe_div(np.sum(iat < 0.005), len(iat)),
                "iat_lt_010_ratio": safe_div(np.sum(iat < 0.010), len(iat)),
                "iat_lt_020_ratio": safe_div(np.sum(iat < 0.020), len(iat)),
                "iat_lt_050_ratio": safe_div(np.sum(iat < 0.050), len(iat)),
                "iat_gt_100_ratio": safe_div(np.sum(iat > 0.100), len(iat)),
                "iat_gt_200_ratio": safe_div(np.sum(iat > 0.200), len(iat)),
            }
        )

    # -------------------------
    # F. Burst features: 10
    # -------------------------
    bursts = compute_bursts(t, lengths, threshold=burst_threshold)
    if bursts:
        burst_sizes = np.asarray([b[0] for b in bursts], dtype=float)
        burst_bytes = np.asarray([b[1] for b in bursts], dtype=float)
        burst_durs = np.asarray([b[2] for b in bursts], dtype=float)
        features.update(
            {
                "burst_count_100ms": float(len(bursts)),
                "burst_count_ratio_100ms": safe_div(len(bursts), n),
                "burst_size_mean_100ms": float(np.mean(burst_sizes)),
                "burst_size_std_100ms": float(np.std(burst_sizes)),
                "burst_size_max_100ms": float(np.max(burst_sizes)),
                "burst_bytes_mean_100ms": float(np.mean(burst_bytes)),
                "burst_bytes_std_100ms": float(np.std(burst_bytes)),
                "burst_bytes_max_100ms": float(np.max(burst_bytes)),
                "burst_duration_mean_100ms": float(np.mean(burst_durs)),
                "first_burst_byte_fraction_100ms": safe_div(bursts[0][1], total_bytes),
            }
        )

    # -------------------------
    # G. Early-window features: 12
    # -------------------------
    for w in (5, 10, 20):
        k = min(w, n)
        w_lengths = lengths[:k]
        w_duration = float(t[k - 1] - t[0]) if k > 1 else 0.0
        w_bytes = float(np.sum(w_lengths))
        features.update(
            {
                f"first_{w}_pkt_len_mean": float(np.mean(w_lengths)) if k > 0 else 0.0,
                f"first_{w}_bytes_sum": w_bytes,
                f"first_{w}_duration": w_duration,
                f"first_{w}_bytes_fraction": safe_div(w_bytes, total_bytes),
            }
        )

    # -------------------------
    # H. Time-bin features: 10
    # 5 equal-width time bins, each with packet-count ratio and byte fraction.
    # -------------------------
    if duration <= EPS:
        bin_idx = np.zeros(n, dtype=int)
    else:
        normalized_time = (t - start_time) / duration
        bin_idx = np.floor(normalized_time * n_time_bins).astype(int)
        bin_idx = np.clip(bin_idx, 0, n_time_bins - 1)

    for i in range(n_time_bins):
        mask = bin_idx == i
        features[f"bin_{i + 1}_pkt_count_ratio"] = safe_div(np.sum(mask), n)
        features[f"bin_{i + 1}_byte_fraction"] = safe_div(np.sum(lengths[mask]), total_bytes)

    # -------------------------
    # I. Cumulative-shape features: 4
    # -------------------------
    if total_bytes > EPS:
        cum_bytes = np.cumsum(lengths)
        cum_frac = cum_bytes / total_bytes

        def time_to_pct(p: float) -> float:
            idx = int(np.searchsorted(cum_frac, p, side="left"))
            idx = min(max(idx, 0), n - 1)
            return safe_div(t[idx] - start_time, duration)

        if duration > EPS:
            norm_t = (t - start_time) / duration
            early_mask = norm_t <= 0.20
            # Integral of cumulative byte fraction over normalized time.
            x_auc = np.concatenate([[0.0], norm_t])
            y_auc = np.concatenate([[0.0], cum_frac])
            auc_norm = float(np.trapezoid(y_auc, x_auc))
        else:
            early_mask = np.ones(n, dtype=bool)
            auc_norm = 0.0

        features.update(
            {
                "time_to_50pct_bytes_ratio": time_to_pct(0.50),
                "time_to_80pct_bytes_ratio": time_to_pct(0.80),
                "early_20pct_time_byte_fraction": safe_div(np.sum(lengths[early_mask]), total_bytes),
                "cumulative_bytes_auc_norm": auc_norm,
            }
        )

    # Ensure finite numeric output and exact 100 features.
    features = {name: finite_or_zero(features.get(name, 0.0)) for name in FEATURE_NAMES_100}
    assert len(features) == 100, len(features)
    return features


# =========================================================
# 4. Dataset loading and feature table generation
# =========================================================

def load_json_dataset(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("The input JSON must be a list of records.")
    return data


def build_feature_dataframe(
    data: List[Dict[str, Any]],
    burst_threshold: float = 0.1,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for idx, item in enumerate(data):
        feats = extract_traffic_features_100(
            item.get("traffic", []),
            burst_threshold=burst_threshold,
        )
        row: Dict[str, Any] = {
            "sample_id": idx,
            "url": item.get("url", ""),
            "label": item.get("label", None),
            "phish_type": item.get("phish_type", None),
            "risk_score": item.get("risk_score", None),
        }
        row.update(feats)
        rows.append(row)

    df = pd.DataFrame(rows)
    ordered_cols = ["sample_id", "url", "label", "phish_type", "risk_score"] + FEATURE_NAMES_100
    return df[ordered_cols]


# =========================================================
# 5. Feature effectiveness analysis
# =========================================================

def rank_auc_binary(y_true: np.ndarray, score: np.ndarray) -> float:
    """
    ROC-AUC implemented by pairwise comparison.
    This avoids requiring scikit-learn and is fine for this dataset scale.
    Returns P(score_pos > score_neg) + 0.5 * P(tie).
    """
    y_true = np.asarray(y_true)
    score = np.asarray(score, dtype=float)
    pos = score[y_true == 1]
    neg = score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5

    greater = 0.0
    ties = 0.0
    for v in pos:
        greater += float(np.sum(v > neg))
        ties += float(np.sum(v == neg))
    return float((greater + 0.5 * ties) / (len(pos) * len(neg)))


def fdr_bh(p_values: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg FDR correction."""
    p = np.asarray([1.0 if (pd.isna(v) or np.isinf(v)) else float(v) for v in p_values], dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q_ranked = np.empty(n, dtype=float)
    prev = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        q = ranked[i] * n / rank
        q = min(q, prev)
        q_ranked[i] = q
        prev = q
    q = np.empty(n, dtype=float)
    q[order] = np.clip(q_ranked, 0.0, 1.0)
    return q


def analyze_feature_effectiveness(
    feature_df: pd.DataFrame,
    label_col: str = "label",
    q_threshold: float = 0.05,
    cliff_threshold: float = 0.147,
    auc_threshold: float = 0.60,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compare phishing(label=1) vs benign(label=0) for each traffic feature.

    Output feature-level table columns include:
        mean/median by class, Mann-Whitney p, KS p,
        FDR q-value, Cliff's delta, single-feature AUC, is_effective.
    """
    if label_col not in feature_df.columns:
        raise ValueError(f"Missing label column: {label_col}")

    y = feature_df[label_col].astype(int).values
    results: List[Dict[str, Any]] = []

    for name in FEATURE_NAMES_100:
        x = feature_df[name].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).values
        x_phish = x[y == 1]
        x_benign = x[y == 0]

        if len(x_phish) == 0 or len(x_benign) == 0:
            mw_p = np.nan
            ks_p = np.nan
        else:
            if mannwhitneyu is not None:
                try:
                    mw_p = float(mannwhitneyu(x_phish, x_benign, alternative="two-sided").pvalue)
                except Exception:
                    mw_p = np.nan
            else:
                mw_p = np.nan

            if ks_2samp is not None:
                try:
                    ks_p = float(ks_2samp(x_phish, x_benign, alternative="two-sided").pvalue)
                except Exception:
                    ks_p = np.nan
            else:
                ks_p = np.nan

        auc_raw = rank_auc_binary(y, x)
        single_auc = max(auc_raw, 1.0 - auc_raw)
        cliffs_delta = 2.0 * auc_raw - 1.0

        results.append(
            {
                "feature": name,
                "category": FEATURE_CATEGORY.get(name, "unknown"),
                "phishing_mean": float(np.mean(x_phish)) if len(x_phish) else 0.0,
                "benign_mean": float(np.mean(x_benign)) if len(x_benign) else 0.0,
                "phishing_median": float(np.median(x_phish)) if len(x_phish) else 0.0,
                "benign_median": float(np.median(x_benign)) if len(x_benign) else 0.0,
                "mean_diff_phish_minus_benign": float(np.mean(x_phish) - np.mean(x_benign)) if len(x_phish) and len(x_benign) else 0.0,
                "median_diff_phish_minus_benign": float(np.median(x_phish) - np.median(x_benign)) if len(x_phish) and len(x_benign) else 0.0,
                "mannwhitney_p_value": mw_p,
                "ks_p_value": ks_p,
                "auc_direction_phish_larger": auc_raw,
                "single_feature_auc": single_auc,
                "cliffs_delta": cliffs_delta,
            }
        )

    analysis_df = pd.DataFrame(results)
    analysis_df["q_value"] = fdr_bh(analysis_df["mannwhitney_p_value"].values)
    analysis_df["is_effective"] = (
        (analysis_df["q_value"] < q_threshold)
        & (analysis_df["cliffs_delta"].abs() >= cliff_threshold)
        & (analysis_df["single_feature_auc"] >= auc_threshold)
    )

    analysis_df = analysis_df.sort_values(
        by=["is_effective", "single_feature_auc", "q_value"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    summary_df = (
        analysis_df.groupby("category", as_index=False)
        .agg(
            total_features=("feature", "count"),
            effective_features=("is_effective", "sum"),
            mean_single_feature_auc=("single_feature_auc", "mean"),
            max_single_feature_auc=("single_feature_auc", "max"),
            mean_abs_cliffs_delta=("cliffs_delta", lambda s: float(np.mean(np.abs(s)))),
        )
        .sort_values(by="effective_features", ascending=False)
    )
    summary_df["effective_ratio"] = summary_df["effective_features"] / summary_df["total_features"]

    total_row = pd.DataFrame(
        [
            {
                "category": "ALL",
                "total_features": int(len(analysis_df)),
                "effective_features": int(analysis_df["is_effective"].sum()),
                "mean_single_feature_auc": float(analysis_df["single_feature_auc"].mean()),
                "max_single_feature_auc": float(analysis_df["single_feature_auc"].max()),
                "mean_abs_cliffs_delta": float(np.mean(np.abs(analysis_df["cliffs_delta"]))),
                "effective_ratio": float(analysis_df["is_effective"].mean()),
            }
        ]
    )
    summary_df = pd.concat([summary_df, total_row], ignore_index=True)
    return analysis_df, summary_df


# =========================================================
# 6. CLI
# =========================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract and analyze 100 traffic time-series features.")
    parser.add_argument("--input", default = "train.json", help="Input JSON dataset path.")
    parser.add_argument("--out_features", default="traffic_100_features.csv", help="Output feature CSV path.")
    parser.add_argument("--out_analysis", default="traffic_feature_analysis.csv", help="Output feature-effectiveness CSV path.")
    parser.add_argument("--out_summary", default="traffic_feature_category_summary.csv", help="Output category summary CSV path.")
    parser.add_argument("--burst_threshold", type=float, default=0.1, help="Burst IAT threshold in seconds. Default: 0.1 = 100 ms.")
    parser.add_argument("--label_col", default="label", help="Label column name. Default: label.")
    parser.add_argument("--q_threshold", type=float, default=0.05, help="FDR q-value threshold.")
    parser.add_argument("--cliff_threshold", type=float, default=0.147, help="Absolute Cliff's delta threshold.")
    parser.add_argument("--auc_threshold", type=float, default=0.60, help="Single-feature AUC threshold.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data = load_json_dataset(args.input)
    feature_df = build_feature_dataframe(data, burst_threshold=args.burst_threshold)

    feature_df.to_csv(args.out_features, index=False, encoding="utf-8-sig")

    analysis_df, summary_df = analyze_feature_effectiveness(
        feature_df,
        label_col=args.label_col,
        q_threshold=args.q_threshold,
        cliff_threshold=args.cliff_threshold,
        auc_threshold=args.auc_threshold,
    )
    analysis_df.to_csv(args.out_analysis, index=False, encoding="utf-8-sig")
    summary_df.to_csv(args.out_summary, index=False, encoding="utf-8-sig")

    n_records = len(feature_df)
    n_phish = int((feature_df[args.label_col] == 1).sum()) if args.label_col in feature_df.columns else -1
    n_benign = int((feature_df[args.label_col] == 0).sum()) if args.label_col in feature_df.columns else -1
    effective_total = int(analysis_df["is_effective"].sum())
    effective_ratio = float(analysis_df["is_effective"].mean())

    print("Done.")
    print(f"Input records: {n_records}")
    print(f"Benign(label=0): {n_benign}, Phishing(label=1): {n_phish}")
    print(f"Extracted traffic features: {len(FEATURE_NAMES_100)}")
    print(f"Effective features: {effective_total}/{len(FEATURE_NAMES_100)} = {effective_ratio:.2%}")
    print(f"Saved features: {args.out_features}")
    print(f"Saved analysis: {args.out_analysis}")
    print(f"Saved summary: {args.out_summary}")


if __name__ == "__main__":
    main()
