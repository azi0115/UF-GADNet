"""Independent flow-only 30-feature LightGBM experiment."""

from __future__ import annotations

import ast
import csv
import importlib.util
import json
import logging
import math
import os
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import RobustScaler, StandardScaler

logger = logging.getLogger(__name__)

FLOW_30_FEATURE_NAMES: list[str] = [
    "pkt_len_kurtosis",
    "first_10_pkt_len_mean",
    "first_10_bytes_sum",
    "first_20_bytes_sum",
    "first_20_pkt_len_mean",
    "burst_count_ratio_100ms",
    "burst_size_mean_100ms",
    "packet_count",
    "iat_count",
    "burst_size_max_100ms",
    "iat_lt_050_ratio",
    "burst_bytes_mean_100ms",
    "total_bytes",
    "log_total_bytes",
    "burst_bytes_max_100ms",
    "iat_gt_100_ratio",
    "burst_size_std_100ms",
    "burst_duration_mean_100ms",
    "largest_pkt_fraction",
    "pkt_len_iqr",
    "iat_skew",
    "bin_2_pkt_count_ratio",
    "iat_kurtosis",
    "bin_2_byte_fraction",
    "burst_bytes_std_100ms",
    "iat_gt_200_ratio",
    "first_5_bytes_fraction",
    "pkt_len_std",
    "pkt_len_var",
    "iat_lt_020_ratio",
]


def _require_lightgbm():
    if importlib.util.find_spec("lightgbm") is None:
        raise ImportError(
            "lightgbm is not installed. Install it first with: pip install lightgbm"
        )
    import lightgbm as lgb

    return lgb


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def setup_logging(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "train.log")
    formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    root.addHandler(stream_handler)
    root.addHandler(file_handler)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def _try_parse_traffic(value: Any) -> list[list[float]] | None:
    if isinstance(value, list):
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return None
    return parsed if isinstance(parsed, list) else None


def _infer_first_present(keys: Iterable[str], candidates: Sequence[str]) -> str | None:
    key_set = {str(key) for key in keys}
    for candidate in candidates:
        if candidate in key_set:
            return candidate
    return None


def _normalize_packet_rows(raw_traffic: Sequence[Any]) -> list[tuple[float, float]]:
    rows: list[tuple[float, float]] = []
    for item in raw_traffic or []:
        if isinstance(item, (list, tuple)):
            if len(item) >= 2:
                timestamp = _safe_float(item[0])
                packet_length = _safe_float(item[1])
            elif len(item) == 1:
                timestamp = _safe_float(item[0])
                packet_length = 0.0
            else:
                continue
        elif isinstance(item, dict):
            timestamp = _safe_float(item.get("timestamp", item.get("time", 0.0)))
            packet_length = _safe_float(
                item.get("packet_length", item.get("pkt_len", item.get("length", item.get("size", 0.0))))
            )
        else:
            continue
        rows.append((timestamp, max(packet_length, 0.0)))
    return rows


def _normalize_timestamps_to_seconds(rows: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if len(rows) < 2:
        return rows
    timestamps = np.asarray([item[0] for item in rows], dtype=np.float64)
    diffs = np.diff(np.sort(timestamps))
    positive_diffs = diffs[diffs > 0]
    if positive_diffs.size == 0:
        return rows
    if float(np.max(timestamps)) > 10_000 or float(np.median(positive_diffs)) > 10:
        return [(timestamp / 1000.0, packet_length) for timestamp, packet_length in rows]
    return rows


def _sample_level_from_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not records:
        return []
    first = records[0]
    if "traffic" in first:
        label_key = _infer_first_present(first.keys(), ["label", "target", "y"])
        if label_key is None:
            raise ValueError("Sample-level records must contain a label field.")
        sample_level: list[dict[str, Any]] = []
        for index, record in enumerate(records):
            sample_id = record.get("sample_id", record.get("flow_id", record.get("id", index)))
            traffic = _try_parse_traffic(record.get("traffic"))
            sample_level.append(
                {
                    "sample_id": sample_id,
                    "label": int(record[label_key]),
                    "traffic": traffic or [],
                }
            )
        return sample_level

    sample_id_key = _infer_first_present(first.keys(), ["sample_id", "flow_id", "id"])
    label_key = _infer_first_present(first.keys(), ["label", "target", "y"])
    timestamp_key = _infer_first_present(first.keys(), ["timestamp", "time"])
    pkt_len_key = _infer_first_present(first.keys(), ["packet_length", "pkt_len", "length", "size"])
    if sample_id_key is None or label_key is None or timestamp_key is None or pkt_len_key is None:
        raise ValueError(
            "Unable to infer grouped flow schema. Expected sample_id/flow_id, label, timestamp/time, and packet_length/pkt_len/length/size."
        )

    grouped: dict[str, dict[str, Any]] = {}
    for row in records:
        sample_id = row[sample_id_key]
        grouped.setdefault(
            str(sample_id),
            {"sample_id": sample_id, "label": int(row[label_key]), "traffic": []},
        )
        grouped[str(sample_id)]["traffic"].append(
            [_safe_float(row[timestamp_key]), _safe_float(row[pkt_len_key])]
        )
    return list(grouped.values())


def load_flow_samples(path: str) -> list[dict[str, Any]]:
    suffix = Path(path).suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        if isinstance(payload, tuple):
            payload = list(payload)
        if not isinstance(payload, list):
            raise ValueError(f"Expected a pickled list in {path}, got {type(payload).__name__}.")
        return _sample_level_from_records(payload)

    if suffix == ".csv":
        with open(path, "r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            return _sample_level_from_records(list(reader))

    with open(path, "r", encoding="utf-8") as handle:
        content = handle.read().strip()
    if not content:
        return []
    if content[0] == "[":
        return _sample_level_from_records(json.loads(content))
    return _sample_level_from_records([json.loads(line) for line in content.splitlines() if line.strip()])


def _safe_skew(values: np.ndarray) -> float:
    if values.size < 3:
        return 0.0
    std = float(np.std(values))
    if std <= 0:
        return 0.0
    mean = float(np.mean(values))
    centered = (values - mean) / std
    result = float(np.mean(np.power(centered, 3)))
    return result if math.isfinite(result) else 0.0


def _safe_kurtosis(values: np.ndarray) -> float:
    if values.size < 4:
        return 0.0
    std = float(np.std(values))
    if std <= 0:
        return 0.0
    mean = float(np.mean(values))
    centered = (values - mean) / std
    result = float(np.mean(np.power(centered, 4)) - 3.0)
    return result if math.isfinite(result) else 0.0


def extract_flow_features(flow_record: dict[str, Any]) -> Dict[str, float]:
    rows = _normalize_packet_rows(flow_record.get("traffic", []))
    rows = _normalize_timestamps_to_seconds(rows)
    rows = sorted(rows, key=lambda item: item[0])

    if not rows:
        return {name: 0.0 for name in FLOW_30_FEATURE_NAMES}

    timestamps = np.asarray([row[0] for row in rows], dtype=np.float64)
    packet_lengths = np.asarray([row[1] for row in rows], dtype=np.float64)
    total_bytes = float(packet_lengths.sum())
    packet_count = int(packet_lengths.size)

    iat = np.diff(timestamps) if packet_count >= 2 else np.asarray([], dtype=np.float64)
    if iat.size:
        iat = np.clip(iat, a_min=0.0, a_max=None)

    q1 = float(np.quantile(packet_lengths, 0.25)) if packet_count else 0.0
    q3 = float(np.quantile(packet_lengths, 0.75)) if packet_count else 0.0

    first_5_sum = float(packet_lengths[:5].sum()) if packet_count else 0.0
    first_10_sum = float(packet_lengths[:10].sum()) if packet_count else 0.0
    first_20_sum = float(packet_lengths[:20].sum()) if packet_count else 0.0

    first_10_mean = float(packet_lengths[:10].mean()) if packet_count else 0.0
    first_20_mean = float(packet_lengths[:20].mean()) if packet_count else 0.0

    duration = float(timestamps[-1] - timestamps[0]) if packet_count >= 2 else 0.0

    burst_sizes: list[int] = []
    burst_bytes: list[float] = []
    burst_durations: list[float] = []
    if packet_count >= 1:
        burst_start = 0
        for index in range(1, packet_count):
            if float(timestamps[index] - timestamps[index - 1]) > 0.1:
                burst_sizes.append(index - burst_start)
                burst_bytes.append(float(packet_lengths[burst_start:index].sum()))
                burst_durations.append(float(timestamps[index - 1] - timestamps[burst_start]))
                burst_start = index
        burst_sizes.append(packet_count - burst_start)
        burst_bytes.append(float(packet_lengths[burst_start:].sum()))
        burst_durations.append(float(timestamps[-1] - timestamps[burst_start]) if packet_count - burst_start > 1 else 0.0)

    burst_sizes_arr = np.asarray(burst_sizes, dtype=np.float64) if burst_sizes else np.asarray([], dtype=np.float64)
    burst_bytes_arr = np.asarray(burst_bytes, dtype=np.float64) if burst_bytes else np.asarray([], dtype=np.float64)
    burst_durations_arr = np.asarray(burst_durations, dtype=np.float64) if burst_durations else np.asarray([], dtype=np.float64)

    if duration > 0:
        normalized_time = (timestamps - timestamps[0]) / duration
        bin_edges = np.linspace(0.0, 1.0, 6)
        bin_index = np.digitize(normalized_time, bin_edges[1:-1], right=False)
        second_bin_mask = bin_index == 1
    else:
        second_bin_mask = np.zeros(packet_count, dtype=bool)

    features = {
        "pkt_len_kurtosis": _safe_kurtosis(packet_lengths),
        "first_10_pkt_len_mean": first_10_mean,
        "first_10_bytes_sum": first_10_sum,
        "first_20_bytes_sum": first_20_sum,
        "first_20_pkt_len_mean": first_20_mean,
        "burst_count_ratio_100ms": _safe_divide(float(len(burst_sizes)), packet_count),
        "burst_size_mean_100ms": float(burst_sizes_arr.mean()) if burst_sizes_arr.size else 0.0,
        "packet_count": float(packet_count),
        "iat_count": float(max(packet_count - 1, 0)),
        "burst_size_max_100ms": float(burst_sizes_arr.max()) if burst_sizes_arr.size else 0.0,
        "iat_lt_050_ratio": _safe_divide(float((iat < 0.50).sum()), iat.size),
        "burst_bytes_mean_100ms": float(burst_bytes_arr.mean()) if burst_bytes_arr.size else 0.0,
        "total_bytes": total_bytes,
        "log_total_bytes": float(np.log1p(total_bytes)),
        "burst_bytes_max_100ms": float(burst_bytes_arr.max()) if burst_bytes_arr.size else 0.0,
        "iat_gt_100_ratio": _safe_divide(float((iat > 1.00).sum()), iat.size),
        "burst_size_std_100ms": float(burst_sizes_arr.std()) if burst_sizes_arr.size >= 2 else 0.0,
        "burst_duration_mean_100ms": float(burst_durations_arr.mean()) if burst_durations_arr.size else 0.0,
        "largest_pkt_fraction": _safe_divide(float(packet_lengths.max()) if packet_count else 0.0, total_bytes),
        "pkt_len_iqr": q3 - q1,
        "iat_skew": _safe_skew(iat),
        "bin_2_pkt_count_ratio": _safe_divide(float(second_bin_mask.sum()), packet_count),
        "iat_kurtosis": _safe_kurtosis(iat),
        "bin_2_byte_fraction": _safe_divide(float(packet_lengths[second_bin_mask].sum()), total_bytes),
        "burst_bytes_std_100ms": float(burst_bytes_arr.std()) if burst_bytes_arr.size >= 2 else 0.0,
        "iat_gt_200_ratio": _safe_divide(float((iat > 2.00).sum()), iat.size),
        "first_5_bytes_fraction": _safe_divide(first_5_sum, total_bytes),
        "pkt_len_std": float(packet_lengths.std()) if packet_count >= 2 else 0.0,
        "pkt_len_var": float(packet_lengths.var()) if packet_count >= 2 else 0.0,
        "iat_lt_020_ratio": _safe_divide(float((iat < 0.20).sum()), iat.size),
    }
    return {name: (float(value) if math.isfinite(float(value)) else 0.0) for name, value in features.items()}


def sanitize_feature_matrix(matrix: np.ndarray) -> np.ndarray:
    features = np.asarray(matrix, dtype=np.float32).copy()
    features[~np.isfinite(features)] = 0.0
    return features


def clip_with_train_quantiles(train_matrix: np.ndarray, matrix: np.ndarray, lower_q: float = 0.01, upper_q: float = 0.99) -> np.ndarray:
    lower = np.quantile(train_matrix, lower_q, axis=0)
    upper = np.quantile(train_matrix, upper_q, axis=0)
    return np.clip(matrix, lower, upper)


def build_scaler(kind: str):
    if kind == "robust":
        return RobustScaler()
    return StandardScaler()


@dataclass
class PreparedSplit:
    sample_ids: list[str]
    labels: np.ndarray
    features: np.ndarray


def _rows_from_samples(samples: list[dict[str, Any]], split_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, sample in enumerate(samples):
        features = extract_flow_features(sample)
        row = {
            "sample_id": str(sample.get("sample_id", index)),
            "label": int(sample["label"]),
            "split": split_name,
        }
        row.update(features)
        rows.append(row)
    return rows


def prepare_split(samples: list[dict[str, Any]], split_name: str) -> PreparedSplit:
    rows = _rows_from_samples(samples, split_name)
    feature_matrix = np.asarray([[row[name] for name in FLOW_30_FEATURE_NAMES] for row in rows], dtype=np.float32)
    labels = np.asarray([row["label"] for row in rows], dtype=np.int64)
    sample_ids = [row["sample_id"] for row in rows]
    return PreparedSplit(sample_ids=sample_ids, labels=labels, features=sanitize_feature_matrix(feature_matrix))


def save_feature_table(output_dir: str, splits: dict[str, PreparedSplit]) -> str:
    path = os.path.join(output_dir, "flow_30_features.csv")
    os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        fieldnames = ["sample_id", "label", "split", *FLOW_30_FEATURE_NAMES]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for split_name, split in splits.items():
            for sample_id, label, feature_vector in zip(split.sample_ids, split.labels, split.features, strict=True):
                row = {
                    "sample_id": sample_id,
                    "label": int(label),
                    "split": split_name,
                }
                row.update({name: float(value) for name, value in zip(FLOW_30_FEATURE_NAMES, feature_vector, strict=True)})
                writer.writerow(row)
    return path


def save_scaler(output_dir: str, scaler) -> str:
    path = os.path.join(output_dir, "scaler.pkl")
    with open(path, "wb") as handle:
        pickle.dump(scaler, handle)
    return path


def save_model(output_dir: str, model) -> str:
    path = os.path.join(output_dir, "best_flow_30_feature_lightgbm.pkl")
    with open(path, "wb") as handle:
        pickle.dump(model, handle)
    return path


def save_metrics(output_dir: str, metrics: dict[str, Any]) -> str:
    path = os.path.join(output_dir, "test_metrics.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    return path


def append_train_log(output_dir: str, rows: list[dict[str, Any]]) -> str:
    path = os.path.join(output_dir, "train_log.csv")
    if not rows:
        return path
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def compute_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(np.int64)
    try:
        auc = roc_auc_score(labels, probabilities)
    except ValueError:
        auc = 0.0
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "auc": float(auc),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
        "classification_report": classification_report(labels, predictions, zero_division=0, output_dict=True),
    }


def run_sanity_checks(train_split: PreparedSplit, val_split: PreparedSplit, test_split: PreparedSplit) -> None:
    logger.info("train/val/test sizes: %d / %d / %d", len(train_split.labels), len(val_split.labels), len(test_split.labels))
    logger.info("label distribution train: %s", {int(v): int((train_split.labels == v).sum()) for v in np.unique(train_split.labels)})
    logger.info("first 5 samples' features:")
    for index in range(min(5, len(train_split.sample_ids))):
        logger.info(
            "sample_id=%s label=%d features=%s",
            train_split.sample_ids[index],
            int(train_split.labels[index]),
            train_split.features[index].tolist(),
        )
    for feature_index, feature_name in enumerate(FLOW_30_FEATURE_NAMES):
        column = train_split.features[:, feature_index]
        logger.info(
            "feature=%s mean=%.6f std=%.6f min=%.6f max=%.6f",
            feature_name,
            float(column.mean()),
            float(column.std()),
            float(column.min()),
            float(column.max()),
        )
    logger.info("has_nan=%s has_inf=%s", bool(np.isnan(train_split.features).any()), bool(np.isinf(train_split.features).any()))
    logger.info("LightGBM input shape per sample: [30]")


def overfit_debug(
    train_split: PreparedSplit,
    seed: int,
    num_estimators: int,
    learning_rate: float,
    num_leaves: int,
    max_depth: int,
) -> None:
    logger.info("Running optional 100-sample overfit debug.")
    lgb = _require_lightgbm()
    subset_size = min(100, len(train_split.labels))
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=num_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        max_depth=max_depth,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(train_split.features[:subset_size], train_split.labels[:subset_size])
    probabilities = model.predict_proba(train_split.features[:subset_size])[:, 1]
    metrics = compute_metrics(train_split.labels[:subset_size], probabilities)
    logger.info(
        "overfit_debug acc=%.4f precision=%.4f recall=%.4f f1=%.4f auc=%.4f",
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
        metrics["auc"],
    )


def run_flow_30_feature_transformer_experiment(
    train_path: str,
    val_path: str,
    test_path: str,
    output_dir: str,
    batch_size: int = 64,
    lr: float = 1e-4,
    epochs: int = 30,
    weight_decay: float = 1e-4,
    seed: int = 42,
    scaler_name: str = "standard",
    d_model: int = 128,
    nhead: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 256,
    dropout: float = 0.1,
    best_metric: str = "f1",
    refit_features: bool = True,
    overfit_100_debug: bool = False,
    num_estimators: int = 1000,
    num_leaves: int = 63,
    max_depth: int = -1,
    min_child_samples: int = 20,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
) -> dict[str, Any]:
    del batch_size, epochs, weight_decay, d_model, nhead, num_layers, dim_feedforward, dropout, refit_features
    lgb = _require_lightgbm()
    setup_logging(output_dir)
    set_seed(seed)
    logger.info("Using LightGBM for the independent 30-feature flow-only experiment.")

    train_samples = load_flow_samples(train_path)
    val_samples = load_flow_samples(val_path)
    test_samples = load_flow_samples(test_path)

    train_split_raw = prepare_split(train_samples, "train")
    val_split_raw = prepare_split(val_samples, "val")
    test_split_raw = prepare_split(test_samples, "test")

    run_sanity_checks(train_split_raw, val_split_raw, test_split_raw)
    save_feature_table(output_dir, {"train": train_split_raw, "val": val_split_raw, "test": test_split_raw})

    train_features = sanitize_feature_matrix(train_split_raw.features)
    val_features = sanitize_feature_matrix(val_split_raw.features)
    test_features = sanitize_feature_matrix(test_split_raw.features)
    train_features = clip_with_train_quantiles(train_features, train_features)
    val_features = clip_with_train_quantiles(train_features, val_features)
    test_features = clip_with_train_quantiles(train_features, test_features)

    scaler = build_scaler(scaler_name)
    scaler.fit(train_features)
    save_scaler(output_dir, scaler)

    train_split = PreparedSplit(
        sample_ids=train_split_raw.sample_ids,
        labels=train_split_raw.labels,
        features=scaler.transform(train_features).astype(np.float32),
    )
    val_split = PreparedSplit(
        sample_ids=val_split_raw.sample_ids,
        labels=val_split_raw.labels,
        features=scaler.transform(val_features).astype(np.float32),
    )
    test_split = PreparedSplit(
        sample_ids=test_split_raw.sample_ids,
        labels=test_split_raw.labels,
        features=scaler.transform(test_features).astype(np.float32),
    )

    if overfit_100_debug:
        overfit_debug(
            train_split=train_split,
            seed=seed,
            num_estimators=min(num_estimators, 200),
            learning_rate=max(lr, 1e-3),
            num_leaves=num_leaves,
            max_depth=max_depth,
        )

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=num_estimators,
        learning_rate=lr,
        num_leaves=num_leaves,
        max_depth=max_depth,
        min_child_samples=min_child_samples,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(
        train_split.features,
        train_split.labels,
        eval_set=[(train_split.features, train_split.labels), (val_split.features, val_split.labels)],
        eval_names=["train", "val"],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(50)],
    )

    train_eval = model.evals_result_.get("train", {}).get("binary_logloss", [])
    val_eval = model.evals_result_.get("val", {}).get("binary_logloss", [])
    best_iteration = getattr(model, "best_iteration_", None) or len(val_eval) or num_estimators

    val_probabilities = model.predict_proba(val_split.features)[:, 1]
    val_metrics = compute_metrics(val_split.labels, val_probabilities)
    threshold = 0.5
    append_train_log(
        output_dir,
        [
            {
                "iteration": index + 1,
                "train_loss": float(train_eval[index]) if index < len(train_eval) else None,
                "val_loss": float(val_eval[index]) if index < len(val_eval) else None,
                "val_acc": val_metrics["accuracy"] if index + 1 == best_iteration else None,
                "val_precision": val_metrics["precision"] if index + 1 == best_iteration else None,
                "val_recall": val_metrics["recall"] if index + 1 == best_iteration else None,
                "val_f1": val_metrics["f1"] if index + 1 == best_iteration else None,
                "val_auc": val_metrics["auc"] if index + 1 == best_iteration else None,
            }
            for index in range(max(len(train_eval), len(val_eval)))
        ],
    )

    best_score = float(val_metrics["auc"] if best_metric.lower() == "auc" else val_metrics["f1"])
    best_model_path = save_model(output_dir, model)
    logger.info(
        "validation metrics | acc=%.4f precision=%.4f recall=%.4f f1=%.4f auc=%.4f best_iteration=%d",
        val_metrics["accuracy"],
        val_metrics["precision"],
        val_metrics["recall"],
        val_metrics["f1"],
        val_metrics["auc"],
        best_iteration,
    )

    test_probabilities = model.predict_proba(test_split.features)[:, 1]
    test_metrics = compute_metrics(test_split.labels, test_probabilities, threshold=threshold)
    final_metrics = {
        "best_metric": best_metric,
        "best_score": best_score,
        "best_iteration": int(best_iteration),
        "model_path": best_model_path,
        "accuracy": test_metrics["accuracy"],
        "precision": test_metrics["precision"],
        "recall": test_metrics["recall"],
        "f1": test_metrics["f1"],
        "auc": test_metrics["auc"],
        "confusion_matrix": test_metrics["confusion_matrix"],
        "classification_report": test_metrics["classification_report"],
    }
    save_metrics(output_dir, final_metrics)
    logger.info("Test metrics: %s", final_metrics)
    return final_metrics
