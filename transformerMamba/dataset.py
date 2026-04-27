"""Data loading, URL n-gram encoding, and batch collation utilities."""

from __future__ import annotations

import ipaddress
import json
import logging
import os
import pickle
from collections import Counter
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_ID = 0
UNK_ID = 1
POSITION_VOCAB_KIND = "position-aware"
PLAIN_VOCAB_KIND = "plain"

logger = logging.getLogger(__name__)


def _describe_value(value: Any) -> str:
    """Generate a compact string summary for schema errors."""
    text = repr(value)
    if len(text) > 80:
        text = text[:77] + "..."
    return f"{type(value).__name__}({text})"


def _raise_schema_error(index: int, field: str, expected: str, value: Any) -> None:
    """Raise a uniform schema validation error."""
    raise ValueError(
        f"Invalid sample at index {index}: field '{field}' expected {expected}, got {_describe_value(value)}."
    )


def _validate_traffic_payload(traffic_raw: Any, index: int, required: bool) -> Sequence[Any]:
    """Validate the top-level traffic payload."""
    if traffic_raw is None:
        if required:
            _raise_schema_error(index, "traffic", "a sequence of packets", traffic_raw)
        return []
    if not isinstance(traffic_raw, (list, tuple)):
        _raise_schema_error(index, "traffic", "a list or tuple", traffic_raw)

    for packet_index, item in enumerate(traffic_raw):
        if isinstance(item, (list, tuple)):
            if len(item) not in {1, 2}:
                _raise_schema_error(index, f"traffic[{packet_index}]", "a list/tuple of length 1 or 2", item)
            for value_index, value in enumerate(item):
                if not isinstance(value, (int, float)):
                    _raise_schema_error(
                        index,
                        f"traffic[{packet_index}][{value_index}]",
                        "an int or float",
                        value,
                    )
        elif not isinstance(item, (int, float)):
            _raise_schema_error(index, f"traffic[{packet_index}]", "an int, float, list, or tuple", item)
    return traffic_raw


def validate_record_schema(
    record: Dict[str, Any],
    index: int,
    require_targets: bool,
    allow_missing_traffic: bool = False,
) -> None:
    """Validate one sample for the current runtime scenario."""
    if not isinstance(record, dict):
        _raise_schema_error(index, "<record>", "a dictionary", record)

    url = record.get("url")
    if not isinstance(url, str) or not url.strip():
        _raise_schema_error(index, "url", "a non-empty string", url)

    traffic_raw = record.get("traffic")
    if traffic_raw is None and allow_missing_traffic:
        traffic_raw = []
    _validate_traffic_payload(traffic_raw, index, required=not allow_missing_traffic)

    if require_targets:
        for field_name in ("label", "phish_type", "risk_score"):
            if field_name not in record:
                _raise_schema_error(index, field_name, "a required field", None)

        if not isinstance(record["label"], int):
            _raise_schema_error(index, "label", "an int", record["label"])
        if not isinstance(record["phish_type"], int):
            _raise_schema_error(index, "phish_type", "an int", record["phish_type"])
        if not isinstance(record["risk_score"], (int, float)):
            _raise_schema_error(index, "risk_score", "an int or float", record["risk_score"])


def load_records(path: str) -> List[Dict[str, Any]]:
    """Load samples from pickle, JSON arrays, or JSONL files."""
    extension = os.path.splitext(path)[1].lower()

    if extension in {".pkl", ".pickle"}:
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        if isinstance(payload, tuple):
            payload = list(payload)
        if not isinstance(payload, list):
            raise ValueError(f"Expected a pickled list in {path}, got {type(payload).__name__}.")
        return payload

    with open(path, "r", encoding="utf-8") as handle:
        content = handle.read().strip()
    if not content:
        return []
    if content[0] == "[":
        payload = json.loads(content)
        if not isinstance(payload, list):
            raise ValueError(f"Expected a JSON list in {path}")
        return payload
    return [json.loads(line) for line in content.splitlines() if line.strip()]


def _get_vocab_meta_path(path: str) -> str:
    """Build the side-car metadata path for the vocabulary file."""
    root, ext = os.path.splitext(path)
    return f"{root}_meta{ext or '.json'}"


def _sanitize_vocab_payload(vocabs: Dict[str, Any]) -> Dict[str, Any]:
    """Drop heavy side-car records from the main saved payload."""
    return {key: value for key, value in vocabs.items() if key != "__meta_records__"}


def save_url_vocabs(vocabs: Dict[str, Any], path: str) -> None:
    """Save the vocabulary bundle to disk."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = _sanitize_vocab_payload(vocabs)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    metadata = payload.get("__metadata__", {})
    meta_records = vocabs.get("__meta_records__", [])
    if metadata.get("vocab_kind") == POSITION_VOCAB_KIND and metadata.get("save_position_vocab_meta", False):
        meta_path = metadata.get("meta_path") or _get_vocab_meta_path(path)
        with open(meta_path, "w", encoding="utf-8") as handle:
            json.dump(meta_records, handle, ensure_ascii=False, indent=2)


def load_url_vocabs(path: str) -> Dict[str, Any]:
    """Load the vocabulary bundle from disk."""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_url(url: str, lowercase_url: bool = False) -> str:
    """Trim spaces and optionally lowercase the URL."""
    normalized = (url or "").strip()
    return normalized.lower() if lowercase_url else normalized


def _normalize_ngram_values(ngram_range: Tuple[int, int] | Sequence[int]) -> List[int]:
    """Expand inclusive n-gram range bounds into concrete n values."""
    if len(ngram_range) != 2:
        raise ValueError(f"ngram_range must contain exactly two values, got {ngram_range}.")
    start, end = int(ngram_range[0]), int(ngram_range[1])
    if start <= 0 or end < start:
        raise ValueError(f"Invalid ngram_range={ngram_range}.")
    return list(range(start, end + 1))


def extract_ngrams(url: str, n: int, lowercase_url: bool = False) -> List[str]:
    """Extract plain character n-grams from a URL."""
    text = normalize_url(url, lowercase_url=lowercase_url)
    if not text:
        return []
    if len(text) < n:
        return [text]
    return [text[idx : idx + n] for idx in range(len(text) - n + 1)]


def _expand_merged_label(label: str) -> List[str]:
    """Expand a merged label such as A+B into ordered atomic labels."""
    return [part for part in label.split("+") if part]


def get_position_class(labels: List[str]) -> str:
    """Merge ordered character labels into one merged component class."""
    ordered: List[str] = []
    for label in labels:
        for part in _expand_merged_label(label):
            if part not in ordered:
                ordered.append(part)
    return "+".join(ordered) if ordered else "PATH"


def _assign_range(labels: List[str | None], start: int, end: int, label: str) -> None:
    """Assign one label to a half-open character span."""
    for index in range(max(start, 0), min(end, len(labels))):
        labels[index] = label


def _is_ip_host(host: str) -> bool:
    """Check whether the host is a valid IP address."""
    try:
        ipaddress.ip_address(host.strip("[]"))
        return True
    except ValueError:
        return False


def _split_host_component_types(host: str) -> List[Tuple[str, str]]:
    """Split a host into SUBDOMAIN/DOMAIN/TLD components."""
    if not host:
        return []
    if _is_ip_host(host):
        # IP hosts are mapped to DOMAIN so we keep the existing component set
        # instead of introducing a new HOST branch into the baseline pipeline.
        return [(host, "DOMAIN")]

    parts = host.split(".")
    if len(parts) == 1:
        return [(parts[0], "DOMAIN")]
    if len(parts) == 2:
        return [(parts[0], "DOMAIN"), (parts[1], "TLD")]

    segments: List[Tuple[str, str]] = []
    for index, part in enumerate(parts):
        if index < len(parts) - 2:
            segments.append((part, "SUBDOMAIN"))
        elif index == len(parts) - 2:
            segments.append((part, "DOMAIN"))
        else:
            segments.append((part, "TLD"))
    return segments


def parse_url_to_char_labels(url: str, granularity: str = "fine") -> List[Tuple[str, str]]:
    """Align every original URL character with one position label."""
    if not url:
        return []
    if granularity != "fine":
        logger.warning("Unsupported position_granularity=%s. Falling back to 'fine'.", granularity)

    labels: List[str | None] = [None] * len(url)
    scheme_sep_index = url.find("://")
    remainder_start = 0
    current_prev_component: str | None = None
    first_host_component: str | None = None
    last_host_component: str | None = None

    if scheme_sep_index > 0:
        _assign_range(labels, 0, scheme_sep_index, "SCHEME")
        remainder_start = scheme_sep_index + 3
        current_prev_component = "SCHEME"

    fragment_index = url.find("#", remainder_start)
    fragment_boundary = fragment_index if fragment_index != -1 else len(url)
    query_index = url.find("?", remainder_start, fragment_boundary)
    query_boundary = query_index if query_index != -1 else fragment_boundary
    path_index = url.find("/", remainder_start, query_boundary)
    authority_end = min(index for index in (path_index, query_index, fragment_index, len(url)) if index != -1)

    authority = url[remainder_start:authority_end]
    host = authority
    port = ""
    port_separator_index: int | None = None

    if authority:
        if authority.startswith("[") and "]" in authority:
            closing = authority.rfind("]")
            if closing + 1 < len(authority) and authority[closing + 1] == ":":
                candidate_port = authority[closing + 2 :]
                if candidate_port.isdigit():
                    host = authority[: closing + 1]
                    port = candidate_port
                    port_separator_index = remainder_start + closing + 1
        elif authority.count(":") == 1:
            candidate_host, candidate_port = authority.rsplit(":", 1)
            if candidate_port.isdigit():
                host = candidate_host
                port = candidate_port
                port_separator_index = remainder_start + len(candidate_host)

        host_segments = _split_host_component_types(host)
        cursor = remainder_start
        for index, (segment, component) in enumerate(host_segments):
            segment_start = cursor
            segment_end = segment_start + len(segment)
            _assign_range(labels, segment_start, segment_end, component)
            if first_host_component is None:
                first_host_component = component
            last_host_component = component
            cursor = segment_end
            if index < len(host_segments) - 1 and cursor < remainder_start + len(host):
                next_component = host_segments[index + 1][1]
                labels[cursor] = get_position_class([component, next_component])
                cursor += 1

        current_prev_component = last_host_component

        if port and port_separator_index is not None:
            labels[port_separator_index] = get_position_class([last_host_component or "DOMAIN", "PORT"])
            _assign_range(labels, port_separator_index + 1, port_separator_index + 1 + len(port), "PORT")
            current_prev_component = "PORT"

    if scheme_sep_index > 0:
        separator_label = get_position_class(["SCHEME", first_host_component or current_prev_component or "PATH"])
        _assign_range(labels, scheme_sep_index, remainder_start, separator_label)

    if path_index != -1 and path_index < query_boundary:
        path_end = query_boundary
        labels[path_index] = get_position_class([current_prev_component or "PATH", "PATH"])
        _assign_range(labels, path_index + 1, path_end, "PATH")
        current_prev_component = "PATH"

    if query_index != -1 and query_index < fragment_boundary:
        labels[query_index] = get_position_class([current_prev_component or "PATH", "QUERY_KEY"])
        cursor = query_index + 1
        trailing_component = "QUERY_KEY"
        while cursor < fragment_boundary:
            while cursor < fragment_boundary and url[cursor] not in "=&":
                labels[cursor] = "QUERY_KEY"
                cursor += 1

            if cursor < fragment_boundary and url[cursor] == "=":
                labels[cursor] = get_position_class(["QUERY_KEY", "QUERY_VALUE"])
                cursor += 1
                trailing_component = "QUERY_VALUE"
                while cursor < fragment_boundary and url[cursor] != "&":
                    labels[cursor] = "QUERY_VALUE"
                    cursor += 1
            else:
                trailing_component = "QUERY_KEY"

            if cursor < fragment_boundary and url[cursor] == "&":
                labels[cursor] = get_position_class([trailing_component, "QUERY_KEY"])
                cursor += 1
        current_prev_component = trailing_component

    if fragment_index != -1:
        labels[fragment_index] = get_position_class([current_prev_component or "PATH", "FRAGMENT"])
        _assign_range(labels, fragment_index + 1, len(url), "FRAGMENT")

    unresolved = False
    for index, label in enumerate(labels):
        if label is None:
            labels[index] = current_prev_component or "PATH"
            unresolved = True

    if unresolved:
        logger.warning("Fell back to default labels for unresolved URL characters: %s", url)

    return list(zip(url, [label for label in labels if label is not None]))


def generate_position_ngrams(
    url: str,
    ngram_range: Tuple[int, int] = (1, 3),
    granularity: str = "fine",
    include_boundary_tokens: bool = True,
) -> List[str]:
    """Generate a flat list of position-aware n-gram tokens."""
    labeled_chars = parse_url_to_char_labels(url, granularity=granularity)
    n_values = _normalize_ngram_values(ngram_range)
    tokens: List[str] = []

    if not labeled_chars:
        return tokens

    characters = [character for character, _ in labeled_chars]
    labels = [label for _, label in labeled_chars]

    for n in n_values:
        if len(characters) < n:
            position_class = get_position_class(labels)
            if include_boundary_tokens or "+" not in position_class:
                tokens.append(f"{position_class}::{''.join(characters)}")
            continue

        for start in range(len(characters) - n + 1):
            position_class = get_position_class(labels[start : start + n])
            if not include_boundary_tokens and "+" in position_class:
                continue
            tokens.append(f"{position_class}::{''.join(characters[start:start + n])}")
    return tokens


def _generate_position_ngram_sequences(
    url: str,
    ngram_range: Tuple[int, int],
    granularity: str,
    include_boundary_tokens: bool,
) -> Dict[str, List[str]]:
    """Generate grouped position-aware n-gram sequences keyed by n."""
    labeled_chars = parse_url_to_char_labels(url, granularity=granularity)
    characters = [character for character, _ in labeled_chars]
    labels = [label for _, label in labeled_chars]
    grouped = {f"{n}gram": [] for n in (1, 2, 3)}

    if not characters:
        return grouped

    for n in _normalize_ngram_values(ngram_range):
        key = f"{n}gram"
        if len(characters) < n:
            position_class = get_position_class(labels)
            if include_boundary_tokens or "+" not in position_class:
                grouped[key].append(f"{position_class}::{''.join(characters)}")
            continue

        for start in range(len(characters) - n + 1):
            position_class = get_position_class(labels[start : start + n])
            if not include_boundary_tokens and "+" in position_class:
                continue
            grouped[key].append(f"{position_class}::{''.join(characters[start:start + n])}")
    return grouped


def build_ngram_vocab(
    urls: Iterable[str],
    n: int,
    max_size: int,
    min_freq: int = 1,
    lowercase_url: bool = False,
) -> Dict[str, int]:
    """Build the original plain n-gram vocabulary for one n value."""
    counter: Counter[str] = Counter()
    for url in urls:
        counter.update(extract_ngrams(url, n, lowercase_url=lowercase_url))

    vocab = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID}
    for token, freq in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
        if freq < min_freq:
            continue
        if len(vocab) >= max_size:
            break
        vocab[token] = len(vocab)
    return vocab


def _build_plain_url_vocabs(urls: Iterable[str], config) -> Dict[str, Any]:
    """Build the original plain URL vocabularies."""
    url_list = list(urls)
    return {
        "1gram": build_ngram_vocab(
            url_list,
            1,
            config.vocab_1gram_max_size,
            config.ngram_min_freq,
            lowercase_url=config.lowercase_url,
        ),
        "2gram": build_ngram_vocab(
            url_list,
            2,
            config.vocab_2gram_max_size,
            config.ngram_min_freq,
            lowercase_url=config.lowercase_url,
        ),
        "3gram": build_ngram_vocab(
            url_list,
            3,
            config.vocab_3gram_max_size,
            config.ngram_min_freq,
            lowercase_url=config.lowercase_url,
        ),
        "__metadata__": {
            "vocab_kind": PLAIN_VOCAB_KIND,
            "lowercase_url": config.lowercase_url,
        },
    }


def build_position_ngram_vocab(urls: Iterable[str], config) -> Dict[str, Any]:
    """Build the grouped position-aware vocabulary bundle."""
    url_list = list(urls)
    grouped_counts = {f"{n}gram": Counter() for n in (1, 2, 3)}
    grouped_df = {f"{n}gram": Counter() for n in (1, 2, 3)}
    position_class_frequency: Counter[str] = Counter()
    ngram_length_frequency: Counter[int] = Counter()
    examples: List[Dict[str, Any]] = []

    for index, url in enumerate(url_list):
        grouped_tokens = _generate_position_ngram_sequences(
            url,
            ngram_range=config.ngram_range,
            granularity=config.position_granularity,
            include_boundary_tokens=config.include_boundary_tokens,
        )
        if index < 3:
            examples.append(
                {
                    "url": url,
                    "tokens": {key: value[:20] for key, value in grouped_tokens.items()},
                }
            )

        for key, tokens in grouped_tokens.items():
            grouped_counts[key].update(tokens)
            grouped_df[key].update(set(tokens))
            for token in tokens:
                position, ngram = token.split("::", 1)
                position_class_frequency[position] += 1
                ngram_length_frequency[len(ngram)] += 1

    max_sizes = {
        "1gram": config.vocab_1gram_max_size,
        "2gram": config.vocab_2gram_max_size,
        "3gram": config.vocab_3gram_max_size,
    }

    vocabs: Dict[str, Any] = {}
    meta_records: List[Dict[str, Any]] = []
    for key in ("1gram", "2gram", "3gram"):
        vocab = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID}
        for token, count in sorted(grouped_counts[key].items(), key=lambda item: (-item[1], item[0])):
            if count < config.ngram_min_freq:
                continue
            if len(vocab) >= max_sizes[key]:
                break
            vocab[token] = len(vocab)
            position, ngram = token.split("::", 1)
            meta_records.append(
                {
                    "token": token,
                    "token_id": vocab[token],
                    "position": position,
                    "ngram": ngram,
                    "ngram_len": len(ngram),
                    "count": int(count),
                    "df": int(grouped_df[key][token]),
                    "bucket": key,
                }
            )
        vocabs[key] = vocab

    vocabs["__metadata__"] = {
        "vocab_kind": POSITION_VOCAB_KIND,
        "use_position_ngram_vocab": True,
        "ngram_range": list(config.ngram_range),
        "position_granularity": config.position_granularity,
        "include_boundary_tokens": config.include_boundary_tokens,
        "save_position_vocab_meta": config.save_position_vocab_meta,
        "lowercase_url": config.lowercase_url,
        "position_class_frequency": dict(position_class_frequency),
        "ngram_length_frequency": dict(ngram_length_frequency),
        "sample_examples": examples,
        "meta_path": _get_vocab_meta_path(config.vocab_path),
    }
    if config.save_position_vocab_meta:
        vocabs["__meta_records__"] = meta_records
    return vocabs


def build_url_vocabs(urls: Iterable[str], config) -> Dict[str, Any]:
    """Build either the original plain or the position-aware URL vocabularies."""
    if getattr(config, "use_position_ngram_vocab", False):
        return build_position_ngram_vocab(urls, config)
    return _build_plain_url_vocabs(urls, config)


def validate_vocab_compatibility(vocabs: Dict[str, Any], config) -> None:
    """Validate that the loaded vocabulary matches the expected structure."""
    metadata = vocabs.get("__metadata__", {})
    vocab_kind = metadata.get("vocab_kind", PLAIN_VOCAB_KIND)
    expected_position_vocab = bool(getattr(config, "use_position_ngram_vocab", False))

    if expected_position_vocab and vocab_kind != POSITION_VOCAB_KIND:
        raise ValueError(
            "The vocabulary structure has changed. This checkpoint is incompatible and the model must be retrained."
        )
    if not expected_position_vocab and vocab_kind == POSITION_VOCAB_KIND:
        raise ValueError(
            "The loaded vocabulary is position-aware but the current configuration expects the plain baseline vocabulary."
        )


def load_vocab_for_runtime(config, checkpoint_vocabs: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Resolve the vocabulary bundle used by evaluation or prediction."""
    if getattr(config, "use_position_ngram_vocab", False):
        if not os.path.exists(config.vocab_path):
            raise FileNotFoundError(
                f"Position-aware vocabulary file is missing: {config.vocab_path}. "
                "Validation/test must load the saved training vocabulary."
            )
        vocabs = load_url_vocabs(config.vocab_path)
    elif checkpoint_vocabs is not None:
        vocabs = checkpoint_vocabs
    elif os.path.exists(config.vocab_path):
        vocabs = load_url_vocabs(config.vocab_path)
    else:
        raise FileNotFoundError(f"Vocabulary file not found: {config.vocab_path}")

    validate_vocab_compatibility(vocabs, config)
    return vocabs


def encode_url_with_position_vocab(
    url: str,
    vocabs: Dict[str, Dict[str, int]],
    max_len: int,
    ngram_range: Tuple[int, int] = (1, 3),
    granularity: str = "fine",
    include_boundary_tokens: bool = True,
) -> Dict[str, List[int]]:
    """Encode one raw URL with the position-aware vocabulary bundle."""
    grouped_tokens = _generate_position_ngram_sequences(
        url,
        ngram_range=ngram_range,
        granularity=granularity,
        include_boundary_tokens=include_boundary_tokens,
    )
    encoded: Dict[str, List[int]] = {}
    for key, tokens in grouped_tokens.items():
        ids = [vocabs[key].get(token, UNK_ID) for token in tokens][:max_len]
        encoded[key] = ids or [PAD_ID]
    return encoded


def encode_url_to_ngrams(
    url: str,
    vocabs: Dict[str, Any],
    max_url_len: int,
    lowercase_url: bool = False,
    use_position_ngram_vocab: bool = True,
    ngram_range: Tuple[int, int] = (1, 3),
    granularity: str = "fine",
    include_boundary_tokens: bool = True,
) -> Dict[str, List[int]]:
    """Encode one URL into model-compatible grouped token IDs."""
    if use_position_ngram_vocab:
        return encode_url_with_position_vocab(
            url,
            vocabs=vocabs,
            max_len=max_url_len,
            ngram_range=ngram_range,
            granularity=granularity,
            include_boundary_tokens=include_boundary_tokens,
        )

    normalized = normalize_url(url, lowercase_url=lowercase_url)[:max_url_len]
    encoded: Dict[str, List[int]] = {}
    for n in (1, 2, 3):
        key = f"{n}gram"
        grams = extract_ngrams(normalized, n, lowercase_url=False)
        ids = [vocabs[key].get(token, UNK_ID) for token in grams]
        encoded[key] = ids or [PAD_ID]
    return encoded


def parse_traffic_sequence(traffic_raw: Sequence[Any], max_traffic_len: int) -> torch.Tensor:
    """Convert the raw traffic payload into a [T, 2] tensor."""
    if not traffic_raw:
        return torch.zeros(1, 2, dtype=torch.float32)

    rows: List[List[float]] = []
    previous_timestamp = 0.0
    for item in traffic_raw[:max_traffic_len]:
        if isinstance(item, (list, tuple)):
            if len(item) >= 2:
                timestamp = float(item[0])
                size = float(item[1])
            elif len(item) == 1:
                timestamp = float(item[0])
                size = 0.0
            else:
                continue
        else:
            timestamp = float(item)
            size = 0.0

        if timestamp >= previous_timestamp:
            delta_time = timestamp - previous_timestamp
            previous_timestamp = timestamp
        else:
            delta_time = abs(timestamp)
            previous_timestamp += delta_time
        rows.append([delta_time, size])

    return torch.tensor(rows or [[0.0, 0.0]], dtype=torch.float32)


def debug_position_aware_tokenization(
    url: str,
    ngram_range: Tuple[int, int] = (1, 3),
    granularity: str = "fine",
    include_boundary_tokens: bool = True,
) -> Dict[str, List[str]]:
    """Expose grouped position-aware tokenization for debugging and tests."""
    return _generate_position_ngram_sequences(
        url,
        ngram_range=ngram_range,
        granularity=granularity,
        include_boundary_tokens=include_boundary_tokens,
    )


class PhishingDataset(Dataset):
    """Unified dataset for training, evaluation, and prediction."""

    def __init__(
        self,
        data: str | Sequence[Dict[str, Any]],
        vocabs: Dict[str, Any],
        max_url_len: int = 256,
        max_traffic_len: int = 512,
        lowercase_url: bool = False,
        require_targets: bool = True,
        allow_missing_traffic: bool = False,
        use_position_ngram_vocab: bool = True,
        ngram_range: Tuple[int, int] = (1, 3),
        position_granularity: str = "fine",
        include_boundary_tokens: bool = True,
    ) -> None:
        super().__init__()
        self.records = load_records(data) if isinstance(data, str) else list(data)
        self.vocabs = vocabs
        self.max_url_len = max_url_len
        self.max_traffic_len = max_traffic_len
        self.lowercase_url = lowercase_url
        self.require_targets = require_targets
        self.allow_missing_traffic = allow_missing_traffic
        self.use_position_ngram_vocab = use_position_ngram_vocab
        self.ngram_range = ngram_range
        self.position_granularity = position_granularity
        self.include_boundary_tokens = include_boundary_tokens

        for index, record in enumerate(self.records):
            validate_record_schema(
                record,
                index=index,
                require_targets=self.require_targets,
                allow_missing_traffic=self.allow_missing_traffic,
            )

    def __len__(self) -> int:
        return len(self.records)

    def get_class_weights(self) -> torch.Tensor:
        positive = sum(int(item.get("label", 0)) for item in self.records)
        negative = len(self.records) - positive
        total = max(len(self.records), 1)
        return torch.tensor(
            [
                total / max(negative, 1),
                total / max(positive, 1),
            ],
            dtype=torch.float32,
        )

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self.records[index]
        encoded = encode_url_to_ngrams(
            item["url"],
            self.vocabs,
            self.max_url_len,
            lowercase_url=self.lowercase_url,
            use_position_ngram_vocab=self.use_position_ngram_vocab,
            ngram_range=self.ngram_range,
            granularity=self.position_granularity,
            include_boundary_tokens=self.include_boundary_tokens,
        )
        traffic_raw = item["traffic"] if "traffic" in item else []
        return {
            "ids_1gram": torch.tensor(encoded.get("1gram", [PAD_ID]), dtype=torch.long),
            "ids_2gram": torch.tensor(encoded.get("2gram", [PAD_ID]), dtype=torch.long),
            "ids_3gram": torch.tensor(encoded.get("3gram", [PAD_ID]), dtype=torch.long),
            "traffic": parse_traffic_sequence(traffic_raw, self.max_traffic_len),
            "label": torch.tensor(int(item["label"]) if self.require_targets else 0, dtype=torch.long),
            "phish_type": torch.tensor(int(item["phish_type"]) if self.require_targets else 0, dtype=torch.long),
            "risk_score": torch.tensor(
                float(item["risk_score"]) if self.require_targets else 0.0,
                dtype=torch.float32,
            ),
            "url": item["url"],
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pad a variable-length sample list into a batch dictionary."""
    ids_1 = pad_sequence([item["ids_1gram"] for item in batch], batch_first=True, padding_value=PAD_ID)
    ids_2 = pad_sequence([item["ids_2gram"] for item in batch], batch_first=True, padding_value=PAD_ID)
    ids_3 = pad_sequence([item["ids_3gram"] for item in batch], batch_first=True, padding_value=PAD_ID)
    traffic = pad_sequence([item["traffic"] for item in batch], batch_first=True, padding_value=0.0)
    url_mask = ids_1.ne(PAD_ID)
    traffic_mask = torch.zeros(traffic.size(0), traffic.size(1), dtype=torch.bool)
    for index, item in enumerate(batch):
        traffic_mask[index, : item["traffic"].size(0)] = True
    return {
        "ids_1gram": ids_1,
        "ids_2gram": ids_2,
        "ids_3gram": ids_3,
        "url_mask": url_mask,
        "traffic_feats": traffic,
        "traffic_mask": traffic_mask,
        "label": torch.stack([item["label"] for item in batch]),
        "phish_type": torch.stack([item["phish_type"] for item in batch]),
        "risk_score": torch.stack([item["risk_score"] for item in batch]),
        "urls": [item["url"] for item in batch],
    }


def build_dataloader(
    data: str | Sequence[Dict[str, Any]],
    config,
    vocabs: Dict[str, Any],
    shuffle: bool = False,
    require_targets: bool = True,
    allow_missing_traffic: bool = False,
) -> DataLoader:
    """Create the project DataLoader using the current configuration."""
    dataset = PhishingDataset(
        data=data,
        vocabs=vocabs,
        max_url_len=config.max_url_len,
        max_traffic_len=config.max_traffic_len,
        lowercase_url=config.lowercase_url,
        require_targets=require_targets,
        allow_missing_traffic=allow_missing_traffic,
        use_position_ngram_vocab=getattr(config, "use_position_ngram_vocab", False),
        ngram_range=tuple(getattr(config, "ngram_range", (1, 3))),
        position_granularity=getattr(config, "position_granularity", "fine"),
        include_boundary_tokens=getattr(config, "include_boundary_tokens", True),
    )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
    )
