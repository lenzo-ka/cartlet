"""Data loading functions for cartlet."""

from __future__ import annotations

import csv
import json
import logging
from collections.abc import Iterator
from typing import Any, TextIO

from .utils import (
    detect_delimiter,
    detect_format,
    format_to_delimiter,
    normalize_text,
    normalize_value,
    try_numeric,
)

_logger = logging.getLogger("cartlet")


# =============================================================================
# Target-column resolution (shared by every loader)
# =============================================================================


def _resolve_target_idx(target_col: str | int | None, header: list[str]) -> int:
    """Resolve a target reference to a positional index for tabular data.

    Precedence, shared by every tabular loader so a column literally named "2"
    behaves identically everywhere: explicit int index, then an exact
    header-name match, then a digit string treated as an index.
    """
    if target_col is None:
        return len(header) - 1
    if isinstance(target_col, int):
        idx = target_col
    elif target_col in header:
        idx = header.index(target_col)
    elif str(target_col).isdigit():
        idx = int(target_col)
    else:
        raise ValueError(f"Target column '{target_col}' not found in header: {header}")
    if idx < 0 or idx >= len(header):
        raise ValueError(f"Target index {idx} out of range (0-{len(header) - 1})")
    return idx


def _resolve_target_key(target_col: str | int | None, keys: list[str]) -> str:
    """Resolve a target reference to a record key (JSONL), mirroring
    :func:`_resolve_target_idx`'s precedence for string keys."""
    if target_col is None:
        return keys[-1]
    if isinstance(target_col, int):
        if target_col < 0 or target_col >= len(keys):
            raise ValueError(f"Target index {target_col} out of range")
        return keys[target_col]
    if target_col in keys:
        return target_col
    if str(target_col).isdigit():
        idx = int(target_col)
        if idx < 0 or idx >= len(keys):
            raise ValueError(f"Target index {idx} out of range")
        return keys[idx]
    raise ValueError(f"Target column '{target_col}' not found. Available: {keys}")


# =============================================================================
# High-level data loading
# =============================================================================


def load_training_data(
    path: str,
    delimiter: str | None = None,
    has_header: bool = True,
    target_col: str | int | None = None,
    column_names: list[str] | None = None,
) -> tuple[list[list[Any]], list[Any], list[str], str]:
    """
    Load training data from CSV/TSV/SSV/JSONL file.

    This is the high-level loader for training data. It:
    - Auto-detects format and delimiter
    - Handles column name overrides
    - Converts numeric strings to int/float
    - Separates features from target

    Args:
        path: Path to data file
        delimiter: Column delimiter for CSV/TSV (auto-detect if None)
        has_header: Whether first row is header (CSV/TSV/SSV only)
        target_col: Target column name or index (default: last column)
        column_names: Explicit column names (overrides header/auto-generated)

    Returns:
        Tuple of (X, y, feature_names, target_name)

    Raises:
        ValueError: If file is empty or target column not found
    """
    # Check for JSONL format
    file_format = detect_format(path)
    if file_format == "jsonl":
        return _load_jsonl_training_data(path, target_col)

    if delimiter is None:
        delimiter = detect_delimiter(path)

    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delimiter)
        rows = [[normalize_text(cell) for cell in row] for row in reader]

    if not rows:
        raise ValueError(f"Empty file: {path}")

    # Extract header
    if has_header:
        header = rows[0]
        data_rows = rows[1:]
    else:
        # 1-indexed like Unix cut/paste
        header = [str(i) for i in range(1, len(rows[0]) + 1)]
        data_rows = rows

    # Override header with explicit column names if provided
    if column_names:
        if len(column_names) != len(rows[0]):
            raise ValueError(
                f"Column names count ({len(column_names)}) doesn't match "
                f"data columns ({len(rows[0])})"
            )
        header = column_names

    if not data_rows:
        raise ValueError(f"No data rows in file: {path}")

    # Determine target column index
    target_idx = _resolve_target_idx(target_col, header)

    # Split into features and target
    target_name = header[target_idx]
    feature_names = [h for i, h in enumerate(header) if i != target_idx]
    X: list[list[Any]] = []
    y: list[Any] = []

    for row_num, row in enumerate(data_rows, start=2 if has_header else 1):
        if len(row) != len(header):
            _logger.warning(
                "Skipping malformed row %d: expected %d columns, got %d",
                row_num,
                len(header),
                len(row),
            )
            continue
        features = [row[i] for i in range(len(row)) if i != target_idx]
        target = row[target_idx]

        # Convert numeric values
        converted_features = [try_numeric(v) for v in features]
        target_val = try_numeric(target)

        X.append(converted_features)
        y.append(target_val)

    return X, y, feature_names, target_name


def _load_jsonl_training_data(
    path: str,
    target_col: str | int | None = None,
) -> tuple[list[list[Any]], list[Any], list[str], str]:
    """Load training data from JSONL file."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = normalize_text(line.strip())
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {line_num}: {e}") from e

    if not records:
        raise ValueError(f"Empty file: {path}")

    # Get all keys from first record
    all_keys = list(records[0].keys())

    # Determine target column
    target_key = _resolve_target_key(target_col, all_keys)

    feature_names = [k for k in all_keys if k != target_key]

    X = []
    y = []
    for record in records:
        features = [try_numeric(record.get(k)) for k in feature_names]
        target = try_numeric(record.get(target_key))
        X.append(features)
        y.append(target)

    return X, y, feature_names, target_key


# =============================================================================
# Read Vectors (low-level)
# =============================================================================


def read_vectors(
    source: str | TextIO,
    format: str | None = None,
    delimiter: str | None = None,
    has_header: bool = True,
    target_col: str | int | None = None,
    labeled: bool = True,
) -> tuple[list[list[Any]], list[Any] | None, list[str], str | None]:
    """
    Read vectors from any format.

    Args:
        source: File path or file object
        format: Format (csv/tsv/ssv/jsonl) - auto-detect if None
        delimiter: Override delimiter
        has_header: Whether first row is header (CSV/TSV/SSV)
        target_col: Target column name or index (None = last)
        labeled: Whether data has labels (False for prediction input)

    Returns:
        (X, y, feature_names, target_name)
        y and target_name are None if labeled=False
    """
    # Handle file path vs file object
    if isinstance(source, str):
        format = format or detect_format(source)
        with open(source, encoding="utf-8") as f:
            return _read_vectors(f, format, delimiter, has_header, target_col, labeled)

    format = format or "csv"
    return _read_vectors(source, format, delimiter, has_header, target_col, labeled)


def _read_vectors(
    f: TextIO,
    format: str,
    delimiter: str | None,
    has_header: bool,
    target_col: str | int | None,
    labeled: bool,
) -> tuple[list[list[Any]], list[Any] | None, list[str], str | None]:
    """Internal reader implementation."""
    if format == "jsonl":
        return _read_jsonl(f, target_col, labeled)

    delimiter = delimiter or format_to_delimiter(format)
    reader = csv.reader(f, delimiter=delimiter)
    # Normalize all text values to NFC
    rows = [
        [normalize_text(cell) if isinstance(cell, str) else cell for cell in row]
        for row in reader
    ]

    if not rows:
        raise ValueError("Empty input")

    # Header
    if has_header:
        header, data = rows[0], rows[1:]
    else:
        header = [str(i) for i in range(1, len(rows[0]) + 1)]  # 1-indexed
        data = rows

    if not data:
        raise ValueError("No data rows")

    if not labeled:
        X = [row for row in data if len(row) == len(header)]
        return X, None, header, None

    target_idx = _resolve_target_idx(target_col, header)

    target_name = header[target_idx]
    feature_names = [h for i, h in enumerate(header) if i != target_idx]
    X = [
        [row[i] for i in range(len(row)) if i != target_idx]
        for row in data
        if len(row) == len(header)
    ]
    y = [row[target_idx] for row in data if len(row) == len(header)]
    return X, y, feature_names, target_name


def _read_jsonl(
    f: TextIO,
    target_col: str | int | None,
    labeled: bool,
) -> tuple[list[list[Any]], list[Any] | None, list[str], str | None]:
    """Read JSONL format."""
    records = [json.loads(line) for line in f if line.strip()]
    if not records:
        raise ValueError("Empty input")

    keys = list(records[0].keys())

    if not labeled:
        X = [[normalize_value(r.get(k)) for k in keys] for r in records]
        return X, None, keys, None

    target_key = _resolve_target_key(target_col, keys)
    feature_names = [k for k in keys if k != target_key]
    X = [[normalize_value(r.get(k)) for k in feature_names] for r in records]
    y = [normalize_value(r.get(target_key)) for r in records]
    return X, y, feature_names, target_key


# =============================================================================
# Streaming Interface (for large files)
# =============================================================================


def iter_vectors(
    source: str | TextIO,
    format: str | None = None,
    delimiter: str | None = None,
    has_header: bool = True,
    labeled: bool = True,
) -> Iterator[tuple[list[Any], Any | None]]:
    """
    Stream vectors one at a time (memory-efficient for large files).

    Yields:
        (features, label) tuples - label is None if labeled=False
    """
    if isinstance(source, str):
        format = format or detect_format(source)
        with open(source, encoding="utf-8") as f:
            yield from _iter_vectors(f, format, delimiter, has_header, labeled)
    else:
        format = format or "csv"
        yield from _iter_vectors(source, format, delimiter, has_header, labeled)


def _iter_vectors(
    f: TextIO,
    format: str,
    delimiter: str | None,
    has_header: bool,
    labeled: bool,
) -> Iterator[tuple[list[Any], Any | None]]:
    """Internal streaming implementation."""
    if format == "jsonl":
        first = True
        target_key = None
        feature_keys: list[str] = []
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                _logger.warning("Skipping malformed JSONL line %d: %s", line_num, e)
                continue

            if first:
                keys = list(record.keys())
                target_key = keys[-1] if labeled else None
                feature_keys = [k for k in keys if k != target_key] if labeled else keys
                first = False
            # Normalize values to match the batch reader (_read_jsonl).
            features = [normalize_value(record.get(k)) for k in feature_keys]
            label = normalize_value(record.get(target_key)) if labeled else None
            yield features, label
    else:
        delimiter = delimiter or format_to_delimiter(format)
        reader = csv.reader(f, delimiter=delimiter)

        # Track the expected row width so malformed rows are skipped exactly as
        # the batch reader (_read_vectors) does, rather than yielding ragged
        # vectors that silently disagree with the non-streaming path.
        expected_width: int | None = None
        if has_header:
            try:
                header = next(reader)
            except StopIteration:
                return
            expected_width = len(header)

        for row in reader:
            if not row:
                continue
            row = [normalize_text(c) if isinstance(c, str) else c for c in row]
            if expected_width is None:
                expected_width = len(row)
            if len(row) != expected_width:
                _logger.warning(
                    "Skipping malformed row: expected %d columns, got %d",
                    expected_width,
                    len(row),
                )
                continue
            if labeled:
                features = row[:-1]
                label = row[-1]
            else:
                features = row
                label = None
            yield features, label
