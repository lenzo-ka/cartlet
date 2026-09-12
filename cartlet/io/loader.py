"""Data loading functions for cartlet."""

from __future__ import annotations

import csv
import logging
from collections.abc import Iterator
from itertools import chain
from typing import Any, TextIO

from .jsonl import iter_jsonl_records, require_target
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


def _tabular_data(
    source: TextIO,
    delimiter: str,
    has_header: bool,
    column_names: list[str] | None = None,
    *,
    require_data: bool = True,
) -> tuple[list[str], Iterator[list[str]]]:
    """Read the schema and iterate valid records without retaining raw data.

    Empty CSV records are ignored, including before the header. Record numbers
    count CSV records (a quoted multiline field still belongs to one record).
    """
    records = (
        (number, row)
        for number, row in enumerate(csv.reader(source, delimiter=delimiter), 1)
        if row
    )
    first = next(records, None)
    if first is None:
        if require_data:
            raise ValueError("Empty input")
        return [], iter(())
    _, first_row = first
    header = (
        [normalize_text(cell) for cell in first_row]
        if has_header
        else [str(i) for i in range(1, len(first_row) + 1)]
    )
    if column_names:
        if len(column_names) != len(header):
            raise ValueError(
                f"Column names count ({len(column_names)}) doesn't match "
                f"data columns ({len(header)})"
            )
        header = column_names
    if has_header:
        first = next(records, None)
        if first is None:
            if require_data:
                raise ValueError("No data rows")
            return header, iter(())
    return header, _valid_tabular_rows(chain([first], records), len(header))


def _valid_tabular_rows(
    records: Iterator[tuple[int, list[str]]], width: int
) -> Iterator[list[str]]:
    """Apply the shared ragged-record and Unicode policy one row at a time."""
    for number, row in records:
        if len(row) != width:
            _logger.warning(
                "Skipping malformed row %d: expected %d columns, got %d",
                number,
                width,
                len(row),
            )
            continue
        yield [normalize_text(cell) for cell in row]


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

    with open(path, encoding="utf-8") as source:
        header, rows = _tabular_data(source, delimiter, has_header, column_names)
        target_idx = _resolve_target_idx(target_col, header)
        target_name = header[target_idx]
        feature_names = [h for i, h in enumerate(header) if i != target_idx]
        X: list[list[Any]] = []
        y: list[Any] = []
        for row in rows:
            X.append(
                [try_numeric(value) for i, value in enumerate(row) if i != target_idx]
            )
            y.append(try_numeric(row[target_idx]))
    return X, y, feature_names, target_name


def _load_jsonl_training_data(
    path: str,
    target_col: str | int | None = None,
) -> tuple[list[list[Any]], list[Any], list[str], str]:
    """Load training data from JSONL file."""
    with open(path, encoding="utf-8") as source:
        records = iter_jsonl_records(source)
        try:
            first_line, first = next(records)
        except StopIteration:
            raise ValueError(f"Empty file: {path}") from None
        all_keys = list(first)
        target_key = _resolve_target_key(target_col, all_keys)
        feature_names = [k for k in all_keys if k != target_key]
        X, y = [], []
        for line_number, record in chain([(first_line, first)], records):
            X.append(
                [try_numeric(normalize_value(record.get(k))) for k in feature_names]
            )
            y.append(
                try_numeric(
                    normalize_value(require_target(record, target_key, line_number))
                )
            )
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
    header, rows = _tabular_data(f, delimiter, has_header)
    if not labeled:
        return list(rows), None, header, None

    target_idx = _resolve_target_idx(target_col, header)
    target_name = header[target_idx]
    feature_names = [h for i, h in enumerate(header) if i != target_idx]
    X: list[list[Any]] = []
    y: list[Any] = []
    for row in rows:
        X.append([value for i, value in enumerate(row) if i != target_idx])
        y.append(row[target_idx])
    return X, y, feature_names, target_name


def _read_jsonl(
    f: TextIO,
    target_col: str | int | None,
    labeled: bool,
) -> tuple[list[list[Any]], list[Any] | None, list[str], str | None]:
    """Read JSONL format."""
    records = iter_jsonl_records(f)
    try:
        first_line, first = next(records)
    except StopIteration:
        raise ValueError("Empty input") from None
    keys = list(first)
    target_key = _resolve_target_key(target_col, keys) if labeled else None
    feature_names = [k for k in keys if k != target_key]
    X, y = [], []
    for line_number, record in chain([(first_line, first)], records):
        X.append([normalize_value(record.get(k)) for k in feature_names])
        if labeled:
            y.append(normalize_value(require_target(record, target_key, line_number)))
    return X, y if labeled else None, feature_names, target_key


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
        for line_number, record in iter_jsonl_records(f):
            if first:
                keys = list(record.keys())
                target_key = keys[-1] if labeled else None
                feature_keys = [k for k in keys if k != target_key] if labeled else keys
                first = False
            # Normalize values to match the batch reader (_read_jsonl).
            features = [normalize_value(record.get(k)) for k in feature_keys]
            label = (
                normalize_value(require_target(record, target_key, line_number))
                if labeled
                else None
            )
            yield features, label
    else:
        delimiter = delimiter or format_to_delimiter(format)
        _, rows = _tabular_data(f, delimiter, has_header, require_data=False)
        for row in rows:
            if labeled:
                yield row[:-1], row[-1]
            else:
                yield row, None
