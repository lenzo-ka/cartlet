"""Data writing functions for cartlet."""

from __future__ import annotations

import csv
import json
from typing import Any, TextIO

from .utils import format_to_delimiter


def _output_format_from_ext(path: str) -> str:
    """Map an output path to a write format by extension only.

    Output paths often do not exist yet, so we must never content-sniff them
    (as ``detect_format`` does for inputs). Unknown extensions default to CSV.
    """
    for ext, fmt in (
        (".jsonl", "jsonl"),
        (".json", "json"),
        (".tsv", "tsv"),
        (".ssv", "ssv"),
        (".csv", "csv"),
    ):
        if path.endswith(ext):
            return fmt
    return "csv"


def write_vectors(
    dest: str | TextIO,
    data: list[list[Any]] | list[Any],
    header: list[str] | None = None,
    format: str | None = None,
    delimiter: str | None = None,
) -> None:
    """
    Write vectors or results to any format.

    Args:
        dest: File path or file object
        data: List of rows or list of values
        header: Column names (optional)
        format: Output format (csv/tsv/ssv/jsonl/json)
        delimiter: Override delimiter
    """
    if isinstance(dest, str):
        fmt = format or _output_format_from_ext(dest)
        with open(dest, "w", encoding="utf-8") as f:
            _write_vectors(f, data, header, fmt, delimiter)
    else:
        _write_vectors(dest, data, header, format or "csv", delimiter)


def _write_vectors(
    f: TextIO,
    data: list[list[Any]] | list[Any],
    header: list[str] | None,
    format: str,
    delimiter: str | None,
) -> None:
    """Internal writer implementation."""
    # Handle single values (predictions)
    if data and not isinstance(data[0], list):
        data = [[v] for v in data]

    if format == "json":
        if header:
            json.dump([dict(zip(header, row)) for row in data], f)
        else:
            json.dump(data, f)
        return

    if format == "jsonl":
        for row in data:
            if header:
                f.write(json.dumps(dict(zip(header, row))) + "\n")
            else:
                f.write(json.dumps(row) + "\n")
        return

    # CSV/TSV/SSV
    delimiter = delimiter or format_to_delimiter(format)
    writer = csv.writer(f, delimiter=delimiter)
    if header:
        writer.writerow(header)
    writer.writerows(data)
