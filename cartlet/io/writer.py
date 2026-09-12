"""Data writing functions for cartlet."""

from __future__ import annotations

import csv
import json
from typing import Any, TextIO

from .utils import atomic_output_path, format_to_delimiter


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
        with (
            atomic_output_path(dest) as output,
            open(output, "w", encoding="utf-8") as f,
        ):
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

    if data:
        width = len(data[0])
        if header is not None and len(header) != width:
            raise ValueError("Header columns do not match row columns")
        if any(len(row) != width for row in data):
            raise ValueError("All rows must have the same number of columns")
    if format == "jsonl" and header is None and data:
        header = [str(i) for i in range(1, len(data[0]) + 1)]
    if format == "jsonl" and data and not header:
        raise ValueError("JSONL rows must have at least one column")
    if format == "jsonl" and header is not None and len(set(header)) != len(header):
        raise ValueError("JSONL header columns must be unique")

    if format == "json":
        if header:
            json.dump([dict(zip(header, row, strict=False)) for row in data], f)
        else:
            json.dump(data, f)
        return

    if format == "jsonl":
        for row in data:
            if header:
                f.write(json.dumps(dict(zip(header, row, strict=False))) + "\n")
            else:
                f.write(json.dumps(row) + "\n")
        return

    # CSV/TSV/SSV
    delimiter = delimiter or format_to_delimiter(format)
    writer = csv.writer(f, delimiter=delimiter)
    if header:
        writer.writerow(header)
    writer.writerows(data)
