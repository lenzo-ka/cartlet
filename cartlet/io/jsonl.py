"""The shared JSONL vector schema: nonempty object records, strict line errors."""

import json
from collections.abc import Iterator
from typing import Any, TextIO


def iter_jsonl_records(source: TextIO) -> Iterator[tuple[int, dict[str, Any]]]:
    """Read object records; skip blank lines, reject invalid shapes and JSON."""
    for line_number, line in enumerate(source, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON at line {line_number}: {e.msg}") from e
        if not isinstance(record, dict) or not record:
            raise ValueError(f"JSONL line {line_number} must be a nonempty object")
        yield line_number, record


def require_target(record: dict[str, Any], target: str, line_number: int) -> Any:
    """Training labels must be present and non-null; missing features are allowed."""
    if target not in record or record[target] is None:
        raise ValueError(f"Missing target {target!r} at JSONL line {line_number}")
    return record[target]
