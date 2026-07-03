"""
Data I/O adaptors for cartlet.

Vector I/O: read any format -> internal representation -> write any format.

Supported formats: CSV, TSV, SSV (space-separated), JSONL, JSON.
Model I/O lives in the model classes themselves
(``DecisionTree.load_model`` / ``.export``, etc.) and in the zero-dependency
``cartlet.runner.load_model``.
"""

from .loader import (
    load_training_data,
    read_vectors,
)
from .utils import (
    detect_delimiter,
    detect_format,
    format_to_delimiter,
    normalize_text,
    open_file,
    open_file_binary,
    resolve_format,
)
from .writer import write_vectors

__all__ = [
    "detect_delimiter",
    "detect_format",
    "format_to_delimiter",
    "load_training_data",
    "normalize_text",
    "open_file",
    "open_file_binary",
    "read_vectors",
    "resolve_format",
    "write_vectors",
]
