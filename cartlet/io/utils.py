"""Format detection and text normalization utilities."""

import csv
import gzip
import os
import tempfile
import unicodedata
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import IO, Any

_SNIFF_BUFFER_SIZE = 4096


def _parse_model_ext(path: str) -> tuple[str, bool]:
    """
    Parse model file extension, handling .gz suffix.

    Returns:
        (extension, use_gzip) — e.g. (".cart", True) for "model.cart.gz"
    """
    if path.endswith(".gz"):
        return os.path.splitext(path[:-3])[1].lower(), True
    return os.path.splitext(path)[1].lower(), False


# Short format names accepted by the ``format=`` parameter of load_model /
# export / convert. Maps to the canonical extension keys used by the dispatch
# tables, so callers using non-standard file suffixes (e.g. phonebox's `.g2p`)
# can still route through the correct codec.
_FORMAT_ALIASES = {
    "cart": ".cart",
    "json": ".json",
    "jsonl": ".jsonl",
    "pkl": ".pkl",
    "pickle": ".pkl",
    "skl": ".skl",
    "joblib": ".skl",
}


def resolve_format(path: str, format: str | None = None) -> tuple[str, bool]:
    """
    Resolve a (extension, use_gzip) pair from either an explicit ``format=``
    override or the file extension.

    Args:
        path: File path; used to detect a trailing ``.gz`` suffix and as the
            extension source when ``format`` is None.
        format: Optional explicit format name (e.g. "jsonl", "cart"); when set,
            takes precedence over the file extension. ``use_gzip`` is still
            inferred from the ``.gz`` suffix on ``path``.

    Returns:
        ``(ext, use_gzip)`` where ``ext`` is the canonical leading-dot
        extension (``.cart``, ``.json``, ...).

    Raises:
        ValueError: If ``format`` is provided but unrecognized.
    """
    if format is None:
        return _parse_model_ext(path)
    key = format.lower().lstrip(".")
    if key not in _FORMAT_ALIASES:
        raise ValueError(
            f"Unknown format {format!r}. Use one of: {sorted(_FORMAT_ALIASES)}"
        )
    return _FORMAT_ALIASES[key], path.endswith(".gz")


def require_joblib() -> Any:
    """
    Return the imported `joblib` module, or raise a uniform ImportError.

    Used by every code path that needs to load/save sklearn models so the
    install hint stays in lockstep across the package.
    """
    try:
        import joblib
    except ImportError as e:
        raise ImportError(
            "joblib is required for sklearn models. Install with: pip install joblib"
        ) from e
    return joblib


def normalize_text(text: str) -> str:
    """Normalize text to NFC form."""
    return unicodedata.normalize("NFC", text)


def detect_format(path: str) -> str:
    """Detect format from extension or content."""
    for ext, fmt in [
        (".jsonl", "jsonl"),
        (".tsv", "tsv"),
        (".csv", "csv"),
        (".ssv", "ssv"),
    ]:
        if path.endswith(ext):
            return fmt
    # Sniff content
    with open(path, encoding="utf-8") as f:
        line = f.readline().strip()
    if line.startswith("{"):
        return "jsonl"
    if "\t" in line:
        return "tsv"
    if "," in line:
        return "csv"
    return "ssv"


def format_to_delimiter(fmt: str) -> str:
    """Get delimiter for format."""
    return {"csv": ",", "tsv": "\t", "ssv": " "}.get(fmt, ",")


def detect_delimiter(path: str) -> str:
    """Detect delimiter from file extension or content sniffing."""
    fmt = detect_format(path)
    if fmt in ("csv", "tsv", "ssv"):
        return format_to_delimiter(fmt)
    # Unknown format - try sniffing
    with open(path, encoding="utf-8") as f:
        sample = f.read(_SNIFF_BUFFER_SIZE)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t|; ")
        return dialect.delimiter
    except csv.Error:
        return ","  # Default to CSV


def try_numeric(val: Any) -> Any:
    """Try to convert a value to int or float."""
    if not isinstance(val, str):
        return val
    try:
        return float(val) if "." in val else int(val)
    except ValueError:
        return val


def normalize_value(v: Any) -> Any:
    """Normalize string values to NFC."""
    return normalize_text(v) if isinstance(v, str) else v


def gzip_file(src_path: str, dest_path: str, cleanup: bool = True) -> None:
    """Gzip a file from src_path to dest_path, optionally removing the source."""
    with open(src_path, "rb") as f_in, gzip.open(dest_path, "wb") as f_out:
        f_out.write(f_in.read())
    if cleanup:
        os.unlink(src_path)


def write_with_optional_gzip(
    path: str, use_gzip: bool, write_fn: Callable[[str], None]
) -> None:
    """Write via write_fn, compressing with gzip through a temp file if requested."""
    if not use_gzip:
        write_fn(path)
        return
    with tempfile.NamedTemporaryFile(delete=False, suffix=".cart") as tmp:
        tmp_path = tmp.name
    try:
        write_fn(tmp_path)
        gzip_file(tmp_path, path, cleanup=False)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@contextmanager
def open_file(path: str, mode: str = "r") -> Iterator[IO[str]]:
    """
    Open a text file, auto-detecting gzip compression from extension.

    Args:
        path: File path (.gz extension triggers gzip mode)
        mode: "r" for read, "w" for write

    Yields:
        Text file handle (always text mode with UTF-8 encoding)

    Example:
        with open_file("model.jsonl.gz", "r") as f:
            content = f.read()

        with open_file("output.json", "w") as f:
            f.write(data)
    """
    if path.endswith(".gz"):
        with gzip.open(path, mode + "t", encoding="utf-8") as handle:
            yield handle  # type: ignore[misc]
    else:
        with open(path, mode, encoding="utf-8") as handle:
            yield handle


@contextmanager
def open_file_binary(path: str, mode: str = "rb") -> Iterator[IO[bytes]]:
    """
    Open a binary file, auto-detecting gzip compression from extension.

    Args:
        path: File path (.gz extension triggers gzip mode)
        mode: "rb" for read, "wb" for write

    Yields:
        Binary file handle

    Example:
        with open_file_binary("model.cart.gz", "rb") as f:
            data = f.read()

        with open_file_binary("model.pkl.gz", "wb") as f:
            pickle.dump(data, f)
    """
    if path.endswith(".gz"):
        with gzip.open(path, mode) as handle:
            yield handle  # type: ignore[misc]
    else:
        with open(path, mode) as handle:
            yield handle
