"""Read access to the frozen canonical one-minute market spines."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MARKET_SCHEMA = "g3-market-spine/1"
PARSER_VERSION = "chicago-end-label-to-eastern-open-v1"
SOURCE_SPECS = {
    "ES": {"rows": 6_855_058, "sha256": "ca2d8aa64977801f01d8f153574fa16cf8fa0dff7fcdfe75c58c5025df441c3b"},
    "NQ": {"rows": 6_418_541, "sha256": "7404eb845350c84361ce2316edfffd830b3eea7761e301a8f1e6368c9178934a"},
    "YM": {"rows": 6_438_663, "sha256": "d249ab94ed8ee372b7c347ac39de76e4529ab660f42e78210d8f09fcec379ddd"},
}
ARRAY_DTYPES = {
    "source_ts_local_ns": np.dtype("int64"),
    "close_ts_utc_ns": np.dtype("int64"),
    "open": np.dtype("float64"),
    "high": np.dtype("float64"),
    "low": np.dtype("float64"),
    "close": np.dtype("float64"),
    "volume": np.dtype("int64"),
    "session_id": np.dtype("int32"),
    "source_row_pos": np.dtype("int64"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class MarketSpine:
    root: Path
    manifest: dict
    source_ts_local_ns: np.ndarray
    close_ts_utc_ns: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    session_id: np.ndarray
    source_row_pos: np.ndarray
    sessions: dict[str, np.ndarray]

    @classmethod
    def open_store(cls, root: Path, mmap_mode: str = "r") -> "MarketSpine":
        root = root.resolve()
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        arrays = {name: np.load(root / f"{name}.npy", mmap_mode=mmap_mode) for name in ARRAY_DTYPES}
        with np.load(root / "sessions.npz") as z:
            sessions = {name: z[name].copy() for name in z.files}
        return cls(root=root, manifest=manifest, sessions=sessions, **arrays)

    def position_of_close_ns(self, timestamp_ns: int) -> int:
        pos = int(np.searchsorted(self.close_ts_utc_ns, timestamp_ns))
        return pos if pos < len(self.close_ts_utc_ns) and int(self.close_ts_utc_ns[pos]) == timestamp_ns else -1

    def window_positions(self, anchors: np.ndarray, before: int, after: int) -> np.ndarray:
        anchors = np.asarray(anchors, dtype=np.int64)
        offsets = np.arange(-before, after + 1, dtype=np.int64)
        positions = anchors[:, None] + offsets[None, :]
        valid = (positions >= 0) & (positions < len(self.close))
        return np.where(valid, positions, -1)
