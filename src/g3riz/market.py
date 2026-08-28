"""Immutable canonical one-minute market spines."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

MARKET_SCHEMA = "g3-market-spine/1"
PARSER_VERSION = "chicago-end-label-to-eastern-open-v1"
EXPECTED_COLUMNS = ["Date", "Time", "Open", "High", "Low", "Close", "Volume"]
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


def _json_dump(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _day_numbers(open_et: pd.Series) -> np.ndarray:
    # Session date is the open-label calendar date before 18:00 ET, otherwise
    # the following date. This is the accepted V2/Golden session convention.
    normalized = open_et.dt.tz_localize(None).dt.normalize()
    session_dates = normalized + pd.to_timedelta((open_et.dt.hour >= 18).astype("int8"), unit="D")
    session_ns = session_dates.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    return (session_ns // 86_400_000_000_000).astype(np.int32)


def ingest_source(source: Path, out_dir: Path, instrument: str, *, chunksize: int = 500_000) -> dict:
    """Verify and atomically convert one pinned CSV to memory-mappable arrays."""
    instrument = instrument.upper()
    if instrument not in SOURCE_SPECS:
        raise ValueError(f"instrument must be one of {sorted(SOURCE_SPECS)}")
    source = source.resolve()
    spec = SOURCE_SPECS[instrument]
    actual_sha = sha256_file(source)
    if actual_sha != spec["sha256"]:
        raise ValueError(f"{instrument} SHA-256 mismatch: {actual_sha} != {spec['sha256']}")

    out_dir = out_dir.resolve()
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{instrument}.market.", dir=out_dir.parent))
    arrays = {
        name: np.lib.format.open_memmap(tmp / f"{name}.npy", mode="w+", dtype=dtype, shape=(spec["rows"],))
        for name, dtype in ARRAY_DTYPES.items()
    }
    session_days: list[int] = []
    session_open_ns: list[int] = []
    session_first: list[int] = []
    session_stop: list[int] = []
    session_lookup: dict[int, int] = {}
    cursor = 0
    previous_close = None
    dtypes = {"Date": "string", "Time": "string", "Open": "float64", "High": "float64",
              "Low": "float64", "Close": "float64", "Volume": "int64"}
    try:
        for frame in pd.read_csv(source, dtype=dtypes, chunksize=chunksize):
            if list(frame.columns) != EXPECTED_COLUMNS:
                raise ValueError(f"expected exact schema {EXPECTED_COLUMNS}, got {list(frame.columns)}")
            n = len(frame)
            stop = cursor + n
            if stop > spec["rows"]:
                raise ValueError(f"source exceeds pinned row count {spec['rows']}")
            label = pd.to_datetime(frame["Date"] + " " + frame["Time"],
                                   format="%Y-%m-%d %H:%M:%S", errors="raise")
            localized = label.dt.tz_localize("America/Chicago", ambiguous="raise", nonexistent="raise")
            open_et = localized.dt.tz_convert("US/Eastern") - pd.Timedelta(minutes=1)
            close_ns = (localized.dt.tz_convert("UTC").dt.tz_localize(None)
                        .to_numpy(dtype="datetime64[ns]").astype(np.int64))
            source_ns = label.to_numpy(dtype="datetime64[ns]").astype(np.int64)
            if previous_close is not None and close_ns[0] <= previous_close:
                raise ValueError("canonical timestamps are not strictly ascending")
            if n > 1 and np.any(close_ns[1:] <= close_ns[:-1]):
                raise ValueError("canonical timestamps are not unique and ascending")
            previous_close = int(close_ns[-1])
            o = frame["Open"].to_numpy()
            h = frame["High"].to_numpy()
            low = frame["Low"].to_numpy()
            c = frame["Close"].to_numpy()
            if np.any(low > np.minimum(o, c)) or np.any(h < np.maximum(o, c)) or np.any(h < low):
                raise ValueError("OHLC invariant failed")
            if not np.isfinite(np.column_stack((o, h, low, c))).all():
                raise ValueError("non-finite OHLC value")

            days = _day_numbers(open_et)
            ids = np.empty(n, dtype=np.int32)
            for day in np.unique(days):
                day_i = int(day)
                sid = session_lookup.get(day_i)
                positions = np.flatnonzero(days == day)
                if sid is None:
                    sid = len(session_days)
                    session_lookup[day_i] = sid
                    session_days.append(day_i)
                    # Midnight ET of the session date minus six hours.
                    day_stamp = pd.Timestamp(day_i * 86_400_000_000_000)
                    anchor_et = day_stamp.tz_localize("US/Eastern") - pd.Timedelta(hours=6)
                    session_open_ns.append(int(anchor_et.tz_convert("UTC").value))
                    session_first.append(cursor + int(positions[0]))
                    session_stop.append(cursor + int(positions[-1]) + 1)
                else:
                    session_stop[sid] = cursor + int(positions[-1]) + 1
                ids[positions] = sid

            arrays["source_ts_local_ns"][cursor:stop] = source_ns
            arrays["close_ts_utc_ns"][cursor:stop] = close_ns
            arrays["open"][cursor:stop] = o
            arrays["high"][cursor:stop] = h
            arrays["low"][cursor:stop] = low
            arrays["close"][cursor:stop] = c
            arrays["volume"][cursor:stop] = frame["Volume"].to_numpy()
            arrays["session_id"][cursor:stop] = ids
            arrays["source_row_pos"][cursor:stop] = np.arange(cursor, stop, dtype=np.int64)
            cursor = stop

        if cursor != spec["rows"]:
            raise ValueError(f"row count mismatch: {cursor} != {spec['rows']}")
        for arr in arrays.values():
            arr.flush()
            mmap = getattr(arr, "_mmap", None)
            if mmap is not None:
                mmap.close()
        del arrays
        np.savez(
            tmp / "sessions.npz",
            session_id=np.arange(len(session_days), dtype=np.int32),
            session_date_local_days=np.asarray(session_days, dtype=np.int32),
            session_open_utc_ns=np.asarray(session_open_ns, dtype=np.int64),
            first_minute_pos=np.asarray(session_first, dtype=np.int64),
            stop_minute_pos=np.asarray(session_stop, dtype=np.int64),
        )
        array_hashes = {p.name: sha256_file(p) for p in sorted(tmp.glob("*.npy"))}
        array_hashes["sessions.npz"] = sha256_file(tmp / "sessions.npz")
        close_mm = np.load(tmp / "close_ts_utc_ns.npy", mmap_mode="r")
        first_close_ns = int(close_mm[0])
        last_close_ns = int(close_mm[-1])
        mmap = getattr(close_mm, "_mmap", None)
        if mmap is not None:
            mmap.close()
        del close_mm
        manifest = {
            "schema": MARKET_SCHEMA,
            "instrument": instrument,
            "rows": cursor,
            "sessions": len(session_days),
            "source": {
                "path_at_build": str(source),
                "dataset": "danielharkin21/futuresss",
                "sha256": actual_sha,
                "feed_origin": None,
                "license": None,
                "timezone_metadata": None,
                "roll_rule": None,
            },
            "transform": {
                "version": PARSER_VERSION,
                "source_interpretation": "America/Chicago end-labelled minute",
                "canonical_open_interpretation": "source instant converted to US/Eastern minus one minute",
                "canonical_close_storage": "UTC epoch nanoseconds of the source-labelled close instant",
                "session_anchor": "18:00 US/Eastern; final and gapped windows contain observed bars only",
                "synthetic_minutes": False,
            },
            "first_close_utc_ns": first_close_ns,
            "last_close_utc_ns": last_close_ns,
            "array_sha256": array_hashes,
        }
        manifest["corpus_id"] = hashlib.sha256(
            f"{MARKET_SCHEMA}|{instrument}|{actual_sha}|{PARSER_VERSION}".encode()
        ).hexdigest()
        _json_dump(tmp / "manifest.json", manifest)
        if out_dir.exists():
            backup = out_dir.with_name(out_dir.name + ".previous")
            if backup.exists():
                shutil.rmtree(backup)
            os.replace(out_dir, backup)
            os.replace(tmp, out_dir)
            shutil.rmtree(backup)
        else:
            os.replace(tmp, out_dir)
        return manifest
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


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
