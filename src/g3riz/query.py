"""Thin bulk/vectorized read surface over an already-built field."""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .market import MarketSpine
from .field import build_identity


def _riz_exists_at(passports: pa.Table, minute_pos: int) -> pa.Array:
    """Research-object existence starts at T0 and ends at canonical deletion."""
    started = pa.compute.less_equal(passports["t0_spine_pos"], minute_pos)
    not_deleted = pa.compute.or_kleene(
        pa.compute.is_null(passports["c1_deletion_spine_pos"]),
        pa.compute.greater(passports["c1_deletion_spine_pos"], minute_pos),
    )
    return pa.compute.and_(started, not_deleted)


class Field:
    def __init__(self, repo_root: Path, instrument: str):
        self.repo_root = repo_root.resolve()
        self.instrument = instrument.upper()
        self.market = MarketSpine.open_store(self.repo_root / "data" / "market" / self.instrument)
        self.field_root = self.repo_root / "data" / "field" / self.instrument

    def _fact_files(self, name: str, tf: int | None) -> Path | list[Path]:
        def current_path(value: int) -> Path | None:
            cell = self.field_root / "cells" / f"tf_{value:04d}"
            path = cell / f"{name}.parquet"
            manifest_path = cell / "manifest.json"
            if not path.exists() or not manifest_path.exists():
                return None
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None
            expected = build_identity(self.market, self.instrument, value)
            if (manifest.get("status") != "complete"
                    or manifest.get("tf_minutes") != value
                    or manifest.get("build_identity") != expected):
                return None
            return path

        if tf is not None:
            path = current_path(tf)
            if path is None:
                raise FileNotFoundError(f"no current complete field cell for {self.instrument} TF {tf}")
            return path
        paths = [path for value in range(1, 1441)
                 if (path := current_path(value)) is not None]
        if not paths:
            raise FileNotFoundError(f"no current materialized {name} cells under {self.field_root}")
        return paths

    def passports(self, *, tf: int | None = None, tf_min: int | None = None,
                  tf_max: int | None = None, t0_start_ns: int | None = None,
                  t0_stop_ns: int | None = None, censored: bool | None = None,
                  x3_ignited: bool | None = None,
                  min_confirmed_nx: int | None = None) -> pa.Table:
        filters: list[tuple[str, str, object]] = []
        if tf is not None:
            filters.append(("tf_minutes", "=", tf))
        if tf_min is not None:
            filters.append(("tf_minutes", ">=", tf_min))
        if tf_max is not None:
            filters.append(("tf_minutes", "<=", tf_max))
        if t0_start_ns is not None:
            filters.append(("t0_ts_ns", ">=", t0_start_ns))
        if t0_stop_ns is not None:
            filters.append(("t0_ts_ns", "<", t0_stop_ns))
        if censored is not None:
            filters.append(("censored", "=", censored))
        table = pq.read_table(self._fact_files("passports", tf), filters=filters or None)
        if x3_ignited is not None:
            present = pa.compute.invert(pa.compute.is_null(table["x3_t0_ts_ns"]))
            table = table.filter(present if x3_ignited else pa.compute.invert(present))
        if min_confirmed_nx is not None:
            table = table.filter(pa.compute.greater_equal(table["final_span_count"],
                                                           min_confirmed_nx))
        return table

    def events(self, *, tf: int | None = None, riz_ids: pa.Array | list[str] | None = None,
               kinds: list[str] | None = None) -> pa.Table:
        table = pq.read_table(self._fact_files("events", tf),
                              filters=[("event_kind", "in", kinds)] if kinds else None)
        if riz_ids is not None:
            ids = pa.array(riz_ids)
            table = table.filter(pa.compute.is_in(table["riz_id"], value_set=ids))
        return table

    def minute_windows(self, anchor_positions: np.ndarray, before: int, after: int,
                       columns: tuple[str, ...] = ("open", "high", "low", "close", "volume")) -> dict:
        positions = self.market.window_positions(anchor_positions, before, after)
        valid = positions >= 0
        safe = np.where(valid, positions, 0)
        out = {"positions": positions, "valid": valid,
               "close_ts_utc_ns": np.asarray(self.market.close_ts_utc_ns[safe])}
        for name in columns:
            values = np.asarray(getattr(self.market, name)[safe])
            if np.issubdtype(values.dtype, np.floating):
                values = np.where(valid, values, np.nan)
            out[name] = values
        return out

    def objects_at(self, minute_pos: int, state: str = "alive") -> pa.Table:
        passports = self.passports()
        mask = _riz_exists_at(passports, minute_pos)
        if state == "blue":
            candidates = passports.filter(mask)
            eligible = pa.compute.or_kleene(
                pa.compute.is_null(candidates["blue_eligibility_end_spine_pos"]),
                pa.compute.greater(candidates["blue_eligibility_end_spine_pos"], minute_pos),
            )
            return candidates.filter(eligible)
        elif state == "breaker":
            candidates = passports.filter(mask)
            events = self.events(riz_ids=candidates["riz_id"], kinds=["breaker_entry"])
            entered = events.filter(pa.compute.less_equal(events["market_spine_pos"], minute_pos))
            ids = pa.compute.unique(entered["riz_id"])
            return candidates.filter(pa.compute.is_in(candidates["riz_id"], value_set=ids))
        elif state != "alive":
            raise ValueError("state must be alive, blue, or breaker")
        return passports.filter(mask)
