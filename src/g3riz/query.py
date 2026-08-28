"""Thin bulk/vectorized read surface over an already-built field."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .market import MarketSpine


class Field:
    def __init__(self, repo_root: Path, instrument: str):
        self.repo_root = repo_root.resolve()
        self.instrument = instrument.upper()
        self.market = MarketSpine.open_store(self.repo_root / "data" / "market" / self.instrument)
        self.field_root = self.repo_root / "data" / "field" / self.instrument

    def _fact_files(self, name: str, tf: int | None) -> Path | list[Path]:
        if tf is not None:
            path = self.field_root / "cells" / f"tf_{tf:04d}" / f"{name}.parquet"
            if not path.exists():
                raise FileNotFoundError(f"materialized field file not found: {path}")
            return path
        consolidated = self.field_root / "consolidated" / f"{name}.parquet"
        if consolidated.exists():
            return consolidated
        paths = sorted((self.field_root / "cells").glob(f"tf_*/{name}.parquet"))
        if not paths:
            raise FileNotFoundError(f"no materialized {name} cells under {self.field_root}")
        return paths

    def passports(self, *, tf: int | None = None, tf_min: int | None = None,
                  tf_max: int | None = None, t0_start_ns: int | None = None,
                  t0_stop_ns: int | None = None, censored: bool | None = None) -> pa.Table:
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
        return pq.read_table(self._fact_files("passports", tf), filters=filters or None)

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
        born = pa.compute.less_equal(passports["birth_known_spine_pos"], minute_pos)
        deleted = pa.compute.or_(pa.compute.is_null(passports["c1_deletion_spine_pos"]),
                                 pa.compute.greater(passports["c1_deletion_spine_pos"], minute_pos))
        mask = pa.compute.and_(born, deleted)
        if state == "blue":
            candidates = passports.filter(mask)
            eligible = pa.compute.or_(
                pa.compute.is_null(candidates["blue_eligibility_end_spine_pos"]),
                pa.compute.greater(candidates["blue_eligibility_end_spine_pos"], minute_pos),
            )
            candidates = candidates.filter(eligible)
            confirmed = pa.compute.fill_null(
                pa.compute.less_equal(candidates["first_confirmation_spine_pos"], minute_pos),
                False,
            )
            confirmed_rows = candidates.filter(confirmed)
            pre_confirmation = candidates.filter(pa.compute.invert(confirmed))
            events = self.events(riz_ids=pre_confirmation["riz_id"],
                                 kinds=["preview", "cancellation"])
            events = events.filter(pa.compute.less_equal(events["market_spine_pos"], minute_pos))
            if not events.num_rows:
                return confirmed_rows
            events = events.sort_by([
                ("riz_id", "ascending"), ("market_spine_pos", "ascending"),
                ("event_seq", "ascending"),
            ])
            event_ids = np.asarray(events["riz_id"].combine_chunks().to_numpy(zero_copy_only=False))
            is_last = np.r_[event_ids[1:] != event_ids[:-1], True]
            last = events.filter(pa.array(is_last))
            is_active = pa.compute.equal(last["event_kind"], "preview")
            active_ids = last.filter(is_active)["riz_id"]
            preview_rows = pre_confirmation.filter(pa.compute.is_in(
                pre_confirmation["riz_id"], value_set=active_ids))
            return pa.concat_tables([confirmed_rows, preview_rows])
        elif state == "breaker":
            candidates = passports.filter(mask)
            events = self.events(riz_ids=candidates["riz_id"], kinds=["breaker_entry"])
            entered = events.filter(pa.compute.less_equal(events["market_spine_pos"], minute_pos))
            ids = pa.compute.unique(entered["riz_id"])
            return candidates.filter(pa.compute.is_in(candidates["riz_id"], value_set=ids))
        elif state != "alive":
            raise ValueError("state must be alive, blue, or breaker")
        return passports.filter(mask)
