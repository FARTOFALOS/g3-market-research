"""Thin bulk/vectorized read surface over an already-built field."""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .market import MarketSpine
from .identity import build_identity
from .film import Film, build_film, passport_rows


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
        # A request without a timeframe asks for the whole population, not for
        # whatever happened to load. One missing or stale cell out of 1,440
        # would return a perfectly normal-looking table over 1,439, and a tool
        # failure would have become a market result.
        resolved = {value: current_path(value) for value in range(1, 1441)}
        missing = [value for value, path in resolved.items() if path is None]
        if missing:
            shown = missing if len(missing) <= 12 else missing[:12] + ["..."]
            raise FileNotFoundError(
                f"incomplete current field for {self.instrument}: {len(missing)} of 1440 "
                f"{name} cells are absent or stale ({shown}). This is a tool or data "
                f"failure, not a market answer; run `g3-riz status` before reading.")
        return [resolved[value] for value in range(1, 1441)]

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
        # Index zero is a gather placeholder, never an observation. Every
        # payload it produced is masked, because a minute that does not exist
        # must not carry a real timestamp or a plausible volume: floats become
        # NaN, integral columns become -1, which no epoch or volume can be.
        out = {"positions": positions, "valid": valid,
               "close_ts_utc_ns": np.where(
                   valid, np.asarray(self.market.close_ts_utc_ns[safe]), -1)}
        for name in columns:
            values = np.asarray(getattr(self.market, name)[safe])
            fill = np.nan if np.issubdtype(values.dtype, np.floating) else -1
            out[name] = np.where(valid, values, fill)
        return out

    def films(self, passports: pa.Table, *, stop: str = "deletion",
              pre_roll: int = 0, max_bars: int | None = None,
              end_positions=None, end_reason: str | None = None):
        """Yield one Film per passport row. Nothing is written anywhere.

        Every film is anchored at that RIZ's T0 and cut from the open spine,
        used, and dropped. To end films where a study's own rule says — its
        first retest, its own exhaustion condition — pass `end_positions`, one
        spine position per row, together with the phrase naming that rule. This
        method never evaluates a study's predicate.
        """
        rows = passport_rows(passports)
        if end_positions is not None and len(end_positions) != len(rows):
            raise ValueError("one end position per passport row is required")
        for index, row in enumerate(rows):
            yield build_film(
                self.market, row, stop=stop, pre_roll=pre_roll,
                max_bars=max_bars,
                end_position=(None if end_positions is None
                              else int(end_positions[index])),
                end_reason=end_reason)

    def film(self, riz_id: str, *, tf: int, stop: str = "deletion",
             pre_roll: int = 0, max_bars: int | None = None,
             end_position: int | None = None,
             end_reason: str | None = None) -> Film:
        """One Film by RIZ identity. `tf` is required: `riz_id` does not carry it."""
        table = self.passports(tf=tf)
        match = table.filter(pa.compute.equal(table["riz_id"], riz_id))
        if match.num_rows != 1:
            raise KeyError(f"{riz_id!r} is not a single RIZ on TF {tf}")
        return build_film(self.market, passport_rows(match)[0], stop=stop,
                          pre_roll=pre_roll, max_bars=max_bars,
                          end_position=end_position, end_reason=end_reason)

    def t0_minute_groups(self, passports: pa.Table | None = None) -> pa.Table:
        """Group passports by (T0 minute, exit side). A DIAGNOSTIC, not a unit.

        This answers one narrow question: how many distinct T0 minutes stand
        behind a pile of rows. The same tape is replayed 1,440 times per
        instrument, so 1.27 million passports sit on far fewer moments, and
        that ratio is worth knowing.

        IT IS NOT THE RESEARCH OBJECT, and an earlier version of this docstring
        said it was. The minute path after T0 does NOT belong to the moment,
        because the line the trader watches after T0 — the boundary the RIZ left
        through — belongs to the RIZ, not to the minute. Measured on NQ: 44.6%
        of T0 minutes carry more than one RIZ, and 56.6% of those carry two or
        more DIFFERENT exit boundaries (up to 170 on one minute). Among the
        minutes with distinct boundaries, the first post-T0 contact with the
        exit boundary lands on a different minute for 46.1% of them, p90 spread
        22 minutes, worst case 599. One pair: 2006-06-12 07:13 UTC, TF 41 at
        1576.00 and TF 44 at 1575.75, first contact +1 vs +205.

        So a group here is one moment several films started at, never one film.
        Films are per `riz_id`, always. A study that needs an independent unit
        defines its own and says what the definition throws away; grouping by
        T0 minute alone is not that definition.

        Returned columns are the group key, its layer count, and the list of
        `riz_id` behind it, so nothing about the layers is thrown away — no
        canonical layer is chosen and no boundary is collapsed.

        On the 2026 field a T0 minute carries ONE exit side, never both,
        checked on all 224,097 groups of ES, NQ and YM. The key still includes
        the side so that a future generation which breaks the invariant splits
        into two groups instead of silently merging two markets.
        """
        table = self.passports() if passports is None else passports
        pos = table["t0_spine_pos"].to_numpy()
        side = np.asarray(table["t0_exit_side"])
        north = side == "north"
        key = pos.astype(np.int64) * 2 + north.astype(np.int64)
        order = np.argsort(key, kind="stable")
        sorted_key = key[order]
        starts = np.flatnonzero(np.r_[True, sorted_key[1:] != sorted_key[:-1]])
        counts = np.diff(np.r_[starts, sorted_key.size])
        uniq = sorted_key[starts]
        ids = np.asarray(table["riz_id"])[order]
        grouped = [ids[a:a + c].tolist() for a, c in zip(starts, counts)]
        ev_pos = (uniq // 2).astype(np.int64)
        ev_north = (uniq % 2).astype(bool)
        return pa.table({
            "instrument": pa.array([self.instrument] * uniq.size, pa.string()),
            "t0_spine_pos": pa.array(ev_pos),
            "t0_exit_side": pa.array(np.where(ev_north, "north", "south")),
            "n_layers": pa.array(counts.astype(np.int32)),
            "riz_ids": pa.array(grouped, pa.list_(pa.string())),
        })

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
