"""Resumable materialization of one canonical (instrument, timeframe) cell."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .identity import (FROZEN_MATERIALIZER_CODE_IDENTITY, build_identity,
                       event_id, riz_id)
from .fast import facts_to_rows, replay_fast
from .machine import (
    TIER_ACTIVE_BOTH, TIER_ACTIVE_ONE_SIDED, TIER_BREAKER, TIER_LATENT,
    NativeBar, Zone, advance_one, blue_confirmed, body_bounds, detect_births,
    spans, zone_key,
)
from .market import MarketSpine, sha256_file
from .native import build_native_arrays, iter_native_bars
from .resources import process_peak_working_set_bytes
from .schema import (CELL_SCHEMA_VERSION, EVENT_SCHEMA, EVENT_SCHEMA_VERSION,
                     PASSPORT_SCHEMA, PASSPORT_SCHEMA_VERSION, SEMANTIC_VERSION)


def _snap(zone: Zone) -> tuple[int, bool, int, bool, bool, int, bool]:
    return (zone.tier, zone.activated, zone.span_count, zone.north_alive,
            zone.south_alive, zone.last_span_index, zone.broke_south)


@dataclass(slots=True)
class RuntimeZone:
    zone: Zone
    riz_id: str
    precursor_formed_ts_ns: int
    precursor_formed_spine_pos: int
    precursor_native_bar_index: int
    events: list[dict] = field(default_factory=list)
    qualifies: bool = False
    t0_ts_ns: int | None = None
    t0_spine_pos: int | None = None
    t0_kind: str | None = None
    t0_exit_side: str | None = None
    t0_close: float | None = None
    t0_span_count: int | None = None
    t0_native_bar_index: int | None = None
    x3_t0_ts_ns: int | None = None
    x3_t0_spine_pos: int | None = None
    x3_t0_native_bar_index: int | None = None
    x3_t0_close: float | None = None
    x3_t0_exit_side: str | None = None
    x3_t0_confirmed: bool | None = None
    native_blue_confirmation_ts_ns: int | None = None
    native_blue_confirmation_spine_pos: int | None = None
    native_blue_confirmation_native_bar_index: int | None = None
    blue_eligibility_end_ts_ns: int | None = None
    blue_eligibility_end_spine_pos: int | None = None
    c1_deletion_ts_ns: int | None = None
    c1_deletion_spine_pos: int | None = None
    c1_deletion_native_bar_index: int | None = None


class CellBuilder:
    def __init__(self, market: MarketSpine, instrument: str, tf_minutes: int):
        self.market = market
        self.instrument = instrument
        self.tf = tf_minutes
        self.corpus_id = market.manifest["corpus_id"]
        self.live: list[RuntimeZone] = []
        self.was_blue: set[tuple] = set()
        self.passports: list[dict] = []
        self.events: list[dict] = []
        self.native_bars = 0
        self.zones_born = 0

    def add_event(self, rt: RuntimeZone, family: str, kind: str, event_ts_ns: int,
                  spine_pos: int, bar: NativeBar, *, terminal: bool = False,
                  deletion_cause: str | None = None,
                  breaker_direction: str | None = None,
                  snapshot: tuple | None = None) -> dict:
        z = rt.zone
        state = snapshot or _snap(z)
        seq = len(rt.events)
        row = {
            "event_id": event_id(rt.riz_id, family, kind, event_ts_ns, bar.index, seq),
            "riz_id": rt.riz_id, "instrument": self.instrument, "tf_minutes": self.tf,
            "event_family": family, "event_kind": kind, "event_ts_ns": event_ts_ns,
            "market_spine_pos": spine_pos,
            "native_bar_index": bar.index, "event_seq": seq,
            "terminal": terminal, "deletion_cause": deletion_cause,
            "breaker_direction": breaker_direction,
            "tier": state[0], "activated": state[1], "span_count": state[2],
            "north_alive": state[3], "south_alive": state[4],
            "last_span_index": state[5], "broke_south": state[6],
        }
        rt.events.append(row)
        return row

    @staticmethod
    def _stage(zone: Zone) -> str | None:
        if zone.tier == TIER_LATENT:
            return "candidate"
        if zone.tier in (TIER_ACTIVE_BOTH, TIER_ACTIVE_ONE_SIDED):
            return "preview"
        return None

    @staticmethod
    def _truth(zone: Zone, stage: str, dev_open: float, closes: np.ndarray,
               bar_index: int) -> np.ndarray:
        if stage == "preview" and not (
            zone.north_alive and zone.south_alive and bar_index >= zone.last_span_index + 2
        ):
            return np.zeros(len(closes), dtype=bool)
        if dev_open < zone.bottom:
            return closes > zone.top
        if dev_open > zone.top:
            return closes < zone.bottom
        return np.zeros(len(closes), dtype=bool)

    def _ladder_before_close(self, bar: NativeBar) -> dict[str, dict]:
        states: dict[str, dict] = {}
        if bar.minute_stop - bar.minute_start <= 1:
            return states
        dev_open = float(self.market.open[bar.minute_start])
        positions = np.arange(bar.minute_start, bar.minute_stop - 1, dtype=np.int64)
        closes = np.asarray(self.market.close[positions])
        min_close = float(np.min(closes))
        max_close = float(np.max(closes))
        for rt in self.live:
            zone = rt.zone
            # A potential next accepted span uses the canonical body-span and
            # cr+2 predicates. Unlike Blue/T0 eligibility it does not require
            # both boundaries to remain alive.
            if (rt.x3_t0_ts_ns is None and zone.span_count == 2
                    and zone.tier in (TIER_ACTIVE_BOTH, TIER_ACTIVE_ONE_SIDED)
                    and bar.index >= zone.last_span_index + 2):
                if dev_open < zone.bottom:
                    x3_truth = closes > zone.top
                elif dev_open > zone.top:
                    x3_truth = closes < zone.bottom
                else:
                    x3_truth = np.zeros(len(closes), dtype=bool)
                first = np.flatnonzero(x3_truth)
                if len(first):
                    ix = int(first[0])
                    self._set_x3(rt, int(self.market.close_ts_utc_ns[positions[ix]]),
                                 int(positions[ix]), bar, float(closes[ix]))

            stage = self._stage(rt.zone)
            if stage is None:
                continue
            if stage == "preview" and not (
                zone.north_alive and zone.south_alive
                and bar.index >= zone.last_span_index + 2
            ):
                continue
            if dev_open < zone.bottom:
                if max_close <= zone.top:
                    continue
                truth = closes > zone.top
                full_truth = bar.close > zone.top
            elif dev_open > zone.top:
                if min_close >= zone.bottom:
                    continue
                truth = closes < zone.bottom
                full_truth = bar.close < zone.bottom
            else:
                continue
            changes = np.flatnonzero(np.r_[truth[0], truth[1:] != truth[:-1]])
            active = False
            for ix in changes:
                now = bool(truth[ix])
                pos = int(positions[ix])
                ts_ns = int(self.market.close_ts_utc_ns[pos])
                if now:
                    active = True
                    if stage == "preview":
                        rt.qualifies = True
                        if rt.t0_ts_ns is None:
                            row = self.add_event(rt, "anchor", "t0", ts_ns, pos, bar)
                            self._set_t0(rt, row, float(closes[ix]), "minute_ignition")
                else:
                    active = False
            states[rt.riz_id] = {
                "stage": stage, "active": active, "fired": True,
                "full_truth": bool(full_truth),
            }
        return states

    def _set_t0(self, rt: RuntimeZone, row: dict, close: float, kind: str) -> None:
        if rt.t0_ts_ns is not None:
            return
        z = rt.zone
        rt.t0_ts_ns = row["event_ts_ns"]
        rt.t0_spine_pos = row["market_spine_pos"]
        rt.t0_kind = kind
        rt.t0_close = close
        rt.t0_exit_side = "north" if close > z.top else "south"
        rt.t0_span_count = z.span_count
        rt.t0_native_bar_index = row["native_bar_index"]

    def _set_x3(self, rt: RuntimeZone, ts: int, pos: int, bar: NativeBar,
                close: float, snapshot: tuple | None = None) -> None:
        if rt.x3_t0_ts_ns is not None:
            return
        self.add_event(rt, "anchor", "x3_t0", ts, pos, bar, snapshot=snapshot)
        rt.x3_t0_ts_ns = ts
        rt.x3_t0_spine_pos = pos
        rt.x3_t0_native_bar_index = bar.index
        rt.x3_t0_close = close
        rt.x3_t0_exit_side = "north" if close > rt.zone.top else "south"
        rt.x3_t0_confirmed = False

    def _state_changes(self, rt: RuntimeZone, before: tuple, bar: NativeBar,
                       survived: bool) -> None:
        z = rt.zone
        t0, act0, sc0, n0, s0, ls0, _bs0 = before
        pos = bar.minute_stop - 1
        ts = bar.close_ts_ns
        if z.span_count > sc0:
            if sc0 == 2:
                if rt.x3_t0_ts_ns is None:
                    self._set_x3(rt, ts, pos, bar, bar.close, snapshot=before)
                if rt.x3_t0_native_bar_index == bar.index:
                    rt.x3_t0_confirmed = True
            self.add_event(rt, "native_state", "accepted_span", ts, pos, bar)
        if n0 and not z.north_alive:
            self.add_event(rt, "native_state", "north_boundary_retired", ts, pos, bar)
            self._set_blue_end(rt, ts, pos)
        if s0 and not z.south_alive:
            self.add_event(rt, "native_state", "south_boundary_retired", ts, pos, bar)
            self._set_blue_end(rt, ts, pos)
        if t0 != TIER_BREAKER and z.tier == TIER_BREAKER:
            self.add_event(rt, "native_state", "breaker_entry", ts, pos, bar,
                           breaker_direction="south" if z.broke_south else "north")
            self._set_blue_end(rt, ts, pos)
        if not survived:
            bmin, bmax = body_bounds(bar)
            erratum = spans(z, bmin, bmax) and act0 and bar.index < ls0 + 2
            cause = ("too_early_respan" if erratum else
                     "breaker_return" if t0 == TIER_BREAKER else "boundary_exhaustion")
            self.add_event(rt, "native_state", "deleted", ts, pos, bar,
                           terminal=True, deletion_cause=cause)
            self._set_blue_end(rt, ts, pos)
            rt.c1_deletion_ts_ns = ts
            rt.c1_deletion_spine_pos = pos
            rt.c1_deletion_native_bar_index = bar.index

    @staticmethod
    def _set_blue_end(rt: RuntimeZone, ts: int, pos: int) -> None:
        if rt.blue_eligibility_end_ts_ns is None:
            rt.blue_eligibility_end_ts_ns = ts
            rt.blue_eligibility_end_spine_pos = pos

    def _finalize(self, rt: RuntimeZone, last_ts: int, last_pos: int, censored: bool) -> None:
        if not rt.qualifies:
            return
        z = rt.zone
        if censored:
            final_bar = NativeBar(
                index=self.native_bars - 1,
                open_ts_ns=last_ts, close_ts_ns=last_ts, open=0.0, high=0.0,
                low=0.0, close=0.0, minute_start=last_pos, minute_stop=last_pos + 1,
            )
            self.add_event(rt, "archive", "archive_censor", last_ts,
                           last_pos, final_bar, terminal=True)
        self.passports.append({
            "riz_id": rt.riz_id, "corpus_id": self.corpus_id,
            "instrument": self.instrument, "tf_minutes": self.tf,
            "zone_top": z.top, "zone_bottom": z.bottom, "zone_width": z.top - z.bottom,
            "direction": z.direction, "bullish": z.bullish,
            "precursor_formed_ts_ns": rt.precursor_formed_ts_ns,
            "precursor_formed_spine_pos": rt.precursor_formed_spine_pos,
            "precursor_native_bar_index": rt.precursor_native_bar_index,
            "t0_ts_ns": rt.t0_ts_ns, "t0_spine_pos": rt.t0_spine_pos,
            "t0_kind": rt.t0_kind, "t0_exit_side": rt.t0_exit_side,
            "t0_close": rt.t0_close, "t0_span_count": rt.t0_span_count,
            "t0_native_bar_index": rt.t0_native_bar_index,
            "native_blue_confirmation_ts_ns": rt.native_blue_confirmation_ts_ns,
            "native_blue_confirmation_spine_pos": rt.native_blue_confirmation_spine_pos,
            "native_blue_confirmation_native_bar_index": rt.native_blue_confirmation_native_bar_index,
            "x3_t0_ts_ns": rt.x3_t0_ts_ns,
            "x3_t0_spine_pos": rt.x3_t0_spine_pos,
            "x3_t0_native_bar_index": rt.x3_t0_native_bar_index,
            "x3_t0_close": rt.x3_t0_close,
            "x3_t0_exit_side": rt.x3_t0_exit_side,
            "x3_t0_confirmed": rt.x3_t0_confirmed,
            "blue_eligibility_end_ts_ns": rt.blue_eligibility_end_ts_ns,
            "blue_eligibility_end_spine_pos": rt.blue_eligibility_end_spine_pos,
            "c1_deletion_ts_ns": rt.c1_deletion_ts_ns,
            "c1_deletion_spine_pos": rt.c1_deletion_spine_pos,
            "c1_deletion_native_bar_index": rt.c1_deletion_native_bar_index,
            "censored": censored, "still_alive_at_archive_end": censored,
            "last_observed_ts_ns": last_ts, "last_observed_spine_pos": last_pos,
            "final_tier": z.tier, "final_activated": z.activated,
            "final_span_count": z.span_count, "final_north_alive": z.north_alive,
            "final_south_alive": z.south_alive, "final_last_span_index": z.last_span_index,
            "final_broke_south": z.broke_south,
        })
        self.events.extend(rt.events)

    def run(self) -> tuple[list[dict], list[dict], int, int]:
        prior: list[NativeBar] = []
        for bar in iter_native_bars(self.market, self.tf):
            self.native_bars += 1
            self._ladder_before_close(bar)
            pre = list(self.live)
            before = {rt.riz_id: _snap(rt.zone) for rt in pre}
            survivors: list[RuntimeZone] = []
            retired: list[RuntimeZone] = []
            for rt in pre:
                (survivors if advance_one(rt.zone, bar) else retired).append(rt)
            self.live = survivors

            survived_ids = {rt.riz_id for rt in survivors}
            for rt in retired:
                self.was_blue.discard(zone_key(rt.zone))
            # Native state mutates before the post-update confirmation test.
            # Preserve that causal order in the per-zone event sequence.
            for rt in pre:
                self._state_changes(rt, before[rt.riz_id], bar,
                                    rt.riz_id in survived_ids)
            for rt in survivors:
                key = zone_key(rt.zone)
                if blue_confirmed(rt.zone):
                    if key not in self.was_blue:
                        rt.qualifies = True
                        if rt.t0_ts_ns is None:
                            row = self.add_event(rt, "anchor", "t0", bar.close_ts_ns,
                                                 bar.minute_stop - 1, bar)
                            self._set_t0(rt, row, bar.close, "native_confirmation")
                        else:
                            self.add_event(rt, "native_state", "native_confirmation",
                                           bar.close_ts_ns, bar.minute_stop - 1, bar)
                        if rt.native_blue_confirmation_ts_ns is None:
                            rt.native_blue_confirmation_ts_ns = bar.close_ts_ns
                            rt.native_blue_confirmation_spine_pos = bar.minute_stop - 1
                            rt.native_blue_confirmation_native_bar_index = bar.index
                    self.was_blue.add(key)
                else:
                    self.was_blue.discard(key)
            for rt in retired:
                self._finalize(rt, bar.close_ts_ns, bar.minute_stop - 1, censored=False)

            prior.append(bar)
            if len(prior) > 3:
                prior.pop(0)
            if len(prior) == 3:
                for zone in detect_births(prior[0], prior[1], prior[2]):
                    zid = riz_id(self.corpus_id, self.instrument, self.tf,
                                 bar.close_ts_ns,
                                 zone.top, zone.bottom, zone.direction)
                    rt = RuntimeZone(zone, zid, bar.close_ts_ns, bar.minute_stop - 1, bar.index)
                    self.add_event(rt, "precursor", "precursor_formed", bar.close_ts_ns,
                                   bar.minute_stop - 1, bar)
                    self.live.append(rt)
                    self.zones_born += 1

        if self.native_bars:
            last_ts = int(self.market.close_ts_utc_ns[-1])
            last_pos = len(self.market.close_ts_utc_ns) - 1
            for rt in self.live:
                self._finalize(rt, last_ts, last_pos, censored=True)
        self.passports.sort(key=lambda r: r["riz_id"])
        self.events.sort(key=lambda r: (r["riz_id"], r["event_seq"]))
        return self.passports, self.events, self.native_bars, self.zones_born


def build_cell(market_root: Path, field_root: Path, instrument: str, tf_minutes: int,
               *, force: bool = False) -> dict:
    market = MarketSpine.open_store(market_root)
    identity = build_identity(market, instrument, tf_minutes)
    target = field_root / instrument / "cells" / f"tf_{tf_minutes:04d}"
    manifest_path = target / "manifest.json"
    if manifest_path.exists() and not force:
        current = json.loads(manifest_path.read_text(encoding="utf-8"))
        if current.get("status") == "complete" and current.get("build_identity") == identity:
            return {**current, "reused": True}
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".tf_{tf_minutes:04d}.", dir=target.parent))
    started = time.perf_counter()
    try:
        native = build_native_arrays(market, tf_minutes)
        facts = replay_fast(native, market.open, market.close, market.close_ts_utc_ns)
        passports, events = facts_to_rows(
            facts, market.manifest["corpus_id"], instrument, tf_minutes,
            int(market.close_ts_utc_ns[-1]), len(market.close_ts_utc_ns) - 1,
            native.close_ts_ns, native.minute_stop,
        )
        ids = [row["riz_id"] for row in passports]
        if len(ids) != len(set(ids)):
            raise ValueError(f"riz_id collision in {instrument} TF {tf_minutes}")
        native_bars, zones_born = len(native), facts.zone_count
        pq.write_table(pa.Table.from_pylist(passports, schema=PASSPORT_SCHEMA),
                       tmp / "passports.parquet", compression="zstd", use_dictionary=True)
        pq.write_table(pa.Table.from_pylist(events, schema=EVENT_SCHEMA),
                       tmp / "events.parquet", compression="zstd", use_dictionary=True)
        manifest = {
            "schema": CELL_SCHEMA_VERSION, "passport_schema": PASSPORT_SCHEMA_VERSION,
            "event_schema": EVENT_SCHEMA_VERSION, "semantic_version": SEMANTIC_VERSION,
            "status": "complete", "build_identity": identity,
            "engine": "compiled-array-replay/1",
            "materializer_code_identity": FROZEN_MATERIALIZER_CODE_IDENTITY,
            "market_corpus_id": market.manifest["corpus_id"], "instrument": instrument,
            "tf_minutes": tf_minutes, "market_rows": market.manifest["rows"],
            "source_first_close_utc_ns": market.manifest["first_close_utc_ns"],
            "source_last_close_utc_ns": market.manifest["last_close_utc_ns"],
            "native_bars": native_bars, "zones_born": zones_born,
            "passports": len(passports), "events": len(events),
            "riz_id_collisions": 0,
            "elapsed_seconds": time.perf_counter() - started,
            "process_peak_working_set_bytes": process_peak_working_set_bytes(),
        }
        manifest["output_sha256"] = {
            "passports.parquet": sha256_file(tmp / "passports.parquet"),
            "events.parquet": sha256_file(tmp / "events.parquet"),
        }
        (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                            encoding="utf-8")
        if target.exists():
            old = target.with_name(target.name + ".previous")
            if old.exists():
                shutil.rmtree(old)
            os.replace(target, old)
            os.replace(tmp, target)
            shutil.rmtree(old)
        else:
            os.replace(tmp, target)
        return {**manifest, "reused": False}
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
