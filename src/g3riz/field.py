"""Resumable materialization of one canonical (instrument, timeframe) cell."""

from __future__ import annotations

import hashlib
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

from .identity import event_id, riz_id
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
    birth_known_ts_ns: int
    birth_known_spine_pos: int
    birth_native_bar_index: int
    events: list[dict] = field(default_factory=list)
    qualifies: bool = False
    first_activation_ts_ns: int | None = None
    first_activation_spine_pos: int | None = None
    first_activation_native_bar_index: int | None = None
    t0_ts_ns: int | None = None
    t0_spine_pos: int | None = None
    t0_kind: str | None = None
    t0_exit_side: str | None = None
    t0_close: float | None = None
    t0_span_count: int | None = None
    t0_native_bar_index: int | None = None
    t0_preview_event_seq: int | None = None
    first_confirmation_ts_ns: int | None = None
    first_confirmation_spine_pos: int | None = None
    first_confirmation_native_bar_index: int | None = None
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
                  known_at_ns: int, spine_pos: int, bar: NativeBar, *,
                  occurrence: int | None = None, terminal: bool = False,
                  cancelled_stage: str | None = None, preview_outcome: str | None = None) -> dict:
        z = rt.zone
        seq = len(rt.events)
        row = {
            "event_id": event_id(rt.riz_id, family, kind, event_ts_ns, bar.index, seq),
            "riz_id": rt.riz_id, "instrument": self.instrument, "tf_minutes": self.tf,
            "event_family": family, "event_kind": kind, "event_ts_ns": event_ts_ns,
            "known_at_ns": known_at_ns, "market_spine_pos": spine_pos,
            "native_bar_index": bar.index, "event_seq": seq,
            "occurrence_index": occurrence, "terminal": terminal,
            "cancelled_stage": cancelled_stage, "preview_outcome": preview_outcome,
            "preview_outcome_known_at_ns": None,
            "preview_outcome_known_at_spine_pos": None,
            "tier": z.tier, "activated": z.activated, "span_count": z.span_count,
            "north_alive": z.north_alive, "south_alive": z.south_alive,
            "last_span_index": z.last_span_index, "broke_south": z.broke_south,
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
            stage = self._stage(rt.zone)
            if stage is None:
                continue
            zone = rt.zone
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
            occurrence = 0
            active = False
            bar_events: list[dict] = []
            for ix in changes:
                now = bool(truth[ix])
                pos = int(positions[ix])
                ts_ns = int(self.market.close_ts_utc_ns[pos])
                if now:
                    row = self.add_event(rt, "ladder", stage, ts_ns, ts_ns, pos, bar,
                                         occurrence=occurrence)
                    bar_events.append(row)
                    active = True
                    if stage == "preview":
                        rt.qualifies = True
                        self._set_t0(rt, row, float(closes[ix]), "preview")
                else:
                    row = self.add_event(rt, "ladder", "cancellation", ts_ns, ts_ns,
                                         pos, bar, occurrence=occurrence,
                                         cancelled_stage=stage)
                    bar_events.append(row)
                    active = False
                    occurrence += 1
            states[rt.riz_id] = {
                "stage": stage, "active": active, "fired": True,
                "occurrence": occurrence, "events": bar_events,
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
        if kind == "preview":
            rt.t0_preview_event_seq = row["event_seq"]

    def _state_changes(self, rt: RuntimeZone, before: tuple, bar: NativeBar,
                       survived: bool) -> None:
        z = rt.zone
        t0, act0, sc0, n0, s0, ls0, _bs0 = before
        pos = bar.minute_stop - 1
        ts = bar.close_ts_ns
        if not act0 and z.activated:
            self.add_event(rt, "zone_state", "activation", ts, ts, pos, bar)
            if rt.first_activation_ts_ns is None:
                rt.first_activation_ts_ns = ts
                rt.first_activation_spine_pos = pos
                rt.first_activation_native_bar_index = bar.index
        if z.span_count > sc0:
            self.add_event(rt, "zone_state", "span", ts, ts, pos, bar)
        if n0 and not z.north_alive:
            self.add_event(rt, "zone_state", "north_boundary_dead", ts, ts, pos, bar)
            self._set_blue_end(rt, ts, pos)
        if s0 and not z.south_alive:
            self.add_event(rt, "zone_state", "south_boundary_dead", ts, ts, pos, bar)
            self._set_blue_end(rt, ts, pos)
        if t0 != TIER_BREAKER and z.tier == TIER_BREAKER:
            self.add_event(rt, "zone_state", "breaker_entry", ts, ts, pos, bar)
            self._set_blue_end(rt, ts, pos)
        if not survived:
            bmin, bmax = body_bounds(bar)
            erratum = spans(z, bmin, bmax) and act0 and bar.index < ls0 + 2
            self.add_event(rt, "zone_state", "erratum1_respan" if erratum else "deleted",
                           ts, ts, pos, bar, terminal=True)
            self._set_blue_end(rt, ts, pos)
            rt.c1_deletion_ts_ns = ts
            rt.c1_deletion_spine_pos = pos
            rt.c1_deletion_native_bar_index = bar.index

    @staticmethod
    def _set_blue_end(rt: RuntimeZone, ts: int, pos: int) -> None:
        if rt.blue_eligibility_end_ts_ns is None:
            rt.blue_eligibility_end_ts_ns = ts
            rt.blue_eligibility_end_spine_pos = pos

    def _classify_preview_events(self, state: dict, confirmation: dict | None,
                                 terminal_cancel: dict | None, bar: NativeBar) -> None:
        events = list(state.get("events", []))
        if confirmation is not None:
            events.append(confirmation)
        if terminal_cancel is not None:
            events.append(terminal_cancel)
        previews = [i for i, event in enumerate(events) if event["event_kind"] == "preview"]
        for i in previews:
            outcome = "carried_no_event"
            for later in events[i + 1:]:
                if later["event_kind"] == "cancellation":
                    outcome = "cancelled"
                    break
                if later["event_kind"] == "confirmation":
                    outcome = "confirmed"
                    break
                if later["event_kind"] == "preview":
                    break
            events[i]["preview_outcome"] = outcome
            events[i]["preview_outcome_known_at_ns"] = bar.close_ts_ns
            events[i]["preview_outcome_known_at_spine_pos"] = bar.minute_stop - 1

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
            self.add_event(rt, "archive", "archive_censor", last_ts, last_ts,
                           last_pos, final_bar, terminal=True)
        preview_outcome = None
        preview_outcome_known_at_ns = None
        preview_outcome_known_at_pos = None
        if rt.t0_preview_event_seq is not None:
            preview = rt.events[rt.t0_preview_event_seq]
            preview_outcome = preview["preview_outcome"]
            preview_outcome_known_at_ns = preview["preview_outcome_known_at_ns"]
            preview_outcome_known_at_pos = preview["preview_outcome_known_at_spine_pos"]
        self.passports.append({
            "riz_id": rt.riz_id, "corpus_id": self.corpus_id,
            "instrument": self.instrument, "tf_minutes": self.tf,
            "zone_top": z.top, "zone_bottom": z.bottom, "zone_width": z.top - z.bottom,
            "direction": z.direction, "bullish": z.bullish,
            "birth_label_ts_ns": z.birth_label_ts_ns,
            "birth_known_ts_ns": rt.birth_known_ts_ns,
            "birth_known_spine_pos": rt.birth_known_spine_pos,
            "birth_native_bar_index": rt.birth_native_bar_index,
            "first_activation_ts_ns": rt.first_activation_ts_ns,
            "first_activation_spine_pos": rt.first_activation_spine_pos,
            "first_activation_native_bar_index": rt.first_activation_native_bar_index,
            "t0_ts_ns": rt.t0_ts_ns, "t0_spine_pos": rt.t0_spine_pos,
            "t0_kind": rt.t0_kind, "t0_exit_side": rt.t0_exit_side,
            "t0_close": rt.t0_close, "t0_span_count": rt.t0_span_count,
            "t0_native_bar_index": rt.t0_native_bar_index,
            "t0_preview_outcome": preview_outcome,
            "t0_preview_outcome_known_at_ns": preview_outcome_known_at_ns,
            "t0_preview_outcome_known_at_spine_pos": preview_outcome_known_at_pos,
            "first_confirmation_ts_ns": rt.first_confirmation_ts_ns,
            "first_confirmation_spine_pos": rt.first_confirmation_spine_pos,
            "first_confirmation_native_bar_index": rt.first_confirmation_native_bar_index,
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
            ladder = self._ladder_before_close(bar)
            pre = list(self.live)
            before = {rt.riz_id: _snap(rt.zone) for rt in pre}
            survivors: list[RuntimeZone] = []
            retired: list[RuntimeZone] = []
            for rt in pre:
                (survivors if advance_one(rt.zone, bar) else retired).append(rt)
            self.live = survivors

            alive_keys = {zone_key(rt.zone) for rt in survivors}
            survived_ids = {rt.riz_id for rt in survivors}
            for rt in retired:
                self.was_blue.discard(zone_key(rt.zone))
            # Native state mutates before the post-update confirmation test.
            # Preserve that causal order in the per-zone event sequence.
            for rt in pre:
                self._state_changes(rt, before[rt.riz_id], bar,
                                    rt.riz_id in survived_ids)
            confirmed_now: dict[str, dict] = {}
            for rt in survivors:
                key = zone_key(rt.zone)
                if blue_confirmed(rt.zone):
                    if key not in self.was_blue:
                        row = self.add_event(rt, "ladder", "confirmation", bar.close_ts_ns,
                                             bar.close_ts_ns, bar.minute_stop - 1, bar,
                                             occurrence=ladder.get(rt.riz_id, {}).get("occurrence", 0))
                        confirmed_now[rt.riz_id] = row
                        rt.qualifies = True
                        self._set_t0(rt, row, bar.close, "confirmation")
                        if rt.first_confirmation_ts_ns is None:
                            rt.first_confirmation_ts_ns = bar.close_ts_ns
                            rt.first_confirmation_spine_pos = bar.minute_stop - 1
                            rt.first_confirmation_native_bar_index = bar.index
                    self.was_blue.add(key)
                else:
                    self.was_blue.discard(key)

            terminal_cancels: dict[str, dict] = {}
            for rt in pre:
                state = ladder.get(rt.riz_id)
                if not state or not state["fired"] or rt.riz_id in confirmed_now:
                    continue
                if not state["active"]:
                    continue
                alive = zone_key(rt.zone) in alive_keys
                if state["full_truth"] and alive:
                    continue
                terminal_cancels[rt.riz_id] = self.add_event(
                    rt, "ladder", "cancellation", bar.close_ts_ns, bar.close_ts_ns,
                    bar.minute_stop - 1, bar, occurrence=state["occurrence"], terminal=True,
                    cancelled_stage=state["stage"],
                )

            for rt in pre:
                if rt.riz_id in ladder:
                    self._classify_preview_events(ladder[rt.riz_id],
                                                  confirmed_now.get(rt.riz_id),
                                                  terminal_cancels.get(rt.riz_id), bar)
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
                    self.add_event(rt, "zone_state", "birth_known", bar.close_ts_ns,
                                   bar.close_ts_ns, bar.minute_stop - 1, bar)
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


def materializer_code_identity() -> str:
    source_dir = Path(__file__).resolve().parent
    code = hashlib.sha256()
    for name in ("fast.py", "identity.py", "machine.py", "native.py", "schema.py"):
        code.update(name.encode())
        code.update((source_dir / name).read_bytes())
    return code.hexdigest()


def build_identity(market: MarketSpine, instrument: str, tf_minutes: int) -> str:
    value = (f"{CELL_SCHEMA_VERSION}|{SEMANTIC_VERSION}|{materializer_code_identity()}|"
             f"{market.manifest['corpus_id']}|{instrument}|{tf_minutes}")
    return hashlib.sha256(value.encode()).hexdigest()


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
            "materializer_code_identity": materializer_code_identity(),
            "market_corpus_id": market.manifest["corpus_id"], "instrument": instrument,
            "tf_minutes": tf_minutes, "market_rows": market.manifest["rows"],
            "source_first_close_utc_ns": market.manifest["first_close_utc_ns"],
            "source_last_close_utc_ns": market.manifest["last_close_utc_ns"],
            "native_bars": native_bars, "zones_born": zones_born,
            "passports": len(passports), "events": len(events),
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
