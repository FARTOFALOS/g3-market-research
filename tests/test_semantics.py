from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from g3riz.fast import facts_to_rows, replay_fast
from g3riz.field import CellBuilder
from g3riz.machine import (TIER_ACTIVE_BOTH, TIER_BREAKER, NativeBar, Zone,
                           advance_one, blue_confirmed, detect_births)
from g3riz.market import _day_numbers
from g3riz.native import NativeArrays


def bar(index, o, h, low, c, start=0, stop=1):
    return NativeBar(index, index * 300, index * 300 + 299, o, h, low, c, start, stop)


def test_session_day_conversion_is_explicitly_nanosecond_safe():
    values = pd.Series(pd.to_datetime(["2006-01-05 17:59", "2006-01-05 18:00"]) \
                       .tz_localize("US/Eastern"))
    days = _day_numbers(values)
    assert days[1] == days[0] + 1
    assert days[0] > 10_000


def test_machine_strict_span_spacing_and_erratum_path():
    z = Zone(20_010.0, 20_000.0, 0, True)
    assert advance_one(z, bar(10, 19_995, 20_016, 19_994, 20_015))
    assert z.tier == TIER_ACTIVE_BOTH and z.span_count == 1
    assert not blue_confirmed(z)
    assert not advance_one(z, bar(11, 20_015, 20_016, 19_994, 19_990))
    assert not z.north_alive and not z.south_alive


def test_breaker_return_is_true_deletion():
    z = Zone(20_010, 20_000, 0, True, north_alive=False, south_alive=False,
             activated=True, tier=TIER_BREAKER, broke_south=True)
    assert advance_one(z, bar(11, 19_990, 19_998, 19_980, 19_985))
    assert not advance_one(z, bar(12, 19_990, 20_001, 19_989, 19_995))


def test_birth_geometry_and_label_are_deterministic():
    c1 = bar(0, 100, 103, 99, 102)
    c2 = bar(1, 104, 109, 103, 107)
    c3 = bar(2, 110, 113, 108, 112)
    born = detect_births(c1, c2, c3)
    assert len(born) == 1
    assert (born[0].bottom, born[0].top, born[0].birth_label_ts_ns) == (102, 110, c1.open_ts_ns)


def _synthetic_market_and_bars():
    # Same load-bearing sequence as the late G2 ladder fixture: birth, native
    # activation, then a minute preview and a separate native confirmation.
    o = np.array([100, 104, 110, 100, 99, 112, 112, 112, 111, 99], dtype=float)
    h = np.array([103, 109, 113, 101, 112, 113, 114, 112.5, 111, 100], dtype=float)
    low = np.array([99, 103, 108, 99, 99, 111, 111.5, 111, 99, 98], dtype=float)
    close = np.array([102, 107, 112, 99, 112, 112, 113, 111, 99, 99], dtype=float)
    ts = np.arange(1, 11, dtype=np.int64) * 60_000_000_000
    market = SimpleNamespace(
        manifest={"corpus_id": "fixture"}, open=o, high=h, low=low, close=close,
        close_ts_utc_ns=ts,
    )
    bars = [
        NativeBar(0, 0, int(ts[0]), 100, 103, 99, 102, 0, 1),
        NativeBar(1, 300, int(ts[1]), 104, 109, 103, 107, 1, 2),
        NativeBar(2, 600, int(ts[2]), 110, 113, 108, 112, 2, 3),
        NativeBar(3, 900, int(ts[5]), 100, 113, 99, 112, 3, 6),
        NativeBar(4, 1200, int(ts[6]), 112, 114, 111.5, 113, 6, 7),
        NativeBar(5, 1500, int(ts[9]), 112, 112.5, 98, 99, 7, 10),
    ]
    return market, bars


def test_two_clocks_precursor_and_direct_addresses_survive():
    market, bars = _synthetic_market_and_bars()
    builder = CellBuilder(market, "NQ", 5)
    with patch("g3riz.field.iter_native_bars", return_value=iter(bars)):
        passports, events, native_count, born = builder.run()
    assert native_count == 6 and born == 1 and len(passports) == 1
    passport = passports[0]
    assert passport["birth_label_ts_ns"] == bars[0].open_ts_ns
    assert passport["birth_known_ts_ns"] == bars[2].close_ts_ns
    assert passport["first_activation_native_bar_index"] == 3
    assert passport["t0_kind"] == "preview"
    assert passport["t0_spine_pos"] == 8
    assert passport["t0_native_bar_index"] == 5
    assert passport["first_confirmation_native_bar_index"] == 5
    assert passport["t0_preview_outcome"] == "confirmed"
    assert passport["t0_preview_outcome_known_at_ns"] == bars[5].close_ts_ns
    assert passport["t0_preview_outcome_known_at_spine_pos"] == bars[5].minute_stop - 1
    assert passport["censored"]
    kinds = [event["event_kind"] for event in events]
    assert events[0]["event_ts_ns"] == passport["birth_known_ts_ns"]
    assert events[0]["known_at_ns"] == passport["birth_known_ts_ns"]
    assert passport["birth_label_ts_ns"] < passport["birth_known_ts_ns"]
    assert kinds == ["birth_known", "candidate", "activation", "span",
                     "preview", "span", "confirmation", "archive_censor"]
    preview = next(event for event in events if event["event_kind"] == "preview")
    assert preview["known_at_ns"] < preview["preview_outcome_known_at_ns"]
    assert all(event["riz_id"] == passport["riz_id"] for event in events)


def test_compiled_replay_is_logically_identical_on_two_clock_fixture():
    market, bars = _synthetic_market_and_bars()
    reference = CellBuilder(market, "NQ", 5)
    with patch("g3riz.field.iter_native_bars", return_value=iter(bars)):
        rp, re, _n, _born = reference.run()
    native = NativeArrays(
        open_ts_ns=np.array([b.open_ts_ns for b in bars], dtype=np.int64),
        close_ts_ns=np.array([b.close_ts_ns for b in bars], dtype=np.int64),
        open=np.array([b.open for b in bars]), high=np.array([b.high for b in bars]),
        low=np.array([b.low for b in bars]), close=np.array([b.close for b in bars]),
        minute_start=np.array([b.minute_start for b in bars], dtype=np.int64),
        minute_stop=np.array([b.minute_stop for b in bars], dtype=np.int64),
    )
    facts = replay_fast(native, market.open, market.close, market.close_ts_utc_ns)
    fp, fe = facts_to_rows(facts, "fixture", "NQ", 5,
                           int(market.close_ts_utc_ns[-1]), len(market.close) - 1,
                           native.close_ts_ns, native.minute_stop)
    assert fp == rp
    assert fe == re


def test_indexed_replay_preserves_breaker_deletion_across_a_price_gap():
    market, bars = _synthetic_market_and_bars()
    extra_o = np.array([105.0, 105.0, 105.0])
    extra_h = np.array([111.0, 106.0, 106.0])
    extra_l = np.array([105.0, 99.0, 104.0])
    extra_c = np.array([106.0, 100.0, 105.0])
    market.open = np.r_[market.open, extra_o]
    market.high = np.r_[market.high, extra_h]
    market.low = np.r_[market.low, extra_l]
    market.close = np.r_[market.close, extra_c]
    market.close_ts_utc_ns = np.arange(1, 14, dtype=np.int64) * 60_000_000_000
    for j in range(3):
        p = 10 + j
        bars.append(NativeBar(6 + j, 1800 + j * 300,
                              int(market.close_ts_utc_ns[p]), extra_o[j],
                              extra_h[j], extra_l[j], extra_c[j], p, p + 1))
    reference = CellBuilder(market, "NQ", 5)
    with patch("g3riz.field.iter_native_bars", return_value=iter(bars)):
        rp, re, _n, _born = reference.run()
    native = NativeArrays(
        open_ts_ns=np.array([b.open_ts_ns for b in bars], dtype=np.int64),
        close_ts_ns=np.array([b.close_ts_ns for b in bars], dtype=np.int64),
        open=np.array([b.open for b in bars]), high=np.array([b.high for b in bars]),
        low=np.array([b.low for b in bars]), close=np.array([b.close for b in bars]),
        minute_start=np.array([b.minute_start for b in bars], dtype=np.int64),
        minute_stop=np.array([b.minute_stop for b in bars], dtype=np.int64),
    )
    facts = replay_fast(native, market.open, market.close, market.close_ts_utc_ns)
    fp, fe = facts_to_rows(facts, "fixture", "NQ", 5,
                           int(market.close_ts_utc_ns[-1]), len(market.close) - 1,
                           native.close_ts_ns, native.minute_stop)
    assert fp == rp and fe == re
    assert not fp[0]["censored"]
    assert fp[0]["c1_deletion_native_bar_index"] == 8


def test_one_minute_clock_has_confirmation_but_no_intrabar_preview():
    market, bars = _synthetic_market_and_bars()
    # Bar membership is already one minute for the first three; the semantic
    # fact under test is structural: a one-minute native bar has no pre-close
    # observation position and therefore cannot emit candidate/preview.
    one_minute_bars = [NativeBar(b.index, b.open_ts_ns, b.close_ts_ns, b.open,
                                 b.high, b.low, b.close, b.minute_start,
                                 b.minute_start + 1) for b in bars]
    builder = CellBuilder(market, "NQ", 1)
    with patch("g3riz.field.iter_native_bars", return_value=iter(one_minute_bars)):
        _passports, events, _native_count, _born = builder.run()
    assert not any(event["event_kind"] in ("candidate", "preview") for event in events)
