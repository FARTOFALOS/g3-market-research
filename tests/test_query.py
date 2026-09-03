"""What the read surface promises: no partial population, no invented minute.

Both failures here have the same dangerous shape — they return numbers that
look perfectly normal. An incomplete field read and an out-of-tape window can
each turn a tool failure into a market answer, which is the one thing the read
contract must not allow.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import needs_field


@needs_field
def test_a_whole_population_read_returns_every_timeframe(nq_field):
    """Asking without a timeframe asks for all 1,440 cells, not for some."""
    paths = nq_field._fact_files("passports", None)
    assert len(paths) == 1440
    assert len(set(paths)) == 1440


@needs_field
def test_an_incomplete_field_refuses_instead_of_answering_with_what_it_found(
        nq_field, tmp_path, monkeypatch):
    """1,439 good cells plus one stale one is a tool failure, not a population.

    Silently dropping the unreadable cell would hand back a normal Arrow table
    and normal statistics over a population nobody chose.
    """
    monkeypatch.setattr(nq_field, "field_root", tmp_path / "empty")
    with pytest.raises(FileNotFoundError, match="incomplete current field"):
        nq_field._fact_files("passports", None)


@needs_field
def test_a_named_timeframe_that_is_not_current_refuses(
        nq_field, tmp_path, monkeypatch):
    monkeypatch.setattr(nq_field, "field_root", tmp_path / "empty")
    with pytest.raises(FileNotFoundError, match="no current complete field cell"):
        nq_field._fact_files("passports", 10)


@needs_field
def test_a_window_off_the_start_of_the_tape_carries_no_plausible_minute(nq_field):
    """Index zero is a gather placeholder; it must never reach the caller.

    A caller who forgets `valid` should get an impossible value, not the first
    minute of 2006 wearing the costume of a real observation.
    """
    window = nq_field.minute_windows(np.array([2]), before=5, after=2)
    invalid = ~window["valid"][0]
    assert invalid.sum() == 3

    assert np.all(window["close_ts_utc_ns"][0][invalid] == -1)
    assert np.all(window["volume"][0][invalid] == -1)
    for name in ("open", "high", "low", "close"):
        assert np.all(np.isnan(window[name][0][invalid]))

    # The real minutes are untouched.
    assert np.all(window["close_ts_utc_ns"][0][~invalid] > 0)
    assert np.all(window["volume"][0][~invalid] >= 0)


@needs_field
def test_a_window_off_the_end_of_the_tape_is_missing_the_same_way(nq_field):
    last = len(nq_field.market.close) - 1
    window = nq_field.minute_windows(np.array([last - 1]), before=1, after=4)
    invalid = ~window["valid"][0]
    assert invalid.any()
    assert np.all(window["close_ts_utc_ns"][0][invalid] == -1)
    assert np.all(np.isnan(window["close"][0][invalid]))


@needs_field
def test_t0_minute_groups_keep_every_layer(nq_field):
    """Grouping by T0 minute is a diagnostic and must lose no RIZ identity."""
    passports = nq_field.passports(tf=15)
    groups = nq_field.t0_minute_groups(passports)
    assert groups.num_rows <= passports.num_rows
    assert int(groups["n_layers"].to_numpy().sum()) == passports.num_rows
    flat = [r for row in groups["riz_ids"].to_pylist() for r in row]
    assert sorted(flat) == sorted(passports["riz_id"].to_pylist())


@needs_field
def test_t0_minute_groups_carry_one_exit_side_per_minute(nq_field):
    """The empirical invariant the 008 corpus rests on, re-checked on a slice."""
    passports = nq_field.passports(tf_min=10, tf_max=20)
    groups = nq_field.t0_minute_groups(passports)
    pos = groups["t0_spine_pos"].to_numpy()
    assert len(set(pos.tolist())) == len(pos), "a T0 minute carried both sides"


@needs_field
def test_a_shared_t0_minute_is_not_a_shared_exit_boundary(nq_field):
    """The guard against re-collapsing films by T0 minute.

    A group of layers on one minute is not one film, because the line each
    film is about — its exit boundary — differs across the layers. If this
    ever stops finding such minutes on a slice this large, the field changed
    and every per-minute grouping in the repository needs re-reading.
    """
    passports = nq_field.passports(tf_min=30, tf_max=90)
    groups = nq_field.t0_minute_groups(passports)
    boundary = {
        rid: (top if side == "north" else bottom)
        for rid, side, top, bottom in zip(
            passports["riz_id"].to_pylist(),
            passports["t0_exit_side"].to_pylist(),
            passports["zone_top"].to_pylist(),
            passports["zone_bottom"].to_pylist())
    }
    multi = shared = split = 0
    for ids in groups["riz_ids"].to_pylist():
        if len(ids) < 2:
            continue
        multi += 1
        if len({boundary[r] for r in ids}) > 1:
            split += 1
        else:
            shared += 1
    assert multi > 100, "no multi-layer T0 minutes in this slice"
    assert split > 0, "every shared T0 minute shared one exit boundary"
    assert shared > 0, "the reverse case must also exist"
