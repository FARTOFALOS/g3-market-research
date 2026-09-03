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


def test_t0_events_group_rows_without_losing_layers(nq_field):
    """Rows are not events: the minute path belongs to the moment, not the TF."""
    passports = nq_field.passports(tf=15)
    events = nq_field.t0_events(passports)
    assert events.num_rows <= passports.num_rows
    assert int(events["n_layers"].to_numpy().sum()) == passports.num_rows
    flat = [r for row in events["riz_ids"].to_pylist() for r in row]
    assert sorted(flat) == sorted(passports["riz_id"].to_pylist())


def test_t0_events_carry_one_exit_side_per_minute(nq_field):
    """The empirical invariant the 008 corpus rests on, re-checked on a slice."""
    passports = nq_field.passports(tf_min=10, tf_max=20)
    events = nq_field.t0_events(passports)
    pos = events["t0_spine_pos"].to_numpy()
    assert len(set(pos.tolist())) == len(pos), "a T0 minute carried both sides"
