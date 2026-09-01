"""What the thirteen core tracks promise: independence, and no look-ahead."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import make_film, needs_field
from g3riz.tracks import CORE_TRACKS, core_tracks


def test_one_minute_answers_several_questions_at_once():
    # Rising highs with falling closes: the minute raises its high AND closes
    # down. Neither reading is the minute's "real" symbol.
    film = make_film([5, 6, 7], [1, 1, 1], open_=[4, 4, 4], close=[3, 2, 1])
    t = core_tracks(film)
    assert t["H"].tolist() == ["?", "+", "+"]
    assert t["C"].tolist() == ["?", "-", "-"]
    assert t["BODY"].tolist() == ["down", "down", "down"]


def test_every_core_track_is_present_and_named():
    film = make_film([5, 6], [1, 2])
    assert sorted(core_tracks(film)) == sorted(CORE_TRACKS)
    assert len(CORE_TRACKS) == 13


def test_the_first_minute_is_undefined_never_imputed():
    t = core_tracks(make_film([5, 6, 7], [1, 2, 3]))
    assert t["H"][0] == "?"
    assert t["OVERLAP"][0] == "?"
    assert np.isnan(t["RANGE_D"][0])
    assert np.isnan(t["BODY_D"][0])


def test_running_extremes_start_at_t0_not_at_the_film_start():
    # The pre-roll high of 100 is the highest bar in the film, yet the first
    # post-T0 minute still sets a running maximum, because the run is measured
    # from T0.
    film = make_film([100, 5, 6], [90, 1, 2], pre_roll=1)
    t = core_tracks(film)
    assert t["RUN_MAX"].tolist() == [False, True, True]


def test_excursion_is_measured_from_the_t0_close_in_the_exit_direction():
    film = make_film([5, 9, 7], [1, 2, 3], close=[4, 8, 6], top=10.0, bottom=0.0)
    t = core_tracks(film)
    assert t["EXC_PTS"].tolist() == [1.0, 5.0, 3.0]
    assert t["EXC_W"].tolist() == [0.1, 0.5, 0.3]

    down = make_film([5, 9, 7], [1, 2, 3], close=[4, 8, 6], exit_up=False)
    assert core_tracks(down)["EXC_PTS"].tolist() == [3.0, 2.0, 1.0]


def test_gaps_and_overlaps_are_told_apart():
    film = make_film([5, 9, 3], [4, 6, 1])
    assert core_tracks(film)["OVERLAP"].tolist() == ["?", "gap_up", "gap_down"]


def test_every_core_track_is_stable_under_truncation():
    """The point-in-time property, demonstrated rather than asserted.

    A prefix function cannot read a later bar, so reading it on a film cut at
    `p` must give exactly what reading the whole film gave up to `p`. This is
    what buys core tracks out of the per-read rebuild that a retrospective
    object costs.
    """
    rng = np.random.default_rng(7)
    low = np.cumsum(rng.normal(size=60)) + 20
    high = low + np.abs(rng.normal(size=60)) + 0.5
    film = make_film(high, low, pre_roll=5, top=22.0, bottom=18.0)
    full = core_tracks(film)

    for cursor in (0, 1, 7, 23, 54):
        seen = core_tracks(film.truncate(cursor))
        keep = int(np.searchsorted(film.bar_ord, cursor, side="right"))
        for name in CORE_TRACKS:
            a, b = full[name][:keep], seen[name]
            assert a.shape == b.shape, name
            if a.dtype.kind == "f":
                assert np.array_equal(a, b, equal_nan=True), name
            else:
                assert np.array_equal(a, b), name


@needs_field
def test_core_tracks_are_stable_under_truncation_on_real_films(nq_field):
    passports = nq_field.passports(tf=10).slice(0, 40)
    checked = 0
    for film in nq_field.films(passports, stop="deletion", max_bars=400):
        if film.n < 20:
            continue
        full = core_tracks(film)
        cursor = int(film.bar_ord[film.n // 2])
        seen = core_tracks(film.truncate(cursor))
        keep = int(np.searchsorted(film.bar_ord, cursor, side="right"))
        for name in CORE_TRACKS:
            a, b = full[name][:keep], seen[name]
            if a.dtype.kind == "f":
                assert np.array_equal(a, b, equal_nan=True), (film.riz_id, name)
            else:
                assert np.array_equal(a, b), (film.riz_id, name)
        checked += 1
    assert checked > 0, "no real film was long enough to check"
