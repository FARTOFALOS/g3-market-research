"""What a film promises: one T0, an end it can explain, and no future in it."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import make_film, needs_field
from g3riz.film import PIT_TRUNCATION, build_film, film_bounds


def test_t0_is_ordinal_zero_and_pre_roll_is_negative():
    film = make_film([5, 6, 7, 8], [1, 2, 3, 4], pre_roll=2)
    assert film.bar_ord.tolist() == [-2, -1, 0, 1]
    assert film.post == slice(2, 4)
    assert film.high[film.post].tolist() == [7.0, 8.0]


def test_a_budget_that_bites_says_so_instead_of_looking_like_a_death():
    row = {"riz_id": "r", "t0_spine_pos": 100, "c1_deletion_spine_pos": 5000,
           "last_observed_spine_pos": 9000, "censored": False}
    start, t0, end, reason = film_bounds(row, "deletion", 0, 240, 10_000)
    assert (start, t0, end) == (100, 100, 339)
    assert reason == "observation_budget"


def test_a_stop_the_field_never_recorded_falls_back_and_admits_it():
    row = {"riz_id": "r", "t0_spine_pos": 10, "c1_deletion_spine_pos": None,
           "last_observed_spine_pos": 900, "censored": True}
    _start, _t0, end, reason = film_bounds(row, "deletion", 0, None, 10_000)
    assert (end, reason) == (900, "archive_edge")


def test_a_riz_alive_at_the_archive_edge_is_censored_whatever_column_was_asked():
    row = {"riz_id": "r", "t0_spine_pos": 10,
           "blue_eligibility_end_spine_pos": 800,
           "last_observed_spine_pos": 800, "censored": True}
    _start, _t0, end, reason = film_bounds(row, "blue_end", 0, None, 10_000)
    assert (end, reason) == (800, "archive_edge")


def test_a_reached_stop_reports_none():
    row = {"riz_id": "r", "t0_spine_pos": 10, "c1_deletion_spine_pos": 60,
           "last_observed_spine_pos": 900, "censored": False}
    _start, _t0, end, reason = film_bounds(row, "deletion", 0, None, 10_000)
    assert (end, reason) == (60, "none")


def test_an_unknown_stop_names_what_it_knows_and_points_at_the_study_route():
    row = {"riz_id": "r", "t0_spine_pos": 10, "last_observed_spine_pos": 90}
    with pytest.raises(ValueError, match="end_position"):
        film_bounds(row, "first_retest", 0, None, 10_000)


def test_a_study_defined_end_must_carry_the_reason_it_cut_there():
    spine = _spine(200)
    row = _row()
    with pytest.raises(ValueError, match="end_reason"):
        build_film(spine, row, end_position=150)
    film = build_film(spine, row, end_position=150,
                      end_reason="first_close_inside")
    assert film.stop == "first_close_inside"
    assert film.end_reason == "first_close_inside"
    # The study's end was asked for and observed, so the film reached its stop.
    assert film.reached_its_stop
    assert film.spine_pos[-1] == 150


def test_an_end_before_t0_is_refused_not_quietly_raised_to_t0():
    """Coercion would hand back a one-bar film still labelled with the event."""
    with pytest.raises(ValueError, match="before T0"):
        build_film(_spine(200), _row(), end_position=50,
                   end_reason="first_close_inside")


def test_a_study_end_the_tape_never_reached_reports_the_tape_not_the_event():
    """The invariant: an end that was not observed is never claimed as reached.

    Otherwise a film that never saw its retest would still name the retest, and
    a censored episode would enter a study as an observed one. Refusing outright
    would honour the same invariant; this build instead returns the film with
    the reason that actually ended observation, which the last two assertions
    record as this build's choice rather than as the requirement.
    """
    film = build_film(_spine(200), _row(), end_position=5_000,
                      end_reason="first_close_inside")
    assert not film.reached_its_stop
    assert film.end_reason != "first_close_inside"

    assert film.end_reason == "archive_edge"
    assert film.spine_pos[-1] == 199


def test_a_budget_outranks_a_study_end_and_says_so():
    film = build_film(_spine(200), _row(), end_position=150,
                      end_reason="first_close_inside", max_bars=10)
    assert not film.reached_its_stop
    assert film.end_reason != "first_close_inside"
    assert film.end_reason == "observation_budget"


def test_an_exit_side_outside_its_closed_domain_is_refused():
    """Unknown must not become "not north" and silently invert excursions."""
    row = _row() | {"t0_exit_side": "sideways"}
    with pytest.raises(ValueError, match="t0_exit_side"):
        build_film(_spine(200), row)


def test_truncation_removes_the_future_from_the_metadata_too():
    film = make_film(np.arange(10.0) + 5, np.arange(10.0),
                     pre_roll=2, stop="deletion", end_reason="none")
    seen = film.truncate(3)

    assert seen.n == 6
    assert seen.bar_ord.tolist() == [-2, -1, 0, 1, 2, 3]
    assert seen.n_pre_roll == 2
    # On minute 3 nobody knew this RIZ would be deleted, or that the film would
    # reach its stop at all. Both statements are gone.
    assert seen.stop == PIT_TRUNCATION
    assert seen.end_reason == PIT_TRUNCATION
    assert not seen.reached_its_stop
    # What T0 already established survives.
    assert (seen.riz_id, seen.tf_minutes) == (film.riz_id, film.tf_minutes)
    assert (seen.zone_top, seen.zone_bottom) == (film.zone_top, film.zone_bottom)
    assert seen.exit_up == film.exit_up


def test_truncating_at_the_end_changes_nothing():
    film = make_film([5, 6, 7], [1, 2, 3])
    assert film.truncate(2) is film


def test_truncating_before_the_film_is_refused_not_guessed():
    film = make_film([5, 6, 7], [1, 2, 3], pre_roll=1)
    with pytest.raises(ValueError, match="before the film starts"):
        film.truncate(-5)


def test_signed_distances_are_orientation_free():
    film = make_film([12, 8], [11, 2], open_=[11.5, 3.0], close=[11.8, 7.0],
                     top=10.0, bottom=0.0)
    d = film.signed_distances()
    assert d["d_high_top"].tolist() == [2.0, -2.0]
    assert d["d_low_bottom"].tolist() == [11.0, 2.0]
    assert d["d_body_max_top"].tolist() == pytest.approx([1.8, -3.0])
    # Nothing in here consults exit_up.
    flipped = make_film([12, 8], [11, 2], open_=[11.5, 3.0],
                        close=[11.8, 7.0], exit_up=False)
    for key, value in flipped.signed_distances().items():
        assert value.tolist() == d[key].tolist()


@needs_field
def test_a_real_film_starts_at_t0_and_ends_where_the_passport_says(nq_field):
    passports = nq_field.passports(tf=10).slice(0, 50)
    rows = passports.to_pylist()
    films = list(nq_field.films(passports, stop="deletion"))
    assert len(films) == 50
    for row, film in zip(rows, films):
        assert film.spine_pos[film.n_pre_roll] == row["t0_spine_pos"]
        assert film.bar_ord[film.n_pre_roll] == 0
        assert film.close[film.n_pre_roll] == pytest.approx(row["t0_close"])
        if row["c1_deletion_spine_pos"] is not None and not row["censored"]:
            assert film.spine_pos[-1] == row["c1_deletion_spine_pos"]
            assert film.end_reason == "none"


@needs_field
def test_a_budget_bounds_a_real_film_and_labels_it(nq_field):
    passports = nq_field.passports(tf=10).slice(0, 200)
    films = list(nq_field.films(passports, stop="deletion", max_bars=240))
    assert films
    for film in films:
        assert film.n - film.n_pre_roll <= 240
    assert any(f.end_reason == "observation_budget" for f in films)


def _spine(n: int):
    class Spine:
        close_ts_utc_ns = np.arange(n, dtype=np.int64)
        open = np.ones(n)
        high = np.ones(n) * 2
        low = np.zeros(n)
        close = np.ones(n)
        volume = np.ones(n, dtype=np.int64)
        session_id = np.zeros(n, dtype=np.int32)

    return Spine()


def _row():
    return {"riz_id": "r", "instrument": "NQ", "tf_minutes": 10,
            "zone_top": 10.0, "zone_bottom": 0.0, "t0_spine_pos": 100,
            "t0_exit_side": "north", "c1_deletion_spine_pos": 190,
            "last_observed_spine_pos": 199, "censored": False}
