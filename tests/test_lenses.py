"""What the lenses promise: four different questions, and an honest swing delay."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import make_film, needs_field
from g3riz.lenses import interaction, swing


def test_the_four_interaction_predicates_are_not_a_ladder():
    """A body can span the whole zone without closing inside it.

    The previous repository shipped these as a strictness ladder and then
    measured that the order does not exist. One minute is enough to show it:
    an engulfing bar whose body runs from below the zone to above it is a
    full traversal and is not a close-inside.
    """
    film = make_film([12.0], [-2.0], open_=[-1.0], close=[11.0],
                     top=10.0, bottom=0.0)
    assert interaction.full_traversal_v1(film).tolist() == [True]
    assert interaction.close_inside_v1(film).tolist() == [False]

    # And the converse: a close inside a zone the body never spanned.
    inside = make_film([6.0], [4.0], open_=[4.5], close=[5.5],
                       top=10.0, bottom=0.0)
    assert interaction.close_inside_v1(inside).tolist() == [True]
    assert interaction.full_traversal_v1(inside).tolist() == [False]


def test_a_wick_touch_is_not_a_body_entry():
    # The wick reaches down into the zone; the body stays entirely above it.
    film = make_film([11.0], [9.5], open_=[10.9], close=[10.8],
                     top=10.0, bottom=0.0)
    assert interaction.shadow_touch_v1(film).tolist() == [True]
    assert interaction.body_entry_v1(film).tolist() == [False]
    assert interaction.close_inside_v1(film).tolist() == [False]


def test_an_engagement_is_a_run_and_its_onset_is_its_first_minute():
    mask = np.array([False, True, True, False, False, True, False])
    onsets, lengths = interaction.runs_v1(mask)
    assert onsets.tolist() == [1, 5]
    assert lengths.tolist() == [2, 1]


def test_no_interaction_at_all_is_zero_runs_not_an_error():
    onsets, lengths = interaction.runs_v1(np.zeros(5, dtype=bool))
    assert onsets.size == 0 and lengths.size == 0


def test_a_swing_belongs_to_its_pivot_and_is_knowable_one_bar_later():
    film = make_film([1, 5, 2, 3], [0, 4, 1, 2])
    is_high, _is_low = swing.strict3_v1(film)
    assert np.flatnonzero(is_high).tolist() == [1]
    assert swing.confirmed_at(np.array([1])).tolist() == [2]


def test_the_ends_of_a_film_cannot_be_swings():
    film = make_film([1, 5, 2], [0, 4, 1])
    is_high, is_low = swing.strict3_v1(film)
    assert not is_high[0] and not is_high[-1]
    assert not is_low[0] and not is_low[-1]


def test_a_tie_confirms_nothing():
    film = make_film([1, 5, 5, 2], [0, 4, 4, 1])
    is_high, _ = swing.strict3_v1(film)
    assert not is_high.any()


def test_a_film_shorter_than_the_window_has_no_swing():
    is_high, is_low = swing.strict3_v1(make_film([1, 2], [0, 1]))
    assert not is_high.any() and not is_low.any()


def test_vs_last_swing_reads_the_confirmation_never_the_pivot():
    film = make_film([1, 5, 2, 6, 3], [0, 4, 1, 5, 2])
    tracks = swing.vs_last_swing_v1(film)
    # No swing is confirmed until index 2, so the first two minutes are
    # undefined rather than compared against a pivot nobody could see yet.
    assert tracks["VS_LAST_SH"][:2].tolist() == ["?", "?"]
    assert tracks["VS_LAST_SH"][2] == "-"      # high 2 against confirmed 5
    assert tracks["VS_LAST_SH"][3] == "+"      # high 6 against confirmed 5


def test_vs_last_swing_is_stable_under_truncation():
    rng = np.random.default_rng(11)
    low = np.cumsum(rng.normal(size=80)) + 30
    high = low + np.abs(rng.normal(size=80)) + 0.3
    film = make_film(high, low)
    full = swing.vs_last_swing_v1(film)
    for cursor in (5, 19, 44, 70):
        seen = swing.vs_last_swing_v1(film.truncate(cursor))
        keep = cursor + 1
        for name in ("VS_LAST_SH", "VS_LAST_SL"):
            assert np.array_equal(full[name][:keep], seen[name]), (name, cursor)


@needs_field
def test_every_predicate_runs_on_real_films_and_answers_per_minute(nq_field):
    """An integration sentinel, deliberately carrying no market assertion.

    That the four predicates encode different definitions is proved above, by
    synthetic counterexamples that cannot depend on which corpus is loaded.
    Asserting here that their aggregate counts differ on NQ would put an
    empirical property of one corpus inside the test suite, where it has no
    territory, denominator or rerun path — and where a green run would slowly
    start reading as evidence about the market. Equal counts on some corpus
    would not make two definitions the same, and different counts would confirm
    nothing.

    So this checks only what a substrate test can honestly check: real films
    build, every predicate executes on them, and each answers once per minute.
    """
    passports = nq_field.passports(tf=10).slice(0, 300)
    films = 0
    for film in nq_field.films(passports, stop="deletion", max_bars=500):
        films += 1
        for name, fn in interaction.V1.items():
            answers = fn(film)
            assert answers.shape == (film.n,), (name, film.riz_id)
            assert answers.dtype == bool, name
    assert films > 0


# --- legs -----------------------------------------------------------------

def _closes(values, exit_up=True):
    """A film whose closes are exactly `values`; bodies sit inside the bars."""
    z = np.asarray(values, dtype=float)
    return make_film(z + 0.5, z - 0.5, close=z, exit_up=exit_up)


def test_a_clean_run_produces_no_leg_at_all():
    """The defect this file exists to keep out.

    Price that only travels has ended no run, so there is nothing to segment.
    The original lens measured the give-back against an extreme still pinned at
    the T0 close, so it emitted a leg of amplitude exactly zero pointing against
    the move — on 79% of NQ films at theta = 0.5 ATR. That fabricated symbol sat
    first in every sequence read off this lens, made the first real leg the
    second symbol, and made the first leg-size ratio a division by zero.
    """
    from g3riz.lenses.legs import directional_change_v1

    for path in ([0, 1, 2, 3, 4], [0, -1, -2, -3, -4]):
        legs = directional_change_v1(_closes(path), theta=1.0)
        assert len(legs) == 0, path
        assert legs.opening_turn_ord == -1, path


def test_the_span_from_t0_to_the_first_turn_is_not_a_leg():
    """It is travel no give-back has measured, so it is reported apart."""
    from g3riz.lenses.legs import directional_change_v1

    # up 3, then back 1.1: one turn, and it opens the sequence rather than
    # being a leg of it.
    legs = directional_change_v1(_closes([0, 3, 1.9]), theta=1.0)
    assert len(legs) == 0
    assert legs.opening_amplitude == 3.0
    assert (legs.opening_turn_ord, legs.opening_conf_ord) == (1, 2)

    # and the first real leg starts at that turn, not at T0
    legs = directional_change_v1(_closes([0, 2, 0.75, 1.75]), theta=1.0)
    assert len(legs) == 1
    assert legs.opening_turn_ord == 1
    assert legs.start_ord.tolist() == [1]
    assert legs.direction.tolist() == [-1]
    assert legs.amplitude.tolist() == [-1.25]
    assert legs.turn_ord.tolist() == [2] and legs.conf_ord.tolist() == [3]


def test_one_tick_of_noise_cannot_open_a_leg():
    """A tick up before a large move down must not become a leg of that tick.

    Freezing the run direction at the first bar the extremes separate does
    exactly that. Starting the sequence at the first turn does not.
    """
    from g3riz.lenses.legs import directional_change_v1

    legs = directional_change_v1(_closes([0, 0.1, -5]), theta=1.0)
    assert len(legs) == 0
    assert legs.opening_amplitude == pytest.approx(0.1)


def test_every_leg_costs_at_least_theta_and_points_the_way_it_moved():
    """Both follow from the definition, so a violation means fabrication."""
    from g3riz.lenses.legs import directional_change_v1

    rng = np.random.default_rng(4)
    for seed_path in rng.normal(size=(40, 90)):
        z = np.cumsum(seed_path)
        for theta in (0.3, 0.8, 1.7):
            legs = directional_change_v1(_closes(z), theta=theta)
            if not len(legs):
                continue
            assert np.all(np.sign(legs.amplitude) == legs.direction)
            assert np.all(np.abs(legs.amplitude) >= theta - 1e-9)
            assert np.all(legs.direction[1:] != legs.direction[:-1])
            assert np.all(legs.confirmation_lag >= 0)
            # a leg begins where the previous one turned; the first at the
            # opening turn, never at T0
            assert legs.start_ord[0] == legs.opening_turn_ord
            assert np.array_equal(legs.start_ord[1:], legs.turn_ord[:-1])


def test_a_turn_is_dated_to_the_earliest_bar_holding_the_extreme():
    """Ties go to the first bar, or a truncated rebuild would disagree."""
    from g3riz.lenses.legs import directional_change_v1

    legs = directional_change_v1(_closes([0, 2, 2, 1, 2, 0.9]), theta=1.0)
    assert legs.opening_turn_ord == 1          # not 2, though both closed at 2
    assert legs.turn_ord.tolist() == [3, 4]
    assert legs.amplitude.tolist() == [-1.0, 1.0]


def test_the_next_run_carries_the_extreme_the_confirming_bar_set():
    """Carrying the ended run's extreme instead dates the next turn too late.

    Here the down leg's true low is at bar 4, and the old lens reported bar 5
    with an amplitude short by 0.1, because it had seeded the low with the high
    the leg had just left.
    """
    from g3riz.lenses.legs import directional_change_v1

    legs = directional_change_v1(_closes([0, 1, 2, 3, 2.4, 2.5, 3.05]), theta=0.5)
    assert legs.turn_ord.tolist() == [4]
    assert legs.amplitude[0] == pytest.approx(-0.6)


@needs_field
def test_legs_alternate_and_confirm_after_the_turn(nq_field):
    from g3riz.lenses.legs import directional_change_v1

    zones = nq_field.passports(tf=15).slice(0, 40)
    seen = 0
    for film in nq_field.films(zones, stop="archive_edge", max_bars=240):
        width = film.width or 1.0
        legs = directional_change_v1(film, theta=0.5 * width)
        if len(legs) < 2:
            continue
        seen += 1
        # a directional change reverses; two legs never point the same way
        assert np.all(legs.direction[1:] != legs.direction[:-1])
        # confirmation is never before the turn it confirms
        assert np.all(legs.confirmation_lag >= 0)
        # legs are ordered in time and confirmed inside the film
        assert np.all(np.diff(legs.conf_ord) > 0)
        assert legs.conf_ord[-1] <= film.bar_ord[-1]
    assert seen >= 5


@needs_field
def test_legs_are_decidable_when_they_confirm(nq_field):
    """The lens claims a leg is knowable at `conf_ord`. Rebuild there and check."""
    from g3riz.lenses.legs import directional_change_v1

    zones = nq_field.passports(tf=15).slice(0, 40)
    checked = 0
    for film in nq_field.films(zones, stop="archive_edge", max_bars=240):
        width = film.width or 1.0
        legs = directional_change_v1(film, theta=0.5 * width)
        if len(legs) < 2:
            continue
        k = 1
        cut = directional_change_v1(film.truncate(int(legs.conf_ord[k])),
                                    theta=0.5 * width)
        assert len(cut) > k
        assert cut.direction[k] == legs.direction[k]
        assert cut.turn_ord[k] == legs.turn_ord[k]
        assert cut.conf_ord[k] == legs.conf_ord[k]
        checked += 1
    assert checked >= 5


@needs_field
def test_legs_never_read_the_pre_roll(nq_field):
    from g3riz.lenses.legs import directional_change_v1

    zones = nq_field.passports(tf=15).slice(0, 20)
    for film in nq_field.films(zones, stop="archive_edge", pre_roll=30,
                               max_bars=240):
        legs = directional_change_v1(film, theta=0.5 * (film.width or 1.0))
        if len(legs):
            assert legs.turn_ord.min() >= 0


# --- the exit boundary and the trader's first film -------------------------


def test_exit_boundary_is_the_side_the_riz_left_through():
    up = make_film([12, 11], [11, 10], top=10.0, bottom=0.0, exit_up=True)
    down = make_film([1, 2], [-2, -1], top=10.0, bottom=0.0, exit_up=False)
    assert up.exit_boundary == 10.0 and up.far_boundary == 0.0
    assert down.exit_boundary == 0.0 and down.far_boundary == 10.0


def test_first_exit_contact_never_returns_the_t0_bar():
    """T0's own range straddles the exit boundary; that is not a contact."""
    from g3riz.lenses.interaction import (exit_boundary_touch_v1,
                                          first_exit_contact_v1)

    # T0 spans the line, +1 stays away, +2 wicks back to it.
    film = make_film(high=[12.0, 8.0, 10.5], low=[9.0, 6.0, 7.0],
                     top=10.0, bottom=0.0, exit_up=True)
    touch = exit_boundary_touch_v1(film)
    assert bool(touch[0]) and not bool(touch[1]) and bool(touch[2])
    assert first_exit_contact_v1(film) == 2


def test_first_exit_contact_is_none_when_price_never_comes_back():
    from g3riz.lenses.interaction import first_exit_contact_v1

    film = make_film(high=[12.0, 9.5, 9.0], low=[9.0, 8.0, 7.0],
                     top=10.0, bottom=0.0, exit_up=True)
    assert first_exit_contact_v1(film) is None


def test_first_exit_contact_counts_a_wick_and_ignores_the_far_side():
    """A wick to the exit line counts; a body through the far side does not."""
    from g3riz.lenses.interaction import first_exit_contact_v1

    # South exit at 0.0. +1 dives further; +2 gaps clean over the zone and
    # sits entirely above the far boundary, so it engages the zone without
    # ever meeting the line this RIZ actually left through.
    film = make_film(high=[1.0, -1.0, 12.0], low=[-2.0, -4.0, 11.0],
                     top=10.0, bottom=0.0, exit_up=False)
    assert first_exit_contact_v1(film) is None


def test_the_exit_boundary_is_not_the_zone():
    """`shadow_touch` fires on the far side too; the exit predicate does not."""
    from g3riz.lenses.interaction import (exit_boundary_touch_v1,
                                          shadow_touch_v1)

    # North exit at 10.0. The last bar sits on the far boundary only.
    film = make_film(high=[12.0, 11.0, 1.0], low=[9.0, 10.5, -1.0],
                     top=10.0, bottom=0.0, exit_up=True)
    assert bool(shadow_touch_v1(film)[2])
    assert not bool(exit_boundary_touch_v1(film)[2])


@needs_field
def test_t0_closes_beyond_its_exit_boundary_and_no_other(nq_field):
    """The one invariant that lets a cold agent find the line without a detector.

    T0 is NOT "a minute candle whose body spanned the zone": the intrabar
    2X preview reads the NATIVE bar's running body, so at TF 54 only ~40% of
    T0 minutes have a minute body that spans the zone. What always holds is
    the close being strictly beyond the recorded exit side.
    """
    import numpy as np

    for tf in (5, 54, 240):
        p = nq_field.passports(tf=tf)
        pos = p["t0_spine_pos"].to_numpy()
        close = np.asarray(nq_field.market.close[pos], dtype=np.float64)
        top = p["zone_top"].to_numpy()
        bottom = p["zone_bottom"].to_numpy()
        north = np.asarray(p["t0_exit_side"]) == "north"
        beyond_exit = np.where(north, close > top, close < bottom)
        beyond_far = np.where(north, close < bottom, close > top)
        assert beyond_exit.all(), f"TF {tf}: a T0 closed inside its own zone"
        assert not beyond_far.any(), f"TF {tf}: a T0 closed beyond both sides"


@needs_field
def test_film1_can_be_cut_at_its_own_first_contact(nq_field):
    """The end-to-end path a study uses: find the contact, cut the film there."""
    from g3riz.lenses.interaction import first_exit_contact_v1

    zones = nq_field.passports(tf=54).slice(0, 60)
    cut = 0
    for film in nq_field.films(zones, stop="archive_edge", max_bars=600):
        ord_ = first_exit_contact_v1(film)
        if ord_ is None:
            continue
        end_pos = int(film.spine_pos[film.n_pre_roll]) + ord_
        one = nq_field.film(film.riz_id, tf=54, end_position=end_pos,
                            end_reason="first_exit_contact_v1")
        assert one.bar_ord[-1] == ord_
        assert one.stop == one.end_reason == "first_exit_contact_v1"
        assert first_exit_contact_v1(one) == ord_
        cut += 1
    assert cut >= 20
