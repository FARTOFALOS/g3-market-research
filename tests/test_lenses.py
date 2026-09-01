"""What the lenses promise: four different questions, and an honest swing delay."""

from __future__ import annotations

import numpy as np

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
def test_the_predicates_disagree_on_the_real_corpus(nq_field):
    passports = nq_field.passports(tf=10).slice(0, 300)
    counts = {name: 0 for name in interaction.PREDICATES}
    films = 0
    for film in nq_field.films(passports, stop="deletion", max_bars=500):
        films += 1
        for name, fn in interaction.V1.items():
            counts[name] += int(fn(film)[film.post].sum())
    assert films > 0
    # Four different questions give four different answers; none is "the" one.
    assert len(set(counts.values())) > 1, counts
