"""Ways to ask a minute whether it engaged the RIZ, or its exit boundary.

THEY ARE NOT A LADDER
---------------------
The previous repository shipped these as an ordered ladder from loose to
strict, and then measured that the order does not exist: `full_traversal`
reaches zones that `close_inside` never does, because a body spanning the whole
zone need not close inside it. Anything that treated the list as nested was
wrong by construction, so the list is not offered as one here. These are
separate questions, and a study names the one it means.

None of them is "the" retest. Retest is a research term: a study that uses the
word must say which predicate it stands for, and whether it means the minute,
the maximal run of such minutes, or that run's first minute.

    shadow_touch    the bar's RANGE meets the zone
    body_entry      the bar's BODY meets the zone
    close_inside    the CLOSE is not strictly beyond either boundary
    full_traversal  the BODY spans the zone from below to above

THE ZONE AND THE EXIT BOUNDARY ARE DIFFERENT QUESTIONS
------------------------------------------------------
The four above ask about the zone as a whole — either boundary counts. The
trader's post-T0 language is about ONE line: the boundary the RIZ left through.
While price stays outside the zone the two readings almost coincide, which is
why the confusion survives: on 300 NQ TF 54 films the FIRST post-T0 minute is
the same for both in 99.3% of cases. But across the whole film the two masks
disagree in 70.7% of films, because once price crosses into and past the zone,
`shadow_touch` keeps firing on the far boundary while the exit boundary is
untouched. A study counting anything beyond the first contact with
`shadow_touch` is counting a different market event in most films.

So `exit_boundary_touch_v1` is its own predicate, and `first_exit_contact_v1`
is the end of the trader's first film.

`first_exit_contact_v1` is here rather than in a study because two independent
studies have needed exactly it: [`002`](../../../base/002-vozvrat-k-vyhodnoy-granitse.md)
measured it by hand, and the Film-1 semantics do too.

All predicates read only the signed distances the film computes from stored
prices, so all are prefix functions and cost no point-in-time rebuild.
"""

from __future__ import annotations

import numpy as np

from ..film import Film

PREDICATES = ("shadow_touch", "body_entry", "close_inside", "full_traversal",
              "exit_boundary_touch")


def shadow_touch_v1(film: Film) -> np.ndarray:
    """The bar's range meets the zone: `high >= bottom and low <= top`."""
    d = film.signed_distances()
    return (d["d_high_bottom"] >= 0) & (d["d_low_top"] <= 0)


def body_entry_v1(film: Film) -> np.ndarray:
    """The bar's body meets the zone."""
    d = film.signed_distances()
    return (d["d_body_max_bottom"] >= 0) & (d["d_body_min_top"] <= 0)


def close_inside_v1(film: Film) -> np.ndarray:
    """The close is not strictly beyond either boundary."""
    d = film.signed_distances()
    return (d["d_close_bottom"] >= 0) & (d["d_close_top"] <= 0)


def full_traversal_v1(film: Film) -> np.ndarray:
    """The body spans the whole zone."""
    d = film.signed_distances()
    return (d["d_body_min_bottom"] <= 0) & (d["d_body_max_top"] >= 0)


def exit_boundary_touch_v1(film: Film) -> np.ndarray:
    """The bar's RANGE contains the boundary this RIZ left through at T0.

    A wick is enough — this is the trader's rule, stated as "any subsequent
    touch of the same exit boundary, wick included". It is deliberately not a
    close and not a body: those are different questions and get their own
    names when a study needs them.

    True at the T0 bar itself by construction, because the T0 minute's range
    straddles the line it closed beyond. Callers who mean "afterwards" must
    say so; `first_exit_contact_v1` does.
    """
    e = film.exit_boundary
    return (film.low <= e) & (film.high >= e)


def first_exit_contact_v1(film: Film) -> int | None:
    """`bar_ord` of the first minute AFTER T0 whose range meets the exit
    boundary, or None if the film ends without one.

    This is the end of the trader's Film-1: `T0 .. this minute inclusive`. Feed
    the returned ordinal's spine position to `Field.films(end_positions=...)`
    with a reason of your own if you want the film cut there.

    NO DEPARTURE IS REQUIRED, and the word `retest` is avoided for that reason.
    `retest` smuggles in "price first went away", and here it usually did not:
    on 40,000 sampled NQ RIZ the first contact is at +1 minute in 57.7% of
    cases, within 3 minutes in 73.0%, within 15 in 88.4%, and never happens
    inside 600 minutes in 4.0%. A study that wants a departure must add it to
    its selection and say how many cases that drops — it is a narrower object,
    not a cleaner version of this one.

    Bar ordinals, not indices: ordinal 0 is T0 and pre-roll is negative, so the
    return value reads as minutes after T0 with no off-by-one to get wrong.
    """
    mask = exit_boundary_touch_v1(film) & (film.bar_ord >= 1)
    hit = np.flatnonzero(mask)
    return int(film.bar_ord[hit[0]]) if hit.size else None


V1 = {"shadow_touch": shadow_touch_v1, "body_entry": body_entry_v1,
      "close_inside": close_inside_v1, "full_traversal": full_traversal_v1,
      "exit_boundary_touch": exit_boundary_touch_v1}


def runs_v1(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`(onsets, lengths)` of the maximal runs of consecutive true minutes.

    An engagement is a run, not a bar: consecutive interacting minutes are one
    event, and its onset is the run's first minute. There is no minimum length,
    deliberately — a threshold on run length would be a tuned magnitude with
    nothing behind it, and a study that wants one can filter the lengths.

    A session break does not split a run. The session id is on the film; a study
    that believes the overnight gap matters can condition on it, which is not
    the same as this function deciding the question in advance.

    The onset is known at its own close; the LENGTH is not known until the run
    ends, so a study reading a length has made a claim about a later minute.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    starts = np.flatnonzero(mask & ~np.concatenate(([False], mask[:-1])))
    ends = np.flatnonzero(mask & ~np.concatenate((mask[1:], [False])))
    return starts, (ends - starts + 1)
