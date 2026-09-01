"""Four ways to ask a minute whether it engaged the RIZ.

THEY ARE NOT A LADDER
---------------------
The previous repository shipped these as an ordered ladder from loose to
strict, and then measured that the order does not exist: `full_traversal`
reaches zones that `close_inside` never does, because a body spanning the whole
zone need not close inside it. Anything that treated the list as nested was
wrong by construction, so the list is not offered as one here. These are four
different questions, and a study names the one it means.

None of them is "the" retest. Retest is a research term: a study that uses the
word must say which predicate it stands for, and whether it means the minute,
the maximal run of such minutes, or that run's first minute.

    shadow_touch    the bar's RANGE meets the zone
    body_entry      the bar's BODY meets the zone
    close_inside    the CLOSE is not strictly beyond either boundary
    full_traversal  the BODY spans the zone from below to above

All four read only the signed distances the film computes from stored prices,
so all four are prefix functions and cost no point-in-time rebuild.
"""

from __future__ import annotations

import numpy as np

from ..film import Film

PREDICATES = ("shadow_touch", "body_entry", "close_inside", "full_traversal")


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


V1 = {"shadow_touch": shadow_touch_v1, "body_entry": body_entry_v1,
      "close_inside": close_inside_v1, "full_traversal": full_traversal_v1}


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
