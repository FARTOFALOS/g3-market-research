"""Swings — the one retrospective reading kept in the first slice.

A swing is a label a minute earns from its neighbours on BOTH sides, so it is
knowable only after the right-hand neighbours close. That delay is the whole
difficulty and it is carried explicitly: `strict3_v1` says where the pivots are,
`confirmed_at` says when each became knowable, and every reading that looks
forward from a swing must use the second, never the first.

The word HH is not used here for the same reason the previous repository banned
it: three different things wore that name — this minute's high above the last
minute's high, a new running extreme since the anchor, and a confirmed swing
high above the previous confirmed swing high. The first two are core tracks
(`H`, `RUN_MAX`); only the third is a swing.

Bars within `width // 2` of either end of the film have no neighbour on one
side and are therefore not swings. That is undefined, never imputed, and it is
the mechanical reason a two-minute film has an extreme but cannot have a swing.
"""

from __future__ import annotations

import numpy as np

from ..film import Film

WIDTH_V1 = 3


def strict3_v1(film: Film) -> tuple[np.ndarray, np.ndarray]:
    """Strict three-bar fractal pivots: `(is_swing_high, is_swing_low)`.

    Strict on both sides. A bar tied with a neighbour is locally maximal but
    confirms nothing, and the rule declines it rather than inventing a
    tie-break.
    """
    return _fractal(film.high, film.low, WIDTH_V1)


def _fractal(high: np.ndarray, low: np.ndarray,
             width: int) -> tuple[np.ndarray, np.ndarray]:
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    n = high.size
    k = width // 2
    is_high = np.zeros(n, dtype=bool)
    is_low = np.zeros(n, dtype=bool)
    if n < width:
        return is_high, is_low
    centre_h, centre_l = high[k:n - k], low[k:n - k]
    above = np.ones(centre_h.shape, dtype=bool)
    below = np.ones(centre_l.shape, dtype=bool)
    for j in range(1, k + 1):
        above &= (centre_h > high[k - j:n - k - j]) & (centre_h > high[k + j:n - k + j])
        below &= (centre_l < low[k - j:n - k - j]) & (centre_l < low[k + j:n - k + j])
    is_high[k:n - k] = above
    is_low[k:n - k] = below
    return is_high, is_low


def confirmed_at(pivot_indices: np.ndarray, width: int = WIDTH_V1) -> np.ndarray:
    """The index at which each pivot became knowable: `pivot + width // 2`."""
    return np.asarray(pivot_indices, dtype=np.int64) + width // 2


def vs_last_swing_v1(film: Film) -> dict[str, np.ndarray]:
    """`{+, 0, -}` of each minute against the last CONFIRMED swing price.

    `?` where no swing has been confirmed yet — undefined, never imputed. The
    comparison is made at the confirmation index and never at the pivot, so it
    reads nothing a minute could not have known.

    Returns the two tracks `VS_LAST_SH` and `VS_LAST_SL`.
    """
    is_high, is_low = strict3_v1(film)
    return {"VS_LAST_SH": _against(film.high, is_high, WIDTH_V1 // 2, invert=False),
            "VS_LAST_SL": _against(film.low, is_low, WIDTH_V1 // 2, invert=True)}


def _against(price: np.ndarray, is_pivot: np.ndarray, lag: int,
             invert: bool) -> np.ndarray:
    """Compare each minute to the newest pivot price already confirmed by it."""
    n = price.size
    out = np.full(n, "?", dtype="<U1")
    pivots = np.flatnonzero(is_pivot)
    if pivots.size == 0:
        return out
    known_at = pivots + lag
    # For each minute, the newest pivot whose confirmation has already arrived.
    slot = np.searchsorted(known_at, np.arange(n), side="right") - 1
    seen = slot >= 0
    if not np.any(seen):
        return out
    reference = np.empty(n, dtype=np.float64)
    reference[seen] = price[pivots[slot[seen]]]
    delta = np.where(seen, price - reference, np.nan)
    higher, lower = ("-", "+") if invert else ("+", "-")
    out[seen] = np.where(delta[seen] > 0, higher,
                         np.where(delta[seen] < 0, lower, "0"))
    return out
