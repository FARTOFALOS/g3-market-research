"""Core tracks — thirteen independent readings of the same minute.

THE RULE THIS MODULE EXISTS TO KEEP
-----------------------------------
A minute is never one symbol. The same minute can raise its high, lower its
close, sit beyond the zone and set a new running extreme; those are four
answers to four questions, not four candidates for one label. So every track is
its own array over the same index, and nothing here composes them.

WHY EXACTLY THESE THIRTEEN
--------------------------
They are mechanical: each is a function of the bars up to and including the
minute being read, and of two fixed zone prices. None of them needs a
threshold, a width, a confirmation delay or a named rule, so none of them is a
research choice wearing the clothes of a coordinate.

Everything that DOES need such a choice — is this minute interacting with the
zone, is this a swing, where does an episode begin — lives in `lenses/` and
must be asked for by name.

`EXC_PTS` and `EXC_W` read the T0 exit side. That side is a stored fact of the
passport, not a study's decision, which is why they belong here; the signed
distances stay orientation-free regardless.

BUT THEY ARE ANCHORED ON `t0_close`, NOT ON THE EXIT BOUNDARY
------------------------------------------------------------
The T0 close is where the minute happened to close. The exit boundary is the
line the trader watches, and it is the one the whole post-T0 language is about.
These two anchors are a fixed distance apart within one film, so nothing here
is wrong — but excursion "in points from T0" is not distance from the level,
and a study that reports the first and says the second has swapped the object.
Distance from either boundary is already mechanical and already here: `D_TOP`
and `D_BOTTOM`. Contact with the exit boundary is a lens,
`lenses.interaction.exit_boundary_touch_v1`.

WHAT THEY ARE NOT
-----------------
Thirteen convenient derived readings of bars the field already stored — not the
market's alphabet, not an ontology, and not the list of questions worth asking.
A fourteenth appears when a study actually needs it, never to complete a set,
and a study is free to read the film directly instead.

Nothing here carries epistemic standing. A track computes; it never assigns a
conclusion, an evidence basis or a setup status. Those live in `base/` and
`setups/`, and a green test over a track says the predicate is what it claims,
not that the question was worth asking.

POINT-IN-TIME
-------------
Every track is a prefix function: it reads bar `i` and the bars before it, and
never a bar after. That makes the whole set stable under truncation by
construction, which is what buys studies out of the per-read rebuild that any
retrospective object costs. The property is pinned by a test, not asserted.
"""

from __future__ import annotations

import numpy as np

from .film import Film

CORE_TRACKS = ("H", "L", "C", "BODY", "RANGE_D", "BODY_D", "OVERLAP",
               "RUN_MAX", "RUN_MIN", "EXC_PTS", "EXC_W", "D_TOP", "D_BOTTOM")


def _step_sign(values: np.ndarray) -> np.ndarray:
    """`{+, 0, -}` against the previous minute; the first minute is `?`."""
    out = np.full(values.size, "?", dtype="<U1")
    if values.size < 2:
        return out
    delta = values[1:] - values[:-1]
    out[1:] = np.where(delta > 0, "+", np.where(delta < 0, "-", "0"))
    return out


def core_tracks(film: Film) -> dict[str, np.ndarray]:
    """All thirteen core tracks of one film, each as its own array."""
    high, low, close, open_ = film.high, film.low, film.close, film.open
    n = film.n
    t0 = film.n_pre_roll          # the index of ordinal 0
    out: dict[str, np.ndarray] = {}

    out["H"] = _step_sign(high)
    out["L"] = _step_sign(low)
    out["C"] = _step_sign(close)

    body = close - open_
    out["BODY"] = np.where(body > 0, "up", np.where(body < 0, "down", "doji"))

    span = high - low
    out["RANGE_D"] = np.concatenate(([np.nan], span[1:] - span[:-1]))
    magnitude = np.abs(body)
    out["BODY_D"] = np.concatenate(([np.nan], magnitude[1:] - magnitude[:-1]))

    overlap = np.full(n, "?", dtype="<U10")
    if n > 1:
        overlap[1:] = np.where(low[1:] > high[:-1], "gap_up",
                               np.where(high[1:] < low[:-1], "gap_down",
                                        "overlap"))
    out["OVERLAP"] = overlap

    # Running extremes are measured from T0, never from the film start, so
    # pre-roll can never manufacture one.
    run_max = np.zeros(n, dtype=bool)
    run_min = np.zeros(n, dtype=bool)
    if n > t0:
        rising = np.maximum.accumulate(high[t0:])
        falling = np.minimum.accumulate(low[t0:])
        run_max[t0:] = np.concatenate(([True], rising[1:] > rising[:-1]))
        run_min[t0:] = np.concatenate(([True], falling[1:] < falling[:-1]))
    out["RUN_MAX"], out["RUN_MIN"] = run_max, run_min

    t0_close = float(close[t0]) if n > t0 else np.nan
    sign = 1.0 if film.exit_up else -1.0
    reach = high if film.exit_up else low
    out["EXC_PTS"] = sign * (reach - t0_close)
    width = film.width
    out["EXC_W"] = (out["EXC_PTS"] / width if width
                    else np.full(n, np.nan, dtype=np.float64))

    out["D_TOP"] = close - film.zone_top
    out["D_BOTTOM"] = close - film.zone_bottom
    return out
