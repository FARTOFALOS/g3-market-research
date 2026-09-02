"""Legs — a film cut into alternating moves by a named retracement amplitude.

WHY THIS IS A LENS AND NOT A TRACK
----------------------------------
It needs one number: how far price must retrace from a running extreme before
the move is called over. A different reasonable number gives a different
segmentation, so it is a choice and must be asked for by name.

WHAT IT IS FOR
--------------
Minute-level pivots are not morphology. On the 2006-2026 field the median film
carries about 25 strict three-bar pivots and 49 close-to-close sign changes in
120 minutes, which is a property of one-minute noise. Sweeping this threshold
instead showed no natural scale either: leg count falls smoothly from 41 to 5.4
as theta goes from 0.10 to 3.00 ATR, with no plateau. So there is no correct
theta, and a study that uses one leg scale should say why and preferably run a
second (see cards 017 to 019).

CLOSES ONLY
-----------
Legs are cut on closes. Minute OHLC does not record intraminute order, so a
zigzag routed through highs and lows would invent a sequence the tape never
had.

TWO CLOCKS, KEPT APART
----------------------
`turn` is the minute a leg's extreme occurred. `conf` is the minute the tape
completed the retracement that declared it over. Only `conf` is knowable live,
and it is always at or after `turn`. Any reading that looks forward from a leg
must start at `conf`; starting at `turn` back-dates knowledge by a median of
one to five minutes depending on theta. The gap is returned, never hidden.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..film import Film

MAX_LEGS_V1 = 64


@dataclass(frozen=True)
class Legs:
    """One film's legs. Arrays are trimmed to the number actually found."""

    direction: np.ndarray      # +1 with the T0 exit side, -1 against it
    amplitude: np.ndarray      # signed price change from the previous anchor
    turn_ord: np.ndarray       # bar ordinal of the leg's extreme
    conf_ord: np.ndarray       # bar ordinal the retracement completed
    theta: float

    def __len__(self) -> int:
        return int(self.direction.size)

    @property
    def confirmation_lag(self) -> np.ndarray:
        return self.conf_ord - self.turn_ord


def directional_change_v1(film: Film, theta: float,
                          max_legs: int = MAX_LEGS_V1) -> Legs:
    """Cut the post-T0 closes into legs that reverse after a `theta` retracement.

    `theta` is in the film's own price units; a study that wants volatility
    units divides by its own scale before calling and says so. Pre-roll is never
    read: a leg cannot begin before T0.
    """
    if theta <= 0:
        raise ValueError("theta must be positive")
    close = film.close[film.post]
    ords = film.bar_ord[film.post]
    n = close.size
    if n < 2:
        empty = np.zeros(0)
        return Legs(empty, empty, empty.astype(np.int64),
                    empty.astype(np.int64), float(theta))

    sign = 1.0 if film.exit_up else -1.0
    z = sign * (close - close[0])

    d_dir, d_amp, d_turn, d_conf = [], [], [], []
    mode = 0
    hi = lo = anchor = z[0]
    hi_t = lo_t = 0
    for i in range(1, n):
        v = z[i]
        if v > hi:
            hi, hi_t = v, i
        if v < lo:
            lo, lo_t = v, i
        down = mode >= 0 and (hi - v) >= theta
        up = mode <= 0 and (v - lo) >= theta and not down
        if not (down or up):
            continue
        ext, ext_t = (hi, hi_t) if down else (lo, lo_t)
        d_dir.append(1 if down else -1)
        d_amp.append(float(ext - anchor))
        d_turn.append(int(ords[ext_t]))
        d_conf.append(int(ords[i]))
        anchor = ext
        mode = -1 if down else 1
        hi, hi_t = (v, i) if down else (ext, ext_t)
        lo, lo_t = (ext, ext_t) if down else (v, i)
        if len(d_dir) >= max_legs:
            break

    return Legs(np.asarray(d_dir, dtype=np.int8),
                np.asarray(d_amp, dtype=np.float64),
                np.asarray(d_turn, dtype=np.int64),
                np.asarray(d_conf, dtype=np.int64),
                float(theta))
