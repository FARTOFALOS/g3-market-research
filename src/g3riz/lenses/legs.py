"""Legs — a film cut into alternating moves by a named retracement amplitude.

WHY THIS IS A LENS AND NOT A TRACK
----------------------------------
It needs one number: how far price must retrace from a running extreme before
the move is called over. A different reasonable number gives a different
segmentation, so it is a choice and must be asked for by name.

WHAT IT IS FOR, AND WHAT IT IS NOT FOR
--------------------------------------
Minute-level pivots are not morphology. On the 2006-2026 field the median film
carries about 25 strict three-bar pivots and 49 close-to-close sign changes in
120 minutes, which is a property of one-minute noise. Sweeping this threshold
instead showed no natural scale either: leg count falls smoothly from 41 to 5.4
as theta goes from 0.10 to 3.00 ATR, with no plateau.

Read that carefully, and do not over-read it. No plateau shows there is no single
privileged scale; it does not show the knob is meaningless, because a genuinely
multiscale tape would produce exactly this smooth decay. What it does mean is
that `theta` is a free search dimension whose multiplicity has to be paid for
like any other.

Three arguments against this lens were tried in 2026-09 and all three failed, so
they are recorded here to keep them from being tried again. That latency
disqualifies it: no — every signal is observed after something happened, S-01
waits fifteen minutes, and 017 correctly measured forward from `conf`; the
give-back is a headwind of known size, not a disqualification. That a leg
discards speed, adverse travel and zone geometry: no — `Legs` is an insufficient
feature object, not a lossy primitive, and the film it came from still holds all
of it. That the surviving setups (S-01, S-02) come from counting and level events
instead: true but weak, being two cases, and those are built on three thresholds
where this lens has one.

What is actually established is narrower. This lens was broken until 2026-09-03
and cards 017 to 019 stand on the broken version; the corrected re-measurement
put 017's survivors at chance width. So the honest status of legs as a discovery
primitive is unknown, not refuted. Earning promotion needs a leg-based episode
whose forward path geometry is asymmetric at a frequency worth trading — which
has not been shown, and was not shown before either.

CLOSES ONLY
-----------
Legs are cut on closes. Minute OHLC does not record intraminute order, so a
zigzag routed through highs and lows would invent a sequence the tape never
had.

THE DEFINITION, WITHOUT REFERENCE TO CODE
-----------------------------------------
Let `z[0..N-1]` be the post-T0 closes signed so that positive is the T0 exit
side, and fix `theta > 0`.

A **turn** is a bar at which the tape has given back `theta` from an extreme.
The first one needs no direction to be named:

    c[1] = least i with  max(z[0..i]) > z[0]  and  max(z[0..i]) - z[i] >= theta
                     or  min(z[0..i]) < z[0]  and  z[i] - min(z[0..i]) >= theta

The `> z[0]` and `< z[0]` guards are not decoration. Without them a tape running
cleanly upward gives back `theta` from a "low" that is still the T0 close, and
the machine reports a turn where price only travelled. That is the original
phantom in its most durable disguise: it survives being renamed.

Exactly one of those can hold at `c[1]` — never both. (If both held, the range
would span `2*theta` and the earlier of the two extremes would already have
produced a give-back of `2*theta` at the bar attaining the later one, so the
turn would have fired before `c[1]`.) The extreme it names, at its earliest
attaining bar, is `e[1]`, and the run leaving `e[1]` heads the other way.

Thereafter the direction alternates, and each turn is found from the previous
extreme:

    c[k+1] = least i > c[k] with the give-back from the running extreme of
             `z[e[k]..i]`, measured in the current run's direction, >= theta
    e[k+1] = the bar attaining that extreme, earliest first

**Leg `k` spans `e[k] -> e[k+1]`**, is confirmed at `c[k+1]`, has direction
`sign(z[e[k+1]] - z[e[k]])` and that same signed amplitude. Every leg therefore
runs from one turn to the next.

The span `0 -> e[1]` is **not a leg**. Nothing has ever given anything back at
the T0 close, so it is not a completed run; it is the opening excursion, and it
is returned under its own name.

WHY THE OPENING SPAN MUST NOT BE A LEG
--------------------------------------
This is the whole of the old defect, and two later repair attempts failed on it.

Calling `0 -> e[1]` a leg forces a convention about a direction the tape has not
yet declared, and every available convention is wrong somewhere:

* Anchor the leg at T0 and let either give-back fire. Then a film that runs
  cleanly one way opens with a leg whose extreme is still the T0 close itself —
  amplitude exactly zero, direction against its own first move. That was the
  original bug, and it seated a fabricated symbol at the front of every affected
  film's sequence.
* Require `theta` of travel from T0 before a direction exists. Then on
  `0, +0.6*theta, -0.6*theta` a run rose and gave back `1.2*theta`, and the
  segmentation reports nothing at all.
* Re-derive the direction each bar from whichever extreme is more recent. Then
  a new high above the anchor silently cancels a pending turn, the sequence
  emits two legs the same way in a row, and alternation — which is definitional
  — breaks. On real NQ films this disagreed with the alternating reading on 878
  of 1056 segmentations.
* Freeze the direction at the first bar the extremes separate. Then
  `0, +0.1, -5*theta` opens with a leg of amplitude `0.1` invented by one tick
  of noise before a large move the other way.

Starting the sequence at `e[1]` removes the choice instead of making it. Every
leg then spans turn to turn, `sign(amplitude) == direction` and
`|amplitude| >= theta` hold for all of them without exception, and a study
comparing each leg with the one before it draws both from the same
distribution.

The second half of the old defect was separate: on a downward turn it set the
high to the confirming close and the low to the old high, assigning the extreme
irrelevant to the new run correctly and the relevant one to a value that run had
already passed. That dated later turns too late and under-measured them.

TWO CLOCKS, KEPT APART
----------------------
`turn` is the minute a leg's extreme occurred. `conf` is the minute the tape
completed the retracement that declared it over. Only `conf` is knowable live,
and it is always at or after `turn`. Any reading that looks forward from a leg
must start at `conf`; starting at `turn` back-dates knowledge by a median of one
to five minutes depending on theta. The gap is returned, never hidden.

`start` is where the leg began — the previous leg's extreme, or the T0 close for
the first one. It is returned rather than left for the caller to rebuild from
`turn_ord[k-1]`, because that rebuild is silently wrong at `k = 0`.

WHERE THE NEXT RUN'S EXTREME COMES FROM
---------------------------------------
The confirming close *is* it, and no search over the intervening bars is
needed. Take an up-run with extreme `H` at bar `e`, confirmed at the first bar
`c` where `H - z_c >= theta`. For any bar `b` strictly between them, "first"
means `H - z_b < theta`, so `z_b > H - theta >= z_c`. Every close between a
run's extreme and its confirmation therefore sits strictly above the confirming
close, and the minimum over `[e, c]` is `z_c` itself. Downward is the mirror.

So the running extremes need no window search at any point: the anchor side
keeps the extreme that just ended the previous leg, and the active side starts
at the confirming close. One comparison per bar is the whole online state.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..film import Film

MAX_LEGS_V1 = 64

_UNDETERMINED = 0


@dataclass(frozen=True)
class Legs:
    """One film's legs. Arrays are trimmed to the number actually found."""

    direction: np.ndarray      # +1 with the T0 exit side, -1 against it
    amplitude: np.ndarray      # signed price change from the leg's start
    start_ord: np.ndarray      # bar ordinal the leg began at, always a turn
    turn_ord: np.ndarray       # bar ordinal of the leg's extreme
    conf_ord: np.ndarray       # bar ordinal the retracement completed
    theta: float

    # The T0 close to the first turn: travel that no give-back has measured, so
    # not a leg. NaN / -1 when the film never turned at all.
    opening_amplitude: float = float("nan")
    opening_turn_ord: int = -1
    opening_conf_ord: int = -1

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
        f, i8 = np.zeros(0), np.zeros(0, dtype=np.int64)
        return Legs(i8.astype(np.int8), f, i8, i8, i8, float(theta))

    sign = 1.0 if film.exit_up else -1.0
    z = sign * (close - close[0])

    d_dir, d_amp, d_start, d_turn, d_conf = [], [], [], [], []
    run_dir = _UNDETERMINED
    anchor, anchor_t = z[0], 0
    hi, hi_t = z[0], 0
    lo, lo_t = z[0], 0
    open_amp, open_turn, open_conf = float("nan"), -1, -1

    for i in range(1, n):
        v = z[i]
        if v > hi:
            hi, hi_t = v, i
        if v < lo:
            lo, lo_t = v, i

        if run_dir == _UNDETERMINED:
            # No direction is named yet, so watch both sides. Whichever gives
            # back theta first names the opening extreme; only one can.
            if hi > z[0] and (hi - v) >= theta:
                ext, ext_t, run_dir = hi, hi_t, -1
            elif lo < z[0] and (v - lo) >= theta:
                ext, ext_t, run_dir = lo, lo_t, 1
            else:
                continue
            # The span T0 -> ext is not a leg. Legs start here.
            open_amp = float(ext - z[0])
            open_turn, open_conf = int(ords[ext_t]), int(ords[i])
            anchor, anchor_t = ext, ext_t
            hi, hi_t = (v, i) if run_dir > 0 else (ext, ext_t)
            lo, lo_t = (ext, ext_t) if run_dir > 0 else (v, i)
            continue

        if run_dir > 0:
            turned = (hi - v) >= theta
            ext, ext_t = hi, hi_t
        else:
            turned = (v - lo) >= theta
            ext, ext_t = lo, lo_t
        if not turned:
            continue

        d_dir.append(run_dir)
        d_amp.append(float(ext - anchor))
        d_start.append(int(ords[anchor_t]))
        d_turn.append(int(ords[ext_t]))
        d_conf.append(int(ords[i]))

        anchor, anchor_t = ext, ext_t
        run_dir = -run_dir
        # The anchor side keeps the extreme that just ended the leg; the side
        # the new run travels starts at this close, which the lemma above shows
        # is already its extreme.
        if run_dir > 0:
            hi, hi_t = v, i
        else:
            lo, lo_t = v, i

        if len(d_dir) >= max_legs:
            break

    return Legs(np.asarray(d_dir, dtype=np.int8),
                np.asarray(d_amp, dtype=np.float64),
                np.asarray(d_start, dtype=np.int64),
                np.asarray(d_turn, dtype=np.int64),
                np.asarray(d_conf, dtype=np.int64),
                float(theta), open_amp, open_turn, open_conf)
