"""Film — the minute life of one RIZ around its T0, as a view that nothing stores.

ONE RIZ, ONE T0, FOREVER
------------------------
T0 is the zero minute of every film. Before it the ordinals run `-1, -2, -3`;
after it `+1, +2, +3`. Nothing else is ever a zero. 3X ignition, native blue
confirmation, breaker entry, a retest — those are marks INSIDE this film, found
by whoever needs them, and they never re-anchor the clock. The moment a second
origin is allowed, "T0" starts meaning "T0 of the 3X" in someone's sentence and
the trader's language is gone.

THE END BELONGS TO THE STUDY, NOT TO THE FILM
---------------------------------------------
A film does not decide how long a RIZ lives. It stops where it was asked to
stop and says why:

    stop="blue_end"      the minute blue eligibility ended
    stop="deletion"      canonical C1 deletion
    stop="archive_edge"  the last minute the field observed
    max_bars=240         a study's observation budget
    end_position=...     a position the study computed itself

The last form is how a study says "cut at my first retest": it finds its own
retest under its own definition, then hands over the position and the reason.
The film never runs a study's predicate, because the moment it did, one
definition of retest would quietly become the repository's.

This matters more here than it did in the previous repository, which
materialized its trajectories and so had to declare one global cap. Measured on
NQ, the median life to deletion is 162 minutes while p95 is 30,044 and p99 is
1,017,896. Any single cap would discard almost every observed minute while
looking like a property of the object. So a cap is always a named budget, never
a default, and when it bites the film reports `observation_budget`.

WHAT A FILM CARRIES
-------------------
Stored primitives and nothing derived: spine positions, close timestamps,
OHLCV, session id, the two fixed zone prices, the T0 exit side, and a signed
`bar_ord`. Negative ordinals are declared pre-roll: they may inform a lens's
geometry, and no statistic about the post-T0 life may count them.

Categories, interactions, swings and episodes are not here. They are lenses
over this view, asked for by name.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pyarrow as pa

#: Named ends, each a passport column holding a spine position.
STOPS: dict[str, str] = {
    "deletion": "c1_deletion_spine_pos",
    "blue_end": "blue_eligibility_end_spine_pos",
    "archive_edge": "last_observed_spine_pos",
}

#: Reasons this module produces on its own. A study passing an explicit
#: `end_position` supplies its own reason and is not limited to these.
FIELD_END_REASONS = ("none", "observation_budget", "archive_edge")

#: The reason a truncated film carries. It replaces both the stop and the end
#: reason, because both of those are knowledge from after the cursor.
PIT_TRUNCATION = "pit_truncation"

PASSPORT_COLUMNS = (
    "riz_id", "instrument", "tf_minutes", "zone_top", "zone_bottom",
    "t0_spine_pos", "t0_exit_side", "c1_deletion_spine_pos",
    "blue_eligibility_end_spine_pos", "last_observed_spine_pos", "censored",
)


@dataclass(frozen=True)
class Film:
    """One RIZ's minutes around T0. Immutable, ephemeral, stored nowhere."""

    riz_id: str
    instrument: str
    tf_minutes: int
    zone_top: float
    zone_bottom: float
    exit_up: bool
    stop: str
    end_reason: str
    n_pre_roll: int
    spine_pos: np.ndarray
    close_ts_utc_ns: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    session_id: np.ndarray
    bar_ord: np.ndarray

    def __post_init__(self) -> None:
        n = self.spine_pos.size
        for name in ("close_ts_utc_ns", "open", "high", "low", "close",
                     "volume", "session_id", "bar_ord"):
            if getattr(self, name).size != n:
                raise ValueError(f"film column {name!r} has the wrong length")
        if not self.stop or not self.end_reason:
            raise ValueError("a film must say where it stopped and why")
        if self.n_pre_roll > n:
            raise ValueError("pre-roll longer than the film")

    @property
    def n(self) -> int:
        return int(self.spine_pos.size)

    @property
    def post(self) -> slice:
        """The post-T0 part. Pre-roll is excluded from every statistic."""
        return slice(self.n_pre_roll, self.n)

    @property
    def width(self) -> float:
        return self.zone_top - self.zone_bottom

    @property
    def reached_its_stop(self) -> bool:
        """True only when the film ran to the end it asked for."""
        return self.end_reason == "none"

    def signed_distances(self) -> dict[str, np.ndarray]:
        """`d = price - boundary` for five measures against both zone prices.

        Orientation-free: the sign says above or below, never with or against.
        A direction-relative reading is a study's derivation, not a fact here.
        """
        body_max = np.maximum(self.open, self.close)
        body_min = np.minimum(self.open, self.close)
        values = {"high": self.high, "low": self.low, "close": self.close,
                  "body_max": body_max, "body_min": body_min}
        out: dict[str, np.ndarray] = {}
        for name, value in values.items():
            out[f"d_{name}_top"] = value - self.zone_top
            out[f"d_{name}_bottom"] = value - self.zone_bottom
        return out

    def truncate(self, cursor_ord: int) -> "Film":
        """The world as it looked at `cursor_ord`, and nothing later.

        The bars after the cursor go, and so does every piece of metadata that
        was knowledge from after it: the stop this film was cut to and the
        reason it ended both become `pit_truncation`, because on that minute
        nobody knew this RIZ would be deleted, would end blue, or would run out
        of budget. What survives is what T0 already established — identity,
        timeframe, the two zone prices, the exit side.

        This is what makes a point-in-time claim testable without any clock
        stored on a derived object: a condition a study declares decidable at
        ordinal `p` must return the same value on `film.truncate(p)`, with the
        lens recomputed from scratch. It rebuilds, so it belongs to the
        qualification of a lens, not to every research run.
        """
        keep = int(np.searchsorted(self.bar_ord, cursor_ord, side="right"))
        if keep <= 0:
            raise ValueError(f"cursor {cursor_ord} is before the film starts")
        if keep >= self.n:
            return self
        return replace(
            self, stop=PIT_TRUNCATION, end_reason=PIT_TRUNCATION,
            n_pre_roll=min(self.n_pre_roll, keep),
            spine_pos=self.spine_pos[:keep],
            close_ts_utc_ns=self.close_ts_utc_ns[:keep],
            open=self.open[:keep], high=self.high[:keep], low=self.low[:keep],
            close=self.close[:keep], volume=self.volume[:keep],
            session_id=self.session_id[:keep], bar_ord=self.bar_ord[:keep])


def film_bounds(row: dict, stop, pre_roll: int, max_bars: int | None,
                n_spine: int) -> tuple[int, int, int, str]:
    """`(start, t0_pos, end, end_reason)` in spine positions.

    The rules, stated once so no caller has to guess:
      * a stop the passport never recorded falls back to the last observed
        minute and reports `archive_edge`, never a natural end;
      * a RIZ still alive when the archive ends reports `archive_edge` too,
        whatever column was asked for;
      * `max_bars` counts post-T0 minutes and reports `observation_budget`;
      * pre-roll is clipped at the start of the tape without comment, because
        the tape's beginning is not a fact about this RIZ.
    """
    t0 = row.get("t0_spine_pos")
    if t0 is None:
        raise ValueError(f"RIZ {row.get('riz_id')!r} has no T0")
    t0 = int(t0)

    if stop not in STOPS:
        raise ValueError(f"unknown stop {stop!r}; known: {sorted(STOPS)}; "
                         f"pass end_position for a study-defined end")
    raw = row.get(STOPS[stop])
    if raw is None:
        end = int(row["last_observed_spine_pos"])
        reason = "archive_edge"
    else:
        end = int(raw)
        reason = "archive_edge" if bool(row.get("censored")) else "none"

    if end < t0:
        end = t0
    if max_bars is not None and (end - t0 + 1) > int(max_bars):
        end = t0 + int(max_bars) - 1
        reason = "observation_budget"
    end = min(end, n_spine - 1)
    return max(0, t0 - int(pre_roll)), t0, end, reason


def build_film(spine, row: dict, *, stop: str = "deletion", pre_roll: int = 0,
               max_bars: int | None = None, end_position: int | None = None,
               end_reason: str | None = None) -> Film:
    """Cut one Film out of an open market spine and one passport row.

    `end_position` lets a study cut at a minute it computed itself — its first
    retest, its own exhaustion rule, anything. It must come with `end_reason`,
    a short phrase naming that rule, so the film never presents a study's cut as
    a fact of the field. `max_bars` still applies on top of it and takes over
    the reason when it bites.
    """
    if end_position is None:
        start, t0, end, reason = film_bounds(
            row, stop, pre_roll, max_bars, len(spine.close))
        label = stop
    else:
        if not end_reason:
            raise ValueError("an explicit end_position needs an end_reason")
        t0 = int(row["t0_spine_pos"])
        end = max(int(end_position), t0)
        reason, label = end_reason, end_reason
        if max_bars is not None and (end - t0 + 1) > int(max_bars):
            end = t0 + int(max_bars) - 1
            reason = "observation_budget"
        end = min(end, len(spine.close) - 1)
        start = max(0, t0 - int(pre_roll))

    sl = slice(start, end + 1)
    n = end - start + 1
    return Film(
        riz_id=str(row["riz_id"]), instrument=str(row["instrument"]),
        tf_minutes=int(row["tf_minutes"]), zone_top=float(row["zone_top"]),
        zone_bottom=float(row["zone_bottom"]),
        exit_up=(str(row["t0_exit_side"]) == "north"),
        stop=label, end_reason=reason, n_pre_roll=int(t0 - start),
        spine_pos=np.arange(start, end + 1, dtype=np.int64),
        close_ts_utc_ns=np.asarray(spine.close_ts_utc_ns[sl]),
        open=np.asarray(spine.open[sl], dtype=np.float64),
        high=np.asarray(spine.high[sl], dtype=np.float64),
        low=np.asarray(spine.low[sl], dtype=np.float64),
        close=np.asarray(spine.close[sl], dtype=np.float64),
        volume=np.asarray(spine.volume[sl]),
        session_id=np.asarray(spine.session_id[sl]),
        bar_ord=np.arange(n, dtype=np.int64) - int(t0 - start))


def passport_rows(table: pa.Table) -> list[dict]:
    """Passport rows as plain dicts, restricted to what a Film needs."""
    columns = [name for name in PASSPORT_COLUMNS if name in table.column_names]
    return table.select(columns).to_pylist()
