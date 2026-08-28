"""Exact session-anchored native-bar reconstruction from the market spine."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from .machine import NativeBar
from .market import MarketSpine

MINUTE_NS = 60_000_000_000


@dataclass(slots=True)
class NativeArrays:
    open_ts_ns: np.ndarray
    close_ts_ns: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    minute_start: np.ndarray
    minute_stop: np.ndarray

    def __len__(self) -> int:
        return len(self.open)


def build_native_arrays(market: MarketSpine, tf_minutes: int) -> NativeArrays:
    """Build one native clock with vectorized per-session reductions.

    Session membership and timestamp parsing are already paid once in the
    market spine. This function constructs each actual native bar once without
    allocating millions of Python minute or bar objects.
    """
    if not 1 <= tf_minutes <= 1440:
        raise ValueError("tf_minutes must be in 1..1440")
    parts = {name: [] for name in ("open_ts_ns", "close_ts_ns", "open", "high", "low",
                                   "close", "minute_start", "minute_stop")}
    sessions = market.sessions
    for sid in range(len(sessions["session_id"])):
        first = int(sessions["first_minute_pos"][sid])
        stop = int(sessions["stop_minute_pos"][sid])
        anchor = int(sessions["session_open_utc_ns"][sid])
        close_ns = np.asarray(market.close_ts_utc_ns[first:stop])
        offsets = (close_ns - anchor) // MINUTE_NS
        if len(offsets) and (offsets[0] < 1 or np.any(offsets[1:] <= offsets[:-1])):
            raise ValueError(f"invalid minute membership in session {sid}")
        keys = (offsets - 1) // tf_minutes
        starts = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1]) + 1].astype(np.int64)
        stops = np.r_[starts[1:], len(keys)].astype(np.int64)
        absolute_starts = starts + first
        absolute_stops = stops + first
        parts["open_ts_ns"].append(anchor + keys[starts] * tf_minutes * MINUTE_NS)
        parts["close_ts_ns"].append(close_ns[stops - 1])
        parts["open"].append(np.asarray(market.open[absolute_starts]))
        parts["high"].append(np.maximum.reduceat(np.asarray(market.high[first:stop]), starts))
        parts["low"].append(np.minimum.reduceat(np.asarray(market.low[first:stop]), starts))
        parts["close"].append(np.asarray(market.close[absolute_stops - 1]))
        parts["minute_start"].append(absolute_starts)
        parts["minute_stop"].append(absolute_stops)
    return NativeArrays(**{
        name: np.concatenate(values) if values else np.empty(0, dtype=np.int64)
        for name, values in parts.items()
    })


def iter_native_bars(market: MarketSpine, tf_minutes: int) -> Iterator[NativeBar]:
    if not 1 <= tf_minutes <= 1440:
        raise ValueError("tf_minutes must be in 1..1440")
    next_index = 0
    sessions = market.sessions
    for sid in range(len(sessions["session_id"])):
        first = int(sessions["first_minute_pos"][sid])
        stop = int(sessions["stop_minute_pos"][sid])
        anchor = int(sessions["session_open_utc_ns"][sid])
        close_ns = np.asarray(market.close_ts_utc_ns[first:stop])
        offsets = (close_ns - anchor) // MINUTE_NS
        if len(offsets) and (offsets[0] < 1 or np.any(offsets[1:] <= offsets[:-1])):
            raise ValueError(f"invalid minute membership in session {sid}")
        keys = (offsets - 1) // tf_minutes
        starts = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1]) + 1]
        stops = np.r_[starts[1:], len(keys)]
        for local_start, local_stop in zip(starts, stops, strict=True):
            a = first + int(local_start)
            b = first + int(local_stop)
            key = int(keys[local_start])
            yield NativeBar(
                index=next_index,
                open_ts_ns=anchor + key * tf_minutes * MINUTE_NS,
                close_ts_ns=int(market.close_ts_utc_ns[b - 1]),
                open=float(market.open[a]),
                high=float(np.max(market.high[a:b])),
                low=float(np.min(market.low[a:b])),
                close=float(market.close[b - 1]),
                minute_start=a,
                minute_stop=b,
            )
            next_index += 1
