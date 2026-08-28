"""Compiled array replay for production-scale field materialization.

The object implementation in :mod:`g3riz.field` is the readable reference.
This module preserves the same machine and ladder predicates while removing
Python bar/minute/zone allocation from the 4,320-cell production path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numba as nb
import numpy as np

from .native import NativeArrays
from .identity import event_id, riz_id

# Transient event matrix columns. Only the factual projection in EVENT_SCHEMA
# is persisted; candidate/preview rows are discarded after anchor extraction.
EV_ZONE, EV_FAMILY, EV_KIND, EV_EVENT_TS, EV_POS, EV_BAR = range(6)
EV_OCC, EV_TERMINAL, EV_CANCEL_STAGE, EV_TIER, EV_ACTIVATED = range(6, 11)
EV_SPAN, EV_NORTH, EV_SOUTH, EV_LAST_SPAN, EV_BROKE, EV_RANK = range(11, 17)
N_EVENT_COLS = 17

F_ZONE = 1
F_LADDER = 2
K_BIRTH = 1
K_ACTIVATION = 2
K_SPAN = 3
K_NORTH_DEAD = 4
K_SOUTH_DEAD = 5
K_BREAKER = 6
K_ERRATUM = 7
K_DELETED = 8
K_CANDIDATE = 9
K_PREVIEW = 10
K_CANCELLATION = 11
K_CONFIRMATION = 12
K_T0 = 13
K_X3_T0 = 14


@dataclass(slots=True)
class FastFacts:
    zone_count: int
    events: np.ndarray
    top: np.ndarray
    bottom: np.ndarray
    birth_label: np.ndarray
    bullish: np.ndarray
    precursor_formed_ts: np.ndarray
    precursor_formed_pos: np.ndarray
    precursor_bar: np.ndarray
    first_activation_ts: np.ndarray
    first_activation_pos: np.ndarray
    first_activation_bar: np.ndarray
    t0_ts: np.ndarray
    t0_pos: np.ndarray
    t0_kind: np.ndarray
    t0_exit_side: np.ndarray
    t0_close: np.ndarray
    t0_span: np.ndarray
    t0_bar: np.ndarray
    x3_t0_ts: np.ndarray
    x3_t0_pos: np.ndarray
    x3_t0_close: np.ndarray
    x3_t0_exit_side: np.ndarray
    x3_t0_bar: np.ndarray
    x3_t0_confirmed: np.ndarray
    native_blue_confirmation_ts: np.ndarray
    native_blue_confirmation_pos: np.ndarray
    native_blue_confirmation_bar: np.ndarray
    blue_end_ts: np.ndarray
    blue_end_pos: np.ndarray
    deletion_ts: np.ndarray
    deletion_pos: np.ndarray
    deletion_bar: np.ndarray
    qualified: np.ndarray
    tier: np.ndarray
    activated: np.ndarray
    span_count: np.ndarray
    north_alive: np.ndarray
    south_alive: np.ndarray
    last_span: np.ndarray
    broke_south: np.ndarray


def count_births(bars: NativeArrays) -> int:
    if len(bars) < 3:
        return 0
    c1o, c1c, c1h, c1l = bars.open[:-2], bars.close[:-2], bars.high[:-2], bars.low[:-2]
    c2o, c2c = bars.open[1:-1], bars.close[1:-1]
    c3o, c3c, c3h, c3l = bars.open[2:], bars.close[2:], bars.high[2:], bars.low[2:]
    c1top, c1bottom = np.maximum(c1o, c1c), np.minimum(c1o, c1c)
    c2top, c2bottom = np.maximum(c2o, c2c), np.minimum(c2o, c2c)
    c3top, c3bottom = np.maximum(c3o, c3c), np.minimum(c3o, c3c)
    bisi_base = (c3l > c1h) & (c2top > c1h) & (c2bottom < c3l)
    bisi_bottom = np.where(c2bottom > c1top, c1top, c1h)
    bisi_top = np.where(c3bottom > c2top, c3bottom, c3l)
    sibi_base = (c3h < c1l) & (c2bottom < c1l) & (c2top > c3h)
    sibi_top = np.where(c2top < c1bottom, c1bottom, c1l)
    sibi_bottom = np.where(c3top < c2bottom, c3top, c3h)
    return int(np.count_nonzero(bisi_base & (bisi_top > bisi_bottom))
               + np.count_nonzero(sibi_base & (sibi_top > sibi_bottom)))


def precompute_births(bars: NativeArrays) -> tuple[np.ndarray, ...]:
    """Vectorize immutable zone geometry before state replay."""
    if len(bars) < 3:
        empty_i = np.empty(0, np.int64); empty_f = np.empty(0, np.float64)
        return empty_f, empty_f.copy(), empty_i, np.empty(0, np.uint8), empty_i.copy(), empty_i.copy(), empty_i.copy()
    idx = np.arange(2, len(bars), dtype=np.int64)
    c1o, c1c, c1h, c1l = bars.open[:-2], bars.close[:-2], bars.high[:-2], bars.low[:-2]
    c2o, c2c = bars.open[1:-1], bars.close[1:-1]
    c3o, c3c, c3h, c3l = bars.open[2:], bars.close[2:], bars.high[2:], bars.low[2:]
    c1top, c1bottom = np.maximum(c1o, c1c), np.minimum(c1o, c1c)
    c2top, c2bottom = np.maximum(c2o, c2c), np.minimum(c2o, c2c)
    c3top, c3bottom = np.maximum(c3o, c3c), np.minimum(c3o, c3c)
    bisi_bottom = np.where(c2bottom > c1top, c1top, c1h)
    bisi_top = np.where(c3bottom > c2top, c3bottom, c3l)
    bisi = (c3l > c1h) & (c2top > c1h) & (c2bottom < c3l) & (bisi_top > bisi_bottom)
    sibi_top = np.where(c2top < c1bottom, c1bottom, c1l)
    sibi_bottom = np.where(c3top < c2bottom, c3top, c3h)
    sibi = (c3h < c1l) & (c2bottom < c1l) & (c2top > c3h) & (sibi_top > sibi_bottom)
    precursor_bar = np.concatenate((idx[bisi], idx[sibi]))
    top = np.concatenate((bisi_top[bisi], sibi_top[sibi])).astype(np.float64)
    bottom = np.concatenate((bisi_bottom[bisi], sibi_bottom[sibi])).astype(np.float64)
    bullish = np.concatenate((np.ones(np.count_nonzero(bisi), np.uint8),
                              np.zeros(np.count_nonzero(sibi), np.uint8)))
    order = np.argsort(precursor_bar, kind="stable")
    precursor_bar = precursor_bar[order]
    top, bottom, bullish = top[order], bottom[order], bullish[order]
    birth_label = bars.open_ts_ns[precursor_bar - 2].astype(np.int64)
    precursor_formed_ts = bars.close_ts_ns[precursor_bar].astype(np.int64)
    precursor_formed_pos = (bars.minute_stop[precursor_bar] - 1).astype(np.int64)
    return top, bottom, birth_label, bullish, precursor_bar, precursor_formed_ts, precursor_formed_pos


@nb.njit(inline="always")
def _emit(events, count, zone, family, kind, event_ts, pos, bar,
          occurrence, terminal, cancel_stage, tier, activated, span_count,
          north, south, last_span, broke, rank):
    if count < events.shape[0]:
        events[count, EV_ZONE] = zone
        events[count, EV_FAMILY] = family
        events[count, EV_KIND] = kind
        events[count, EV_EVENT_TS] = event_ts
        events[count, EV_POS] = pos
        events[count, EV_BAR] = bar
        events[count, EV_OCC] = occurrence
        events[count, EV_TERMINAL] = terminal
        events[count, EV_CANCEL_STAGE] = cancel_stage
        events[count, EV_TIER] = tier
        events[count, EV_ACTIVATED] = activated
        events[count, EV_SPAN] = span_count
        events[count, EV_NORTH] = north
        events[count, EV_SOUTH] = south
        events[count, EV_LAST_SPAN] = last_span
        events[count, EV_BROKE] = broke
        events[count, EV_RANK] = rank
    return count + 1


@nb.njit(inline="always")
def _first_developing_span(minute_open, minute_close, start, stop,
                           bottom, top):
    """First completed pre-close minute whose developing body spans a zone."""
    stop_before_close = stop - 1
    if stop_before_close <= start:
        return -1
    dev_open = minute_open[start]
    if dev_open < bottom:
        for p in range(start, stop_before_close):
            if minute_close[p] > top:
                return p
    elif dev_open > top:
        for p in range(start, stop_before_close):
            if minute_close[p] < bottom:
                return p
    return -1


@nb.njit(cache=True)
def _kernel(open_ts, close_ts, bo, bh, bl, bc, mstart, mstop,
            minute_open, minute_close, minute_close_ts, n_zones, event_capacity):
    # Zone state and durable milestone arrays.
    top = np.empty(n_zones, np.float64); bottom = np.empty(n_zones, np.float64)
    birth_label = np.empty(n_zones, np.int64); bullish = np.empty(n_zones, np.uint8)
    precursor_formed_ts = np.empty(n_zones, np.int64); precursor_formed_pos = np.empty(n_zones, np.int64)
    precursor_bar = np.empty(n_zones, np.int64)
    tier = np.zeros(n_zones, np.int8); activated = np.zeros(n_zones, np.uint8)
    span_count = np.zeros(n_zones, np.int32); north = np.ones(n_zones, np.uint8)
    south = np.ones(n_zones, np.uint8); last_span = np.full(n_zones, -1, np.int64)
    broke = np.zeros(n_zones, np.uint8); was_blue = np.zeros(n_zones, np.uint8)
    qualified = np.zeros(n_zones, np.uint8)
    first_activation_ts = np.full(n_zones, -1, np.int64)
    first_activation_pos = np.full(n_zones, -1, np.int64)
    first_activation_bar = np.full(n_zones, -1, np.int64)
    t0_ts = np.full(n_zones, -1, np.int64); t0_pos = np.full(n_zones, -1, np.int64)
    t0_kind = np.zeros(n_zones, np.int8); t0_exit = np.zeros(n_zones, np.int8)
    t0_close = np.full(n_zones, np.nan, np.float64); t0_span = np.full(n_zones, -1, np.int32)
    t0_bar = np.full(n_zones, -1, np.int64)
    x3_ts = np.full(n_zones, -1, np.int64); x3_pos = np.full(n_zones, -1, np.int64)
    x3_close = np.full(n_zones, np.nan, np.float64); x3_exit = np.zeros(n_zones, np.int8)
    x3_bar = np.full(n_zones, -1, np.int64); x3_confirmed = np.zeros(n_zones, np.uint8)
    confirm_ts = np.full(n_zones, -1, np.int64); confirm_pos = np.full(n_zones, -1, np.int64)
    confirm_bar = np.full(n_zones, -1, np.int64)
    blue_end_ts = np.full(n_zones, -1, np.int64); blue_end_pos = np.full(n_zones, -1, np.int64)
    deletion_ts = np.full(n_zones, -1, np.int64); deletion_pos = np.full(n_zones, -1, np.int64)
    deletion_bar = np.full(n_zones, -1, np.int64)
    live = np.empty(n_zones, np.int64); live_count = 0; zone_count = 0
    events = np.empty((event_capacity, N_EVENT_COLS), np.int64); event_count = 0

    for i in range(len(bo)):
        pos = mstop[i] - 1
        ts = close_ts[i]
        new_live_count = 0
        for li in range(live_count):
            z = live[li]
            # Minute-clock preview on frozen pre-update state.
            fired = False; active = False; occurrence = 0; full_truth = False
            if (x3_ts[z] < 0 and span_count[z] == 2
                    and (tier[z] == 1 or tier[z] == 2)
                    and i >= last_span[z] + 2):
                p3 = _first_developing_span(
                    minute_open, minute_close, mstart[i], mstop[i], bottom[z], top[z])
                if p3 >= 0:
                    x3_ts[z] = minute_close_ts[p3]; x3_pos[z] = p3; x3_bar[z] = i
                    x3_close[z] = minute_close[p3]
                    x3_exit[z] = 1 if minute_close[p3] > top[z] else -1
                    event_count = _emit(events, event_count, z, F_LADDER, K_X3_T0,
                        minute_close_ts[p3], p3, i, -1, 0, 0,
                        tier[z], activated[z], span_count[z], north[z], south[z],
                        last_span[z], broke[z], 12)
            if (tier[z] == 1 or tier[z] == 2) and north[z] and south[z] and i >= last_span[z] + 2:
                a = mstart[i]; stop_before_close = mstop[i] - 1
                if stop_before_close > a:
                    dev_open = minute_open[a]
                    possible = False; mode = 0
                    if dev_open < bottom[z]:
                        for p in range(a, stop_before_close):
                            if minute_close[p] > top[z]: possible = True; break
                        mode = 1
                    elif dev_open > top[z]:
                        for p in range(a, stop_before_close):
                            if minute_close[p] < bottom[z]: possible = True; break
                        mode = -1
                    if possible:
                        for p in range(a, stop_before_close):
                            truth = minute_close[p] > top[z] if mode == 1 else minute_close[p] < bottom[z]
                            if truth and not active:
                                event_count = _emit(events, event_count, z, F_LADDER, K_PREVIEW,
                                    minute_close_ts[p], p, i, occurrence, 0, 0,
                                    tier[z], activated[z], span_count[z], north[z], south[z],
                                    last_span[z], broke[z], 10)
                                active = True; fired = True; qualified[z] = 1
                                if t0_ts[z] < 0:
                                    t0_ts[z] = minute_close_ts[p]; t0_pos[z] = p; t0_kind[z] = 1
                                    t0_exit[z] = 1 if minute_close[p] > top[z] else -1
                                    t0_close[z] = minute_close[p]; t0_span[z] = span_count[z]; t0_bar[z] = i
                                    event_count = _emit(events, event_count, z, F_LADDER, K_T0,
                                        minute_close_ts[p], p, i, -1, 0, 0,
                                        tier[z], activated[z], span_count[z], north[z], south[z],
                                        last_span[z], broke[z], 11)
                            elif not truth and active:
                                event_count = _emit(events, event_count, z, F_LADDER, K_CANCELLATION,
                                    minute_close_ts[p], p, i, occurrence, 0, 2,
                                    tier[z], activated[z], span_count[z], north[z], south[z],
                                    last_span[z], broke[z], 10)
                                active = False; occurrence += 1
                        full_truth = bc[i] > top[z] if mode == 1 else bc[i] < bottom[z]

            # Native-close C1 transition, literal branch order.
            t0 = tier[z]; act0 = activated[z]; sc0 = span_count[z]
            n0 = north[z]; s0 = south[z]; ls0 = last_span[z]
            survived = True
            bmin = bo[i] if bo[i] < bc[i] else bc[i]
            bmax = bo[i] if bo[i] > bc[i] else bc[i]
            spanned = bmin < bottom[z] and bmax > top[z]
            if tier[z] == 3:
                returned = bh[i] >= bottom[z] if broke[z] else bl[i] <= top[z]
                survived = not returned
            elif spanned and (not activated[z] or i >= last_span[z] + 2):
                if not activated[z]:
                    activated[z] = 1; tier[z] = 1 if north[z] and south[z] else 2
                    span_count[z] = 1
                else:
                    span_count[z] += 1
                last_span[z] = i
            else:
                kind = 0
                if not spanned:
                    if south[z] and not north[z] and bo[i] > bottom[z] and bc[i] < bottom[z]: kind = -1
                    elif north[z] and not south[z] and bo[i] < top[z] and bc[i] > top[z]: kind = 1
                    elif north[z] and south[z] and bo[i] > bottom[z] and bc[i] < bottom[z] and bh[i] >= top[z] and bc[i] <= top[z]: kind = -1
                    elif north[z] and south[z] and bo[i] < top[z] and bc[i] > top[z] and bl[i] <= bottom[z] and bc[i] >= bottom[z]: kind = 1
                if kind != 0:
                    broke[z] = 1 if kind == -1 else 0; north[z] = 0; south[z] = 0
                    tier[z] = 3; activated[z] = 1; last_span[z] = i
                else:
                    if bl[i] <= top[z] <= bh[i]: north[z] = 0
                    if bl[i] <= bottom[z] <= bh[i]: south[z] = 0
                    survived = bool(north[z] or south[z])

            if not act0 and activated[z]:
                event_count = _emit(events, event_count, z, F_ZONE, K_ACTIVATION, ts, pos, i,
                    -1, 0, 0, tier[z], activated[z], span_count[z], north[z], south[z], last_span[z], broke[z], 21)
                if first_activation_ts[z] < 0:
                    first_activation_ts[z] = ts; first_activation_pos[z] = pos; first_activation_bar[z] = i
            if span_count[z] > sc0:
                if sc0 == 2:
                    if x3_ts[z] < 0:
                        x3_ts[z] = ts; x3_pos[z] = pos; x3_bar[z] = i
                        x3_close[z] = bc[i]; x3_exit[z] = 1 if bc[i] > top[z] else -1
                        event_count = _emit(events, event_count, z, F_LADDER, K_X3_T0,
                            ts, pos, i, -1, 0, 0, t0, act0, sc0, n0, s0,
                            ls0, broke[z], 12)
                    if x3_bar[z] == i:
                        x3_confirmed[z] = 1
                event_count = _emit(events, event_count, z, F_ZONE, K_SPAN, ts, pos, i,
                    -1, 0, 0, tier[z], activated[z], span_count[z], north[z], south[z], last_span[z], broke[z], 22)
            if n0 and not north[z]:
                event_count = _emit(events, event_count, z, F_ZONE, K_NORTH_DEAD, ts, pos, i,
                    -1, 0, 0, tier[z], activated[z], span_count[z], north[z], south[z], last_span[z], broke[z], 23)
                if blue_end_ts[z] < 0: blue_end_ts[z] = ts; blue_end_pos[z] = pos
            if s0 and not south[z]:
                event_count = _emit(events, event_count, z, F_ZONE, K_SOUTH_DEAD, ts, pos, i,
                    -1, 0, 0, tier[z], activated[z], span_count[z], north[z], south[z], last_span[z], broke[z], 24)
                if blue_end_ts[z] < 0: blue_end_ts[z] = ts; blue_end_pos[z] = pos
            if t0 != 3 and tier[z] == 3:
                event_count = _emit(events, event_count, z, F_ZONE, K_BREAKER, ts, pos, i,
                    -1, 0, 0, tier[z], activated[z], span_count[z], north[z], south[z], last_span[z], broke[z], 25)
                if blue_end_ts[z] < 0: blue_end_ts[z] = ts; blue_end_pos[z] = pos
            if not survived:
                erratum = spanned and bool(act0) and i < ls0 + 2
                event_count = _emit(events, event_count, z, F_ZONE, K_ERRATUM if erratum else K_DELETED,
                    ts, pos, i, -1, 1, 0, tier[z], activated[z], span_count[z], north[z], south[z], last_span[z], broke[z], 26)
                if blue_end_ts[z] < 0: blue_end_ts[z] = ts; blue_end_pos[z] = pos
                deletion_ts[z] = ts; deletion_pos[z] = pos; deletion_bar[z] = i; was_blue[z] = 0
            else:
                live[new_live_count] = z; new_live_count += 1

            confirmed_now = False
            if survived and (tier[z] == 1 or tier[z] == 2) and span_count[z] >= 2 and north[z] and south[z]:
                if not was_blue[z]:
                    event_count = _emit(events, event_count, z, F_LADDER, K_CONFIRMATION, ts,
                        pos, i, occurrence, 0, 0, tier[z], activated[z], span_count[z], north[z], south[z], last_span[z], broke[z], 30)
                    qualified[z] = 1; confirmed_now = True
                    if t0_ts[z] < 0:
                        t0_ts[z] = ts; t0_pos[z] = pos; t0_kind[z] = 2
                        t0_exit[z] = 1 if bc[i] > top[z] else -1
                        t0_close[z] = bc[i]; t0_span[z] = span_count[z]; t0_bar[z] = i
                        event_count = _emit(events, event_count, z, F_LADDER, K_T0,
                            ts, pos, i, -1, 0, 0, tier[z], activated[z],
                            span_count[z], north[z], south[z], last_span[z], broke[z], 31)
                    if confirm_ts[z] < 0:
                        confirm_ts[z] = ts; confirm_pos[z] = pos; confirm_bar[z] = i
                was_blue[z] = 1
            else:
                was_blue[z] = 0
            if fired and active and not confirmed_now and not (full_truth and survived):
                event_count = _emit(events, event_count, z, F_LADDER, K_CANCELLATION, ts,
                    pos, i, occurrence, 1, 2, tier[z], activated[z], span_count[z], north[z], south[z], last_span[z], broke[z], 40)
        live_count = new_live_count

        # Birth pass follows the update pass.
        if i >= 2:
            c1top = bo[i-2] if bo[i-2] > bc[i-2] else bc[i-2]
            c1bottom = bo[i-2] if bo[i-2] < bc[i-2] else bc[i-2]
            c2top = bo[i-1] if bo[i-1] > bc[i-1] else bc[i-1]
            c2bottom = bo[i-1] if bo[i-1] < bc[i-1] else bc[i-1]
            c3top = bo[i] if bo[i] > bc[i] else bc[i]
            c3bottom = bo[i] if bo[i] < bc[i] else bc[i]
            if bl[i] > bh[i-2] and c2top > bh[i-2] and c2bottom < bl[i]:
                bot = c1top if c2bottom > c1top else bh[i-2]
                tp = c3bottom if c3bottom > c2top else bl[i]
                if tp > bot:
                    z = zone_count; zone_count += 1; top[z] = tp; bottom[z] = bot
                    birth_label[z] = open_ts[i-2]; bullish[z] = 1
                    precursor_formed_ts[z] = ts; precursor_formed_pos[z] = pos; precursor_bar[z] = i
                    live[live_count] = z; live_count += 1
                    event_count = _emit(events, event_count, z, F_ZONE, K_BIRTH, ts,
                        pos, i, -1, 0, 0, 0, 0, 0, 1, 1, -1, 0, 20)
            if bh[i] < bl[i-2] and c2bottom < bl[i-2] and c2top > bh[i]:
                tp = c1bottom if c2top < c1bottom else bl[i-2]
                bot = c3top if c3top < c2bottom else bh[i]
                if tp > bot:
                    z = zone_count; zone_count += 1; top[z] = tp; bottom[z] = bot
                    birth_label[z] = open_ts[i-2]; bullish[z] = 0
                    precursor_formed_ts[z] = ts; precursor_formed_pos[z] = pos; precursor_bar[z] = i
                    live[live_count] = z; live_count += 1
                    event_count = _emit(events, event_count, z, F_ZONE, K_BIRTH, ts,
                        pos, i, -1, 0, 0, 0, 0, 0, 1, 1, -1, 0, 20)

    # Candidate episodes are precursor facts. Reconstruct them only for zones
    # that actually qualify, avoiding minute work for discarded latent zones.
    for z in range(zone_count):
        if not qualified[z]: continue
        end_bar = first_activation_bar[z]
        for i in range(precursor_bar[z] + 1, end_bar + 1):
            a = mstart[i]; stop_before_close = mstop[i] - 1
            if stop_before_close <= a: continue
            dev_open = minute_open[a]; mode = 0; possible = False
            if dev_open < bottom[z]:
                mode = 1
                for p in range(a, stop_before_close):
                    if minute_close[p] > top[z]: possible = True; break
            elif dev_open > top[z]:
                mode = -1
                for p in range(a, stop_before_close):
                    if minute_close[p] < bottom[z]: possible = True; break
            if not possible: continue
            active = False; fired = False; occurrence = 0
            for p in range(a, stop_before_close):
                truth = minute_close[p] > top[z] if mode == 1 else minute_close[p] < bottom[z]
                if truth and not active:
                    event_count = _emit(events, event_count, z, F_LADDER, K_CANDIDATE,
                        minute_close_ts[p], p, i, occurrence, 0, 0,
                        0, 0, 0, 1, 1, -1, 0, 10)
                    active = True; fired = True
                elif not truth and active:
                    event_count = _emit(events, event_count, z, F_LADDER, K_CANCELLATION,
                        minute_close_ts[p], p, i, occurrence, 0, 1,
                        0, 0, 0, 1, 1, -1, 0, 10)
                    active = False; occurrence += 1
            full_truth = bc[i] > top[z] if mode == 1 else bc[i] < bottom[z]
            if fired and active and not full_truth:
                pos = mstop[i] - 1; ts = close_ts[i]
                event_count = _emit(events, event_count, z, F_LADDER, K_CANCELLATION,
                    ts, pos, i, occurrence, 1, 1, 0, 0, 0, 1, 1, -1, 0, 40)

    return (zone_count, event_count, events, top, bottom, birth_label, bullish,
            precursor_formed_ts, precursor_formed_pos, precursor_bar, first_activation_ts,
            first_activation_pos, first_activation_bar, t0_ts, t0_pos, t0_kind,
            t0_exit, t0_close, t0_span, t0_bar, x3_ts, x3_pos, x3_close, x3_exit,
            x3_bar, x3_confirmed, confirm_ts, confirm_pos, confirm_bar,
            blue_end_ts, blue_end_pos, deletion_ts, deletion_pos, deletion_bar,
            qualified, tier, activated, span_count, north, south, last_span, broke)


@nb.njit(cache=True)
def _kernel_indexed(open_ts, close_ts, bo, bh, bl, bc, mstart, mstop,
                    minute_open, minute_close, minute_close_ts,
                    top, bottom, birth_label, bullish, precursor_bar,
                    precursor_formed_ts, precursor_formed_pos, top_values, top_zones,
                    bottom_values, bottom_zones, event_capacity):
    """Replay only zones whose immutable boundary is inside the bar range."""
    n_zones = len(top)
    tier = np.zeros(n_zones, np.int8); activated = np.zeros(n_zones, np.uint8)
    span_count = np.zeros(n_zones, np.int32); north = np.ones(n_zones, np.uint8)
    south = np.ones(n_zones, np.uint8); last_span = np.full(n_zones, -1, np.int64)
    broke = np.zeros(n_zones, np.uint8); was_blue = np.zeros(n_zones, np.uint8)
    alive = np.zeros(n_zones, np.uint8); qualified = np.zeros(n_zones, np.uint8)
    first_activation_ts = np.full(n_zones, -1, np.int64)
    first_activation_pos = np.full(n_zones, -1, np.int64)
    first_activation_bar = np.full(n_zones, -1, np.int64)
    t0_ts = np.full(n_zones, -1, np.int64); t0_pos = np.full(n_zones, -1, np.int64)
    t0_kind = np.zeros(n_zones, np.int8); t0_exit = np.zeros(n_zones, np.int8)
    t0_close = np.full(n_zones, np.nan, np.float64); t0_span = np.full(n_zones, -1, np.int32)
    t0_bar = np.full(n_zones, -1, np.int64)
    x3_ts = np.full(n_zones, -1, np.int64); x3_pos = np.full(n_zones, -1, np.int64)
    x3_close = np.full(n_zones, np.nan, np.float64); x3_exit = np.zeros(n_zones, np.int8)
    x3_bar = np.full(n_zones, -1, np.int64); x3_confirmed = np.zeros(n_zones, np.uint8)
    confirm_ts = np.full(n_zones, -1, np.int64); confirm_pos = np.full(n_zones, -1, np.int64)
    confirm_bar = np.full(n_zones, -1, np.int64)
    blue_end_ts = np.full(n_zones, -1, np.int64); blue_end_pos = np.full(n_zones, -1, np.int64)
    deletion_ts = np.full(n_zones, -1, np.int64); deletion_pos = np.full(n_zones, -1, np.int64)
    deletion_bar = np.full(n_zones, -1, np.int64)
    events = np.empty((event_capacity, N_EVENT_COLS), np.int64); event_count = 0
    affected = np.empty(n_zones, np.int64); stamp = np.full(n_zones, -1, np.int64)
    breaker_zones = np.empty(n_zones, np.int64); breaker_count = 0
    next_birth = 0

    for i in range(len(bo)):
        pos = mstop[i] - 1; ts = close_ts[i]; affected_count = 0
        lo = np.searchsorted(top_values, bl[i])
        hi = np.searchsorted(top_values, bh[i], side="right")
        for q in range(lo, hi):
            z = top_zones[q]
            if alive[z] and stamp[z] != i:
                stamp[z] = i; affected[affected_count] = z; affected_count += 1
        lo = np.searchsorted(bottom_values, bl[i])
        hi = np.searchsorted(bottom_values, bh[i], side="right")
        for q in range(lo, hi):
            z = bottom_zones[q]
            if alive[z] and stamp[z] != i:
                stamp[z] = i; affected[affected_count] = z; affected_count += 1
        # A breaker return predicate is one-sided. A market gap can jump wholly
        # beyond its retirement boundary, leaving that price outside the bar's
        # [low, high] interval, so breaker thresholds need this separate index.
        live_breaker_count = 0
        for q in range(breaker_count):
            z = breaker_zones[q]
            if alive[z]:
                breaker_zones[live_breaker_count] = z
                live_breaker_count += 1
            if alive[z] and stamp[z] != i:
                returned = bh[i] >= bottom[z] if broke[z] else bl[i] <= top[z]
                if returned:
                    stamp[z] = i; affected[affected_count] = z; affected_count += 1
        breaker_count = live_breaker_count

        for ai in range(affected_count):
            z = affected[ai]
            fired = False; active = False; occurrence = 0; full_truth = False
            if (x3_ts[z] < 0 and span_count[z] == 2
                    and (tier[z] == 1 or tier[z] == 2)
                    and i >= last_span[z] + 2):
                p3 = _first_developing_span(
                    minute_open, minute_close, mstart[i], mstop[i], bottom[z], top[z])
                if p3 >= 0:
                    x3_ts[z] = minute_close_ts[p3]; x3_pos[z] = p3; x3_bar[z] = i
                    x3_close[z] = minute_close[p3]
                    x3_exit[z] = 1 if minute_close[p3] > top[z] else -1
                    event_count = _emit(events, event_count, z, F_LADDER, K_X3_T0,
                        minute_close_ts[p3], p3, i, -1, 0, 0,
                        tier[z], activated[z], span_count[z], north[z], south[z],
                        last_span[z], broke[z], 12)
            if (tier[z] == 1 or tier[z] == 2) and north[z] and south[z] and i >= last_span[z] + 2:
                a = mstart[i]; stop_before_close = mstop[i] - 1
                if stop_before_close > a:
                    dev_open = minute_open[a]; possible = False; mode = 0
                    if dev_open < bottom[z]:
                        mode = 1
                        for p in range(a, stop_before_close):
                            if minute_close[p] > top[z]: possible = True; break
                    elif dev_open > top[z]:
                        mode = -1
                        for p in range(a, stop_before_close):
                            if minute_close[p] < bottom[z]: possible = True; break
                    if possible:
                        for p in range(a, stop_before_close):
                            truth = minute_close[p] > top[z] if mode == 1 else minute_close[p] < bottom[z]
                            if truth and not active:
                                event_count = _emit(events, event_count, z, F_LADDER, K_PREVIEW,
                                    minute_close_ts[p], p, i, occurrence, 0, 0,
                                    tier[z], activated[z], span_count[z], north[z], south[z], last_span[z], broke[z], 10)
                                active = True; fired = True; qualified[z] = 1
                                if t0_ts[z] < 0:
                                    t0_ts[z] = minute_close_ts[p]; t0_pos[z] = p; t0_kind[z] = 1
                                    t0_exit[z] = 1 if minute_close[p] > top[z] else -1
                                    t0_close[z] = minute_close[p]; t0_span[z] = span_count[z]; t0_bar[z] = i
                                    event_count = _emit(events, event_count, z, F_LADDER, K_T0,
                                        minute_close_ts[p], p, i, -1, 0, 0,
                                        tier[z], activated[z], span_count[z], north[z], south[z],
                                        last_span[z], broke[z], 11)
                            elif not truth and active:
                                event_count = _emit(events, event_count, z, F_LADDER, K_CANCELLATION,
                                    minute_close_ts[p], p, i, occurrence, 0, 2,
                                    tier[z], activated[z], span_count[z], north[z], south[z], last_span[z], broke[z], 10)
                                active = False; occurrence += 1
                        full_truth = bc[i] > top[z] if mode == 1 else bc[i] < bottom[z]

            t0 = tier[z]; act0 = activated[z]; sc0 = span_count[z]
            n0 = north[z]; s0 = south[z]; ls0 = last_span[z]
            bmin = bo[i] if bo[i] < bc[i] else bc[i]
            bmax = bo[i] if bo[i] > bc[i] else bc[i]
            spanned = bmin < bottom[z] and bmax > top[z]
            survived = True
            if tier[z] == 3:
                returned = bh[i] >= bottom[z] if broke[z] else bl[i] <= top[z]
                survived = not returned
            elif spanned and (not activated[z] or i >= last_span[z] + 2):
                if not activated[z]:
                    activated[z] = 1; tier[z] = 1 if north[z] and south[z] else 2
                    span_count[z] = 1
                else: span_count[z] += 1
                last_span[z] = i
            else:
                breaker = 0
                if not spanned:
                    if south[z] and not north[z] and bo[i] > bottom[z] and bc[i] < bottom[z]: breaker = -1
                    elif north[z] and not south[z] and bo[i] < top[z] and bc[i] > top[z]: breaker = 1
                    elif north[z] and south[z] and bo[i] > bottom[z] and bc[i] < bottom[z] and bh[i] >= top[z] and bc[i] <= top[z]: breaker = -1
                    elif north[z] and south[z] and bo[i] < top[z] and bc[i] > top[z] and bl[i] <= bottom[z] and bc[i] >= bottom[z]: breaker = 1
                if breaker:
                    broke[z] = 1 if breaker == -1 else 0; north[z] = 0; south[z] = 0
                    tier[z] = 3; activated[z] = 1; last_span[z] = i
                else:
                    if bl[i] <= top[z] <= bh[i]: north[z] = 0
                    if bl[i] <= bottom[z] <= bh[i]: south[z] = 0
                    survived = bool(north[z] or south[z])

            if not act0 and activated[z]:
                event_count = _emit(events, event_count, z, F_ZONE, K_ACTIVATION, ts, pos, i,
                    -1, 0, 0, tier[z], activated[z], span_count[z], north[z], south[z], last_span[z], broke[z], 21)
                if first_activation_ts[z] < 0:
                    first_activation_ts[z] = ts; first_activation_pos[z] = pos; first_activation_bar[z] = i
            if span_count[z] > sc0:
                if sc0 == 2:
                    if x3_ts[z] < 0:
                        x3_ts[z] = ts; x3_pos[z] = pos; x3_bar[z] = i
                        x3_close[z] = bc[i]; x3_exit[z] = 1 if bc[i] > top[z] else -1
                        event_count = _emit(events, event_count, z, F_LADDER, K_X3_T0,
                            ts, pos, i, -1, 0, 0, t0, act0, sc0, n0, s0,
                            ls0, broke[z], 12)
                    if x3_bar[z] == i:
                        x3_confirmed[z] = 1
                event_count = _emit(events, event_count, z, F_ZONE, K_SPAN, ts, pos, i,
                    -1, 0, 0, tier[z], activated[z], span_count[z], north[z], south[z], last_span[z], broke[z], 22)
            if n0 and not north[z]:
                event_count = _emit(events, event_count, z, F_ZONE, K_NORTH_DEAD, ts, pos, i,
                    -1, 0, 0, tier[z], activated[z], span_count[z], north[z], south[z], last_span[z], broke[z], 23)
                if blue_end_ts[z] < 0: blue_end_ts[z] = ts; blue_end_pos[z] = pos
            if s0 and not south[z]:
                event_count = _emit(events, event_count, z, F_ZONE, K_SOUTH_DEAD, ts, pos, i,
                    -1, 0, 0, tier[z], activated[z], span_count[z], north[z], south[z], last_span[z], broke[z], 24)
                if blue_end_ts[z] < 0: blue_end_ts[z] = ts; blue_end_pos[z] = pos
            if t0 != 3 and tier[z] == 3:
                breaker_zones[breaker_count] = z; breaker_count += 1
                event_count = _emit(events, event_count, z, F_ZONE, K_BREAKER, ts, pos, i,
                    -1, 0, 0, tier[z], activated[z], span_count[z], north[z], south[z], last_span[z], broke[z], 25)
                if blue_end_ts[z] < 0: blue_end_ts[z] = ts; blue_end_pos[z] = pos
            if not survived:
                erratum = spanned and bool(act0) and i < ls0 + 2
                event_count = _emit(events, event_count, z, F_ZONE, K_ERRATUM if erratum else K_DELETED,
                    ts, pos, i, -1, 1, 0, tier[z], activated[z], span_count[z], north[z], south[z], last_span[z], broke[z], 26)
                if blue_end_ts[z] < 0: blue_end_ts[z] = ts; blue_end_pos[z] = pos
                deletion_ts[z] = ts; deletion_pos[z] = pos; deletion_bar[z] = i
                alive[z] = 0; was_blue[z] = 0

            confirmed_now = False
            if survived and (tier[z] == 1 or tier[z] == 2) and span_count[z] >= 2 and north[z] and south[z]:
                if not was_blue[z]:
                    event_count = _emit(events, event_count, z, F_LADDER, K_CONFIRMATION, ts,
                        pos, i, occurrence, 0, 0, tier[z], activated[z], span_count[z], north[z], south[z], last_span[z], broke[z], 30)
                    qualified[z] = 1; confirmed_now = True
                    if t0_ts[z] < 0:
                        t0_ts[z] = ts; t0_pos[z] = pos; t0_kind[z] = 2
                        t0_exit[z] = 1 if bc[i] > top[z] else -1
                        t0_close[z] = bc[i]; t0_span[z] = span_count[z]; t0_bar[z] = i
                        event_count = _emit(events, event_count, z, F_LADDER, K_T0,
                            ts, pos, i, -1, 0, 0, tier[z], activated[z],
                            span_count[z], north[z], south[z], last_span[z], broke[z], 31)
                    if confirm_ts[z] < 0:
                        confirm_ts[z] = ts; confirm_pos[z] = pos; confirm_bar[z] = i
                was_blue[z] = 1
            else: was_blue[z] = 0
            if fired and active and not confirmed_now and not (full_truth and survived):
                event_count = _emit(events, event_count, z, F_LADDER, K_CANCELLATION, ts,
                    pos, i, occurrence, 1, 2, tier[z], activated[z], span_count[z], north[z], south[z], last_span[z], broke[z], 40)

        # Update pass is complete; C3-close precursor formation happens now.
        while next_birth < n_zones and precursor_bar[next_birth] == i:
            z = next_birth; alive[z] = 1
            event_count = _emit(events, event_count, z, F_ZONE, K_BIRTH, precursor_formed_ts[z],
                precursor_formed_pos[z], i, -1, 0, 0, 0, 0, 0, 1, 1, -1, 0, 20)
            next_birth += 1

    for z in range(n_zones):
        if not qualified[z]: continue
        end_bar = first_activation_bar[z]
        for i in range(precursor_bar[z] + 1, end_bar + 1):
            a = mstart[i]; stop_before_close = mstop[i] - 1
            if stop_before_close <= a: continue
            dev_open = minute_open[a]; mode = 0; possible = False
            if dev_open < bottom[z]:
                mode = 1
                for p in range(a, stop_before_close):
                    if minute_close[p] > top[z]: possible = True; break
            elif dev_open > top[z]:
                mode = -1
                for p in range(a, stop_before_close):
                    if minute_close[p] < bottom[z]: possible = True; break
            if not possible: continue
            active = False; fired = False; occurrence = 0
            for p in range(a, stop_before_close):
                truth = minute_close[p] > top[z] if mode == 1 else minute_close[p] < bottom[z]
                if truth and not active:
                    event_count = _emit(events, event_count, z, F_LADDER, K_CANDIDATE,
                        minute_close_ts[p], p, i, occurrence, 0, 0,
                        0, 0, 0, 1, 1, -1, 0, 10)
                    active = True; fired = True
                elif not truth and active:
                    event_count = _emit(events, event_count, z, F_LADDER, K_CANCELLATION,
                        minute_close_ts[p], p, i, occurrence, 0, 1,
                        0, 0, 0, 1, 1, -1, 0, 10)
                    active = False; occurrence += 1
            full_truth = bc[i] > top[z] if mode == 1 else bc[i] < bottom[z]
            if fired and active and not full_truth:
                pos = mstop[i] - 1; ts = close_ts[i]
                event_count = _emit(events, event_count, z, F_LADDER, K_CANCELLATION,
                    ts, pos, i, occurrence, 1, 1, 0, 0, 0, 1, 1, -1, 0, 40)

    return (n_zones, event_count, events, top, bottom, birth_label, bullish,
            precursor_formed_ts, precursor_formed_pos, precursor_bar, first_activation_ts,
            first_activation_pos, first_activation_bar, t0_ts, t0_pos, t0_kind,
            t0_exit, t0_close, t0_span, t0_bar, x3_ts, x3_pos, x3_close, x3_exit,
            x3_bar, x3_confirmed, confirm_ts, confirm_pos, confirm_bar,
            blue_end_ts, blue_end_pos, deletion_ts, deletion_pos, deletion_bar,
            qualified, tier, activated, span_count, north, south, last_span, broke)


def replay_fast(bars: NativeArrays, minute_open: np.ndarray, minute_close: np.ndarray,
                minute_close_ts: np.ndarray, *, initial_event_capacity: int | None = None) -> FastFacts:
    births = precompute_births(bars)
    top, bottom, birth_label, bullish, precursor_bar, precursor_formed_ts, precursor_formed_pos = births
    n_zones = len(top)
    top_zones = np.argsort(top, kind="stable").astype(np.int64)
    bottom_zones = np.argsort(bottom, kind="stable").astype(np.int64)
    top_values = top[top_zones]
    bottom_values = bottom[bottom_zones]
    capacity = initial_event_capacity or max(1024, n_zones * 20)
    while True:
        result = _kernel_indexed(
            bars.open_ts_ns, bars.close_ts_ns, bars.open, bars.high, bars.low, bars.close,
            bars.minute_start, bars.minute_stop, np.asarray(minute_open),
            np.asarray(minute_close), np.asarray(minute_close_ts), top, bottom,
            birth_label, bullish, precursor_bar, precursor_formed_ts, precursor_formed_pos,
            top_values, top_zones, bottom_values, bottom_zones, capacity,
        )
        zone_count, event_count = result[0], result[1]
        if event_count <= capacity:
            break
        capacity = max(event_count, capacity * 2)
    names = [
        "top", "bottom", "birth_label", "bullish", "precursor_formed_ts", "precursor_formed_pos",
        "precursor_bar", "first_activation_ts", "first_activation_pos", "first_activation_bar",
        "t0_ts", "t0_pos", "t0_kind", "t0_exit_side", "t0_close", "t0_span", "t0_bar",
        "x3_t0_ts", "x3_t0_pos", "x3_t0_close", "x3_t0_exit_side", "x3_t0_bar",
        "x3_t0_confirmed",
        "native_blue_confirmation_ts", "native_blue_confirmation_pos",
        "native_blue_confirmation_bar",
        "blue_end_ts", "blue_end_pos", "deletion_ts", "deletion_pos", "deletion_bar",
        "qualified", "tier", "activated", "span_count", "north_alive", "south_alive",
        "last_span", "broke_south",
    ]
    arrays = {name: value[:zone_count] for name, value in zip(names, result[3:], strict=True)}
    return FastFacts(zone_count=zone_count, events=result[2][:event_count].copy(), **arrays)


KIND_NAMES = {
    K_BIRTH: ("precursor", "precursor_formed"),
    K_SPAN: ("native_state", "accepted_span"),
    K_NORTH_DEAD: ("native_state", "north_boundary_retired"),
    K_SOUTH_DEAD: ("native_state", "south_boundary_retired"),
    K_BREAKER: ("native_state", "breaker_entry"),
    K_ERRATUM: ("native_state", "deleted"),
    K_DELETED: ("native_state", "deleted"),
    K_CONFIRMATION: ("native_state", "native_confirmation"),
    K_T0: ("anchor", "t0"),
    K_X3_T0: ("anchor", "x3_t0"),
}


def facts_to_rows(facts: FastFacts, corpus_id: str, instrument: str, tf_minutes: int,
                  archive_last_ts_ns: int, archive_last_pos: int,
                  native_close_ts_ns: np.ndarray, native_minute_stop: np.ndarray,
                  ) -> tuple[list[dict], list[dict]]:
    zone_ids = [
        riz_id(corpus_id, instrument, tf_minutes, int(facts.precursor_formed_ts[z]),
               float(facts.top[z]), float(facts.bottom[z]), 1 if facts.bullish[z] else -1)
        for z in range(facts.zone_count)
    ]
    selected = facts.events[facts.qualified[facts.events[:, EV_ZONE]].astype(bool)]
    order = np.lexsort((selected[:, EV_RANK], selected[:, EV_POS], selected[:, EV_ZONE]))
    selected = selected[order]
    events: list[dict] = []
    by_zone: dict[int, list[dict]] = {}
    for raw in selected:
        z = int(raw[EV_ZONE])
        raw_kind = int(raw[EV_KIND])
        if raw_kind not in KIND_NAMES:
            continue
        # When the first Blue observation is the native close itself, the T0
        # anchor already carries that confirmation. Do not duplicate one fact.
        if (raw_kind == K_CONFIRMATION and facts.t0_kind[z] == 2
                and int(raw[EV_POS]) == int(facts.t0_pos[z])):
            continue
        rows = by_zone.setdefault(z, [])
        seq = len(rows)
        family, kind = KIND_NAMES[raw_kind]
        deletion_cause = None
        if raw_kind == K_ERRATUM:
            deletion_cause = "too_early_respan"
        elif raw_kind == K_DELETED:
            deletion_cause = "breaker_return" if int(raw[EV_TIER]) == 3 else "boundary_exhaustion"
        breaker_direction = None
        if raw_kind == K_BREAKER:
            breaker_direction = "south" if bool(raw[EV_BROKE]) else "north"
        row = {
            "event_id": event_id(zone_ids[z], family, kind, int(raw[EV_EVENT_TS]),
                                 int(raw[EV_BAR]), seq),
            "riz_id": zone_ids[z], "instrument": instrument, "tf_minutes": tf_minutes,
            "event_family": family, "event_kind": kind,
            "event_ts_ns": int(raw[EV_EVENT_TS]),
            "market_spine_pos": int(raw[EV_POS]), "native_bar_index": int(raw[EV_BAR]),
            "event_seq": seq, "terminal": bool(raw[EV_TERMINAL]),
            "deletion_cause": deletion_cause, "breaker_direction": breaker_direction,
            "tier": int(raw[EV_TIER]),
            "activated": bool(raw[EV_ACTIVATED]), "span_count": int(raw[EV_SPAN]),
            "north_alive": bool(raw[EV_NORTH]), "south_alive": bool(raw[EV_SOUTH]),
            "last_span_index": int(raw[EV_LAST_SPAN]), "broke_south": bool(raw[EV_BROKE]),
        }
        rows.append(row)
        events.append(row)

    # Archive exhaustion is a factual observation boundary, not a C1 state
    # change. Persist it explicitly so censoring is never inferred from absence.
    archive_bar = len(native_close_ts_ns) - 1
    for z in np.flatnonzero(facts.qualified & (facts.deletion_ts < 0)):
        z = int(z)
        rows = by_zone[z]
        seq = len(rows)
        row = {
            "event_id": event_id(zone_ids[z], "archive", "archive_censor",
                                 archive_last_ts_ns, archive_bar, seq),
            "riz_id": zone_ids[z], "instrument": instrument, "tf_minutes": tf_minutes,
            "event_family": "archive", "event_kind": "archive_censor",
            "event_ts_ns": archive_last_ts_ns,
            "market_spine_pos": archive_last_pos, "native_bar_index": archive_bar,
            "event_seq": seq, "terminal": True, "deletion_cause": None,
            "breaker_direction": None,
            "tier": int(facts.tier[z]), "activated": bool(facts.activated[z]),
            "span_count": int(facts.span_count[z]),
            "north_alive": bool(facts.north_alive[z]),
            "south_alive": bool(facts.south_alive[z]),
            "last_span_index": int(facts.last_span[z]),
            "broke_south": bool(facts.broke_south[z]),
        }
        rows.append(row)
        events.append(row)

    passports: list[dict] = []
    for z in np.flatnonzero(facts.qualified):
        z = int(z)
        deletion = int(facts.deletion_ts[z])
        censored = deletion < 0
        nullable = lambda value: None if int(value) < 0 else int(value)
        x3_exists = int(facts.x3_t0_ts[z]) >= 0
        passports.append({
            "riz_id": zone_ids[z], "corpus_id": corpus_id, "instrument": instrument,
            "tf_minutes": tf_minutes, "zone_top": float(facts.top[z]),
            "zone_bottom": float(facts.bottom[z]),
            "zone_width": float(facts.top[z] - facts.bottom[z]),
            "direction": 1 if facts.bullish[z] else -1, "bullish": bool(facts.bullish[z]),
            "precursor_formed_ts_ns": int(facts.precursor_formed_ts[z]),
            "precursor_formed_spine_pos": int(facts.precursor_formed_pos[z]),
            "precursor_native_bar_index": int(facts.precursor_bar[z]),
            "t0_ts_ns": int(facts.t0_ts[z]), "t0_spine_pos": int(facts.t0_pos[z]),
            "t0_kind": "minute_ignition" if facts.t0_kind[z] == 1 else "native_confirmation",
            "t0_exit_side": "north" if facts.t0_exit_side[z] == 1 else "south",
            "t0_close": float(facts.t0_close[z]), "t0_span_count": int(facts.t0_span[z]),
            "t0_native_bar_index": int(facts.t0_bar[z]),
            "native_blue_confirmation_ts_ns": nullable(facts.native_blue_confirmation_ts[z]),
            "native_blue_confirmation_spine_pos": nullable(facts.native_blue_confirmation_pos[z]),
            "native_blue_confirmation_native_bar_index": nullable(
                facts.native_blue_confirmation_bar[z]),
            "x3_t0_ts_ns": nullable(facts.x3_t0_ts[z]),
            "x3_t0_spine_pos": nullable(facts.x3_t0_pos[z]),
            "x3_t0_native_bar_index": nullable(facts.x3_t0_bar[z]),
            "x3_t0_close": float(facts.x3_t0_close[z]) if x3_exists else None,
            "x3_t0_exit_side": (("north" if facts.x3_t0_exit_side[z] == 1 else "south")
                                if x3_exists else None),
            "x3_t0_confirmed": bool(facts.x3_t0_confirmed[z]) if x3_exists else None,
            "blue_eligibility_end_ts_ns": nullable(facts.blue_end_ts[z]),
            "blue_eligibility_end_spine_pos": nullable(facts.blue_end_pos[z]),
            "c1_deletion_ts_ns": nullable(facts.deletion_ts[z]),
            "c1_deletion_spine_pos": nullable(facts.deletion_pos[z]),
            "c1_deletion_native_bar_index": nullable(facts.deletion_bar[z]),
            "censored": censored, "still_alive_at_archive_end": censored,
            "last_observed_ts_ns": archive_last_ts_ns if censored else deletion,
            "last_observed_spine_pos": archive_last_pos if censored else int(facts.deletion_pos[z]),
            "final_tier": int(facts.tier[z]), "final_activated": bool(facts.activated[z]),
            "final_span_count": int(facts.span_count[z]),
            "final_north_alive": bool(facts.north_alive[z]),
            "final_south_alive": bool(facts.south_alive[z]),
            "final_last_span_index": int(facts.last_span[z]),
            "final_broke_south": bool(facts.broke_south[z]),
        })
    passports.sort(key=lambda row: row["riz_id"])
    events.sort(key=lambda row: (row["riz_id"], row["event_seq"]))
    return passports, events
