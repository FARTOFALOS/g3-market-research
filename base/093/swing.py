#!/usr/bin/env python3
"""093 — новая action family: вход на развороте от дальнего экстремума, структурный стоп до `b`.

МОТИВАЦИЯ (из открытого вопроса 092)
====================================
092 показал: возврат к собственной `b` частый, но раннее no-stop участие имеет
тяжёлый коррелированный adverse tail. Здесь другой вход: НЕ рано и вслепую, а
после того как цена сделала новый дальний экстремум и повернула назад к `b`.
Риск задаёт сам график — стоп за экстремумом, — а не выдуманное число.

Это ОТДЕЛЬНАЯ declared action family, а не rescue v1. Region/threshold/stop не
подбираются под результат: пороги не вводятся, строится рельеф по геометрии.

FREEZE (до счёта)
=================
Объект/окно/расход — 080A/081 без переопределения (NQ, canonical live Film-1,
03:00 ET → закрытие XNYS, 1 пункт круг).
Recognition (prefix-only), north-фильм: подтверждённый swing-high — бар `p`, чей
high строго больше w=3 соседей с каждой стороны, И новый экстремум фильма
(high[p] > всех high от T0 до p-1). Подтверждение на баре c=p+w; вход open[c+1].
South — зеркально swing-low / новый минимум.
Стоп: структурный, за экстремумом (high[p]+1 тик для short; low[p]-1 тик для long).
Цель: собственная `b`. Гэп через стоп — по худшему open. Оба барьера в баре —
пара границ. Не дошло к закрытию — выход по close (NY-close). Потеря
наблюдаемости — unknown (bounded, не ноль).
Разрешено действие на КАЖДОМ новом экстремум-развороте; первый-на-фильм помечен.
w=3 заморожен; w∈{2,5} — только sensitivity, не отбор.
"""
from __future__ import annotations
import os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit, prange

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = Path(os.environ.get('G3_093_OUT', str(ROOT / 'work/093')))
sys.path.insert(0, str(ROOT / 'base/081'))
sys.path.insert(0, str(ROOT / 'base/080'))
from trading import sessions, bar_session_map                          # noqa: E402
from tape import presence_grid, gap_kinds                              # noqa: E402
from paths import load_films                                           # noqa: E402

TICK = {'NQ': 0.25, 'ES': 0.25, 'YM': 1.0}
COST = {'NQ': 1.00, 'ES': 1.00, 'YM': 4.00}
PV = {'NQ': 20.0, 'ES': 50.0, 'YM': 5.0}

WIN, LOSS, AMBIG, TIME_EXIT, UNKNOWN, NO_EXEC, MISSED = 0, 1, 2, 3, 4, 5, 6
NAMES = {WIN: 'win', LOSS: 'loss', AMBIG: 'ambig', TIME_EXIT: 'time_exit',
         UNKNOWN: 'unknown', NO_EXEC: 'no_exec', MISSED: 'missed_or_invalid'}


@njit(cache=True)
def _is_pivot(high, low, p, w, up, prefix_ext):
    """north: строгий локальный максимум high[p] по [p-w,p+w] И новый экстремум."""
    if up:
        v = high[p]
        if v <= prefix_ext:            # не новый экстремум
            return False
        for k in range(1, w + 1):
            if high[p - k] >= v or high[p + k] >= v:
                return False
        return True
    else:
        v = low[p]
        if v >= prefix_ext:
            return False
        for k in range(1, w + 1):
            if low[p - k] <= v or low[p + k] <= v:
                return False
        return True


@njit(parallel=True, cache=True)
def _count(t0s, Fs, north, w, high, low, cnt):
    for i in prange(t0s.size):
        t0 = t0s[i]; F = Fs[i]; up = north[i]
        ext = -np.inf if up else np.inf
        n = 0
        for p in range(t0, F + 1):
            # обновляем prefix-экстремум ПОсле проверки (prefix = до p)
            if p >= t0 + w and p + w <= F:
                if _is_pivot(high, low, p, w, up, ext):
                    n += 1
            v = high[p] if up else low[p]
            if up:
                if v > ext:
                    ext = v
            else:
                if v < ext:
                    ext = v
        cnt[i] = n


@njit(parallel=True, cache=True)
def _fill(t0s, Fs, ends, es, north, off, w,
          opn, high, low, close, sess, last_bar, kind_gap, tick,
          o_kind, o_g, o_stop, o_ttc, o_dur, o_pay, o_lo, o_hi, o_mae, o_first,
          o_age, o_dclose, close_ns, ts):
    for i in prange(t0s.size):
        t0 = t0s[i]; F = Fs[i]; end_f = ends[i]; e = es[i]; up = north[i]
        base = off[i]
        ext = -np.inf if up else np.inf
        idx = 0
        first = 1
        for p in range(t0, F + 1):
            if p >= t0 + w and p + w <= F and _is_pivot(high, low, p, w, up, ext):
                pos = base + idx; idx += 1
                c = p + w; j = c + 1
                o_first[pos] = first; first = 0
                q = c
                o_age[pos] = q - t0
                o_dclose[pos] = (close[q] - e) if up else (e - close[q])
                sq = sess[q]
                if sq >= 0:
                    o_ttc[pos] = (close_ns[sq] - ts[q]) / 6.0e10
                # исполнимость
                if j > F or kind_gap[j] != 0:
                    o_kind[pos] = NO_EXEC
                else:
                    sj = sess[j]
                    if sj < 0:
                        o_kind[pos] = NO_EXEC
                    else:
                        o = opn[j]
                        g = (o - e) if up else (e - o)
                        stop = (high[p] + tick) if up else (low[p] - tick)
                        sd = (stop - o) if up else (o - stop)
                        if g <= 0.0 or sd <= 0.0:
                            o_kind[pos] = MISSED
                        else:
                            lb = last_bar[sj]
                            limit = lb if lb < end_f else end_f
                            o_g[pos] = g; o_stop[pos] = sd
                            res = UNKNOWN; pay = np.nan; lo = np.nan; hi = np.nan
                            dur = 0; run_adv = 0.0
                            for b in range(j, limit + 1):
                                if up:
                                    adv = high[b] - o
                                    open_through = opn[b] >= stop
                                    stop_touch = high[b] >= stop
                                    tp_touch = low[b] <= e
                                else:
                                    adv = o - low[b]
                                    open_through = opn[b] <= stop
                                    stop_touch = low[b] <= stop
                                    tp_touch = high[b] >= e
                                if adv > run_adv:
                                    run_adv = adv
                                if open_through:
                                    fill = opn[b]
                                    loss = (fill - o) if up else (o - fill)
                                    res = LOSS; pay = -loss; lo = -loss; hi = -loss
                                    dur = b - q; break
                                if stop_touch and tp_touch:
                                    res = AMBIG; lo = -sd; hi = g; dur = b - q; break
                                if stop_touch:
                                    res = LOSS; pay = -sd; lo = -sd; hi = -sd
                                    dur = b - q; break
                                if tp_touch:
                                    res = WIN; pay = g; lo = g; hi = g
                                    dur = b - q; break
                            if res == UNKNOWN:
                                if limit == lb:
                                    px = close[lb]
                                    pay = (o - px) if up else (px - o)
                                    res = TIME_EXIT; lo = pay; hi = pay; dur = lb - q
                                else:
                                    dur = limit - q
                            o_kind[pos] = res; o_pay[pos] = pay
                            o_lo[pos] = lo; o_hi[pos] = hi
                            o_dur[pos] = dur; o_mae[pos] = run_adv
            v = high[p] if up else low[p]
            if up:
                if v > ext:
                    ext = v
            else:
                if v < ext:
                    ext = v


def build(inst='NQ', terr='discovery', w=3):
    m = ROOT / 'data/market' / inst
    opn = np.load(m / 'open.npy'); high = np.load(m / 'high.npy')
    low = np.load(m / 'low.npy'); close = np.load(m / 'close.npy')
    ts = np.load(m / 'close_ts_utc_ns.npy')
    grid, glo, ghi = presence_grid()
    kind_gap, _ = gap_kinds(inst, grid, glo, ghi)
    st, cl = sessions(); cl = cl.astype(np.int64)
    sess, last_bar = bar_session_map(ts, st, cl)

    f = load_films(inst, terr)
    t0 = f.t0_spine_pos.to_numpy().astype(np.int64)
    F = f.certified_fresh_until_pos_strict.to_numpy().astype(np.int64)
    c = f.first_observed_contact_pos.to_numpy().astype(np.int64)
    cert = (f.film1_status.to_numpy() == 'contact_certified')
    end = np.where(cert, c, F)
    e = f.exit_boundary.to_numpy().astype(np.float64)
    north = (f.side.to_numpy() == 'north')

    cnt = np.zeros(len(f), np.int64)
    _count(t0, F, north, w, high, low, cnt)
    off = np.concatenate(([0], np.cumsum(cnt)))
    total = int(off[-1])

    z = lambda dt, v=0: np.full(total, v, dtype=dt)
    o = dict(kind=z(np.int8, NO_EXEC), g=z(np.float32, np.nan), stop=z(np.float32, np.nan),
             ttc=z(np.float32, np.nan), dur=z(np.int32), pay=z(np.float32, np.nan),
             lo=z(np.float32, np.nan), hi=z(np.float32, np.nan), mae=z(np.float32, np.nan),
             first=z(np.int8), age=z(np.int32), dclose=z(np.float32, np.nan))
    t = time.time()
    _fill(t0, F, end, e, north, off[:-1], w,
          opn, high, low, close, sess, last_bar, kind_gap, TICK[inst],
          o['kind'], o['g'], o['stop'], o['ttc'], o['dur'], o['pay'], o['lo'], o['hi'],
          o['mae'], o['first'], o['age'], o['dclose'], cl, ts)
    secs = round(time.time() - t, 2)

    film_ix = np.repeat(np.arange(len(f)), cnt)
    entry_day = np.where(o['kind'] <= TIME_EXIT,
                         ts[np.clip(t0[film_ix] + o['age'] + 1, 0, len(ts) - 1)] // 86400000000000, -1)
    r = pd.DataFrame({
        'film': film_ix.astype(np.int32), 'riz_id': f.riz_id.to_numpy()[film_ix],
        'tf': f.tf_minutes.to_numpy().astype(np.int32)[film_ix], 'north': north[film_ix],
        'first': o['first'], 'age': o['age'], 'kind': o['kind'],
        'd_close': o['dclose'], 'target': o['g'], 'stop_dist': o['stop'],
        'ttc': o['ttc'], 'dur': o['dur'], 'pay': o['pay'], 'lo': o['lo'], 'hi': o['hi'],
        'mae': o['mae'], 't0_ns': f.t0_ts_ns.to_numpy()[film_ix], 'day': entry_day.astype(np.int64)})
    r['rr'] = r.target / r.stop_dist
    r.attrs['seconds'] = secs
    r.attrs['w'] = w
    return r


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    terr = sys.argv[1] if len(sys.argv) > 1 else 'discovery'
    w = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    for inst in (sys.argv[2:3] or ['NQ']):
        r = build(inst, terr, w)
        p = OUT / f'swing_{inst}_{terr}_w{w}.parquet'
        r.to_parquet(p, index=False, compression='zstd')
        vc = {NAMES[k]: int(v) for k, v in r.kind.value_counts().items()}
        print(f'{inst}/{terr}/w{w}: {len(r)} pivots, walk {r.attrs["seconds"]}s, {p.stat().st_size/1e6:.1f}MB')
        print('  kinds:', vc)
