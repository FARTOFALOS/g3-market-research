#!/usr/bin/env python3
"""095 barrier — настоящий first-touch: цель прочь / стоп к b, выход к close NY.

Пост-открытийное продолжение (ТФ>=60, since 0-120). Вход open[t0+1] прочь от b.
Цель T (прочь), стоп S (к b) — в пунктах, читаются из распределения продолжений
(discovery away-MFE / adverse-MFE), морозятся, проверяются на eval. First-touch
по реальным барам; гэп через уровень — по худшему open; оба в баре — пара;
не сработало к close NY — выход по close; гэп ленты — unknown. Cost 1.0. День — единица.
"""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit, prange

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'base/081')); sys.path.insert(0, str(ROOT / 'base/080'))
from trading import sessions, bar_session_map  # noqa
from tape import presence_grid, gap_kinds       # noqa
from paths import load_films                     # noqa
COST = 1.0
WIN, LOSS, AMB, TIME_X, UNK, NOEX = 0, 1, 2, 3, 4, 5


@njit(parallel=True)
def _bar(t0s, es, north, since, tf, opn, high, low, close, sess, last_bar, kg, last, T, S,
         o_kind, o_pay, o_day, ts):
    for i in prange(t0s.size):
        if tf[i] < 60 or since[i] < 0 or since[i] >= 120:
            o_kind[i] = NOEX; continue
        t0 = t0s[i]; up = north[i]; j = t0 + 1
        if j > last or kg[j] != 0 or sess[j] < 0:
            o_kind[i] = NOEX; continue
        sj = sess[j]; o = opn[j]; lb = last_bar[sj]
        o_day[i] = ts[j] // 86400000000000
        tgt = o + T if up else o - T
        stp = o - S if up else o + S
        res = UNK; pay = np.nan
        for b in range(j, lb + 1):
            if kg[b] != 0:
                break
            if up:
                op_thru = opn[b] <= stp; s_t = low[b] <= stp; t_t = high[b] >= tgt
            else:
                op_thru = opn[b] >= stp; s_t = high[b] >= stp; t_t = low[b] <= tgt
            if op_thru:
                fill = opn[b]; pay = (fill - o) if up else (o - fill); res = LOSS; break
            if s_t and t_t:
                res = AMB; break
            if s_t:
                pay = -S; res = LOSS; break
            if t_t:
                pay = T; res = WIN; break
            if b == lb:
                pay = (close[b] - o) if up else (o - close[b]); res = TIME_X
        o_kind[i] = res
        if res == WIN:
            o_pay[i] = T
        elif res == LOSS and np.isnan(pay):
            o_pay[i] = -S
        else:
            o_pay[i] = pay


def run(terr, T, S, cache):
    opn, high, low, close, ts, sess, last_bar, kg, last = cache
    f = load_films('NQ', terr)
    t0 = f.t0_spine_pos.to_numpy().astype(np.int64)
    e = f.exit_boundary.to_numpy().astype(np.float64)
    north = (f.side.to_numpy() == 'north')
    et = pd.to_datetime(f.t0_ts_ns.to_numpy(), utc=True).tz_convert('America/New_York')
    since = (et.hour.to_numpy() * 60 + et.minute.to_numpy() - 570).astype(np.int64)
    tf = f.tf_minutes.to_numpy().astype(np.int64)
    n = len(f)
    ok = np.full(n, NOEX, np.int8); pay = np.full(n, np.nan, np.float32); day = np.zeros(n, np.int64)
    _bar(t0, e, north, since, tf, opn, high, low, close, sess, last_bar, kg, last, float(T), float(S),
         ok, pay, day, ts)
    m = np.isin(ok, [WIN, LOSS, AMB, TIME_X])
    r = pd.DataFrame({'kind': ok[m], 'pay': pay[m], 'day': day[m]})
    # неоднозначные: пара границ -> публикуем fav/adv bound
    def net(amb_as):
        p = r.pay.to_numpy().copy()
        p[r.kind.to_numpy() == AMB] = T if amb_as == 'fav' else -S
        p = p - COST
        dv = pd.DataFrame({'d': r.day.to_numpy(), 'p': p}).groupby('d').p
        return p.mean(), dv.mean().mean(), (dv.sum() > 0).mean()
    ptf, dtf, ftf = net('fav'); pta, dta, fta = net('adv')
    amb = (r.kind == AMB).mean(); win = (r.kind == WIN).mean(); los = (r.kind == LOSS).mean()
    print(f'  {terr:>10} T{T}/S{S}: n={len(r):>6} win={win:.3f} los={los:.3f} amb={amb:.3f} | '
          f'day-mean fav={dtf:+.2f} adv={dta:+.2f} | frac+ {ftf:.2f}/{fta:.2f} | per-trade {ptf:+.2f}/{pta:+.2f}')
    return dtf, dta


if __name__ == '__main__':
    m = ROOT / 'data/market/NQ'
    opn = np.load(m / 'open.npy'); high = np.load(m / 'high.npy'); low = np.load(m / 'low.npy')
    close = np.load(m / 'close.npy'); ts = np.load(m / 'close_ts_utc_ns.npy')
    grid, glo, ghi = presence_grid(); kg, _ = gap_kinds('NQ', grid, glo, ghi)
    st, cl = sessions(); cl = cl.astype(np.int64); sess, last_bar = bar_session_map(ts, st, cl)
    cache = (opn, high, low, close, ts, sess, last_bar, kg, opn.size - 1)
    print('first-touch барьер, day-unit (fav=amb->цель, adv=amb->стоп); frozen T/S из распределения\n')
    for T, S in [(16, 12), (12, 8), (10, 6), (20, 12), (8, 8)]:
        run('discovery', T, S, cache)
        run('evaluation', T, S, cache)
        print()
