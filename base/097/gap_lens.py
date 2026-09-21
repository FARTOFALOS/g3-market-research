#!/usr/bin/env python3
"""Line 097: research LENS over the tape, next to the frozen field (permission of 2026-09-22, this line only).
It replays the pinned machine (reference/pine/RIZ_BLUE_v1.0.pine, lines 119-213) on native bars and keeps EVERY
gap, not only those that become Blue. Nothing is written into the field; the field is used only to check the
instrument: the lens must reproduce the Blue zones of the field.

Logged per gap, all observable at native closes, no price outcome:
  birth bar, bounds, r0 (range of C3), bull
  first contact: bar, kind, and the price facts that exist independently of how Pine archives the object -
     pen  = how deep the bar's extreme went into / through the zone, in zone widths (0 = touched the near edge,
            1 = reached the far edge, >1 = beyond)
     clo  = where the bar CLOSED relative to the near edge, in zone widths (>0 = back on the impulse side,
            0..-1 = inside the zone, <-1 = beyond the far edge)
     kind = 0 span (body covered the whole zone) / 1 closed inside / 2 closed back on the impulse side / 3 other
  run   = how far the impulse ran past the zone before the first contact, in r0
  span1, span2(Blue) bars, deletion bar and cause (1 exhaustion by touches, 2 too early re-span, 3 breaker return)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[2] / 'work' / 'line-097'
MIN = 60_000_000_000


def native_bars(inst, tf, y0=2020):
    mk = ROOT / f'data/market/{inst}'
    O, H, L, C, TS, SID = (np.load(mk / f'{n}.npy') for n in ('open', 'high', 'low', 'close', 'close_ts_utc_ns', 'session_id'))
    z = np.load(mk / 'sessions.npz'); op = dict(zip(z['session_id'].tolist(), z['session_open_utc_ns'].tolist()))
    keep = pd.to_datetime(TS).year >= y0
    pos = np.arange(len(TS))[keep]; sid = SID[keep]
    k = ((TS[keep] - np.array([op[int(s)] for s in sid])) // MIN - 1) // tf
    g = pd.DataFrame(dict(pos=pos, key=sid.astype(np.int64) * 100000 + k, o=O[keep], h=H[keep], l=L[keep], c=C[keep]))
    a = g.groupby('key', sort=True).agg(o=('o', 'first'), h=('h', 'max'), l=('l', 'min'), c=('c', 'last'), end=('pos', 'last'))
    return a.o.to_numpy(), a.h.to_numpy(), a.l.to_numpy(), a.c.to_numpy(), a.end.to_numpy()


@njit(cache=True)
def machine(o, h, l, c):
    n = len(o); M = n // 2 + 16
    zt = np.empty(M); zb = np.empty(M); bull = np.zeros(M, np.bool_); birth = np.zeros(M, np.int64); r0 = np.empty(M)
    nn = np.ones(M, np.bool_); ss = np.ones(M, np.bool_); ac = np.zeros(M, np.bool_); cr = np.full(M, -1, np.int64)
    tier = np.zeros(M, np.int8); t3 = np.zeros(M, np.bool_); nx = np.zeros(M, np.int64)
    fc = np.full(M, -1, np.int64); fkind = np.full(M, -1, np.int8); pen = np.full(M, np.nan); clo = np.full(M, np.nan); run = np.zeros(M)
    s1 = np.full(M, -1, np.int64); blue = np.full(M, -1, np.int64); dead = np.full(M, -1, np.int64); cause = np.zeros(M, np.int8)
    alive = np.empty(M, np.int64); na = 0; m = 0
    for i in range(n):
        bmin = min(o[i], c[i]); bmax = max(o[i], c[i]); j = 0
        while j < na:
            q = alive[j]; kill = False
            if tier[q] == 3:
                if (t3[q] and h[i] >= zb[q]) or ((not t3[q]) and l[i] <= zt[q]):
                    kill = True; cause[q] = 3
            else:
                touch = l[i] <= zt[q] and h[i] >= zb[q]
                if fc[q] < 0:
                    if touch:
                        w = zt[q] - zb[q]; fc[q] = i
                        if bull[q]:
                            pen[q] = (zt[q] - l[i]) / w; clo[q] = (c[i] - zt[q]) / w
                        else:
                            pen[q] = (h[i] - zb[q]) / w; clo[q] = (zb[q] - c[i]) / w
                    else:
                        e = (h[i] - zt[q]) if bull[q] else (zb[q] - l[i])
                        if e / r0[q] > run[q]:
                            run[q] = e / r0[q]
                span = bmin < zb[q] and bmax > zt[q]
                if fc[q] == i:
                    fkind[q] = 0 if span else (1 if -1.0 <= clo[q] <= 0.0 else (2 if clo[q] > 0.0 else 3))
                if span and ((not ac[q]) or i >= cr[q] + 2):
                    if not ac[q]:
                        ac[q] = True; tier[q] = 1 if (nn[q] and ss[q]) else 2; nx[q] = 1; s1[q] = i
                    else:
                        nx[q] += 1
                        if nx[q] >= 2 and nn[q] and ss[q] and blue[q] < 0:
                            blue[q] = i
                    cr[q] = i
                else:
                    sbk = (not span) and ss[q] and (not nn[q]) and o[i] > zb[q] and c[i] < zb[q]
                    nbk = (not span) and nn[q] and (not ss[q]) and o[i] < zt[q] and c[i] > zt[q]
                    sbk2 = (not span) and nn[q] and ss[q] and o[i] > zb[q] and c[i] < zb[q] and h[i] >= zt[q] and c[i] <= zt[q]
                    nbk2 = (not span) and nn[q] and ss[q] and o[i] < zt[q] and c[i] > zt[q] and l[i] <= zb[q] and c[i] >= zb[q]
                    if sbk or nbk or sbk2 or nbk2:
                        nn[q] = False; ss[q] = False; tier[q] = 3; t3[q] = sbk or sbk2; ac[q] = True; cr[q] = i
                    else:
                        if l[i] <= zt[q] and h[i] >= zt[q]:
                            nn[q] = False
                        if l[i] <= zb[q] and h[i] >= zb[q]:
                            ss[q] = False
                        if (not nn[q]) and (not ss[q]):
                            kill = True; cause[q] = 2 if (span and ac[q] and i == cr[q] + 1) else 1
            if kill:
                dead[q] = i; na -= 1; alive[j] = alive[na]
            else:
                j += 1
        if i >= 2 and m < M - 2:
            if l[i] > h[i - 2] and max(o[i - 1], c[i - 1]) > h[i - 2] and min(o[i - 1], c[i - 1]) < l[i]:
                bt1 = max(o[i - 2], c[i - 2]); bt2 = max(o[i - 1], c[i - 1]); c2bb = min(o[i - 1], c[i - 1]); c3bb = min(o[i], c[i])
                nb_ = bt1 if c2bb > bt1 else h[i - 2]; nt_ = c3bb if c3bb > bt2 else l[i]
                if nt_ > nb_:
                    zt[m] = nt_; zb[m] = nb_; bull[m] = True; birth[m] = i; r0[m] = max(h[i] - l[i], 1e-9); alive[na] = m; na += 1; m += 1
            if h[i] < l[i - 2] and min(o[i - 1], c[i - 1]) < l[i - 2] and max(o[i - 1], c[i - 1]) > h[i]:
                bb1 = min(o[i - 2], c[i - 2]); bb2 = min(o[i - 1], c[i - 1]); c2bt = max(o[i - 1], c[i - 1]); c3bt = max(o[i], c[i])
                nt_ = bb1 if c2bt < bb1 else l[i - 2]; nb_ = c3bt if c3bt < bb2 else h[i]
                if nt_ > nb_:
                    zt[m] = nt_; zb[m] = nb_; bull[m] = False; birth[m] = i; r0[m] = max(h[i] - l[i], 1e-9); alive[na] = m; na += 1; m += 1
    return zt[:m], zb[:m], bull[:m], birth[:m], r0[:m], fc[:m], fkind[:m], pen[:m], clo[:m], run[:m], s1[:m], blue[:m], dead[:m], cause[:m]


def run_tf(inst, tf):
    o, h, l, c, end = native_bars(inst, tf)
    r = machine(o, h, l, c)
    d = pd.DataFrame(dict(zip(['zt', 'zb', 'bull', 'birth', 'r0', 'fc', 'fkind', 'pen', 'clo', 'run', 's1', 'blue', 'dead', 'cause'], r)))
    d['tf'] = tf; d['birth_pos'] = end[d.birth.to_numpy()]
    d['blue_pos'] = np.where(d.blue >= 0, end[np.maximum(d.blue.to_numpy(), 0)], -1)
    return d, end


if __name__ == '__main__':
    import pyarrow.parquet as pq
    inst = 'NQ'
    for tf in [int(x) for x in (sys.argv[1:] or ['15'])]:
        d, end = run_tf(inst, tf)
        d.to_parquet(OUT / f'gaps_{inst}_{tf}.parquet')
        p = pq.read_table(ROOT / f'data/field/{inst}/cells/tf_{tf:04d}/passports.parquet', columns=[
            'zone_top', 'zone_bottom', 'native_blue_confirmation_spine_pos', 't0_ts_ns', 't0_kind']).to_pandas()
        p = p[pd.to_datetime(p.t0_ts_ns).dt.year.between(2021, 2025)]
        conf = p[p.native_blue_confirmation_spine_pos > 0]
        mine = d[d.blue >= 0]
        key = lambda a, b, c_: set(zip(np.round(a, 2), np.round(b, 2), c_.astype(np.int64)))
        kf = key(conf.zone_top, conf.zone_bottom, conf.native_blue_confirmation_spine_pos)
        km = key(mine.zt, mine.zb, mine.blue_pos)
        print('TF %d: gaps born %d | lens Blue %d | field zones 2021-2025: %d, of them natively confirmed %d | reproduced by the lens %d (%.3f)'
              % (tf, len(d), len(mine), len(p), len(conf), len(kf & km), len(kf & km) / max(1, len(kf))))
