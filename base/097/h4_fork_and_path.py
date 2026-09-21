#!/usr/bin/env python3
"""097 additional search (after H1-H3), question changed by the lens set of 2026-09-21.

Changed object. Not "a separator at the contact" but "what space of continuations
the path LEFT". For trading, the path survives in the present only as
  (1) levels it created or consumed  -> the fork geometry,
  (2) the scale it revealed,  (3) the clock it spent.
Two approaches that leave the same fork, scale and clock are the same state for
our purpose. So "the path matters" means: at a FIXED fork, the money of the fork
still depends on a relation that cannot be rebuilt from the current slice.

Scene   NQ, certified first contact of own b, contact on T0+3 or later (a real
        departure exists: 30 % of RIZ), 2021-2025, one bet per (minute, side).
Fork    both levels come from the scene, nothing chosen:
        up   = U, the outward extreme made before the contact bar (unfinished)
        down = the far boundary of the zone (not yet reached)
        entry e = open[k+1]; d = U - e, s = e + w; fair line p0 = s / (d + s).
        win: range passes U by a tick (exit at U); loss: range passes far by a
        tick (exit there, worse open on a gap); both in one minute -> loss in the
        primary count, win in the upper bound; else exit at the 16:00 ET close.
        Known in advance from 056: a neighbouring outward fork is fair (gross
        +0.006), so overall G near zero is EXPECTED and is not the question.
Path    three relations, each scale-free, parameter-free, prefix-only, and not
        recoverable from price, time and levels at the contact close:
        asym  = (k - iU) / (k - T0)         when the extreme was made: early/late
        conf  = max close before k / U      how much of U was accepted by bodies
        brake = (C[k-1]-C[k]) / ((U-C[k]) / (k-iU))   contact bar against the
                average pace of the whole return: >1 accelerates into the line,
                <1 brakes, <0 closes back outward
Readout G in points and G/(d+s) (money per unit of fork width), terciles of each
        relation, top minus bottom, day-block t, sign by year. Shape of winners:
        duration and heat as a share of s.
Rule    a relation changes the opportunity only if top-bottom contrast of
        G/(d+s) has |t| >= 3 pooled and one sign in >= 4 of 5 years.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from race_money import cutoffs, O, H, L, C, TS, SID, MIN

OUT = Path(__file__).resolve().parents[2] / 'work' / '097'
TICK = 0.25
COST = 1.00


def build():
    t = pd.read_parquet(OUT / 'contact_rows_NQ.parquet')
    t = t[t.wait >= 3].sort_values(['k', 'side', 't0_spine_pos', 'tf_minutes', 'riz_id'])
    t = t.groupby(['k', 'side']).head(1)
    cut, _ = cutoffs()
    rows = []; skip = dict(no_entry=0, u_consumed=0, no_fork=0)
    for r in t.itertuples():
        k, p = int(r.k), int(r.t0_spine_pos); ce = int(cut[k])
        s_ = 1.0 if r.side == 'north' else -1.0
        B, w, U = r.exit_boundary, r.w, r.U
        if ce <= k or TS[k + 1] - TS[k] != MIN or SID[k + 1] != SID[k]:
            skip['no_entry'] += 1; continue
        if r.k_new_ext > 0:
            skip['u_consumed'] += 1; continue          # contact bar already revisited U
        e = s_ * (O[k + 1] - B)
        d, s = U - e, e + w
        if d <= TICK or s <= TICK:
            skip['no_fork'] += 1; continue
        oh = (H[p:k] - B) if s_ > 0 else (B - L[p:k])
        oc = s_ * (C[p:k + 1] - B)
        iU = p + int(np.argmax(oh))
        asym = (k - iU) / (k - p)
        conf = float(oc[:-1].max()) / U
        pace = (U - oc[-1]) / (k - iU)
        brake = (oc[-2] - oc[-1]) / pace if pace > 0 else np.nan
        seg = slice(k + 1, ce + 1)
        ts = TS[seg]
        brk = np.flatnonzero(np.diff(ts) != MIN)
        n = (brk[0] + 1) if brk.size else len(ts)
        hi = (H[seg][:n] - B) if s_ > 0 else (B - L[seg][:n])
        lo = (L[seg][:n] - B) if s_ > 0 else (B - H[seg][:n])
        op = s_ * (O[seg][:n] - B)
        iw = np.flatnonzero(hi >= U + TICK); il = np.flatnonzero(lo <= -w - TICK)
        jw = iw[0] if iw.size else 10 ** 9; jl = il[0] if il.size else 10 ** 9
        amb = jw == jl and jw < 10 ** 9
        if jw == jl == 10 ** 9:
            kind, g, g_up, j = 'time', s_ * (C[seg][n - 1] - O[k + 1]), None, n - 1
        elif jl <= jw:
            j = jl; kind = 'loss'; g = min(-w - TICK, op[j]) - e
        else:
            j = jw; kind = 'win'; g = U - e
        g_up = (U - e) if amb else g
        heat = (e - lo[:j + 1].min()) / s if kind == 'win' else np.nan
        rows.append(dict(k=k, sid=r.sid, year=r.year, side=r.side, w=w, U=U, e=e, d=d, s=s,
                         p0=s / (d + s), kind=kind, amb=bool(amb), g=g, g_up=g_up, dur=j + 1,
                         heat=heat, asym=asym, conf=conf, brake=brake, wait=r.wait))
    x = pd.DataFrame(rows)
    x.to_parquet(OUT / 'h4_rows.parquet')
    print('skipped:', skip)
    return x


def tday(x, col):
    day = x.groupby('sid')[col].agg(['sum', 'size'])
    m = x[col].mean()
    se = np.sqrt(((day['sum'] - m * day['size']) ** 2).sum()) / day['size'].sum()
    return m, (m / se if se > 0 else np.nan)


def contrast(a, b, col):
    da = a.groupby('sid')[col].mean(); db = b.groupby('sid')[col].mean()
    dd = (da - db).dropna()
    return dd.mean(), dd.mean() / (dd.std(ddof=1) / np.sqrt(len(dd))), len(dd)


if __name__ == '__main__':
    x = build() if 'run' in sys.argv else pd.read_parquet(OUT / 'h4_rows.parquet')
    x['gn'] = x.g / (x.d + x.s)
    d = x[x.year <= 2025]
    print('fork-valid moments 2021-2025:', len(d), '| per session %.1f | sessions with >=1: %.3f of 1284'
          % (len(d) / 1284, d.sid.nunique() / 1284))
    print('geometry, pts: d p50 %.2f  s p50 %.2f  p0 p50 %.3f | ambiguous minute %.3f'
          % (d.d.median(), d.s.median(), d.p0.median(), d.amb.mean()))
    m, t = tday(d, 'g'); mu, tu = tday(d, 'g_up')
    print('WHOLE FORK: win %.3f vs fair %.3f | time-exit %.3f | G %+.3f pt (t %.2f), upper bound %+.3f (t %.2f), net %+.3f'
          % ((d.kind == 'win').mean(), d.p0.mean(), (d.kind == 'time').mean(), m, t, mu, tu, m - COST))
    print('along the fair line (quintiles of p0): win / fair / G')
    d = d.assign(pq=pd.qcut(d.p0, 5, labels=False))
    for q, y in d.groupby('pq'):
        m, t = tday(y, 'g')
        print('   p0 %.2f-%.2f  n %5d  win %.3f  fair %.3f  G %+.3f (t %.2f)'
              % (y.p0.min(), y.p0.max(), len(y), (y.kind == 'win').mean(), y.p0.mean(), m, t))
    print('\nPATH RELATIONS at the netted fork: terciles, G pts | G/(d+s) | winners: duration p50, heat p50')
    for col in ['asym', 'conf', 'brake']:
        y = d.dropna(subset=[col]).copy()
        y['tc'] = pd.qcut(y[col].rank(method='first'), 3, labels=False)
        print(' %s  (cuts %.2f / %.2f)' % (col, y[col].quantile(1 / 3), y[col].quantile(2 / 3)))
        for q, z in y.groupby('tc'):
            m, t = tday(z, 'g'); mn, tn = tday(z, 'gn'); wz = z[z.kind == 'win']
            print('   T%d n %5d  p0 %.3f win %.3f | G %+.3f (t %.2f) | Gn %+.4f (t %.2f) | dur %3.0f heat %.2f'
                  % (q + 1, len(z), z.p0.mean(), (z.kind == 'win').mean(), m, t, mn, tn,
                     wz.dur.median(), wz.heat.median()))
        cm, ct, nd = contrast(y[y.tc == 2], y[y.tc == 0], 'gn')
        signs = [np.sign(contrast(y[(y.tc == 2) & (y.year == yr)], y[(y.tc == 0) & (y.year == yr)], 'gn')[0])
                 for yr in range(2021, 2026)]
        print('   top-bottom Gn %+.4f  t %.2f  days %d | sign by year %s' % (cm, ct, nd, signs))
