#!/usr/bin/env python3
"""097 H5: was "same fork + same scale + same clock = same state" too coarse?

H4 showed only that three path relations do not move the MEAN money of one fork.
A martingale fixes the mean of any stopped bet and nothing else: order of
movement, time to resolution, the adverse tail of winners and how far losers
first travelled are all free. If the path changes those at a fixed fork, the
action space differs although win rate and mean do not, and the equivalence
relation of H4 is too coarse. Declared before the count.

Same population and fork as H4 (U against the far boundary, entry open[k+1]).
Scale  mr = median bar range over T0..k (prefix only); dn = d/mr, sn = s/mr.
Match  cells = p0 quintile x quintile of dn*sn (diffusion time of the fork).
       Every readout is demeaned inside its cell before contrasting.
Read   logT   log bars to resolution (all resolved)
       heat   winners: deepest adverse excursion before the win, share of s
       prog   losers: furthest favourable travel before the loss, share of d
       half   losers that first covered >= half of d (an intermediate decision
              was on offer)
By     asym, conf, brake (H4) and, for comparison, ttc = minutes to 16:00 ET.
Rule   too coarse if any relation shifts any readout, top minus bottom tercile,
       |t| >= 3 by session-day blocks AND one sign in >= 4 of 5 years.
Also   the dangerous check for the thought born after 097 ("the clock breaks the
       fair line"): G of THIS fork by the ET clock of the contact, 5 buckets
       named by the session calendar, not by result.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from race_money import cutoffs, O, H, L, C, TS, SID, MIN

OUT = Path(__file__).resolve().parents[2] / 'work' / '097'
TICK = 0.25


def build():
    x = pd.read_parquet(OUT / 'h4_rows.parquet')
    rows = pd.read_parquet(OUT / 'contact_rows_NQ.parquet')[['k', 'side', 't0_spine_pos', 'exit_boundary', 'tf_minutes', 'riz_id', 'wait']]
    rows = rows[rows.wait >= 3].sort_values(['k', 'side', 't0_spine_pos', 'tf_minutes', 'riz_id']).groupby(['k', 'side']).head(1)
    x = x.merge(rows[['k', 'side', 't0_spine_pos', 'exit_boundary']], on=['k', 'side'], how='left')
    cut, mod = cutoffs()
    mr = np.empty(len(x)); heat = np.full(len(x), np.nan); prog = np.full(len(x), np.nan)
    for i, r in enumerate(x.itertuples()):
        k, p = int(r.k), int(r.t0_spine_pos)
        s_ = 1.0 if r.side == 'north' else -1.0
        mr[i] = float(np.median(H[p:k + 1] - L[p:k + 1]))
        if r.kind == 'time':
            continue
        a, b = k + 1, k + 1 + int(r.dur)
        hi = (H[a:b] - r.exit_boundary) if s_ > 0 else (r.exit_boundary - L[a:b])
        lo = (L[a:b] - r.exit_boundary) if s_ > 0 else (r.exit_boundary - H[a:b])
        if r.kind == 'win':
            heat[i] = max(0.0, r.e - lo.min()) / r.s
        else:
            prog[i] = max(0.0, hi[:-1].max() - r.e) / r.d if len(hi) > 1 else 0.0
    x['mr'] = np.maximum(mr, TICK); x['heat'] = heat; x['prog'] = prog
    x['half'] = np.where(x.kind == 'loss', (x.prog >= 0.5).astype(float), np.nan)
    x['logT'] = np.where(x.kind != 'time', np.log(x.dur), np.nan)
    x['ttc'] = (16 * 60 - mod[x.k.to_numpy()]) % (24 * 60)
    x['clock'] = mod[x.k.to_numpy()]
    x.to_parquet(OUT / 'h5_rows.parquet')
    return x


def contrast(y, col, rel):
    z = y.dropna(subset=[col, rel]).copy()
    z['adj'] = z[col] - z.groupby('cell')[col].transform('mean')
    z['tc'] = pd.qcut(z[rel].rank(method='first'), 3, labels=False)

    def one(w):
        a = w[w.tc == 2].groupby('sid').adj.mean(); b = w[w.tc == 0].groupby('sid').adj.mean()
        dd = (a - b).dropna()
        return (dd.mean(), dd.mean() / (dd.std(ddof=1) / np.sqrt(len(dd)))) if len(dd) > 20 else (np.nan, np.nan)
    m, t = one(z)
    signs = ''.join('+' if one(z[z.year == yr])[0] > 0 else '-' for yr in range(2021, 2026))
    raw = [z[z.tc == q][col].mean() for q in range(3)]
    return m, t, signs, raw


if __name__ == '__main__':
    x = build() if 'run' in sys.argv else pd.read_parquet(OUT / 'h5_rows.parquet')
    d = x[x.year <= 2025].copy()
    d['cell'] = (pd.qcut(d.p0, 5, labels=False).astype(str) + '_' +
                 pd.qcut(np.log(d.d / d.mr * d.s / d.mr), 5, labels=False).astype(str))
    print('moments', len(d), '| cells', d.cell.nunique(), '| resolved %.3f' % (d.kind != 'time').mean())
    print('%-6s %-5s | %9s %6s %6s | raw tercile means' % ('by', 'read', 'top-bot', 't', 'years'))
    for rel in ['asym', 'conf', 'brake', 'ttc']:
        for col in ['logT', 'heat', 'prog', 'half']:
            m, t, sg, raw = contrast(d, col, rel)
            print('%-6s %-5s | %+9.4f %6.2f %6s | %.3f  %.3f  %.3f' % (rel, col, m, t, sg, *raw))
    print('\nG of the same fork by the ET clock of the contact (session calendar buckets)')
    bk = [('18:00-03:00 overnight', lambda c: (c > 18 * 60) | (c <= 3 * 60)),
          ('03:00-09:30 Europe/pre-open', lambda c: (c > 3 * 60) & (c <= 9 * 60 + 30)),
          ('09:30-11:00 cash open', lambda c: (c > 9 * 60 + 30) & (c <= 11 * 60)),
          ('11:00-14:00 midday', lambda c: (c > 11 * 60) & (c <= 14 * 60)),
          ('14:00-16:00 into close', lambda c: (c > 14 * 60) & (c <= 16 * 60))]
    for name, f in bk:
        y = d[f(d.clock)]
        day = y.groupby('sid').g.agg(['sum', 'size']); m = y.g.mean()
        se = np.sqrt(((day['sum'] - m * day['size']) ** 2).sum()) / day['size'].sum()
        print('   %-28s n %5d  win %.3f fair %.3f  G %+.3f (t %.2f)  width p50 %.1f  bars p50 %.0f'
              % (name, len(y), (y.kind == 'win').mean(), y.p0.mean(), m, m / se,
                 (y.d + y.s).median(), y.dur.median()))
