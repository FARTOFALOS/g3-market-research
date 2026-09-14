"""075: the dossier of the one sign that beat its own search ceiling.

sign, all three parts read at the close of the push-out minute:
   the minute closes in the last fifth of its own range;
   the distance from the entry to the cleared border is under one ATR30;
   the minute closes between 13:00 and 14:00 New York.
action: buy/sell at the open of the next minute in the direction of the push-out,
   stop at the cleared border, target two risks away, 15 $ a turn, 60 minutes max.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
from predicates import build
from validate import outcome, boot

TGT = 2.0
pd.set_option('display.width', 200)


def load():
    f = pd.read_parquet(common.CACHE / 'feat_all.parquet').reset_index(drop=True)
    run = np.load(common.CACHE / 'run_R.npy')
    st = np.load(common.CACHE / 'stopped.npy')
    o = np.argsort(f.t0.to_numpy(), kind='stable')
    f, run, st = f.iloc[o].reset_index(drop=True), run[o], st[o]
    win, pnl = outcome(f, run, st)
    f['win'] = win; f['pnl'] = pnl
    f['gross_R'] = np.where(win, TGT, -1.0)
    f['day'] = pd.to_datetime(f.t0.to_numpy()).tz_localize('UTC').tz_convert('America/New_York').normalize()
    return f, run, st


def mask(f):
    P = build(f)
    return P['closes in its last fifth'] & (~P['stop at least one ATR30']) & P['13:00-14:00 New York']


def main():
    f, run, st = load()
    m = mask(f)
    g = f[m].reset_index(drop=True)
    base = f.win.mean()
    print(f'population {len(f)}  base 2R rate {base:.4f}  base gross {3*base-1:+.4f} R')
    print(f'sign       {len(g)}  ({len(g)/len(f):.4f} of the population), '
          f'{g.day.nunique()} distinct days, {len(g)/ (f.day.nunique()/252):.0f} per year-equivalent')
    print(f'           2R rate {g.win.mean():.4f}   gross {g.gross_R.mean():+.4f} R   '
          f'median R {np.median(g.risk):.2f} pts (${np.median(g.risk)*20:.0f})')
    lo, hi = boot(g.pnl.to_numpy(), g.day.to_numpy())
    print(f'           net ${g.pnl.mean():+.2f} per trade, day-block 95% [{lo:+.2f}, {hi:+.2f}], '
          f'total ${g.pnl.sum():+,.0f}')

    print('\n--- by epoch: structure first, then money')
    rows = []
    for a, b in [(2006, 2010), (2011, 2015), (2016, 2020), (2021, 2026)]:
        s = g[(g.year >= a) & (g.year <= b)]
        lo, hi = boot(s.pnl.to_numpy(), s.day.to_numpy())
        gl, gh = boot(s.gross_R.to_numpy(), s.day.to_numpy())
        rows.append(dict(epoch=f'{a}-{b}', n=len(s), rate=round(s.win.mean(), 4),
                         gross_R=round(s.gross_R.mean(), 4), gR_lo=round(gl, 3), gR_hi=round(gh, 3),
                         medR_pts=round(float(np.median(s.risk)), 2),
                         cost_in_R=round(15 / (np.median(s.risk) * 20), 3),
                         net=round(float(s.pnl.mean()), 2), lo=round(lo, 2), hi=round(hi, 2)))
    print(pd.DataFrame(rows).to_string(index=False))

    print('\n--- year by year')
    y = g.groupby('year').agg(n=('win', 'size'), rate=('win', 'mean'),
                              gross_R=('gross_R', 'mean'), net=('pnl', 'mean'), total=('pnl', 'sum'))
    y['cum'] = y.total.cumsum()
    print(y.round(3).to_string())
    eq = g.sort_values('t0').pnl.cumsum().to_numpy()
    print(f'\nequity on the sign: end ${eq[-1]:+,.0f}   max drawdown ${(np.maximum.accumulate(eq)-eq).max():,.0f}')

    print('\n--- is the hour a knife edge? the same two candle conditions by hour of New York')
    P = build(f)
    c2 = P['closes in its last fifth'] & (~P['stop at least one ATR30'])
    t = pd.DataFrame(dict(hour=f.minute_of_day // 60, win=f.win, pnl=f.pnl, R=f.gross_R))[c2]
    q = t.groupby('hour').agg(n=('win', 'size'), rate=('win', 'mean'), gross_R=('R', 'mean'), net=('pnl', 'mean'))
    q['lift'] = q.rate - base
    print(q[q.n >= 300].round(4).to_string())

    print('\n--- what weakens the expectation once the trade is on')
    # conditional on still alive at minute m, what the next closed minute changes
    idx = np.flatnonzero(m)
    r, s2 = run[idx], st[idx]
    alive = ~np.maximum.accumulate(s2, axis=1)
    gotR = np.where(np.isfinite(r), r, -9) >= TGT
    final = gotR[:, -1]
    rows = []
    for mm in [1, 2, 3, 5, 10, 20]:
        j = mm - 1
        al = alive[:, j]
        if al.sum() < 40:
            continue
        prog = np.where(np.isfinite(r[:, j]), r[:, j], np.nan)
        hi_ = al & (prog >= 0.5)
        lo_ = al & (prog < 0.0)
        rows.append(dict(minute=mm, alive=round(float(al.mean()), 3),
                         P_target_if_alive=round(float(final[al].mean()), 4),
                         n_ahead=int(hi_.sum()), P_if_already_half_way=round(float(final[hi_].mean()), 4) if hi_.sum() > 30 else None,
                         n_behind=int(lo_.sum()), P_if_still_behind_entry=round(float(final[lo_].mean()), 4) if lo_.sum() > 30 else None))
    print(pd.DataFrame(rows).to_string(index=False))
    g.to_parquet(common.CACHE / 'candidate.parquet')


if __name__ == '__main__':
    main()
