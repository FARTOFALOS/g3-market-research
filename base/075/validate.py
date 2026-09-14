"""075: the composite found by the search, put through the checks it has to pass.

  1 the search is repeated on even years only; the winner is then read on odd
    years, which the search never saw;
  2 the same winner is read epoch by epoch;
  3 the money is counted with the day as the resampling block;
  4 the parts are separated: the clock alone, the candle conditions alone.
"""
from pathlib import Path
import itertools
import json
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
from predicates import build
from compose import literals

TGT = 2.0
FLOOR_DEV = 1200
NPERM = 20
SEED = 751
BOOT = 400


def outcome(f, run, stopped, tgt=TGT):
    r = np.where(np.isfinite(run), run, -9)
    has_t = r[:, -1] >= tgt
    first_t = np.argmax(r >= tgt, axis=1)
    first_s = np.argmax(stopped, axis=1)
    has_s = stopped.any(1)
    win = has_t & (~has_s | (first_t < first_s))
    pnl = np.where(win, tgt * f.risk.to_numpy(), -f.risk.to_numpy()) * 20.0 - 15.0
    return win, pnl


def boot(pnl, day, n=BOOT, seed=SEED):
    g = pd.DataFrame(dict(day=day, pnl=pnl))
    arrs = [v.to_numpy() for _, v in g.groupby('day').pnl]
    rng = np.random.default_rng(seed)
    m = [np.concatenate([arrs[i] for i in rng.integers(0, len(arrs), len(arrs))]).mean()
         for _ in range(n)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def search(arrs, names, y, floor, nperm=NPERM, seed=SEED):
    N = len(y)
    rng = np.random.default_rng(seed)
    sh = rng.integers(N // 20, N - N // 20, nperm)
    Y = np.vstack([np.packbits(y)] + [np.packbits(np.roll(y, int(s))) for s in sh])
    base = np.array([np.bitwise_count(Y[i]).sum() for i in range(len(Y))], float) / N
    A = [np.packbits(a) for a in arrs]
    L = len(names)
    best = np.full(len(Y), -9.0); bname = [None] * len(Y)
    c2 = {}
    rows = []
    for cb in ([(i,) for i in range(L)] + list(itertools.combinations(range(L), 2))
               + list(itertools.combinations(range(L), 3))):
        if len(cb) == 1:
            m = A[cb[0]]
        elif len(cb) == 2:
            m = A[cb[0]] & A[cb[1]]; c2[cb] = m
        else:
            m = c2.get(cb[:2])
            if m is None:
                m = A[cb[0]] & A[cb[1]]; c2[cb[:2]] = m
            m = m & A[cb[2]]
        n = int(np.bitwise_count(m).sum())
        if n < floor:
            continue
        rate = np.bitwise_count(Y & m[None, :]).sum(1).astype(float) / n
        lift = rate - base
        imp = lift > best
        if imp.any():
            nm = ' & '.join(names[t] for t in cb)
            for q in np.flatnonzero(imp):
                bname[q] = nm
            best = np.maximum(best, lift)
        rows.append((cb, n, float(rate[0]), float(lift[0])))
    return best, bname, rows


def main():
    f = pd.read_parquet(common.CACHE / 'feat_all.parquet').reset_index(drop=True)
    run = np.load(common.CACHE / 'run_R.npy')
    stopped = np.load(common.CACHE / 'stopped.npy')
    o = np.argsort(f.t0.to_numpy(), kind='stable')
    f, run, stopped = f.iloc[o].reset_index(drop=True), run[o], stopped[o]
    win, pnl = outcome(f, run, stopped)
    f['win'] = win; f['pnl'] = pnl
    day = pd.to_datetime(f.t0.to_numpy()).tz_localize('UTC').tz_convert('America/New_York').normalize()
    f['day'] = day
    lit = literals(f)
    names = list(lit); arrs = [lit[n] for n in names]
    ev = (f.year % 2 == 0).to_numpy()
    print(f'development = even years ({ev.sum()} events), check = odd years ({(~ev).sum()})')
    best, bname, rows = search([a[ev] for a in arrs], names, f.win.to_numpy()[ev].astype(bool), FLOOR_DEV)
    null = best[1:]
    print(f'best on even years : {best[0]:+.4f}   {bname[0]}')
    print(f'ceiling p95 {np.percentile(null,95):+.4f}  max {null.max():+.4f}   '
          f'p = {float((null>=best[0]).mean()):.3f}')
    t = pd.DataFrame([dict(sign=' & '.join(names[i] for i in cb), n=n, rate=round(r,4), lift=round(l,4))
                      for cb, n, r, l in rows]).sort_values('lift', ascending=False)
    print('\n--- five strongest on even years')
    print(t.head(5).to_string(index=False))
    sel = t.head(5).sign.tolist() + ['13:00-14:00 New York',
                                     'closes in its last fifth',
                                     'closes in its last fifth & 13:00-14:00 New York']
    print('\n--- the same conditions read on odd years, which the search never saw')
    out = []
    for s in dict.fromkeys(sel):
        m = np.ones(len(f), bool)
        for part in s.split(' & '):
            m &= lit[part]
        for tag, sub in (('even', m & ev), ('odd', m & ~ev)):
            g = f[sub]
            if len(g) < 50:
                continue
            lo, hi = boot(g.pnl.to_numpy(), g.day.to_numpy())
            out.append(dict(sign=s[:70], half=tag, n=len(g), win=round(g.win.mean(), 4),
                            lift=round(g.win.mean() - f.win[ev if tag == 'even' else ~ev].mean(), 4),
                            medR=round(float(np.median(g.risk)), 2),
                            net=round(float(g.pnl.mean()), 2), lo=round(lo, 2), hi=round(hi, 2)))
    print(pd.DataFrame(out).to_string(index=False))
    best_sign = t.iloc[0].sign
    m = np.ones(len(f), bool)
    for part in best_sign.split(' & '):
        m &= lit[part]
    print(f'\n--- {best_sign}  by epoch (all years)')
    rows2 = []
    for a, b in [(2006, 2010), (2011, 2015), (2016, 2020), (2021, 2026)]:
        s = m & ((f.year >= a) & (f.year <= b)).to_numpy()
        g = f[s]
        if len(g) < 30:
            rows2.append(dict(epoch=f'{a}-{b}', n=len(g))); continue
        lo, hi = boot(g.pnl.to_numpy(), g.day.to_numpy())
        rows2.append(dict(epoch=f'{a}-{b}', n=len(g), win=round(g.win.mean(), 4),
                          medR=round(float(np.median(g.risk)), 2), net=round(float(g.pnl.mean()), 2),
                          lo=round(lo, 2), hi=round(hi, 2)))
    print(pd.DataFrame(rows2).to_string(index=False))
    json.dump(dict(best_even=bname[0], lift_even=float(best[0]),
                   ceiling_p95=float(np.percentile(null, 95))),
              open(common.CACHE / 'validate.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
