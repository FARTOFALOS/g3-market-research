"""075: the same numbers counted on distinct price paths, with a day block.

The field cuts one row per riz_id: a single T0 minute carries several zones, and
a single film carries several push-outs. Counting each of those as an independent
case multiplies the same tape. Here the headline is recomputed three ways:

  raw          every node event
  one per film every T0 minute contributes only its first push-out
  day block    the trading day is the resampling unit of the bootstrap
"""
from pathlib import Path
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common

TGT = 2.0
BOOT = 400
SEED = 750


def stats(f, run, stopped, tag):
    r = np.where(np.isfinite(run), run, -9)
    has_t = r[:, -1] >= TGT
    first_t = np.argmax(r >= TGT, axis=1)
    first_s = np.argmax(stopped, axis=1)
    has_s = stopped.any(1)
    win = has_t & (~has_s | (first_t < first_s))
    pnl = np.where(win, TGT * f.risk.to_numpy(), -f.risk.to_numpy()) * 20.0 - 15.0
    day = pd.to_datetime(f.t0.to_numpy()).tz_localize('UTC').tz_convert('America/New_York').normalize()
    g = pd.DataFrame(dict(day=day, pnl=pnl, win=win, R=(TGT * win - 1 * ~win)))
    rng = np.random.default_rng(SEED)
    days = g.day.unique()
    by = {d: v for d, v in g.groupby('day').pnl}
    means = []
    arrs = [by[d].to_numpy() for d in days]
    for _ in range(BOOT):
        pick = rng.integers(0, len(arrs), len(arrs))
        means.append(np.concatenate([arrs[i] for i in pick]).mean())
    means = np.array(means)
    print(f'-- {tag:22s} n={len(f):7d}  days={len(days):5d}  win={win.mean():.4f}  '
          f'E={g.R.mean():+.4f} R  net=${pnl.mean():+7.2f}  '
          f'day-block 95% [{np.percentile(means,2.5):+7.2f}, {np.percentile(means,97.5):+7.2f}]')
    return g


def main():
    f = pd.read_parquet(common.CACHE / 'feat_all.parquet').reset_index(drop=True)
    run = np.load(common.CACHE / 'run_R.npy')
    stopped = np.load(common.CACHE / 'stopped.npy')
    print(f'target {TGT}R, stop at the cleared border, 15 $ a turn\n')
    stats(f, run, stopped, 'every node event')
    m = (~f.duplicated(['row', 'k'])).to_numpy()
    stats(f[m].reset_index(drop=True), run[m], stopped[m], 'one per path-minute')
    m2 = (~f.duplicated(['row'])).to_numpy()
    stats(f[m2].reset_index(drop=True), run[m2], stopped[m2], 'first push-out per film')
    print('\nby epoch, one per path-minute:')
    ff = f[m].reset_index(drop=True); rr = run[m]; ss = stopped[m]
    yr = ff.year.to_numpy()
    for a, b in [(2006, 2010), (2011, 2015), (2016, 2020), (2021, 2026)]:
        s = (yr >= a) & (yr <= b)
        stats(ff[s].reset_index(drop=True), rr[s], ss[s], f'{a}-{b}')


if __name__ == '__main__':
    main()
