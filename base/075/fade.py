"""075: the mirror construction. The node candle pushes back out of the zone;
the fade bets that the push fails, with the push candle's own extreme as the stop.

Phases on the candles:
  1  price left the zone at T0 (the scene's own displacement)
  2  a run of minutes whose range meets the zone again
  3  one minute closes entirely clear of the near border again  <- the node
The fade takes the other side of phase 3 at the open of the next minute.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
from bracket import evaluate


def main():
    o, h, l, c, ts, sid = common.tape()
    pos = common.positions(ts)
    f = pd.read_parquet(common.CACHE / 'feat_all.parquet')
    row = f.row.to_numpy(); k = f.k.to_numpy()
    long_node = f.long.to_numpy()
    ar = np.arange(len(f))
    kp = pos[row, np.minimum(k, common.HOR)]
    node_ext = np.where(long_node, h[np.maximum(kp, 0)], l[np.maximum(kp, 0)])
    long_fade = ~long_node
    r = evaluate(pos, o, h, l, c, row, k, long_fade, node_ext.astype(np.float64))
    g = pd.DataFrame(r)
    g = g.assign(kind=f.kind.to_numpy(), year=f.year.to_numpy(), rth=f.rth.to_numpy(),
                 row=row, k=k, scene=f.scene.to_numpy())
    g = g[g.ok].reset_index(drop=True)
    g.to_parquet(common.CACHE / 'fade_all.parquet')
    print('=== fade of the push-out, stop = the push candle\'s own extreme')
    for tag, sub in (('all', g), ('even years', g[g.year % 2 == 0]),
                     ('ZE', g[g.kind == 'ZE']), ('ZF', g[g.kind == 'ZF'])):
        R = sub.risk.to_numpy(); M = sub.mfe.to_numpy()
        e1 = 2 * sub.R1.mean() - 1
        e2 = 3 * sub.R2.mean() - 1
        print(f'-- {tag:11s} n={len(sub):7d}  risk med {np.median(R):5.2f} pts   '
              f'stopped {sub.stopped.mean():.3f}  R1 {sub.R1.mean():.4f}  R2 {sub.R2.mean():.4f}  '
              f'R3 {sub.R3.mean():.4f}   E(1R) {e1:+.4f}R  E(2R) {e2:+.4f}R')
    print('\n-- fade expectancy in dollars by the size of its own stop')
    q = pd.qcut(g.risk, 8, duplicates='drop')
    t = g.groupby(q, observed=True).agg(n=('R2', 'size'), risk=('risk', 'median'),
                                        R1=('R1', 'mean'), R2=('R2', 'mean'), R3=('R3', 'mean'))
    t['E1_net$'] = (2 * t.R1 - 1) * t.risk * 20 - 15
    t['E2_net$'] = (3 * t.R2 - 1) * t.risk * 20 - 15
    print(t.round(4).to_string())


if __name__ == '__main__':
    main()
