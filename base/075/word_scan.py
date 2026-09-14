"""075: every five-candle construction, counted on every one of its occurrences.

For each word the move after the word is complete -- entry at the open of the
minute after its fifth candle, held ten minutes. The ceiling of the same scan is
taken by rotating the move series, which keeps its own serial structure.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
from census import words, decode

FLOOR = 3000
NPERM = 40


def main():
    o, h, l, c, ts, sid = common.tape()
    pos = common.positions(ts)
    code = np.load(common.CACHE / 'letters.npy')
    out = []
    for n in (3, 4, 5):
        ok, wid = words(code, n)
        si, sj = np.nonzero(ok)
        start = sj + 1
        p = pos[np.arange(code.shape[0])]
        ep = p[si, np.minimum(start + n, common.HOR)]
        pp = p[si, np.minimum(start + n + 9, common.HOR)]
        v = np.where((pp >= 0) & (ep >= 0),
                     (c[np.maximum(pp, 0)] - o[np.maximum(ep, 0)]) * 20.0, np.nan)
        m = np.isfinite(v)
        w = wid[si, sj][m]; v = v[m]
        order = np.argsort(si[m] * 100 + start[m], kind='stable')
        w, v = w[order], v[order]
        g = pd.DataFrame(dict(w=w, v=v)).groupby('w').v.agg(['size', 'mean'])
        g = g[g['size'] >= FLOOR]
        obs = float(g['mean'].abs().max())
        who = decode(int(g['mean'].abs().idxmax()), n)
        rng = np.random.default_rng(75)
        null = []
        N = len(v)
        for _ in range(NPERM):
            vs = np.roll(v, int(rng.integers(N // 20, N - N // 20)))
            gg = pd.DataFrame(dict(w=w, v=vs)).groupby('w').v.agg(['size', 'mean'])
            null.append(float(gg[gg['size'] >= FLOOR]['mean'].abs().max()))
        null = np.array(null)
        print(f'{n} candles: {len(g)} words above the floor, {len(v)} occurrences')
        print(f'   strongest word {who}: mean move after completion ${g["mean"].abs().max():+.2f} '
              f'(n={int(g.loc[g["mean"].abs().idxmax(), "size"])})')
        print(f'   ceiling of the same scan: median ${np.median(null):.2f}  '
              f'p95 ${np.percentile(null, 95):.2f}  max ${null.max():.2f}   '
              f'-> above the ceiling: {bool(obs > np.percentile(null, 95))}')
        print(f'   whole population mean: ${v.mean():+.2f}, and the cost of one turn is $15')
        out.append(dict(n=n, best=who, value=round(obs, 2), p95=round(float(np.percentile(null, 95)), 2)))
    pd.DataFrame(out).to_csv(common.CACHE / 'word_scan.csv', index=False)


if __name__ == '__main__':
    main()
