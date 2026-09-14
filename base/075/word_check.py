"""075: the one word that cleared its scan ceiling, put through the same checks.

BUDDU: an outside minute, then one that takes only the high, two that take only
the low, then one that takes the high again. Entry at the open of the minute
after the fifth candle, held ten minutes, no stop.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
from census import words, decode

WORD = 'UUIBU'
BOOT = 400


def code_of(s):
    v = 0
    for ch in s:
        v = v * 4 + 'UDBI'.index(ch)
    return v


def main():
    o, h, l, c, ts, sid = common.tape()
    pos = common.positions(ts)
    code = np.load(common.CACHE / 'letters.npy')
    n = len(WORD)
    ok, wid = words(code, n)
    si, sj = np.nonzero(ok)
    start = sj + 1
    p = pos[np.arange(code.shape[0])]
    ep = p[si, np.minimum(start + n, common.HOR)]
    pp = p[si, np.minimum(start + n + 9, common.HOR)]
    good = (pp >= 0) & (ep >= 0)
    v = np.where(good, (c[np.maximum(pp, 0)] - o[np.maximum(ep, 0)]) * 20.0, np.nan)
    # volatility at the moment of entry: median minute range of the 30 minutes before
    back = np.clip(ep[:, None] - np.arange(1, 31)[None, :], 0, len(h) - 1)
    atr = np.median(h[back] - l[back], axis=1)
    t0 = common.t0_list()[si]
    tt = pd.to_datetime(t0).tz_localize('UTC').tz_convert('America/New_York')
    d = pd.DataFrame(dict(w=wid[si, sj], v=v, atr=atr, year=tt.year.to_numpy(),
                          day=tt.normalize().to_numpy(), film=si, start=start))
    d = d[np.isfinite(d.v) & (d.atr > 0)]
    m = d.w == code_of(WORD)
    g = d[m]
    print(f'{WORD}: {len(g)} occurrences in {g.film.nunique()} films on {g.day.nunique()} days')
    print(f'   mean move ${g.v.mean():+.2f}   in ATR units {(g.v/(g.atr*20)).mean():+.3f}'
          f'   population in ATR units {(d.v/(d.atr*20)).mean():+.3f}')
    rng = np.random.default_rng(75)
    arrs = [x.to_numpy() for _, x in g.groupby('day').v]
    bs = [np.concatenate([arrs[i] for i in rng.integers(0, len(arrs), len(arrs))]).mean()
          for _ in range(BOOT)]
    print(f'   day-block 95% on the dollar mean: [{np.percentile(bs,2.5):+.2f}, {np.percentile(bs,97.5):+.2f}]')
    print('\n   by epoch')
    rows = []
    for a, b in [(2006, 2010), (2011, 2015), (2016, 2020), (2021, 2026)]:
        s = g[(g.year >= a) & (g.year <= b)]
        if len(s) < 30:
            rows.append(dict(epoch=f'{a}-{b}', n=len(s))); continue
        rows.append(dict(epoch=f'{a}-{b}', n=len(s), mean=round(float(s.v.mean()), 2),
                         in_ATR=round(float((s.v / (s.atr * 20)).mean()), 3),
                         median_ATR_pts=round(float(s.atr.median()), 2)))
    print(pd.DataFrame(rows).to_string(index=False))
    print('\n   the same word measured in ATR units, ceiling of the whole five-candle scan')
    dd = d.assign(z=d.v / (d.atr * 20))
    gg = dd.groupby('w').z.agg(['size', 'mean'])
    gg = gg[gg['size'] >= 3000]
    obs = float(abs(gg.loc[code_of(WORD), 'mean']))
    rng2 = np.random.default_rng(751)
    N = len(dd)
    z = dd.z.to_numpy(); w = dd.w.to_numpy()
    null = []
    for _ in range(40):
        zz = np.roll(z, int(rng2.integers(N // 20, N - N // 20)))
        h2 = pd.DataFrame(dict(w=w, z=zz)).groupby('w').z.agg(['size', 'mean'])
        null.append(float(h2[h2['size'] >= 3000]['mean'].abs().max()))
    null = np.array(null)
    print(f'   strongest word in ATR units: {decode(int(gg["mean"].abs().idxmax()), n)} '
          f'{gg["mean"].abs().max():+.4f};  {WORD} {obs:+.4f}')
    print(f'   ceiling: median {np.median(null):.4f}  p95 {np.percentile(null,95):.4f}  max {null.max():.4f}')
    print(f'   -> the dollar winner still clears the ceiling in ATR units: '
          f'{bool(obs > np.percentile(null, 95))}')


if __name__ == '__main__':
    main()
