"""075: the trader's fork, decided on the same candles.

reading A -- exact time from T0: does the same construction behave differently
             depending on which minute after T0 it starts on?
reading B -- order of events: does the same order behave the same when it is
             passed at different speeds?

Both are asked of the move that is still ahead, in dollars, not of frequency.
"""
from pathlib import Path
import collections
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
from census import words, decode, rle

pd.set_option('display.width', 200)


def main():
    o, h, l, c, ts, sid = common.tape()
    pos = common.positions(ts)
    code = np.load(common.CACHE / 'letters.npy')
    ok5, w5 = words(code, 5)
    si, sj = np.nonzero(ok5)
    start = sj + 1
    p = pos[np.arange(code.shape[0])]
    # the word is complete only at the close of its fifth candle, minute start+4;
    # entry is the open of the next minute, so nothing of the word is read ahead
    ep = p[si, np.minimum(start + 5, common.HOR)]
    entry = np.where(ep >= 0, o[np.maximum(ep, 0)], np.nan)
    pp = p[si, np.minimum(start + 14, common.HOR)]
    m10 = np.where((pp >= 0) & (ep >= 0), (c[np.maximum(pp, 0)] - entry) * 20.0, np.nan)
    d = pd.DataFrame(dict(w=w5[si, sj], start=start, m10=m10))
    d = d[np.isfinite(d.m10)]
    d['word'] = [decode(v, 5) for v in d.w.to_numpy()]
    d['order'] = [rle(s) for s in d.word]

    print('=== reading A: the same five-candle word, split by where it starts')
    top = d.word.value_counts().head(12).index
    rows = []
    for w in top:
        s = d[d.word == w]
        b = pd.cut(s.start, [0, 8, 16, 26, 40, 47], labels=['1-8', '9-16', '17-26', '27-40', '41-46'])
        g = s.groupby(b, observed=True).m10.agg(['size', 'mean'])
        r = dict(word=w, n=len(s), all=round(float(s.m10.mean()), 2))
        for i, v in g.iterrows():
            r[str(i)] = round(float(v['mean']), 2)
        r['spread'] = round(float(g['mean'].max() - g['mean'].min()), 2)
        rows.append(r)
    t = pd.DataFrame(rows)
    print(t.to_string(index=False))
    # how big is that spread against the noise of the same split?
    rng = np.random.default_rng(75)
    sh = []
    for w in top:
        s = d[d.word == w]
        for _ in range(20):
            b = pd.Series(rng.permutation(pd.cut(s.start, [0, 8, 16, 26, 40, 47],
                                                 labels=['1-8', '9-16', '17-26', '27-40', '41-46']).to_numpy()))
            g = s.m10.groupby(b.to_numpy(), observed=True).mean()
            sh.append(float(g.max() - g.min()))
    print(f'\nobserved spread across start-minute bands: median {t.spread.median():.2f} $')
    print(f'the same spread when the band label is shuffled: median {np.median(sh):.2f} $, '
          f'p95 {np.percentile(sh, 95):.2f} $')

    print('\n=== reading B: the same order of events, split by how fast it was passed')
    ro = d.order.value_counts()
    rows = []
    for od in ro.head(10).index:
        s = d[d.order == od]
        g = s.groupby('word', observed=True).m10.agg(['size', 'mean'])
        g = g[g['size'] >= 2000]
        if len(g) < 2:
            continue
        rows.append(dict(order=od, exact_words=len(g), n=int(g['size'].sum()),
                         mean=round(float(s.m10.mean()), 2),
                         slowest=round(float(g['mean'].min()), 2),
                         fastest=round(float(g['mean'].max()), 2),
                         spread=round(float(g['mean'].max() - g['mean'].min()), 2)))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    main()
