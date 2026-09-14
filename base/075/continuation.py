"""075: what is still ahead once the early part of a construction is closed.

For a five-candle construction the first three candles are the part that is
already readable at the close of minute s+2; candles four and five are the part
still ahead. Two questions, on the same candles:

  structure : how often does the same ending follow this beginning, against how
              often that ending appears at all;
  money     : what the price does over the minutes that are still ahead,
              entered at the open of minute s+3 and counted in dollars.
"""
from pathlib import Path
import collections
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
from census import letters, words, decode, rle, LEN

pd.set_option('display.width', 220)


def main():
    o, h, l, c, ts, sid = common.tape()
    pos = common.positions(ts)
    rows = np.arange(len(common.t0_list()))
    code = np.load(common.CACHE / 'letters.npy')
    ok5, w5 = words(code, 5)
    ok3, w3 = words(code, 3)
    W = ok5.shape[1]
    si, sj = np.nonzero(ok5)
    pre = w3[si, sj]                      # first three letters
    suf = w5[si, sj] % 16                 # last two letters
    start = sj + 1                        # minute of the first candle
    # price: entry at the open of minute start+3, held h minutes
    p = pos[rows]
    ent_i = np.minimum(start + 3, common.HOR)
    ep = p[si, ent_i]
    entry = np.where(ep >= 0, o[np.maximum(ep, 0)], np.nan)
    out = {}
    for hh in (2, 5, 10):
        j = np.minimum(start + 2 + hh, common.HOR)
        pp = p[si, j]
        out[f'm{hh}'] = np.where((pp >= 0) & (ep >= 0), (c[np.maximum(pp, 0)] - entry) * 20.0, np.nan)
    d = pd.DataFrame(out)
    d['pre'] = pre; d['suf'] = suf; d['start'] = start; d['film'] = si
    d = d[np.isfinite(d.m2)].reset_index(drop=True)
    print(f'five-candle occurrences with a usable continuation: {len(d)}')

    # --- structure: does the beginning select the ending?
    base = d.suf.value_counts(normalize=True)
    tab = pd.crosstab(d.pre, d.suf, normalize='index')
    n_pre = d.pre.value_counts()
    keep = n_pre[n_pre >= 3000].index
    tab = tab.loc[tab.index.isin(keep)]
    dev = (tab - base).abs().max(axis=1)
    print(f'\n=== beginnings with at least 3000 occurrences: {len(tab)}')
    print(f'largest shift of any ending, over all beginnings: {dev.max():.4f}  '
          f'(the ending itself has base rate {base.max():.4f})')
    top = dev.sort_values(ascending=False).head(8)
    rowsout = []
    for pw in top.index:
        j = (tab.loc[pw] - base).abs().idxmax()
        rowsout.append(dict(beginning=decode(pw, 3), ending=decode(j, 2), n=int(n_pre[pw]),
                            rate=round(float(tab.loc[pw, j]), 4), base=round(float(base[j]), 4),
                            lift=round(float(tab.loc[pw, j] - base[j]), 4)))
    print(pd.DataFrame(rowsout).to_string(index=False))

    # --- money: the move over the minutes still ahead, by beginning
    g = d.groupby('pre').agg(n=('m2', 'size'), m2=('m2', 'mean'), m5=('m5', 'mean'), m10=('m10', 'mean'))
    g = g[g.n >= 3000]
    g['t2'] = d.groupby('pre').m2.mean() / (d.groupby('pre').m2.std() / np.sqrt(d.groupby('pre').m2.count()))
    g['t10'] = d.groupby('pre').m10.mean() / (d.groupby('pre').m10.std() / np.sqrt(d.groupby('pre').m10.count()))
    g.index = [decode(i, 3) for i in g.index]
    g = g.sort_values('m10')
    print('\n=== the move still ahead, $ per contract, one row per beginning')
    print(pd.concat([g.head(6), g.tail(6)]).round(3).to_string())
    print(f'\npopulation mean of the same move: m2 {d.m2.mean():.2f}  m5 {d.m5.mean():.2f}  m10 {d.m10.mean():.2f}'
          f'   (n = {len(d)})')

    # --- reading A vs reading B on the same occurrences
    print('\n=== the two readings on the same five candles')
    d['ord'] = [rle(decode(int(v), 5)) for v in np.unique(np.zeros(1))] if False else None
    u = pd.DataFrame(dict(word=w5[si, sj], start=start))
    cnt = u.groupby(['word', 'start']).size()
    byword = u.word.value_counts()
    chi = {}
    for wv in byword.head(20).index:
        s = u.start[u.word == wv]
        h_ = s.value_counts().reindex(range(1, W + 1)).fillna(0)
        exp = len(s) / W
        chi[decode(wv, 5)] = round(float(((h_ - exp) ** 2 / exp).sum() / W), 3)
    print('per-word deviation of the start-minute histogram from flat, chi2/df:')
    print('  ' + '  '.join(f'{k}:{v}' for k, v in chi.items()))
    rmap = collections.Counter()
    for wv, m in byword.items():
        rmap[rle(decode(wv, 5))] += m
    print(f'distinct exact words {len(byword)}, distinct orders {len(rmap)}; '
          f'the commonest order {rmap.most_common(1)[0][0]} gathers '
          f'{rmap.most_common(1)[0][1]} occurrences from '
          f'{sum(1 for wv in byword.index if rle(decode(wv,5))==rmap.most_common(1)[0][0])} exact words')


if __name__ == '__main__':
    main()
