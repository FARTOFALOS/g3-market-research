"""075: the census of repeating candle constructions, in both readings.

Per-minute letter, a pure relation between two adjacent closed candles:
    U  took the previous high, not the previous low
    D  took the previous low, not the previous high
    B  took both (an outside minute)
    I  took neither (an inside minute)

A construction is a run of consecutive letters.
    reading A, exact time from T0 : the word together with the minute it starts on
    reading B, order of events    : the word after run-length compression, so the
                                    same organisation passed at a different speed
                                    counts as the same construction

Nothing about the zone enters the letter; the zone stays available as context.
"""
from pathlib import Path
import collections
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common

ALPHA = np.array(list('.UDBI'))
LEN = 50


def letters(pos, h, l, rows):
    """(n, LEN) codes for minutes 1..LEN; 0 where a minute is missing"""
    p = pos[rows][:, :LEN + 1]
    H = np.where(p >= 0, h[np.maximum(p, 0)], np.nan)
    L = np.where(p >= 0, l[np.maximum(p, 0)], np.nan)
    up = H[:, 1:] > H[:, :-1]
    dn = L[:, 1:] < L[:, :-1]
    code = np.where(up & ~dn, 1, np.where(dn & ~up, 2, np.where(up & dn, 3, 4)))
    code = np.where(np.isfinite(H[:, 1:]) & np.isfinite(H[:, :-1]), code, 0).astype(np.uint8)
    return code


def words(code, n):
    """all contiguous n-letter words with no gap; returns (scene_index, start, word_id)"""
    L = code.shape[1]
    ok = np.ones((code.shape[0], L - n + 1), bool)
    wid = np.zeros((code.shape[0], L - n + 1), np.int32)
    for j in range(n):
        c = code[:, j:L - n + 1 + j]
        ok &= c > 0
        wid = wid * 4 + (c.astype(np.int32) - 1)
    return ok, wid


def decode(wid, n):
    s = ''
    for j in range(n):
        s = 'UDBI'[wid % 4] + s
        wid //= 4
    return s


def rle(s):
    out = s[0]
    for ch in s[1:]:
        if ch != out[-1]:
            out += ch
    return out


def main():
    o, h, l, c, ts, sid = common.tape()
    pos = common.positions(ts)
    rows = np.arange(len(common.t0_list()))
    code = letters(pos, h, l, rows)
    np.save(common.CACHE / 'letters.npy', code)
    print('films:', len(rows), ' minutes per film:', LEN)
    print('letter frequencies:', {ch: round(float((code == i + 1).sum() / (code > 0).sum()), 4)
                                  for i, ch in enumerate('UDBI')})
    for n in (3, 4, 5):
        ok, wid = words(code, n)
        v = wid[ok]
        cnt = collections.Counter(v.tolist())
        tot = len(v)
        print(f'\n=== constructions of {n} candles: {tot} occurrences, '
              f'{len(cnt)} of {4**n} possible words seen')
        print('--- reading A, exact time from T0: the ten commonest words, and where they sit')
        top = cnt.most_common(10)
        for wv, m in top:
            st = np.flatnonzero(ok.ravel())
            pass
        starts = np.tile(np.arange(ok.shape[1]) + 1, (ok.shape[0], 1))[ok]
        df = pd.DataFrame(dict(w=v, s=starts))
        for wv, m in top:
            ss = df.s[df.w == wv]
            hist = ss.value_counts()
            print(f'  {decode(wv, n)}  n={m:8d}  {m/tot:6.4f}   '
                  f'commonest start minute {hist.index[0]} ({hist.iloc[0]/m:.3f}), '
                  f'spread over {ss.nunique()} of {ok.shape[1]} start minutes, '
                  f'max share at one minute {hist.iloc[0]/m:.3f}')
        # reading B
        rc = collections.Counter()
        for wv, m in cnt.items():
            rc[rle(decode(wv, n))] += m
        print(f'--- reading B, order of events: {len(rc)} distinct orders')
        for w, m in rc.most_common(10):
            print(f'  {w:6s}  n={m:8d}  {m/tot:6.4f}')


if __name__ == '__main__':
    main()
