"""075: where the expectation has to be revised, read on closed minutes only.

Progress is the CLOSE relative to the entry in risk units -- what a person sees
at a minute's close. The outcome is the trade's own outcome: the target reached
STRICTLY BEFORE the stop, not the maximum excursion of the window.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
from candidate import load, mask, TGT

pd.set_option('display.width', 210)
BANDS = [(-99, -.25, 'back below entry'), (-.25, .25, 'around entry'),
         (.25, 1., '0.25R..1R ahead'), (1., 99, 'a risk ahead')]


def win_vector(run, st):
    r = np.where(np.isfinite(run), run, -9)
    has_t = r[:, -1] >= TGT
    ft = np.argmax(r >= TGT, axis=1)
    fs = np.argmax(st, axis=1)
    hs = st.any(1)
    return has_t & (~hs | (ft < fs))


def table(run, st, clo, tag):
    alive = ~np.maximum.accumulate(st, axis=1)
    win = win_vector(run, st)
    rows = []
    for mm in [1, 2, 3, 5, 10, 15, 20]:
        j = mm - 1
        al = alive[:, j]
        if al.sum() < 60:
            continue
        r = dict(minute=mm, still_open=round(float(al.mean()), 3),
                 P_win_if_open=round(float(win[al].mean()), 4))
        for lo, hi, nm in BANDS:
            s = al & (clo[:, j] >= lo) & (clo[:, j] < hi) & np.isfinite(clo[:, j])
            r[nm] = f'{win[s].mean():.3f} (n={int(s.sum())})' if s.sum() >= 30 else '-'
        rows.append(r)
    print(f'\n--- {tag}')
    print(pd.DataFrame(rows).to_string(index=False))


def main():
    f, run, st = load()
    raw = pd.read_parquet(common.CACHE / 'feat_all.parquet')
    o = np.argsort(raw.t0.to_numpy(), kind='stable')
    clo = np.load(common.CACHE / 'closeR.npy')[o]
    m = np.asarray(mask(f))
    print(f'unconditional win rate of the family: {win_vector(run, st).mean():.4f}')
    table(run, st, clo, f'whole family, n={len(f)}')
    table(run[m], st[m], clo[m], f'the sign only, n={int(m.sum())}')


if __name__ == '__main__':
    main()
