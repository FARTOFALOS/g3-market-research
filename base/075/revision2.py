"""075: is there a second revision signal, or only the trade's own progress?

Inside bands of equal close progress, still-open trades are split by one candle
event of that same minute that is independent of the stop: the minute took the
previous minute's extreme AGAINST the trade. Meeting the zone itself cannot serve
here -- inside this family meeting the zone is the stop, so no open trade has it.
Outcome as everywhere else: target strictly before stop.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
from candidate import load
from revision import win_vector, BANDS

pd.set_option('display.width', 200)


def main():
    f, run, st = load()
    raw = pd.read_parquet(common.CACHE / 'feat_all.parquet')
    o = np.argsort(raw.t0.to_numpy(), kind='stable')
    clo = np.load(common.CACHE / 'closeR.npy')[o]
    o2, h2, l2, c2, ts2, sid2 = common.tape()
    pos = common.positions(ts2)
    rw = f.row.to_numpy(); k = f.k.to_numpy()
    lg = f.long.to_numpy()
    alive = ~np.maximum.accumulate(st, axis=1)
    win = win_vector(run, st)
    rows = []
    for mm in [1, 2, 3, 5, 10]:
        j = mm - 1
        cur = pos[rw, np.minimum(k + mm, common.HOR)]
        prv = pos[rw, np.minimum(k + mm - 1, common.HOR)]
        okp = (cur >= 0) & (prv >= 0)
        against = np.where(lg, l2[np.maximum(cur, 0)] < l2[np.maximum(prv, 0)],
                           h2[np.maximum(cur, 0)] > h2[np.maximum(prv, 0)]) & okp
        for lo, hi, nm in BANDS:
            base = alive[:, j] & (clo[:, j] >= lo) & (clo[:, j] < hi) & np.isfinite(clo[:, j])
            a, b = base & against, base & ~against
            if a.sum() < 200 or b.sum() < 200:
                continue
            rows.append(dict(minute=mm, band=nm, n_against=int(a.sum()),
                             P_against=round(float(win[a].mean()), 4), n_not=int(b.sum()),
                             P_not=round(float(win[b].mean()), 4),
                             diff=round(float(win[a].mean() - win[b].mean()), 4)))
    t = pd.DataFrame(rows)
    print('P(target before stop), among trades still open, inside bands of equal close progress')
    print(t.to_string(index=False) if len(t) else 'no band keeps both halves above 200 cases')
    if len(t):
        print(f'\nlargest separation produced by the candle event: {t["diff"].abs().max():.4f}')


if __name__ == '__main__':
    main()
