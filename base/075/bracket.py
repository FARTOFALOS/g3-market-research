"""075: one honest bracket evaluator, used for every construction in the cycle.

given a node (row, k, direction, stop level) it returns
  risk   = |open(k+1) - stop|
  mfe    = best excursion in the trade direction before the stop level is touched
  tstop  = the minute of the first touch (a touch counts, the bar order inside a
           minute is unknown, so a minute holding both prices is resolved as a
           stop: the pessimistic reading)
  R1/R2/R3 = whether mfe reached 1R / 2R / 3R strictly before that touch
The entry minute itself is included in the scan.
"""
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common

FWD = 60


def evaluate(pos, o, h, l, c, row, k, long, stop, fwd=FWD):
    n = len(row)
    ar = np.arange(n)
    p = pos[row]
    ent = p[ar, np.minimum(k + 1, common.HOR)]
    ok = (k + 1 <= common.HOR) & (ent >= 0)
    entry = np.where(ok, o[np.maximum(ent, 0)], np.nan).astype(np.float64)
    side_ok = np.where(long, entry < stop, entry > stop) if False else \
        np.where(long, entry > stop, entry < stop)
    idx = np.minimum(k[:, None] + 1 + np.arange(fwd)[None, :], common.HOR)
    fp = p[ar[:, None], idx]
    val = (fp >= 0) & (k[:, None] + 1 + np.arange(fwd)[None, :] <= common.HOR)
    hh = np.where(val, h[np.maximum(fp, 0)], np.nan)
    ll = np.where(val, l[np.maximum(fp, 0)], np.nan)
    fav = np.where(long[:, None], hh - entry[:, None], entry[:, None] - ll)
    touch = np.where(long[:, None], ll <= stop[:, None], hh >= stop[:, None])
    touch = np.where(np.isfinite(hh), touch, False)
    any_t = touch.any(1)
    first = np.where(any_t, touch.argmax(1), fwd)
    before = np.arange(fwd)[None, :] < first[:, None]
    favb = np.where(before & np.isfinite(fav), fav, -np.inf)
    mfe = favb.max(1)
    mfe = np.where(np.isinf(mfe), 0.0, mfe)
    risk = np.abs(entry - stop)
    out = dict(entry=entry, risk=risk, mfe=mfe, stopped=any_t,
               tstop=np.where(any_t, first + 1, -1), ok=ok & np.isfinite(entry) & side_ok)
    for m in (1, 2, 3):
        out[f'R{m}'] = (mfe >= m * risk).astype(np.int8)
    # close-out at the end of the scan when never stopped
    lastp = np.where(val, fp, -1).max(1)
    out['last_close'] = np.where(lastp >= 0, c[np.maximum(lastp, 0)], np.nan)
    return out
