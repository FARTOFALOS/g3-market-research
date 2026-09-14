"""075: the overlay of the family, from which the numbers of a rule are read.

All node films are laid on top of each other by their own node minute and scaled
by their own risk R (the distance from the entry to the cleared border). Then the
picture is read: how far the overlay reaches, by which minute, and what share has
already come back through the border.

Nothing here is chosen in advance: the horizon and the target of the later action
are taken off these curves.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common

FWD = 60


def paths(f, pos, o, h, l, c):
    row = f.row.to_numpy(); k = f.k.to_numpy(); n = len(f)
    ar = np.arange(n)
    p = pos[row]
    ent = p[ar, np.minimum(k + 1, common.HOR)]
    entry = np.where(ent >= 0, o[np.maximum(ent, 0)], np.nan).astype(np.float64)
    idx = np.minimum(k[:, None] + 1 + np.arange(FWD)[None, :], common.HOR)
    fp = p[ar[:, None], idx]
    val = (fp >= 0) & (k[:, None] + 1 + np.arange(FWD)[None, :] <= common.HOR)
    hh = np.where(val, h[np.maximum(fp, 0)], np.nan)
    ll = np.where(val, l[np.maximum(fp, 0)], np.nan)
    cc = np.where(val, c[np.maximum(fp, 0)], np.nan)
    lg = f.long.to_numpy()[:, None]
    fav = np.where(lg, hh - entry[:, None], entry[:, None] - ll)
    adv = np.where(lg, entry[:, None] - ll, hh - entry[:, None])
    clo = np.where(lg, cc - entry[:, None], entry[:, None] - cc)
    return entry, fav, adv, clo


def main():
    o, h, l, c, ts, sid = common.tape()
    pos = common.positions(ts)
    f = pd.read_parquet(common.CACHE / 'feat_all.parquet').reset_index(drop=True)
    entry, fav, adv, clo = paths(f, pos, o, h, l, c)
    R = f.risk.to_numpy()[:, None]
    favR = fav / R
    advR = adv / R
    stopped = np.nan_to_num(advR, nan=-9) >= 1.0
    ever = np.maximum.accumulate(stopped, axis=1)
    run = np.fmax.accumulate(np.where(np.isfinite(favR), favR, -np.inf), axis=1)
    alive = ~ever
    print('=== overlay of the family, aligned on the node minute, scaled by its own R')
    print(f'{"m":>3} {"alive":>7} {"medMFE|alive":>13} {"q75":>7} {"P(1R by m)":>11} '
          f'{"P(2R by m)":>11} {"P(3R by m)":>11} {"medClose|alive":>15}')
    for m in [1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 45, 60]:
        j = m - 1
        al = alive[:, j]
        rr = np.where(al, run[:, j], np.nan)
        p1 = float((np.where(np.isfinite(run[:, j]), run[:, j], -9) >= 1).mean())
        p2 = float((np.where(np.isfinite(run[:, j]), run[:, j], -9) >= 2).mean())
        p3 = float((np.where(np.isfinite(run[:, j]), run[:, j], -9) >= 3).mean())
        cl = np.where(al, clo[:, j] / R[:, 0], np.nan)
        print(f'{m:3d} {al.mean():7.3f} {np.nanmedian(rr):13.2f} {np.nanpercentile(rr,75):7.2f} '
              f'{p1:11.3f} {p2:11.3f} {p3:11.3f} {np.nanmedian(cl):15.2f}')
    print('\n=== expectancy in R of a target at nR with the stop at the border, before costs')
    for tgt in [0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]:
        hit = (np.nan_to_num(run[:, -1], nan=-9) >= tgt)
        # the stop and the target inside one minute are resolved against the trade
        first_t = np.argmax(np.where(np.isfinite(run), run, -9) >= tgt, axis=1)
        has_t = (np.nan_to_num(run[:, -1], nan=-9) >= tgt)
        first_s = np.argmax(stopped, axis=1)
        has_s = stopped.any(1)
        win = has_t & (~has_s | (first_t < first_s))
        e = tgt * win.mean() - 1.0 * (~win).mean()
        med_net = e * np.median(f.risk) * 20 - 15
        print(f'  target {tgt:4.2f}R  win {win.mean():.4f}  E {e:+.4f} R   '
              f'at the median R ({np.median(f.risk):.2f} pts) that is {e*np.median(f.risk)*20:+.2f} $ gross, '
              f'{med_net:+.2f} $ after a 15 $ turn')
    print('\n=== time exit instead of a target: hold m minutes, no stop')
    for m in [1, 2, 3, 5, 10, 20, 30, 60]:
        v = clo[:, m - 1] * 20
        v = v[np.isfinite(v)]
        se = v.std(ddof=1) / np.sqrt(len(v))
        print(f'  hold {m:3d} min   mean {v.mean():+7.2f} $   t {v.mean()/se:+6.2f}   n {len(v)}')
    np.save(common.CACHE / 'run_R.npy', run.astype(np.float32))
    np.save(common.CACHE / 'stopped.npy', stopped)
    np.save(common.CACHE / 'closeR.npy', (clo / R).astype(np.float32))


if __name__ == '__main__':
    main()
