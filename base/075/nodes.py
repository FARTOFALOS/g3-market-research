"""075: the risk geometry of the scene, read off the overlay.

A node is a state transition of the zone word that is observable at its own
closing minute.  For the two transitions that leave the zone -- Z->E (back to the
exit side) and Z->F (through, to the far side) -- the just-cleared border is the
natural invalidation level and the whole move is still ahead.  The overlay
answers, without any invented number: how far does price get before it returns
through that border, and how often does it return at once.

entry  = open of minute k+1 (k is the closing minute of the node)
stop   = the just-cleared border (touch counts as a return)
MFE    = best excursion in the node's direction before that touch
units  = zone widths w = max(top-bottom, 0.25), and NQ points
"""
from pathlib import Path
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
from words import states

FWD = 60          # minutes of forward path read after the node


def collect(kind, d, st, pos, O, H, L, C, maxk=100):
    """kind: 'ZE' or 'ZF'. returns a DataFrame, one row per node occurrence"""
    tgt = 1 if kind == 'ZE' else 3
    prev_ok = st[:, :-1] == 2
    cur = st[:, 1:] == tgt
    # minute index of the node (1-based minute number) = column+2
    hit = prev_ok & cur
    # every minute before the node must be present
    present = np.cumprod(st != 0, axis=1).astype(bool)
    hit &= present[:, 1:]
    hit[:, maxk - 1:] = False
    si, ki = np.nonzero(hit)
    kmin = ki + 2                     # minute number of the node
    r = d.row.to_numpy()[si]
    top = d.zone_top.to_numpy(np.float32)[si]
    bot = d.zone_bottom.to_numpy(np.float32)[si]
    south = (d.t0_exit_side.to_numpy() == 'south')[si]
    w = np.maximum(top - bot, .25)
    # long = +1 if the node's direction is up in price
    if kind == 'ZE':
        border = np.where(south, bot, top)
        long = ~south
    else:
        border = np.where(south, top, bot)
        long = south
    sgn = np.where(long, 1.0, -1.0).astype(np.float32)

    p = pos[r]                        # (n, HOR+1)
    ent_pos = p[np.arange(len(si)), np.minimum(kmin + 1, common.HOR)]
    ok = (kmin + 1 <= common.HOR) & (ent_pos >= 0)
    entry = np.where(ok, O[np.maximum(ent_pos, 0)], np.nan)
    # the entry must open on the node's own side of the cleared border
    side_ok = np.where(long, entry > border, entry < border)

    n = len(si)
    mfe = np.full(n, np.nan, np.float32)
    stopped = np.zeros(n, bool)
    tstop = np.full(n, -1, np.int32)
    tmfe = np.full(n, -1, np.int32)
    fwd = np.arange(0, FWD)        # the entry minute k+1 itself counts
    idx = np.minimum(kmin[:, None] + 1 + fwd[None, :], common.HOR)
    fp = p[np.arange(n)[:, None], idx]
    valid = (fp >= 0) & (kmin[:, None] + 1 + fwd[None, :] <= common.HOR)
    hh = np.where(valid, H[np.maximum(fp, 0)], np.nan)
    ll = np.where(valid, L[np.maximum(fp, 0)], np.nan)
    # excursion in the node's direction, and the touch of the border
    fav = np.where(long[:, None], hh - entry[:, None], entry[:, None] - ll)
    adv_touch = np.where(long[:, None], ll <= border[:, None], hh >= border[:, None])
    adv_touch = np.where(np.isfinite(hh), adv_touch, False)
    first_stop = np.where(adv_touch.any(1), adv_touch.argmax(1), FWD)
    cols = np.arange(FWD)[None, :]
    before = cols < first_stop[:, None]
    favb = np.where(before & np.isfinite(fav), fav, -np.inf)
    mfe = favb.max(1)
    tmfe = favb.argmax(1) + 1
    mfe = np.where(np.isinf(mfe), 0.0, mfe)
    stopped = adv_touch.any(1)
    tstop = np.where(stopped, first_stop + 1, -1)
    # the entry-side risk actually taken: distance from entry to the border
    risk = np.abs(entry - border)
    # full-window excursion regardless of the stop, for reference
    favall = np.where(np.isfinite(fav), fav, -np.inf)
    mfe_all = np.where(np.isinf(favall.max(1)), 0.0, favall.max(1))
    return pd.DataFrame(dict(
        scene=si, k=kmin, row=r, long=long, w=w, entry=entry, border=border,
        risk=risk, mfe=mfe, mfe_all=mfe_all, tmfe=tmfe, stopped=stopped, tstop=tstop,
        ok=ok & np.isfinite(entry) & side_ok,
        t0=d.t0_ts_ns.to_numpy()[si], tf=d.tf_minutes.to_numpy()[si]))


def main():
    o, h, l, c, ts, sid = common.tape()
    pos = common.positions(ts)
    d = common.scene_table()
    st = np.load(common.CACHE / 'states.npy')
    for kind in ('ZE', 'ZF'):
        e = collect(kind, d, st, pos, o, h, l, c)
        e = e[e.ok].reset_index(drop=True)
        e['first'] = ~e.duplicated(['scene'])
        e.to_parquet(common.CACHE / f'nodes_{kind}.parquet')
        for tag, sub in (('all nodes', e), ('first node of the scene', e[e['first']])):
            R = sub.risk.to_numpy(); M = sub.mfe.to_numpy(); W = sub.w.to_numpy()
            print(f'\n=== {kind}  {tag}:  n = {len(sub)}')
            print(f'  risk to the cleared border, points : median {np.median(R):.2f}  '
                  f'q25 {np.percentile(R,25):.2f}  q75 {np.percentile(R,75):.2f}')
            print(f'  risk in zone widths                : median {np.median(R/W):.2f}')
            print(f'  stopped inside {FWD} min            : {sub.stopped.mean():.3f}'
                  f'   median minute of the stop {np.median(sub.tstop[sub.stopped]):.0f}')
            print(f'  MFE before the stop, points        : median {np.median(M):.2f}  '
                  f'q75 {np.percentile(M,75):.2f}  q90 {np.percentile(M,90):.2f}')
            print(f'  MFE / risk                         : median {np.median(M/np.maximum(R,.25)):.2f}'
                  f'  share MFE>=1R {np.mean(M>=R):.3f}  >=2R {np.mean(M>=2*R):.3f}  >=3R {np.mean(M>=3*R):.3f}')


if __name__ == '__main__':
    main()
