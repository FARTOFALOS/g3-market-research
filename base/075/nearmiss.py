"""075: the cases that almost are the construction.

The accepted push-out asks that the whole minute stand clear of the border. The
near miss closes clear of it but its range still touches the zone: one relation
apart, and invisible to anyone who only reads closes. If the two behave alike,
the edge of the predicate is not an edge of the process -- the same lesson 074
drew about its own frozen predicate, tested here on price rather than on labels.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
from bracket import evaluate
from validate import boot

TGT = 2.0


def gather(d, st, pos, o, h, l, c, kind):
    """minutes inside a Z run whose close is already clear of the border"""
    r = d.row.to_numpy()
    top = d.zone_top.to_numpy(np.float32); bot = d.zone_bottom.to_numpy(np.float32)
    south = (d.t0_exit_side.to_numpy() == 'south')
    w = np.maximum(top - bot, .25)
    border = np.where(south, bot, top) if kind == 'ZE' else np.where(south, top, bot)
    long = (~south) if kind == 'ZE' else south
    cur_z = st[:, 1:] == 2
    prev_z = st[:, :-1] == 2
    present = np.cumprod(st != 0, axis=1).astype(bool)
    cand = cur_z & prev_z & present[:, 1:]
    cand[:, 99:] = False
    si, ki = np.nonzero(cand)
    kmin = ki + 2
    p = pos[r[si]]
    kp = p[np.arange(len(si)), kmin]
    ok = kp >= 0
    C = np.where(ok, c[np.maximum(kp, 0)], np.nan)
    clear = np.where(long[si], C > border[si], C < border[si])
    keep = ok & clear
    si, kmin = si[keep], kmin[keep]
    res = evaluate(pos, o, h, l, c, r[si], kmin, long[si], border[si].astype(np.float64))
    g = pd.DataFrame(res)
    g['scene'] = si; g['k'] = kmin; g['row'] = r[si]; g['kind'] = kind
    g['t0'] = d.t0_ts_ns.to_numpy()[si]
    return g[g.ok].reset_index(drop=True)


def score(g, tag, base=None):
    win = g.R2.to_numpy().astype(bool)
    pnl = np.where(win, TGT * g.risk, -g.risk) * 20.0 - 15.0
    day = pd.to_datetime(g.t0.to_numpy()).tz_localize('UTC').tz_convert('America/New_York').normalize()
    lo, hi = boot(pnl, day.to_numpy())
    print(f'{tag:34s} n {len(g):7d}  reaches 2R {win.mean():.4f}'
          f'{"" if base is None else f"  ({win.mean()-base:+.4f})"}  '
          f'stopped {g.stopped.mean():.3f}  medR {np.median(g.risk):5.2f}  '
          f'net ${pnl.mean():+7.2f}  [{lo:+.2f}, {hi:+.2f}]')
    return win.mean()


def main():
    o, h, l, c, ts, sid = common.tape()
    pos = common.positions(ts)
    d = common.scene_table()
    st = np.load(common.CACHE / 'states.npy')
    f = pd.read_parquet(common.CACHE / 'feat_all.parquet')
    win = (f.R2 == 1).to_numpy()
    pnl = np.where(win, TGT * f.risk, -f.risk) * 20.0 - 15.0
    day = pd.to_datetime(f.t0.to_numpy()).tz_localize('UTC').tz_convert('America/New_York').normalize()
    lo, hi = boot(pnl, day.to_numpy())
    print(f'{"accepted: whole minute clear":34s} n {len(f):7d}  reaches 2R {win.mean():.4f}'
          f'          stopped {f.stopped.mean():.3f}  medR {np.median(f.risk):5.2f}  '
          f'net ${pnl.mean():+7.2f}  [{lo:+.2f}, {hi:+.2f}]')
    base = win.mean()
    G = [gather(d, st, pos, o, h, l, c, k) for k in ('ZE', 'ZF')]
    g = pd.concat(G, ignore_index=True)
    score(g, 'near miss: close clear, range not', base)
    g.to_parquet(common.CACHE / 'nearmiss.parquet')
    for kind in ('ZE', 'ZF'):
        score(g[g.kind == kind].reset_index(drop=True), f'   near miss, {kind}', base)


if __name__ == '__main__':
    main()
