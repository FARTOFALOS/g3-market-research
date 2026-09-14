"""075: does the found sign need this zone, or only this hour?

The whole pipeline is run again with the scene's zone displaced by a fixed number
of its own widths along the exit direction. Everything else is identical: the same
T0 minutes, the same widths, the same exit sides, the same candles, the same
conditions, the same bracket. A displaced band is a level the price also passed
through, but it is not a RIZ. If the sign keeps its lift there, the sign belongs
to the tape and the clock, not to the scene.

Run: python -B work/075/control.py <shift in widths>
"""
from pathlib import Path
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
from words import states
import nodes as nd
import features as ft
from predicates import build
from validate import outcome, boot

SIGN = ['closes in its last fifth', 'not stop at least one ATR30', '13:00-14:00 New York']


def shifted_table(shift):
    d = common.scene_table()
    w = np.maximum(d.zone_top - d.zone_bottom, .25).to_numpy()
    s = np.where(d.t0_exit_side.to_numpy() == 'south', -1., 1.) * shift * w
    d = d.copy()
    d['zone_top'] = d.zone_top + s
    d['zone_bottom'] = d.zone_bottom + s
    return d


def run_pipeline(d, tag):
    o, h, l, c, ts, sid = common.tape()
    pos = common.positions(ts)
    st = states(d, pos, h, l)
    F = []
    for kind in ('ZE', 'ZF'):
        e = nd.collect(kind, d, st, pos, o, h, l, c)
        e = e[e.ok].reset_index(drop=True)
        e['first'] = ~e.duplicated(['scene'])
        e.to_parquet(common.CACHE / f'ctl_nodes_{kind}.parquet')
        F.append(kind)
    import features
    feats = []
    for kind in ('ZE', 'ZF'):
        src = common.CACHE / f'nodes_{kind}.parquet'
        bak = common.CACHE / f'bak_{kind}.parquet'
        if src.exists() and not bak.exists():
            src.replace(bak)
        (common.CACHE / f'ctl_nodes_{kind}.parquet').replace(src)
        f = features.build(kind)
        f = f.copy(); f['kind'] = kind
        feats.append(f)
        src.replace(common.CACHE / f'ctl_nodes_{kind}.parquet')
        bak.replace(src)
    f = pd.concat(feats, ignore_index=True).sort_values(['t0', 'k']).reset_index(drop=True)
    # forward path and bracket, exactly as overlay.py
    import overlay
    entry, fav, adv, clo = overlay.paths(f, pos, o, h, l, c)
    R = f.risk.to_numpy()[:, None]
    run = np.fmax.accumulate(np.where(np.isfinite(fav / R), fav / R, -np.inf), axis=1)
    stopped = np.nan_to_num(adv / R, nan=-9) >= 1.0
    win, pnl = outcome(f, run, stopped)
    f['win'] = win; f['pnl'] = pnl
    f['day'] = pd.to_datetime(f.t0.to_numpy()).tz_localize('UTC').tz_convert('America/New_York').normalize()
    lit = build(f)
    lit.update({'not ' + k: ~v for k, v in list(lit.items())})
    m = np.ones(len(f), bool)
    for part in SIGN:
        m &= lit[part]
    base = f.win.mean()
    g = f[m]
    lo, hi = boot(g.pnl.to_numpy(), g.day.to_numpy()) if len(g) > 50 else (np.nan, np.nan)
    print(f'{tag:26s} events {len(f):7d}  base {base:.4f}   sign n {len(g):6d}  '
          f'rate {g.win.mean():.4f}  lift {g.win.mean()-base:+.4f}  '
          f'medR {np.median(g.risk):5.2f}  net ${g.pnl.mean():+7.2f}  [{lo:+.2f}, {hi:+.2f}]')
    return f


def main():
    sh = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
    run_pipeline(shifted_table(sh), f'zone displaced {sh:+g} widths')


if __name__ == '__main__':
    main()
