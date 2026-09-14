"""075: what is observable at a node, on the candles, at the node's own close.

Every column is computed from minutes T0..k only (k = the closing minute of the
node) plus the scene's own zone.  Nothing from k+1 onward enters a feature.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common

ET = np.timedelta64(0, 'ns')


def build(kind):
    o, h, l, c, ts, sid = common.tape()
    pos = common.positions(ts)
    d = common.scene_table()
    st = np.load(common.CACHE / 'states.npy')
    e = pd.read_parquet(common.CACHE / f'nodes_{kind}.parquet')
    si = e.scene.to_numpy()
    k = e.k.to_numpy()
    r = e.row.to_numpy()
    p = pos[r]
    n = len(e)
    ar = np.arange(n)

    kp = p[ar, k]                      # the node minute itself
    km1 = p[ar, np.maximum(k - 1, 0)]
    km2 = p[ar, np.maximum(k - 2, 0)]
    O, H, L, C = o[kp], h[kp], l[kp], c[kp]
    O1, H1, L1, C1 = o[km1], h[km1], l[km1], c[km1]
    H2, L2 = h[km2], l[km2]
    w = e.w.to_numpy(np.float32)
    sgn = np.where(e.long.to_numpy(), 1.0, -1.0).astype(np.float32)
    border = e.border.to_numpy(np.float32)
    rng = np.maximum(H - L, .25)

    f = pd.DataFrame(index=e.index)
    f['k'] = k
    f['risk'] = e.risk.to_numpy()
    f['risk_w'] = e.risk.to_numpy() / w
    f['close_beyond'] = sgn * (C - border) / w       # how far past the border it closed
    f['body'] = sgn * (C - O) / rng                  # body direction inside its own range
    f['close_in_range'] = np.where(e.long, (C - L) / rng, (H - C) / rng)
    f['range_w'] = rng / w
    f['range_pts'] = rng
    # the minute against its predecessor
    f['pushes_prev'] = np.where(e.long, H > H1, L < L1).astype(np.int8)
    f['engulf_prev'] = ((np.maximum(O, C) >= np.maximum(O1, C1)) &
                        (np.minimum(O, C) <= np.minimum(O1, C1))).astype(np.int8)
    f['inside_prev'] = ((H <= H1) & (L >= L1)).astype(np.int8)
    f['prev_pushes'] = np.where(e.long, H1 > H2, L1 < L2).astype(np.int8)
    # the Z run that just ended: its length and how deep it went
    zlen = np.zeros(n, np.int16)
    depth = np.zeros(n, np.float32)
    nprior = np.zeros(n, np.int16)
    been_far = np.zeros(n, np.int8)
    been_exit = np.zeros(n, np.int8)
    S = st[si]
    for i in range(n):
        q = S[i, :k[i] - 1]            # minutes 1..k-1
        j = len(q)
        while j > 0 and q[j - 1] == 2:
            j -= 1
        zlen[i] = len(q) - j
        pre = q[:j]
        nprior[i] = int(np.sum((pre[1:] == 2) & (pre[:-1] != 2))) + (1 if len(pre) and pre[0] == 2 else 0)
        been_far[i] = int((pre == 3).any())
        been_exit[i] = int((pre == 1).any())
    f['zlen'] = zlen
    f['n_prior_Z'] = nprior
    f['been_far'] = been_far
    f['been_exit'] = been_exit
    # deepest penetration of that Z run, in widths, measured from the near border
    top = d.zone_top.to_numpy(np.float32)[si]
    bot = d.zone_bottom.to_numpy(np.float32)[si]
    south = (d.t0_exit_side.to_numpy() == 'south')[si]
    near = np.where(south, bot, top)
    dep = np.zeros(n, np.float32)
    for i in range(n):
        a, b = k[i] - zlen[i], k[i]        # minutes [a, b) are the Z run
        if b <= a:
            continue
        pp = p[i, a:b]
        pp = pp[pp >= 0]
        if not len(pp):
            continue
        dep[i] = (np.max(h[pp]) - near[i]) / w[i] if south[i] else (near[i] - np.min(l[pp])) / w[i]
    f['zdepth'] = dep
    # tape volatility before T0: median minute range of the 30 minutes ending at T0
    base = p[:, 0]
    back = base[:, None] - np.arange(30)[None, :]
    back = np.clip(back, 0, len(h) - 1)
    f['atr30'] = np.median(h[back] - l[back], axis=1)
    f['w_atr'] = w / np.maximum(f.atr30.to_numpy(), .25)
    f['risk_atr'] = e.risk.to_numpy() / np.maximum(f.atr30.to_numpy(), .25)
    # clock
    tt = pd.to_datetime(ts[p[ar, k]]).tz_localize('UTC').tz_convert('America/New_York')
    f['year'] = tt.year.to_numpy()
    f['minute_of_day'] = (tt.hour * 60 + tt.minute).to_numpy()
    f['rth'] = ((f.minute_of_day >= 570) & (f.minute_of_day < 960)).astype(np.int8)
    f['tf'] = e.tf.to_numpy()
    f['long'] = e.long.to_numpy()
    f['first'] = e['first'].to_numpy()
    f['scene'] = si
    f['row'] = r
    f['t0'] = e.t0.to_numpy()
    f['mfe'] = e.mfe.to_numpy()
    f['mfe_all'] = e.mfe_all.to_numpy()
    f['stopped'] = e.stopped.to_numpy()
    f['tstop'] = e.tstop.to_numpy()
    f['R1'] = (f.mfe >= f.risk).astype(np.int8)
    f['R2'] = (f.mfe >= 2 * f.risk).astype(np.int8)
    f['R3'] = (f.mfe >= 3 * f.risk).astype(np.int8)
    f.to_parquet(common.CACHE / f'feat_{kind}.parquet')
    return f


if __name__ == '__main__':
    for kind in ('ZE', 'ZF'):
        f = build(kind)
        print(kind, f.shape, 'R1', round(f.R1.mean(), 4), 'R2', round(f.R2.mean(), 4), 'R3', round(f.R3.mean(), 4))
