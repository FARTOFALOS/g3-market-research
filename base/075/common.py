"""075: shared loading for the construction study.

One scene = one RIZ zone (dedup by t0 + zone borders + exit side), exactly as 074.
Window = T0 .. T0+HOR closed minutes on the exact minute grid; a minute that is
missing from the spine lands as -1 in `pos` and as NaN in the prices.

No Volume anywhere. No future label is ever used to select a scene.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MKT = ROOT / 'data/market/NQ'
CACHE = ROOT / 'work/075'
HOR = 120                      # minutes after T0 kept in the window
MINUTE = 60_000_000_000


def tape():
    """full NQ spine in RAM as float32 + session id"""
    o, h, l, c = [np.load(MKT / f'{n}.npy').astype(np.float32) for n in
                  ('open', 'high', 'low', 'close')]
    ts = np.load(MKT / 'close_ts_utc_ns.npy')
    sid = np.load(MKT / 'session_id.npy')
    return o, h, l, c, ts, sid


def positions(ts):
    """(87600, HOR+1) row positions of T0..T0+HOR, -1 where the minute is absent"""
    p = CACHE / f'pos_{HOR}.npy'
    if p.exists():
        return np.load(p)
    t0 = np.load(ROOT / 'work/074/NQ_prefix51.npz')['t0']
    want = t0[:, None] + np.arange(HOR + 1)[None, :] * MINUTE
    q = np.searchsorted(ts, want)
    safe = np.minimum(q, len(ts) - 1)
    ok = (q < len(ts)) & (ts[safe] == want)
    out = np.where(ok, safe, -1).astype(np.int64)
    np.save(p, out)
    return out


def t0_list():
    return np.load(ROOT / 'work/074/NQ_prefix51.npz')['t0']


def window(arr, pos):
    """price window with NaN where the minute is absent"""
    safe = np.maximum(pos, 0)
    v = arr[safe].astype(np.float32)
    v[pos < 0] = np.nan
    return v


def scene_table():
    """one row per zone; `row` indexes into the 87600 T0 windows"""
    d = pd.read_parquet(ROOT / 'work/074/NQ_zones.parquet')
    d = d.drop_duplicates(['t0_ts_ns', 'zone_top', 'zone_bottom', 't0_exit_side'])
    t0 = t0_list()
    row = pd.Series(np.arange(len(t0)), index=t0.astype('int64'))
    d = d[d.t0_ts_ns.isin(row.index)].copy()
    d['row'] = row.loc[d.t0_ts_ns].to_numpy()
    return d.reset_index(drop=True)
