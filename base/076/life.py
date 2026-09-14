"""Minimal Life-before-scene state L: what happened between the zone's birth and
the current scene. Built from price, not from the event log.

Measured fact that forced this: every RIZ has exactly two events before T0
(precursor_formed and one accepted_span), so the field's event log carries no
count of prior interactions. Any Life state must come from the tape.

Window, declared before the outcome: the last W = 1440 minutes strictly before
T0, or the zone's whole life if it is younger. The window length actually used
is recorded. Age itself is a separate, unbounded coordinate.

No Y. No current-scene features. No Birth features (those are controlled for
separately).
"""
import numpy as np, pyarrow.parquet as pq
from pathlib import Path
import open_y as M, birth as BR

ROOT = Path('C:/Users/Admin/Claude/g3-market-research'); NS = 60_000_000_000
W = 1440


def life_rows(inst, tf):
    A, se = M.market(inst); ts = A['close_ts_utc_ns']
    t = pq.read_table(ROOT / f'data/field/{inst}/cells/tf_{tf:04d}/passports.parquet',
                      columns=['t0_spine_pos', 'precursor_formed_spine_pos',
                               'zone_top', 'zone_bottom']).to_pylist()
    out = {}
    for r in t:
        pf = int(r['precursor_formed_spine_pos']); t0 = int(r['t0_spine_pos'])
        if pf <= 0 or t0 <= pf or t0 >= len(A['close']):
            continue
        age = t0 - pf
        lo = max(pf, t0 - W)
        n = t0 - lo
        if n < 30:
            continue
        hh = np.asarray(A['high'][lo:t0], float)
        ll = np.asarray(A['low'][lo:t0], float)
        cc = np.asarray(A['close'][lo:t0], float)
        if not (np.isfinite(hh).all() and np.isfinite(ll).all() and np.isfinite(cc).all()):
            continue
        zb, zt = r['zone_bottom'], r['zone_top']
        if zt <= zb:
            continue
        touch = (ll <= zt) & (hh >= zb)                 # minute range meets the band
        visits = int(np.count_nonzero(touch[1:] & ~touch[:-1]) + (1 if touch[0] else 0))
        d = np.diff(cc)
        sig = float(np.std(d)) if len(d) >= 2 else np.nan
        if not np.isfinite(sig) or sig <= 0:
            continue
        away = np.maximum(ll - zt, zb - hh)             # distance outside the band
        out[t0] = dict(log_age=float(np.log10(age)),
                       touch_share=float(touch.mean()),
                       visits_k=1000.0 * visits / n,
                       max_exc=float(np.max(away)) / sig,
                       win_len=float(n))
    return out


LF = ['log_age', 'touch_share', 'visits_k', 'max_exc']
BF = ['imp_size', 'imp_shape', 'imp_close_pos']
CUR = ['close_s', 'sigma', 'width']


def attach(inst, tf):
    rows = BR.attach(inst, tf)          # already carries current state + Birth
    L = life_rows(inst, tf)
    # recover t0 per row in the same order BR.attach produced them
    A, se = M.market(inst); ts = A['close_ts_utc_ns']
    p = pq.read_table(ROOT / f'data/field/{inst}/cells/tf_{tf:04d}/passports.parquet',
                      columns=['t0_spine_pos']).to_pylist()
    # BR.attach filtered by the same pipeline; re-derive by matching on width+close_s
    # is fragile, so instead re-run the key extraction here:
    out = []
    keys = BR._last_keys.get((inst, tf))
    for row, t0 in zip(rows, keys):
        l = L.get(t0)
        if l is None:
            continue
        row = dict(row); row.update(l)
        out.append(row)
    return out
