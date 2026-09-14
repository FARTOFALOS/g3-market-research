import numpy as np, pyarrow.parquet as pq
from pathlib import Path
import open_y as M, transition as T, external as X

ROOT = Path('C:/Users/Admin/Claude/g3-market-research'); NS = 60_000_000_000


def build2(inst, tf):
    A, se = M.market(inst); ts = A['close_ts_utc_ns']
    base = T.build(inst, tf)
    rows = pq.read_table(ROOT / f'data/field/{inst}/cells/tf_{tf:04d}/passports.parquet',
                         columns=['zone_bottom', 'zone_top', 't0_spine_pos',
                                  't0_exit_side']).to_pylist()
    rows.sort(key=lambda r: r['t0_spine_pos'])
    meta = []
    for r in rows:
        t0 = int(r['t0_spine_pos']); p = t0 + 2
        si = se['lookup'].get(int(A['session_id'][t0]))
        if si is None: continue
        an = int(se['session_open_utc_ns'][si]); k = ((int(ts[t0]) - an) // NS - 1) // tf
        bg = an + k * tf * NS
        st = int(np.searchsorted(ts[:p + 1], bg, side='right')); tw = np.asarray(ts[st:p + 1])
        if len(tw) < 3 or int(tw[0]) != bg + NS or np.any(np.diff(tw) != NS): continue
        w = r['zone_top'] - r['zone_bottom']
        if w <= 0: continue
        north = r['t0_exit_side'] == 'north'; ex = r['zone_top'] if north else r['zone_bottom']
        hi = st + 240
        if hi + 31 >= len(A['close']): continue
        seg = np.asarray(ts[p + 1:hi])
        if len(seg) == 0 or np.any(np.diff(seg) != NS): continue
        sh = np.asarray(A['high'][p + 1:hi], float); sl = np.asarray(A['low'][p + 1:hi], float)
        ol = (sl - ex) if north else (ex - sh); h = np.where(ol <= 0)[0]
        if len(h) == 0: continue
        q = p + 1 + int(h[0]); tp = np.asarray(ts[st:q + 1])
        if len(tp) < 4 or np.any(np.diff(tp) != NS): continue
        cp = np.asarray(A['close'][st:q + 1], float)
        if not np.isfinite(cp).all(): continue
        d = np.diff(cp)
        if len(d) < 3 or np.std(d) <= 0: continue
        ft = np.asarray(ts[q:q + 31])
        if len(ft) < 31 or np.any(np.diff(ft) != NS): continue
        cf = np.asarray(A['close'][q:q + 31], float)
        if not np.isfinite(cf).all(): continue
        meta.append((q, float(A['close'][q]), int(ts[q])))
    assert len(meta) == len(base), (len(meta), len(base))
    for b, e in zip(base, X.enrich(inst, tf, meta)):
        b.update(e)
    return base


D = {}
for i, tf in [('NQ', 54), ('ES', 54), ('YM', 54), ('NQ', 30)]:
    D[(i, tf)] = build2(i, tf)
    print('%s tf%d n=%d' % (i, tf, len(D[(i, tf)])), flush=True)

ref = D[('NQ', 54)]
CS = np.percentile([x['close_s'] for x in ref], [20, 40, 60, 80])
EXT = ['field_zones', 'cross_mean', 'cross_absdiff', 'session_min', 'session_pos']
print('\nEXTERNAL COORDINATES on NQ tf54:')
for c in EXT:
    v = np.array([x[c] for x in ref], float)
    print('   %-14s finite %.3f  p5=%.2f med=%.2f p95=%.2f'
          % (c, np.mean(np.isfinite(v)), *np.nanpercentile(v, [5, 50, 95])))

MED = {c: float(np.nanmedian([x[c] for x in ref])) for c in EXT}
RNG = np.random.default_rng(11)


def eff(d, c, r, idx=None):
    v = np.array([x['close_s'] for x in d]); y = np.array([x[r] for x in d], float)
    z = np.array([x[c] for x in d], float); b = np.searchsorted(CS, v)
    if idx is not None: v, y, z, b = v[idx], y[idx], z[idx], b[idx]
    num = den = 0.0
    for s in range(5):
        m = (b == s) & np.isfinite(z) & np.isfinite(y)
        hi = m & (z > MED[c]); lo = m & (z <= MED[c])
        if hi.sum() >= 10 and lo.sum() >= 10:
            w = min(hi.sum(), lo.sum()); num += w * (np.mean(y[hi]) - np.mean(y[lo])); den += w
    return num / den if den else np.nan


for r in ['drift', 'ac1', 'volratio']:
    print('\nREADING %s  effect [90%% bootstrap], high minus low within close_s strata' % r)
    print('%-14s %s  verdict' % ('external', ' '.join('%-26s' % ('%s tf%d' % k) for k in D)))
    for c in EXT:
        cells, sgn, excl = [], [], 0
        for k, d in D.items():
            e = eff(d, c, r)
            bs = np.array([eff(d, c, r, RNG.integers(0, len(d), len(d))) for _ in range(400)])
            p5, p95 = np.nanpercentile(bs, [5, 95])
            cells.append('%+.4f[%+.3f,%+.3f]' % (e, p5, p95)); sgn.append(e)
            if np.isfinite(p5) and np.isfinite(p95) and (p5 > 0 or p95 < 0): excl += 1
        ok = (all(np.isfinite(x) for x in sgn)
              and (all(x > 0 for x in sgn) or all(x < 0 for x in sgn)) and excl >= 3)
        print('%-14s %s  %s' % (c, ' '.join('%-26s' % x for x in cells),
                                'PASSES' if ok else 'no'))
