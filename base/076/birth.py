"""Minimal Birth state B of the RIZ, prefix-only. No Y. No current-scene features.

Only the first of the two biography layers is built here: what happened at the
moment the zone was formed. Age, span intervals and prior contacts (the
Life-before-scene layer) are deliberately NOT included; they are a separate
hypothesis.

Two questions, both answered before any outcome is opened:
  1. Is B a function of the current state (position, local sigma, width, kind)?
  2. At comparable current state, is there enough variation in B left?

If B is essentially recoverable from the current state, the branch closes here.
"""
import numpy as np, pyarrow.parquet as pq
from pathlib import Path
import open_y as M, transition as T

ROOT = Path('C:/Users/Admin/Claude/g3-market-research'); NS = 60_000_000_000
BIRTH_LOOK = 60        # minutes before formation used for sigma at birth


def birth_rows(inst, tf):
    """B for every passport, keyed by t0_spine_pos."""
    A, se = M.market(inst); ts = A['close_ts_utc_ns']
    t = pq.read_table(ROOT / f'data/field/{inst}/cells/tf_{tf:04d}/passports.parquet',
                      columns=['t0_spine_pos', 'precursor_formed_spine_pos',
                               'zone_top', 'zone_bottom', 'direction']).to_pylist()
    out = {}
    for r in t:
        pf = int(r['precursor_formed_spine_pos'])
        if pf <= 0 or pf >= len(A['close']):
            continue
        a = pf - BIRTH_LOOK
        lo_bar = pf - tf + 1
        if a < 0 or lo_bar < 0:
            continue
        tq = np.asarray(ts[a:pf + 1])
        if len(tq) < BIRTH_LOOK + 1 or np.any(np.diff(tq) != NS):
            continue
        cq = np.asarray(A['close'][a:pf + 1], float)
        if not np.isfinite(cq).all():
            continue
        sig_b = float(np.std(np.diff(cq)))
        if sig_b <= 0:
            continue
        tb = np.asarray(ts[lo_bar:pf + 1])
        if len(tb) < tf or np.any(np.diff(tb) != NS):
            continue
        o = float(A['open'][lo_bar]); c = float(A['close'][pf])
        h = float(np.max(np.asarray(A['high'][lo_bar:pf + 1], float)))
        l = float(np.min(np.asarray(A['low'][lo_bar:pf + 1], float)))
        if not all(np.isfinite([o, c, h, l])) or h <= l:
            continue
        d = 1.0 if r['direction'] > 0 else -1.0
        w = r['zone_top'] - r['zone_bottom']
        if w <= 0:
            continue
        out[int(r['t0_spine_pos'])] = dict(
            imp_size=d * (c - o) / sig_b,               # displacement that made the zone
            imp_shape=abs(c - o) / (h - l),             # body over range of the forming bar
            imp_close_pos=((c - l) / (h - l)) if d > 0 else ((h - c) / (h - l)),
            w_over_sig_birth=w / sig_b)
    return out


_last_keys = {}


def attach(inst, tf):
    """T0' rows (current state + transition readings) with B attached."""
    base = T.build(inst, tf)
    B = birth_rows(inst, tf)
    # recover t0 for each base row by rebuilding the same passport order
    A, se = M.market(inst); ts = A['close_ts_utc_ns']
    rows = pq.read_table(ROOT / f'data/field/{inst}/cells/tf_{tf:04d}/passports.parquet',
                         columns=['zone_bottom', 'zone_top', 't0_spine_pos',
                                  't0_exit_side']).to_pylist()
    rows.sort(key=lambda r: r['t0_spine_pos'])
    keys = []
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
        keys.append((t0, w))
    assert len(keys) == len(base), (len(keys), len(base))
    out, kept = [], []
    for row, (t0, w) in zip(base, keys):
        b = B.get(t0)
        if b is None:
            continue
        row = dict(row); row.update(b); row['width'] = w
        out.append(row); kept.append(t0)
    _last_keys[(inst, tf)] = kept
    return out


BF = ['imp_size', 'imp_shape', 'imp_close_pos', 'w_over_sig_birth']
CUR = ['close_s', 'sigma', 'width']


def r2(y, X):
    A_ = np.column_stack([np.ones(len(X)), X])
    beta = np.linalg.lstsq(A_, y, rcond=None)[0]
    res = y - A_ @ beta
    ss = ((y - y.mean()) ** 2).sum()
    return 1.0 - (res ** 2).sum() / ss if ss > 0 else np.nan


if __name__ == '__main__':
    D = {}
    for i, tf in [('NQ', 54), ('ES', 54), ('YM', 54), ('NQ', 30)]:
        D[(i, tf)] = attach(i, tf)
        print('%s tf%d  n=%d with birth attached' % (i, tf, len(D[(i, tf)])), flush=True)

    print('\n1. IS B A FUNCTION OF THE CURRENT STATE?')
    print('   R2 of each birth coordinate regressed on close_s, sigma, width, kind-free set')
    print('%-18s %s' % ('birth coord', ' '.join('%-12s' % ('%s tf%d' % k) for k in D)))
    for f in BF:
        cells = []
        for k, d in D.items():
            y = np.array([x[f] for x in d], float)
            X = np.array([[x[c] for c in CUR] for x in d], float)
            ok = np.isfinite(y) & np.isfinite(X).all(1)
            cells.append('%.4f' % r2(y[ok], X[ok]) if ok.sum() > 30 else '--')
        print('%-18s %s' % (f, ' '.join('%-12s' % c for c in cells)))

    print('\n2. HOW MUCH B VARIATION SURVIVES AT COMPARABLE CURRENT STATE?')
    ref = D[('NQ', 54)]
    CS = np.percentile([x['close_s'] for x in ref], [20, 40, 60, 80])
    SG = np.percentile([x['sigma'] for x in ref], [33.3, 66.7])
    print('   cells = 5 close_s strata x 3 sigma terciles, frozen from NQ tf54')
    print('%-18s %s' % ('birth coord', ' '.join('%-16s' % ('%s tf%d' % k) for k in D)))
    for f in BF:
        cells = []
        for k, d in D.items():
            v = np.array([x[f] for x in d], float)
            a = np.searchsorted(CS, [x['close_s'] for x in d])
            b = np.searchsorted(SG, [x['sigma'] for x in d])
            tot = np.nanstd(v)
            within, n = [], []
            for i in range(5):
                for j in range(3):
                    m = (a == i) & (b == j) & np.isfinite(v)
                    if m.sum() >= 10:
                        within.append(np.nanstd(v[m])); n.append(m.sum())
            if not within:
                cells.append('--'); continue
            wsd = float(np.average(within, weights=n))
            cells.append('%.3f / %.3f = %.2f' % (wsd, tot, wsd / tot if tot > 0 else np.nan))
        print('%-18s %s' % (f, ' '.join('%-16s' % c for c in cells)))
    print('\n   cell = within-cell sd / overall sd. Near 1.0 means the birth')
    print('   coordinate is essentially free of the current state.')
