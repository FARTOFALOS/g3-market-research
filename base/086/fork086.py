#!/usr/bin/env python3
"""086 — structural fork of close_break: target b against cancel a = first tick beyond M. FREEZE_086.

Stages (each reads the previous output; the holdout outcomes are computed only in `holdout`):
    selftest   synthetic bars both sides; with a = 2c - b the walker must reproduce the frozen 084 race
    prep       outcome-free map cut points on NQ search -> work/086/cells086.json
    budget     resolution budget from geometry + 084 clustering (no fork outcomes)
    overall    X / G on NQ development, NQ evaluation (decides), ES and YM evaluation
    map        16 frozen cells on NQ search (flag only if >= 50 distinct T0 days)
    select     freeze selected regions and their search signs -> work/086/selected086.json
    holdout    NQ 2025-11-01 ... 2026-05-04, once: raw X / G primary, sign-aligned secondary
"""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit, prange

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'work/086'
OUT84 = ROOT / 'work/084'
sys.path.insert(0, str(ROOT / 'base/084'))
import race084                                                               # noqa: E402
sys.path.insert(0, str(ROOT / 'base/081'))
from trading import sessions, bar_session_map                               # noqa: E402

TICK = {'NQ': 0.25, 'ES': 0.25, 'YM': 1.0}
COST = {'NQ': 1.00, 'ES': 1.00, 'YM': 4.00}
WIN, CANCEL, SAME, GAP_B, GAP_A, LOST, EDGE = 0, 1, 2, 3, 4, 5, 6
NAMES = {WIN: 'win', CANCEL: 'cancel', SAME: 'same_bar', GAP_B: 'gap_through_b', GAP_A: 'gap_through_a',
         LOST: 'lost_observability', EDGE: 'archive_edge'}
BLOCKS = (1, 5, 20)
NBOOT = 2000
SEED = 20260916
PERIODS = {  # name: (instrument, x-ray territory, T0 start, T0 end exclusive)
    'NQ_development': ('NQ', 'discovery', '2006-01-01', '2018-12-25'),
    'NQ_evaluation': ('NQ', 'evaluation', '2019-01-01', '2025-10-25'),
    'ES_evaluation': ('ES', 'evaluation', '2019-01-01', '2025-10-25'),
    'YM_evaluation': ('YM', 'evaluation', '2019-01-01', '2025-10-25'),
    'NQ_search': ('NQ', 'evaluation', '2020-01-01', '2025-10-25'),
    'NQ_holdout': ('NQ', 'evaluation', '2025-11-01', '2026-05-05'),
}
COORDS = ('u', 'k', 'ttc', 'r')
MIN_DAYS_CELL = 50           # small-cell rule, FREEZE_086 §6


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def freeze():
    return sha(HERE / 'FREEZE_086.md')


@njit(parallel=True, cache=True)
def _extreme(t0s, qs, north, high, low, out):
    for i in prange(t0s.size):
        if north[i]:
            v = high[t0s[i]]
            for j in range(t0s[i] + 1, qs[i] + 1):
                if high[j] > v:
                    v = high[j]
        else:
            v = low[t0s[i]]
            for j in range(t0s[i] + 1, qs[i] + 1):
                if low[j] < v:
                    v = low[j]
        out[i] = v


@njit(parallel=True, cache=True)
def _fork(qs, bs, as_, north, high, low, kind, last, code, kbar):
    for i in prange(qs.size):
        q = qs[i]; b = bs[i]; a = as_[i]; up = north[i]
        j = q
        while True:
            j += 1
            if j > last:
                code[i] = 6
                break
            if kind[j] != 0:
                code[i] = 5
                break
            hb = low[j] <= b and b <= high[j]
            ha = low[j] <= a and a <= high[j]
            if hb and ha:
                code[i] = 2
                break
            if hb:
                code[i] = 0
                break
            if ha:
                code[i] = 1
                break
            if up:
                gb = high[j] < b
                ga = low[j] > a
            else:
                gb = low[j] > b
                ga = high[j] < a
            if gb or ga:
                code[i] = 3 if gb else 4
                break
        kbar[i] = j - q


_MARKET = {}


def market(inst):
    if inst not in _MARKET:
        _MARKET[inst] = race084.load_market(inst)
    return _MARKET[inst]


def geometry(period):
    inst, terr, start, end = PERIODS[period]
    p = race084.populations(inst, terr)['close_break']
    ts0 = p.t0_ts_ns.to_numpy()
    p = p[(ts0 >= pd.Timestamp(start, tz='UTC').value) & (ts0 < pd.Timestamp(end, tz='UTC').value)].reset_index(drop=True)
    (high, low, close), _ = market(inst)
    t0 = p.t0_pos.to_numpy().astype(np.int64); q = p.q.to_numpy().astype(np.int64)
    north = (p.side == 'north').to_numpy()
    M = np.empty(len(p)); _extreme(t0, q, north, high, low, M)
    c = close[q]; b = p.exit_boundary.to_numpy()
    d = np.abs(c - b)
    a = np.where(north, M + TICK[inst], M - TICK[inst])
    s = np.abs(a - c)
    assert (d > 0).all() and (s > TICK[inst] - 1e-9).all()
    assert (np.where(north, b < c, b > c) & np.where(north, c < a, c > a)).all(), 'b, c, a out of order'
    cs = np.concatenate(([0.0], np.cumsum(high - low)))
    sigma = (cs[q + 1] - cs[t0]) / (q - t0 + 1)
    ts = np.load(ROOT / 'data/market' / inst / 'close_ts_utc_ns.npy')
    cal = pd.read_parquet(ROOT / 'setups/S-04/calendar.parquet').close_ns.to_numpy().astype(np.int64)
    idx = np.searchsorted(cal, ts[q], side='left')
    ttc = np.where(idx < cal.size, (cal[np.minimum(idx, cal.size - 1)] - ts[q]) / 60e9, np.nan)
    sess, _ = bar_session_map(ts, *sessions())
    g = pd.DataFrame({'riz_id': p.riz_id, 'side': p.side, 't0_pos': t0, 'q': q, 't0_day': p.t0_day.to_numpy(),
                      'north': north, 'b': b, 'c': c, 'M': M, 'a': a, 'd': d, 's': s, 'p0': s / (d + s),
                      'sigma': sigma, 'u': d / sigma, 'k': (q - t0).astype(np.int64), 'ttc': ttc, 'r': s / d,
                      'in_window': sess[q] >= 0})
    g.attrs['instrument'] = inst
    return g


def walk(g):
    inst = g.attrs['instrument']
    (high, low, close), kind = market(inst)
    code = np.full(len(g), -1, np.int8); kbar = np.zeros(len(g), np.int64)
    _fork(g.q.to_numpy(), g.b.to_numpy(), g.a.to_numpy(), g.north.to_numpy(), high, low, kind, close.size - 1, code, kbar)
    assert (code >= 0).all()
    return code, kbar


def per_film(g, code):
    p0 = g.p0.to_numpy(); d = g.d.to_numpy(); s = g.s.to_numpy()
    unk = np.isin(code, (SAME, LOST, EDGE))
    x_lo = np.where(code == WIN, 1 - p0, np.where(code == CANCEL, -p0, np.where(unk, -p0, 0.0)))
    x_hi = np.where(code == WIN, 1 - p0, np.where(code == CANCEL, -p0, np.where(unk, 1 - p0, 0.0)))
    g_lo = np.where(code == WIN, d, np.where(code == CANCEL, -s, np.where(unk, -s, 0.0)))
    g_hi = np.where(code == WIN, d, np.where(code == CANCEL, -s, np.where(unk, d, 0.0)))
    return x_lo, x_hi, g_lo, g_hi


def estimate(x_lo, x_hi, g_lo, g_hi, day, w=None):
    w = np.ones(x_lo.size) if w is None else w
    ud, inv = np.unique(day, return_inverse=True)
    D = ud.size
    sums = {k: np.bincount(inv, weights=w * v, minlength=D) for k, v in
            (('xl', x_lo), ('xh', x_hi), ('gl', g_lo), ('gh', g_hi))}
    n_d = np.bincount(inv, weights=w, minlength=D)
    N = n_d.sum()
    out = {'N_weighted': round(float(N), 2), 'days': int(D),
           'X': [round(float(sums['xl'].sum() / N), 5), round(float(sums['xh'].sum() / N), 5)],
           'G': [round(float(sums['gl'].sum() / N), 4), round(float(sums['gh'].sum() / N), 4)],
           'X_by_block': {}, 'G_by_block': {}}
    for L in BLOCKS:
        rng = np.random.default_rng(SEED + L)
        nb = int(np.ceil(D / L))
        starts = rng.integers(0, max(D - L + 1, 1), size=(NBOOT, nb))
        idx = (starts[:, :, None] + np.arange(L)[None, None, :]).reshape(NBOOT, -1)[:, :D]
        nn = n_d[idx].sum(1)
        rep = {k: v[idx].sum(1) / nn for k, v in sums.items()}
        out['X_by_block'][str(L)] = [round(float(np.quantile(rep['xl'], 0.025)), 5), round(float(np.quantile(rep['xh'], 0.975)), 5)]
        out['G_by_block'][str(L)] = [round(float(np.quantile(rep['gl'], 0.025)), 4), round(float(np.quantile(rep['gh'], 0.975)), 4)]
    out['X_outer_95'] = max(out['X_by_block'].values(), key=lambda v: v[1] - v[0])
    out['G_outer_95'] = max(out['G_by_block'].values(), key=lambda v: v[1] - v[0])
    lo, hi = out['X_outer_95']
    if lo > 0:
        out['class'] = 'positive structural edge toward b'
        out['sign'] = 1
    elif hi < 0:
        out['class'] = 'negative: the reverse side of this binary fork is positive'
        out['sign'] = -1
    else:
        out['class'] = 'unresolved'
        out['sign'] = 0
        out['limiting_cause'] = 'identification' if out['X'][0] <= 0 <= out['X'][1] else 'statistical precision'
    return out


def counts(code):
    return {NAMES[c]: int((code == c).sum()) for c in NAMES}


def cell_ids(g, med):
    bits = np.zeros(len(g), np.int64)
    for i, name in enumerate(COORDS):
        bits |= ((g[name].to_numpy() > med[name]).astype(np.int64) << i)
    return bits


def cell_label(cid):
    return ' '.join(f"{name}{'+' if (cid >> i) & 1 else '-'}" for i, name in enumerate(COORDS))


# ------------------------------------------------------------------ stages

def selftest():
    def case(rows, north, kind_at=None):
        h = np.array([r[0] for r in rows]); lo = np.array([r[1] for r in rows])
        if not north:
            h, lo = 200.0 - lo, 200.0 - h
        kind = np.zeros(len(rows), np.int8)
        if kind_at is not None:
            kind[kind_at] = 2
        b, a = (100.0, 102.25) if north else (100.0, 97.75)
        code = np.full(1, -1, np.int8); kb = np.zeros(1, np.int64)
        _fork(np.array([0]), np.array([b]), np.array([a]), np.array([north]), h, lo, kind, len(rows) - 1, code, kb)
        return int(code[0]), int(kb[0])
    q = (101.25, 100.75); neutral = (101.50, 100.50)          # c = 101, b = 100, M = 102 -> a = 102.25 (north)
    cases = {'win': ([q, neutral, (100.50, 99.75)], None, WIN, 2), 'cancel': ([q, (102.50, 101.75)], None, CANCEL, 1),
             'same': ([q, (102.50, 99.75)], None, SAME, 1), 'gap_b': ([q, (99.75, 99.25)], None, GAP_B, 1),
             'gap_a': ([q, (103.00, 102.50)], None, GAP_A, 1), 'tie_not_cancel': ([q, (102.00, 101.00), (100.25, 100.0)], None, WIN, 2),
             'lost': ([q, neutral, (100.50, 99.75)], 2, LOST, 2), 'edge': ([q, neutral], None, EDGE, 2)}
    for label, (rows, kat, want, wk) in cases.items():
        for north in (True, False):
            got = case(rows, north, kat)
            assert got == (want, wk), (label, north, got)
    # the walker with a = 2c - b must reproduce the frozen 084 race film by film
    for inst, terr in (('NQ', 'discovery'), ('ES', 'evaluation')):
        p = race084.populations(inst, terr)['close_break']
        (high, low, close), kind = market(inst)
        qn = p.q.to_numpy().astype(np.int64); b = p.exit_boundary.to_numpy(); north = (p.side == 'north').to_numpy()
        code = np.full(len(p), -1, np.int8); kb = np.zeros(len(p), np.int64)
        _fork(qn, b, 2 * close[qn] - b, north, high, low, kind, close.size - 1, code, kb)
        st = pd.read_parquet(OUT84 / f'race_{inst}_{terr}.parquet')
        st = st[st.anchor == 'close_break'].set_index('riz_id').loc[p.riz_id]
        assert (st.code.to_numpy() == code).all() and (st.kbar.to_numpy() == kb).all(), f'{inst} {terr}: 084 not reproduced'
        print(f'  {inst} {terr}: fork walker with mirror level reproduces 084 on {len(p)} films')
    print('fork selftest: 8 synthetic scenes x 2 sides, and 084 reproduction, as expected')


def prep():
    g = geometry('NQ_search')
    ok = np.isfinite(g.ttc.to_numpy())
    med = {name: float(np.median(g.loc[ok, name])) for name in COORDS}
    res = {'freeze_sha256': freeze(), 'period': PERIODS['NQ_search'], 'films': int(len(g)),
           'films_without_ttc': int((~ok).sum()), 'medians': med, 'rule': '<= median -> low, > median -> high'}
    p = OUT / 'cells086.json'
    p.write_text(json.dumps(res, indent=1), encoding='utf-8')
    print(json.dumps(res, indent=1))
    print('cells sha256', sha(p))


def cluster_rho(inst, terr):
    """ρ from the frozen 084 mirror race (already opened outcomes): per-film 1{own} - 1{mirror}."""
    p = race084.populations(inst, terr)['close_break']
    st = pd.read_parquet(OUT84 / f'race_{inst}_{terr}.parquet')
    st = st[st.anchor == 'close_break'].set_index('riz_id').loc[p.riz_id]
    v = (st.code.to_numpy() == 0).astype(float) - (st.code.to_numpy() == 1)
    N = v.size
    var_iid = v.var(ddof=1) / N
    ud, inv = np.unique(p.t0_day.to_numpy(), return_inverse=True)
    D = ud.size
    s_d = np.bincount(inv, weights=v, minlength=D); n_d = np.bincount(inv, minlength=D).astype(float)
    var_boot = 0.0
    for L in BLOCKS:
        rng = np.random.default_rng(SEED + 100 + L)
        nb = int(np.ceil(D / L))
        starts = rng.integers(0, max(D - L + 1, 1), size=(NBOOT, nb))
        idx = (starts[:, :, None] + np.arange(L)[None, None, :]).reshape(NBOOT, -1)[:, :D]
        var_boot = max(var_boot, float((s_d[idx].sum(1) / n_d[idx].sum(1)).var(ddof=1)))
    K = float((n_d ** 2).sum() / N)
    deff = var_boot / var_iid
    codes = st.code.to_numpy()
    unk = float(np.isin(codes, (2, 5, 6)).mean())
    return {'deff': round(deff, 3), 'K': round(K, 2), 'rho': round((deff - 1) / (K - 1), 6), 'unknown_share_084': round(unk, 4)}


def budget_unit(g, rho, unk, cost):
    N = len(g)
    m = g.groupby('t0_day').size().to_numpy().astype(float)
    K = float((m ** 2).sum() / N)
    deff = 1 + rho * (K - 1)
    p0 = g.p0.to_numpy(); d = g.d.to_numpy(); s = g.s.to_numpy()
    se_x = np.sqrt(deff * (p0 * (1 - p0)).sum()) / N
    se_g = np.sqrt(deff * (d * s).sum()) / N
    out = {'N': N, 'days': int(m.size), 'deff': round(deff, 2), 'mean_p0': round(float(p0.mean()), 4),
           'mean_d': round(float(d.mean()), 3), 'mean_s': round(float(s.mean()), 3),
           'SE_X': round(float(se_x), 5), 'SE_G': round(float(se_g), 4),
           'cost_hurdle_X': round(float((cost / (d + s)).mean()), 4)}
    for zname, z in (('95', 1.96), ('80power', 2.80)):
        out[f'MDE_X_plus_{zname}'] = round(float(unk * p0.mean() + z * se_x), 4)
        out[f'MDE_X_minus_{zname}'] = round(float(unk * (1 - p0.mean()) + z * se_x), 4)
        out[f'MDE_G_plus_{zname}'] = round(float(unk * s.mean() + z * se_g), 3)
        out[f'MDE_G_minus_{zname}'] = round(float(unk * d.mean() + z * se_g), 3)
    return out


def budget():
    cells = json.loads((OUT / 'cells086.json').read_text(encoding='utf-8'))
    fam = {'NQ_development': ('NQ', 'discovery'), 'NQ_evaluation': ('NQ', 'evaluation'), 'ES_evaluation': ('ES', 'evaluation'),
           'YM_evaluation': ('YM', 'evaluation'), 'NQ_search': ('NQ', 'evaluation'), 'NQ_holdout': ('NQ', 'evaluation')}
    rhos = {}
    res = {'freeze_sha256': freeze(), 'cells_sha256': sha(OUT / 'cells086.json'), 'clustering_084': {}, 'units': {}, 'cells': {}}
    for period, key in fam.items():
        if key not in rhos:
            rhos[key] = cluster_rho(*key)
            res['clustering_084'][f'{key[0]} {key[1]}'] = rhos[key]
        g = geometry(period)
        R = rhos[key]
        res['units'][period] = budget_unit(g, R['rho'], R['unknown_share_084'], COST[g.attrs['instrument']])
        if period == 'NQ_search':
            ok = np.isfinite(g.ttc.to_numpy())
            gg = g[ok].copy(); gg.attrs['instrument'] = 'NQ'
            cid = cell_ids(gg, cells['medians'])
            for c in range(16):
                sub = gg[cid == c]
                if len(sub):
                    res['cells'][cell_label(c)] = budget_unit(sub, R['rho'], R['unknown_share_084'], COST['NQ'])
    (OUT / 'budget086.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print('clustering from 084:', json.dumps(res['clustering_084']))
    cols = ('N', 'days', 'deff', 'mean_p0', 'mean_d', 'mean_s', 'SE_X', 'MDE_X_plus_95', 'MDE_X_minus_95', 'MDE_X_plus_80power',
            'SE_G', 'MDE_G_plus_95', 'MDE_G_minus_95', 'cost_hurdle_X')
    print(f"{'unit':<18}" + ''.join(f'{c:>16}' for c in cols))
    for name, u in list(res['units'].items()) + [(f'cell {k}', v) for k, v in res['cells'].items()]:
        print(f'{name:<18}' + ''.join(f'{u[c]:>16}' for c in cols))


def overall():
    budget_json = OUT / 'budget086.json'
    assert budget_json.exists(), 'budget stage must run first'
    res = {'freeze_sha256': freeze(), 'periods': {}}
    for period in ('NQ_development', 'NQ_evaluation', 'ES_evaluation', 'YM_evaluation'):
        g = geometry(period)
        code, kbar = walk(g)
        xl, xh, gl, gh = per_film(g, code)
        E = estimate(xl, xh, gl, gh, g.t0_day.to_numpy())
        path = pd.Series(list(zip(g.t0_pos, g.side)))
        w = 1.0 / path.map(path.value_counts()).to_numpy()
        Ep = estimate(xl, xh, gl, gh, g.t0_day.to_numpy(), w)
        inst = g.attrs['instrument']
        res['periods'][period] = {'N': int(len(g)), 'counts': counts(code), 'film_weight': E, 'path_weight': Ep,
                                  'cost_points': COST[inst], 'cost_hurdle_X': round(float((COST[inst] / (g.d + g.s)).mean()), 4),
                                  'mean_p0': round(float(g.p0.mean()), 4), 'mean_d': round(float(g.d.mean()), 3),
                                  'mean_s': round(float(g.s.mean()), 3)}
        R = res['periods'][period]
        print(f"\n{period}: N {R['N']}  {R['counts']}")
        print(f"   mean p0 {R['mean_p0']}  mean d {R['mean_d']}  mean s {R['mean_s']}  cost {R['cost_points']} "
              f"(hurdle in X {R['cost_hurdle_X']})")
        for tag, e in (('film', E), ('path', Ep)):
            print(f"   {tag}: X [{e['X'][0]:+.4f}; {e['X'][1]:+.4f}] outer {e['X_outer_95']}  G [{e['G'][0]:+.3f}; {e['G'][1]:+.3f}] "
                  f"outer {e['G_outer_95']}  -> {e['class']}{' (' + e['limiting_cause'] + ')' if 'limiting_cause' in e else ''}")
    (OUT / 'overall086.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')


def map_stage():
    assert (OUT / 'overall086.json').exists(), 'overall stage must run first'
    cells = json.loads((OUT / 'cells086.json').read_text(encoding='utf-8'))
    g = geometry('NQ_search')
    ok = np.isfinite(g.ttc.to_numpy())
    g = g[ok].reset_index(drop=True); g.attrs['instrument'] = 'NQ'
    code, kbar = walk(g)
    xl, xh, gl, gh = per_film(g, code)
    cid = cell_ids(g, cells['medians'])
    res = {'freeze_sha256': freeze(), 'cells_sha256': sha(OUT / 'cells086.json'), 'films': int(len(g)), 'cells': {}}
    print(f"NQ search map: {len(g)} films; medians {cells['medians']}")
    for c in range(16):
        m = cid == c
        if m.sum() == 0:
            continue
        e = estimate(xl[m], xh[m], gl[m], gh[m], g.t0_day.to_numpy()[m])
        if e['days'] < MIN_DAYS_CELL and e['sign'] != 0:
            e['class'] = e['class'] + ' (below small-cell rule: not flagged)'
            e['sign_raw'] = e['sign']
            e['sign'] = 0
        sub = g[m]
        res['cells'][cell_label(c)] = {'cell': c, 'N': int(m.sum()), 'counts': counts(code[m]),
                                       'in_window_share': round(float(sub.in_window.mean()), 4),
                                       'mean_p0': round(float(sub.p0.mean()), 4), 'mean_d': round(float(sub.d.mean()), 3),
                                       'mean_s': round(float(sub.s.mean()), 3),
                                       'cost_hurdle_X': round(float((COST['NQ'] / (sub.d + sub.s)).mean()), 4), **e}
        R = res['cells'][cell_label(c)]
        print(f"  {cell_label(c):<16} N {R['N']:>5} days {e['days']:>4} window {R['in_window_share']:.2f} p0 {R['mean_p0']:.3f} "
              f"X [{e['X'][0]:+.4f}; {e['X'][1]:+.4f}] outer [{e['X_outer_95'][0]:+.4f}; {e['X_outer_95'][1]:+.4f}] "
              f"G outer [{e['G_outer_95'][0]:+.3f}; {e['G_outer_95'][1]:+.3f}] hurdle {R['cost_hurdle_X']:.3f} -> {e['class']}")
    res['flagged'] = {k: v['sign'] for k, v in res['cells'].items() if v['sign'] != 0}
    print('flagged cells:', res['flagged'])
    (OUT / 'map086.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print('map sha256', sha(OUT / 'map086.json'))


def select():
    mp = OUT / 'map086.json'
    assert mp.exists(), 'map stage must run first'
    M = json.loads(mp.read_text(encoding='utf-8'))
    flagged = {k: {'cell': v['cell'], 'sign': v['sign']} for k, v in M['cells'].items() if v['sign'] != 0}
    regions = {}
    for k, v in flagged.items():
        regions.setdefault('positive' if v['sign'] > 0 else 'negative', []).append(v['cell'])
    res = {'freeze_sha256': freeze(), 'map_sha256': sha(mp), 'cells_sha256': sha(OUT / 'cells086.json'),
           'selected_cells': flagged, 'regions': regions,
           'note': 'signs chosen on NQ search before any holdout outcome; never re-chosen on holdout'}
    p = OUT / 'selected086.json'
    p.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(res, ensure_ascii=False, indent=1))
    print('selected sha256', sha(p))


def holdout():
    sp = OUT / 'selected086.json'
    assert sp.exists(), 'select stage must run first'
    S = json.loads(sp.read_text(encoding='utf-8'))
    cells = json.loads((OUT / 'cells086.json').read_text(encoding='utf-8'))
    g = geometry('NQ_holdout')
    code, kbar = walk(g)
    xl, xh, gl, gh = per_film(g, code)
    day = g.t0_day.to_numpy()
    res = {'freeze_sha256': freeze(), 'selected_sha256': sha(sp), 'N': int(len(g)), 'counts': counts(code),
           'primary_raw_overall': estimate(xl, xh, gl, gh, day)}
    ok = np.isfinite(g.ttc.to_numpy())
    cid = np.full(len(g), -1)
    cid[ok] = cell_ids(g[ok], cells['medians'])
    res['films_without_ttc'] = int((~ok).sum())
    res['primary_raw_regions'] = {}
    for rname, cl in S['regions'].items():
        m = np.isin(cid, cl)
        frozen_sign = 1 if rname == 'positive' else -1
        if m.sum() == 0:
            res['primary_raw_regions'][rname] = {'cells': [cell_label(c) for c in cl], 'N': 0, 'confirmed': False}
            continue
        e = estimate(xl[m], xh[m], gl[m], gh[m], day[m])
        lo, hi = e['X_outer_95']
        e['confirmed'] = bool(lo > 0) if frozen_sign > 0 else bool(hi < 0)
        res['primary_raw_regions'][rname] = {'cells': [cell_label(c) for c in cl], 'frozen_sign': frozen_sign, 'N': int(m.sum()), **e}
    sign = np.zeros(len(g))
    for k, v in S['selected_cells'].items():
        sign[cid == v['cell']] = v['sign']
    m = sign != 0
    if m.any():
        pl = np.where(sign > 0, xl, -xh)[m]; ph = np.where(sign > 0, xh, -xl)[m]
        ql = np.where(sign > 0, gl, -gh)[m]; qh = np.where(sign > 0, gh, -gl)[m]
        res['secondary_sign_aligned'] = {'N': int(m.sum()), **estimate(pl, ph, ql, qh, day[m])}
        res['secondary_cells'] = {}
        for k, v in S['selected_cells'].items():
            mm = cid == v['cell']
            if mm.sum():
                res['secondary_cells'][k] = {'N': int(mm.sum()), 'search_sign': v['sign'],
                                             **estimate(xl[mm], xh[mm], gl[mm], gh[mm], day[mm])}
    (OUT / 'holdout086.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    e = res['primary_raw_overall']
    print(f"holdout NQ 2025-11-01..2026-05-04: N {res['N']}  {res['counts']}  (without ttc {res['films_without_ttc']})")
    print(f"   PRIMARY raw overall: X [{e['X'][0]:+.4f}; {e['X'][1]:+.4f}] outer {e['X_outer_95']}  "
          f"G [{e['G'][0]:+.3f}; {e['G'][1]:+.3f}] outer {e['G_outer_95']} -> {e['class']}")
    if not S['regions']:
        print('   no selected regions: region not established on search; holdout reports overall only')
    for rname, R in res['primary_raw_regions'].items():
        if R['N']:
            print(f"   PRIMARY raw region {rname} (frozen sign {R['frozen_sign']:+d}, cells {R['cells']}): N {R['N']}  "
                  f"X [{R['X'][0]:+.4f}; {R['X'][1]:+.4f}] outer {R['X_outer_95']}  G outer {R['G_outer_95']}  confirmed {R['confirmed']}")
        else:
            print(f"   PRIMARY raw region {rname}: no holdout films in its cells")
    if 'secondary_sign_aligned' in res:
        P = res['secondary_sign_aligned']
        print(f"   secondary sign-aligned (search signs): N {P['N']}  X [{P['X'][0]:+.4f}; {P['X'][1]:+.4f}] outer {P['X_outer_95']}")
        for k, v in res['secondary_cells'].items():
            print(f"   secondary {k:<16} search sign {v['search_sign']:+d}  N {v['N']}  X [{v['X'][0]:+.4f}; {v['X'][1]:+.4f}] "
                  f"outer {v['X_outer_95']}")


if __name__ == '__main__':
    stage = sys.argv[1] if len(sys.argv) > 1 else 'selftest'
    {'selftest': selftest, 'prep': prep, 'budget': budget, 'overall': overall, 'map': map_stage, 'select': select,
     'holdout': holdout}[stage]()
