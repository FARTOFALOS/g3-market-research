#!/usr/bin/env python3
"""085 — σ at q_event: direction (ΔB) or speed (ΔR_h). Implements FREEZE_085 exactly.

σ = mean(high - low) over bars T0 ... q_event.  Strata: exact d ticks | side | TF band.
Median split of σ inside the stratum (ties out), MIN_SIDE 30 in each half, stratum weight
min(n_low, n_high). Outcomes recomputed by the frozen race084.walk and checked against 084 files.
"""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'work/085'
OUT84 = ROOT / 'work/084'
sys.path.insert(0, str(ROOT / 'base/084'))
import race084                                                               # noqa: E402
sys.path.insert(0, str(ROOT / 'base/081'))
from trading import sessions, bar_session_map                               # noqa: E402

TERR = [('NQ', 'discovery'), ('NQ', 'evaluation'), ('ES', 'evaluation'), ('YM', 'evaluation')]
TICK = {'NQ': 0.25, 'ES': 0.25, 'YM': 1.0}
TF_EDGES = [0, 2, 5, 15, 60, 240, 1441]
TF_LAB = ['1-2', '3-5', '6-15', '16-60', '61-240', '241-1440']
MIN_SIDE = 30
HS = (5, 15, 60)
BLOCKS = (1, 5, 20)
NBOOT = 2000
SEED = 20260916
OWN, MIRROR, SAME, GAP_B, GAP_M, LOST, EDGE = 0, 1, 2, 3, 4, 5, 6


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def frame(inst, terr, cal):
    p = race084.populations(inst, terr)['close_break']
    market, kind = race084.load_market(inst)
    high, low, close = market
    q = p.q.to_numpy().astype(np.int64); t0 = p.t0_pos.to_numpy().astype(np.int64)
    b = p.exit_boundary.to_numpy()
    north = (p.side == 'north').to_numpy()
    code, kbar = race084.walk(q, b, north, market, kind, blind=False)
    stored = pd.read_parquet(OUT84 / f'race_{inst}_{terr}.parquet')
    stored = stored[stored.anchor == 'close_break'].set_index('riz_id').loc[p.riz_id]
    assert (stored.code.to_numpy() == code).all() and (stored.kbar.to_numpy() == kbar).all(), \
        'outcomes differ from frozen 084 run'
    cs = np.concatenate(([0.0], np.cumsum(high - low)))
    sigma = (cs[q + 1] - cs[t0]) / (q - t0 + 1)
    dist = np.abs(close[q] - b)
    dt = np.rint(dist / TICK[inst]).astype(np.int64)
    assert np.allclose(dt * TICK[inst], dist, atol=1e-9), 'distance off the tick grid'
    ts = np.load(ROOT / 'data/market' / inst / 'close_ts_utc_ns.npy')
    sess, _ = bar_session_map(ts, *cal)
    d = pd.DataFrame({'riz_id': p.riz_id, 'side': p.side, 'tfb': pd.cut(p.tf_minutes, TF_EDGES, labels=TF_LAB).astype(str),
                      'dt': dt, 'sigma': sigma, 'k': (q - t0), 't0_day': p.t0_day.to_numpy(),
                      'in_window': sess[q] >= 0, 'code': code, 'kbar': kbar})
    d['key'] = d.dt.astype(str) + '|' + d.side + '|' + d.tfb
    return d


def split(d):
    med = d.groupby('key').sigma.transform('median')
    d['half'] = np.where(d.sigma < med, 0, np.where(d.sigma > med, 1, -1))
    cnt = d[d.half >= 0].groupby(['key', 'half']).size().unstack(fill_value=0)
    for h in (0, 1):
        if h not in cnt:
            cnt[h] = 0
    kept = cnt[(cnt[0] >= MIN_SIDE) & (cnt[1] >= MIN_SIDE)]
    w = np.minimum(kept[0], kept[1]).astype(float)
    w = w / w.sum()
    sup = {'strata_total': int(d.key.nunique()), 'strata_kept': int(len(kept)), 'films_total': int(len(d)),
           'films_used': int(kept[0].sum() + kept[1].sum()),
           'ties_dropped_in_kept': int(((d.half < 0) & d.key.isin(kept.index)).sum())}
    sup['support_retained'] = round(sup['films_used'] / sup['films_total'], 4)
    return kept.index, w, sup


def quantities(u):
    q = {'n': np.ones(len(u)),
         'S': ((u.code == OWN).astype(float) - (u.code == MIRROR)).to_numpy(),
         'U': u.code.isin([SAME, LOST, EDGE]).astype(float).to_numpy()}
    for h in HS:
        by = (u.kbar <= h)
        q[f'C{h}'] = (u.code.isin([OWN, MIRROR, SAME]) & by).astype(float).to_numpy()
        q[f'UH{h}'] = (u.code.isin([LOST, EDGE]) & by).astype(float).to_numpy()
    return q


def deltas(agg, w):
    """agg[name] has shape (K, 2, ...) : strata x half (0 low, 1 high) x replicates. Returns interval arrays."""
    nL, nH = agg['n'][:, 0], agg['n'][:, 1]
    ok = (nL > 0) & (nH > 0)
    ww = w.reshape(-1, *([1] * (nL.ndim - 1))) * ok
    tot = ww.sum(0)

    def wsum(x):
        x = np.where(ok, x, 0.0)
        return (ww * x).sum(0) / np.where(tot > 0, tot, np.nan)

    with np.errstate(divide='ignore', invalid='ignore'):
        bl = lambda g: (agg['S'][:, g] - agg['U'][:, g]) / agg['n'][:, g]
        bh = lambda g: (agg['S'][:, g] + agg['U'][:, g]) / agg['n'][:, g]
        out = {'dB': (wsum(bl(1) - bh(0)), wsum(bh(1) - bl(0)))}
        for h in HS:
            rl = lambda g: agg[f'C{h}'][:, g] / agg['n'][:, g]
            rh = lambda g: (agg[f'C{h}'][:, g] + agg[f'UH{h}'][:, g]) / agg['n'][:, g]
            out[f'dR{h}'] = (wsum(rl(1) - rh(0)), wsum(rh(1) - rl(0)))
    return out


def analyse(d):
    keys, w, sup = split(d)
    u = d[d.key.isin(keys) & (d.half >= 0)].reset_index(drop=True)
    kidx = pd.Index(keys).get_indexer(u.key)
    K = len(keys)
    qs = quantities(u)
    # full data
    full = {}
    for name, v in qs.items():
        a = np.zeros((K, 2))
        np.add.at(a, (kidx, u.half.to_numpy()), v)
        full[name] = a
    point = deltas(full, w.to_numpy())
    res = {'support': sup, 'identified': {k: [round(float(v[0]), 5), round(float(v[1]), 5)] for k, v in point.items()}}
    # descriptive imbalance, stratum-weighted high - low
    def wdiff(col):
        a = np.zeros((K, 2)); c = np.zeros((K, 2))
        np.add.at(a, (kidx, u.half.to_numpy()), u[col].to_numpy(dtype=float))
        np.add.at(c, (kidx, u.half.to_numpy()), 1.0)
        m = a / c
        return float(w.to_numpy() @ (m[:, 1] - m[:, 0])), float(w.to_numpy() @ m[:, 0]), float(w.to_numpy() @ m[:, 1])
    res['imbalance_high_minus_low'] = {col: [round(x, 4) for x in wdiff(col)] for col in ('sigma', 'k', 'in_window')}
    # bootstrap over days of the used population
    ud, dcode = np.unique(u.t0_day.to_numpy(), return_inverse=True)
    D = ud.size
    rows = kidx * 2 + u.half.to_numpy()
    mats = {name: sparse.csr_matrix((v, (rows, dcode)), shape=(2 * K, D)) for name, v in qs.items()}
    sets = {name: {} for name in point}
    for L in BLOCKS:
        rng = np.random.default_rng(SEED + L)
        nb = int(np.ceil(D / L))
        starts = rng.integers(0, max(D - L + 1, 1), size=(NBOOT, nb))
        idx = (starts[:, :, None] + np.arange(L)[None, None, :]).reshape(NBOOT, -1)[:, :D]
        M = np.zeros((D, NBOOT))
        for r in range(NBOOT):
            M[:, r] = np.bincount(idx[r], minlength=D)
        agg = {name: (m @ M).reshape(K, 2, NBOOT) for name, m in mats.items()}
        rep = deltas(agg, w.to_numpy())
        for name, (lo, hi) in rep.items():
            sets[name][str(L)] = [round(float(np.nanquantile(lo, 0.025)), 5), round(float(np.nanquantile(hi, 0.975)), 5)]
    res['outer_set_by_block'] = sets
    res['outer_set_95'] = {name: max(v.values(), key=lambda x: x[1] - x[0]) for name, v in sets.items()}
    res['days'] = int(D)
    return res


def classify(r):
    s = r['outer_set_95']
    excl = lambda x: x[0] > 0 or x[1] < 0
    sign = lambda x: 1 if x[0] > 0 else (-1 if x[1] < 0 else 0)
    speed = excl(s['dR5']) and excl(s['dR15']) and sign(s['dR5']) == sign(s['dR15'])
    out = {'speed_differs': bool(speed), 'speed_sign_high_minus_low': sign(s['dR5']) if speed else 0,
           'dB_excludes_zero': bool(excl(s['dB']))}
    if excl(s['dB']):
        out['class'] = 'H1 supported: ' + ('high σ shifts balance toward own' if s['dB'][0] > 0 else
                                           'high σ shifts balance toward mirror')
    elif speed:
        out['class'] = 'H2: speed differs, directional difference not established'
    else:
        causes = []
        if r['identified']['dB'][0] <= 0 <= r['identified']['dB'][1]:
            causes.append('identification (ΔB identified interval contains 0)')
        else:
            causes.append('precision on ΔB')
        causes.append(f"support retained {r['support']['support_retained']}")
        if not speed:
            causes.append('speed windows 5 and 15 not both excluding 0')
        out['class'] = 'unresolved'
        out['causes'] = causes
    return out


def main():
    cal = sessions()
    res = {'freeze_sha256': sha(HERE / 'FREEZE_085.md'), 'race084_sha256': sha(ROOT / 'base/084/race084.py'),
           'xray_sha256': sha(ROOT / 'base/084/xray.py'), 'territories': {}}
    for inst, terr in TERR:
        d = frame(inst, terr, cal)
        r = analyse(d)
        r['verdict'] = classify(r)
        res['territories'][f'{inst} {terr}'] = r
        s, i, sup = r['outer_set_95'], r['identified'], r['support']
        print(f"\n{inst} {terr}: strata {sup['strata_kept']}/{sup['strata_total']}  films {sup['films_used']}/{sup['films_total']} "
              f"(support {sup['support_retained']}), ties {sup['ties_dropped_in_kept']}, days {r['days']}")
        print(f"   imbalance high-low (diff, low, high): {r['imbalance_high_minus_low']}")
        for name in ('dB', 'dR5', 'dR15', 'dR60'):
            print(f"   {name:<5} identified [{i[name][0]:+.4f}; {i[name][1]:+.4f}]  outer 95% [{s[name][0]:+.4f}; {s[name][1]:+.4f}]"
                  f"  by block {r['outer_set_by_block'][name]}")
        print(f"   -> {r['verdict']}")
    (OUT / 'sigma085.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print('\nfreeze', res['freeze_sha256'])


if __name__ == '__main__':
    main()
