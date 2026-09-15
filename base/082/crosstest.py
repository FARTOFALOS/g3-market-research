#!/usr/bin/env python3
"""Решающий cross-test с диагностикой «объяснение против потери контраста».

К каждому Δ печатается не только разделение ДЕЛЯЩЕГО признака, но и то,
насколько при этом разошёлся ВТОРОЙ признак — утечка. Три случая различаются
только так:

    объяснение      Δ отклика упал, своё разделение сохранено, утечка мала
    потеря контраста своё разделение упало вместе с Δ
    ложный контроль утечка велика — «фиксация» второго признака не состоялась

Дополнительно результат проверяется на числе корзин фиксации nq ∈ {2, 4, 8}:
вывод, живущий ровно на одном nq, выводом не является.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'work/082'
sys.path.insert(0, str(HERE))
from contrast import (load, MIN_SIDE, NBOOT, BLOCKS, _day_multipliers,   # noqa: E402
                      AGES, HS)

RNG = np.random.default_rng(8209151)


def cross(d, feat, resp, other, refine=None, nq=4, key='base_key'):
    d = d[d.state_readable & d[resp].notna() & d[feat].notna() & d[other].notna()].copy()
    if refine is not None:
        d['_qq'] = (d.groupby(key, observed=True)[refine]
                    .transform(lambda s: pd.qcut(s, nq, labels=False, duplicates='drop')))
        d = d[d._qq.notna()]
        d['_key'] = d[key] + '#' + d._qq.astype(int).astype(str)
    else:
        d['_key'] = d[key]
    if not len(d):
        return None
    sid, keys = pd.factorize(d._key)
    nk = len(keys)
    y = d[resp].to_numpy(np.float64)
    x = d[feat].to_numpy(np.float64)
    o = d[other].to_numpy(np.float64)
    med = pd.Series(x).groupby(sid).transform('median').to_numpy()
    lo_m, hi_m = x < med, x > med
    n_lo = np.bincount(sid[lo_m], minlength=nk)
    n_hi = np.bincount(sid[hi_m], minlength=nk)
    keep = (n_lo >= MIN_SIDE) & (n_hi >= MIN_SIDE)
    if not keep.any():
        return None
    w = np.minimum(n_lo, n_hi).astype(float) * keep
    w = w / w.sum()

    def wm(mask, v):
        s = np.bincount(sid[mask], v[mask], nk)
        c = np.bincount(sid[mask], minlength=nk).astype(float)
        return float(w @ np.divide(s, c, out=np.zeros(nk), where=c > 0))

    def sep_sd(v):
        sd = pd.Series(v).groupby(sid).transform('std').to_numpy()
        g = np.isfinite(sd) & (sd > 0)
        z = np.zeros_like(v); z[g] = v[g] / sd[g]
        return wm(hi_m & g, z) - wm(lo_m & g, z)

    days = d.t0_day.to_numpy()
    uday, dcode = np.unique(days, return_inverse=True)
    cis = {}
    for blk in BLOCKS:
        M = _day_multipliers(uday.size, blk, NBOOT, RNG)
        agg = {}
        for tag, mask in (('lo', lo_m), ('hi', hi_m)):
            S = sparse.csr_matrix((y[mask], (sid[mask], dcode[mask])), (nk, uday.size))
            C = sparse.csr_matrix((np.ones(int(mask.sum())), (sid[mask], dcode[mask])),
                                  (nk, uday.size))
            agg[tag] = (S @ M, C @ M)
        (slo, clo), (shi, chi) = agg['lo'], agg['hi']
        ok = keep[:, None] & (clo > 0) & (chi > 0)
        mlo = np.divide(slo, clo, out=np.zeros_like(slo), where=clo > 0)
        mhi = np.divide(shi, chi, out=np.zeros_like(shi), where=chi > 0)
        ww = w[:, None] * ok
        tot = ww.sum(0)
        v = np.where(tot > 0, (ww * (mhi - mlo)).sum(0) / np.maximum(tot, 1e-12), np.nan)
        v = v[np.isfinite(v)]
        cis[blk] = (float(np.quantile(v, .025)), float(np.quantile(v, .975)))
    widest = max(cis.values(), key=lambda c: c[1] - c[0])
    own, leak = sep_sd(x), sep_sd(o)
    dl = wm(hi_m, y) - wm(lo_m, y)
    return {'delta': dl, 'ci': widest, 'own_sep_sd': own, 'leak_sep_sd': leak,
            'per_own_sd': dl / own if own else np.nan,
            'n': int((lo_m | hi_m)[keep[sid]].sum()), 'strata': int(keep.sum()),
            'support': float((lo_m | hi_m)[keep[sid]].sum() / len(d)),
            'mean_low': wm(lo_m, y), 'mean_high': wm(hi_m, y)}


def fmt(r):
    if r is None:
        return f"{'нет опоры':>72}"
    return (f"Δ {r['delta']:+.4f} CI[{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}] "
            f"на sd {r['per_own_sd']:+.4f} | своё {r['own_sep_sd']:+.2f}sd "
            f"утечка {r['leak_sep_sd']:+.2f}sd | n {r['n']:>6} страт {r['strata']:>3}")


def main(inst='NQ', terr='discovery'):
    res = {}
    for a in AGES:
        d0 = load(inst, terr, a)
        for h in HS:
            s = d0[d0[f'contact_{h}'] >= 0].copy()
            s['rho'] = s[f'away_{h}'] / s.d_close
            s['appr'] = s[f'toward_{h}'] / s.d_close
            for resp in (f'contact_{h}', 'rho', 'appr'):
                print(f"\n=== возраст {a}, окно h={h}, отклик {resp}")
                m0 = cross(s, 'retraced_fraction', resp, 'prefix_sigma')
                m1 = cross(s, 'prefix_sigma', resp, 'retraced_fraction')
                print(f"   {'retrace, без контроля':>34}: {fmt(m0)}")
                print(f"   {'sigma,   без контроля':>34}: {fmt(m1)}")
                res[f'a{a}_h{h}_{resp}_retrace_raw'] = m0
                res[f'a{a}_h{h}_{resp}_sigma_raw'] = m1
                for nq in (2, 4, 8):
                    r0 = cross(s, 'retraced_fraction', resp, 'prefix_sigma',
                               refine='prefix_sigma', nq=nq)
                    r1 = cross(s, 'prefix_sigma', resp, 'retraced_fraction',
                               refine='retraced_fraction', nq=nq)
                    print(f"   {f'retrace | sigma фикс nq={nq}':>34}: {fmt(r0)}")
                    print(f"   {f'sigma   | retrace фикс nq={nq}':>34}: {fmt(r1)}")
                    res[f'a{a}_h{h}_{resp}_retrace_sigmafix{nq}'] = r0
                    res[f'a{a}_h{h}_{resp}_sigma_retracefix{nq}'] = r1
    p = OUT / f'crosstest_{inst}_{terr}.json'
    p.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float),
                 encoding='utf-8')
    print('\n', p, 'written')


if __name__ == '__main__':
    main(*(sys.argv[1:] or ['NQ', 'discovery']))
