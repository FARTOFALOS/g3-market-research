#!/usr/bin/env python3
"""Три политики на общей исходной территории: A (T0), B (простое состояние), C (префикс).

C строится ТОЛЬКО поверх замороженного B: те же D и k, добавлен один порог
`retraced_fraction >= θ` и минимальный возраст 1 (на T0 префикса нет). Это и
есть вопрос «добавляет ли дорога сверх простого положения», заданный как
политика, а не как признак.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from risk_family import score_stop, COST, WIN, LOSS, AMBIG, UNKNOWN     # noqa: E402
from family2 import load                                               # noqa: E402

OUT = ROOT / 'work/081a'
THETAS = (0.25, 0.50, 0.75)


def run_policy(q, f, inst, age_min, d_mult, k, theta=None, cost_mult=1.0):
    c = COST[inst]; N = len(f); nq = f.n_q.to_numpy()
    cond = (q.age.to_numpy() >= age_min) & (q.d_close.to_numpy() > d_mult * c)
    if theta is not None:
        cond &= (q.retraced_fraction.to_numpy() >= theta)
    trig = q[cond].groupby('film', sort=False).head(1)
    missed = int((nq <= age_min).sum())
    alive = int((nq > age_min).sum())
    expired = alive - len(trig)
    ok = trig.usable.to_numpy(); cancelled = int((~ok).sum()); ent = trig[ok]
    kind, lo, hi = score_stop(ent, k, inst, cost_mult)
    g = ent.gross.to_numpy(np.float64); ma = ent.mae_bound.to_numpy(np.float64)
    cst = c * cost_mult
    up = np.where(np.isnan(hi), g - cst, hi); dn = np.where(np.isnan(lo), -ma - cst, lo)
    det = np.isin(kind, (WIN, LOSS))
    n = len(ent)
    return {'policy': {'age_min': age_min, 'd_mult': d_mult, 'k': k, 'theta': theta},
            'stream': {'original': N, 'missed_before_eligible': missed,
                       'expired_without_trigger': expired,
                       'execution_cancelled': cancelled, 'executed': n},
            'executed_share': round(n / N, 4),
            'unknown_share': round(float((kind == UNKNOWN).mean()), 4) if n else None,
            'win': round(float((kind == WIN).mean()), 4) if n else None,
            'loss': round(float((kind == LOSS).mean()), 4) if n else None,
            'lower': round(float(dn.mean()), 4) if n else None,
            'upper': round(float(up.mean()), 4) if n else None,
            'determined': round(float(np.nanmean(lo[det])), 4) if det.any() else None,
            'gross_determined': None}


def with_gross(q, f, inst, **kw):
    r = run_policy(q, f, inst, cost_mult=1.0, **kw)
    r['gross_determined'] = run_policy(q, f, inst, cost_mult=0.0, **kw)['determined']
    return r


def main(inst='NQ', terr='discovery'):
    q, f = load(inst, terr)
    res = {}
    res['A_T0_early'] = with_gross(q, f, inst, age_min=0, d_mult=0.0, k=1.0)
    res['B_simple_frozen'] = with_gross(q, f, inst, age_min=0, d_mult=4.0, k=1.0)
    res['B_prime_age1'] = with_gross(q, f, inst, age_min=1, d_mult=4.0, k=1.0)
    for th in THETAS:
        res[f'C_prefix_theta{th}'] = with_gross(q, f, inst, age_min=1, d_mult=4.0,
                                                k=1.0, theta=th)
    hdr = f"{'политика':>22}{'испол':>8}{'доля':>8}{'unk':>7}{'win':>7}{'loss':>7}{'валовое':>9}{'после':>8}{'нижн':>8}{'верхн':>8}"
    print(hdr)
    for kk, v in res.items():
        print(f"{kk:>22}{v['stream']['executed']:>8}{v['executed_share']:>8.4f}"
              f"{v['unknown_share']:>7.3f}{v['win']:>7.3f}{v['loss']:>7.3f}"
              f"{v['gross_determined']:>9.4f}{v['determined']:>8.4f}{v['lower']:>8.3f}{v['upper']:>8.3f}")
    (OUT / f'candidate_{inst}_{terr}.json').write_text(
        json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    return res


if __name__ == '__main__':
    main(*(sys.argv[1:] or ['NQ', 'discovery']))
