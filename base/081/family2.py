#!/usr/bin/env python3
"""СЕМЬЯ 2 — добавляет ли дорога от T0 до q сверх текущего положения.

Основное чтение заморожено в `FREEZE_FAMILY2.md` до первого взгляда на исход:

    retraced_fraction = 1 − d_close / max_departure_distance     ∈ [0, 1]
    0 = стоит на максимальном удалении, 1 = вернулась к границе почти полностью

ПОРЯДОК, ОБЪЯВЛЕННЫЙ ЗАРАНЕЕ
============================
1. общая поддержка на `(age, d_close, side)`; ТФ — сначала проверка перекоса
   состава, а не дробление ячеек;
2. СОВМЕСТНЫЙ будущий профиль (adverse, gross, время, certified/unknown,
   ambiguity), а не бинарная торговая метка;
3. один q на фильм: срез на объявленном возрасте. q-взвешенное — картография.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from risk_family import score_stop, COST, WIN, LOSS, AMBIG, UNKNOWN      # noqa: E402

OUT = ROOT / 'work/081a'
MIN_CELL = 150                       # опора в КАЖДОЙ группе страты, объявлено до счёта
AGES = (5, 15, 60)
DC_EDGES = [0, 1, 2, 4, 8, np.inf]
DC_LAB = ['<=1c', '1-2c', '2-4c', '4-8c', '>8c']
TF_EDGES = [0, 2, 5, 15, 60, 240, 1441]
TF_LAB = ['1-2', '3-5', '6-15', '16-60', '61-240', '241-1440']


def load(inst='NQ', terr='discovery'):
    q = pd.read_parquet(OUT / f'paths/q_{inst}_{terr}.parquet',
                        columns=['film', 'age', 'outcome', 'usable', 'd_close',
                                 'gross', 'mae_certain', 'mae_bound', 'dur'])
    x = pd.read_parquet(OUT / f'paths/prefix_{inst}_{terr}.parquet',
                        columns=['run_max'])
    f = pd.read_parquet(OUT / f'paths/films_{inst}_{terr}.parquet')
    q['retraced_fraction'] = 1.0 - q.d_close.to_numpy() / x.run_max.to_numpy()
    return q, f


def slice_age(q, f, a, inst):
    s = q[(q.age == a) & q.usable].copy()
    fi = s.film.to_numpy()
    s['side'] = f.side.to_numpy()[fi]
    s['tf'] = f.tf_minutes.to_numpy()[fi]
    c = COST[inst]
    s['dc_bin'] = pd.cut(s.d_close / c, DC_EDGES, labels=DC_LAB).astype(str)
    s['stratum'] = s.dc_bin + '|' + s.side
    return s


def tercile_groups(s, feat='retraced_fraction'):
    """Границы берутся ВНУТРИ страты, чтобы не переносить состав между ними."""
    qs = s.groupby('stratum', observed=True)[feat].quantile([1/3, 2/3]).unstack()
    qs.columns = ['lo', 'hi']
    m = s.stratum.map(qs.lo), s.stratum.map(qs.hi)
    g = pd.Series('mid', index=s.index, dtype=object)
    g[s[feat] <= m[0].to_numpy()] = 'low'
    g[s[feat] >= m[1].to_numpy()] = 'high'
    return g


def profile(s, inst, k=2.0):
    """СОВМЕСТНЫЙ будущий профиль. Ветви разделены, ничего не усредняется через них."""
    c = COST[inst]
    kind, lo, hi = score_stop(s, k, inst)
    cert = s.outcome.to_numpy() == 0
    out = {'n': int(len(s)),
           'certified_share': round(float(cert.mean()), 4),
           'remaining_gross_p50': round(float(s.gross.median()), 3),
           'retraced_fraction_p50': round(float(s.retraced_fraction.median()), 4)}
    for tag, m in (('certified', cert), ('unresolved', ~cert)):
        sub = s[m]
        if not len(sub):
            continue
        out[f'adverse_after_q_{tag}'] = {
            'p50': round(float(sub.mae_bound.median()), 3),
            'p75': round(float(sub.mae_bound.quantile(.75)), 3),
            'p90': round(float(sub.mae_bound.quantile(.9)), 3)}
        if tag == 'certified':
            out['time_to_contact_bars'] = {
                'p50': round(float(sub.dur.median()), 1),
                'p90': round(float(sub.dur.quantile(.9)), 1)}
    out['ambiguity_share_at_k'] = round(float((kind == AMBIG).mean()), 4)
    out['policy_outcome'] = {'win': round(float((kind == WIN).mean()), 4),
                             'loss': round(float((kind == LOSS).mean()), 4),
                             'ambiguous': round(float((kind == AMBIG).mean()), 4),
                             'unknown': round(float((kind == UNKNOWN).mean()), 4)}
    g = s.gross.to_numpy(np.float64); madv = s.mae_bound.to_numpy(np.float64)
    out['bounded_points'] = {
        'lower': round(float(np.where(np.isnan(lo), -madv - c, lo).mean()), 4),
        'upper': round(float(np.where(np.isnan(hi), g - c, hi).mean()), 4)}
    return out


def tf_balance(a_, b_):
    """Перекос состава по ТФ между группами — проверяется, а не режется заранее."""
    fa = pd.cut(a_.tf, TF_EDGES, labels=TF_LAB).value_counts(normalize=True)
    fb = pd.cut(b_.tf, TF_EDGES, labels=TF_LAB).value_counts(normalize=True)
    d = (fb - fa).abs()
    return {'low': {str(i): round(float(v), 4) for i, v in fa.items()},
            'high': {str(i): round(float(v), 4) for i, v in fb.items()},
            'total_variation': round(float(d.sum() / 2), 4)}


def run_age(q, f, a, inst, k=2.0):
    s = slice_age(q, f, a, inst)
    s['grp'] = tercile_groups(s)
    s = s[s.grp != 'mid']
    per, dropped = [], 0
    for st, sub in s.groupby('stratum', observed=True):
        a_, b_ = sub[sub.grp == 'low'], sub[sub.grp == 'high']
        if len(a_) < MIN_CELL or len(b_) < MIN_CELL:
            dropped += len(sub); continue
        per.append({'stratum': st, 'w': min(len(a_), len(b_)),
                    'low': profile(a_, inst, k), 'high': profile(b_, inst, k),
                    'tf_balance': tf_balance(a_, b_),
                    'd_close_p50': {'low': round(float(a_.d_close.median()), 3),
                                    'high': round(float(b_.d_close.median()), 3)},
                    'retraced_p50': {'low': round(float(a_.retraced_fraction.median()), 4),
                                     'high': round(float(b_.retraced_fraction.median()), 4)}})
    if not per:
        return {'age': a, 'strata_kept': 0, 'note': 'общей поддержки нет'}
    w = np.array([p['w'] for p in per], float); w /= w.sum()

    def wm(side, path):
        v = []
        for p in per:
            x = p[side]
            for key in path:
                x = x[key]
            v.append(x)
        return round(float(np.dot(w, v)), 4)

    keys = [('certified_share',), ('remaining_gross_p50',),
            ('adverse_after_q_certified', 'p50'), ('adverse_after_q_certified', 'p90'),
            ('adverse_after_q_unresolved', 'p50'),
            ('time_to_contact_bars', 'p50'),
            ('policy_outcome', 'win'), ('policy_outcome', 'loss'),
            ('policy_outcome', 'unknown'), ('ambiguity_share_at_k',),
            ('bounded_points', 'lower'), ('bounded_points', 'upper')]
    low = {'.'.join(kk): wm('low', kk) for kk in keys}
    high = {'.'.join(kk): wm('high', kk) for kk in keys}
    return {'age': a, 'stop_k': k, 'strata_kept': len(per),
            'films_used': int(len(s)), 'rows_dropped_no_common_support': dropped,
            'weighting': 'равный вес страты, вес = min(n_low, n_high); один q на фильм',
            'tf_total_variation_max': round(max(p['tf_balance']['total_variation']
                                                for p in per), 4),
            'stratum_weighted': {'low_retraced': low, 'high_retraced': high,
                                 'delta_high_minus_low':
                                     {kk: round(high[kk] - low[kk], 4) for kk in low}},
            'per_stratum': per}


if __name__ == '__main__':
    inst = sys.argv[1] if len(sys.argv) > 1 else 'NQ'
    terr = sys.argv[2] if len(sys.argv) > 2 else 'discovery'
    q, f = load(inst, terr)
    res = {}
    for a in AGES:
        r = run_age(q, f, a, inst)
        res[f'age_{a}'] = r
        if r.get('strata_kept'):
            d = r['stratum_weighted']['delta_high_minus_low']
            print(f"age {a:>3}: страт {r['strata_kept']:>2} фильмов {r['films_used']:>7} "
                  f"| Δadv_cert_p50 {d['adverse_after_q_certified.p50']:+.3f} "
                  f"Δunresolved {d['policy_outcome.unknown']:+.4f} "
                  f"Δloss {d['policy_outcome.loss']:+.4f} Δlower {d['bounded_points.lower']:+.4f} "
                  f"| TF перекос max {r['tf_total_variation_max']:.3f}")
    p = OUT / f'family2_{inst}_{terr}.json'
    p.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print(p, 'written')
