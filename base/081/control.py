#!/usr/bin/env python3
"""ЦЕНТРАЛЬНЫЙ КОНТРОЛЬ: «где цена сейчас» против «как она сюда пришла».

Грубые корзины `d_close` оказались недостаточным контролем: внутри корзины
группа с большим `retraced_fraction` систематически стоит БЛИЖЕ к границе
(`2-4c|north`: 3,25 против 2,75 пункта), и состав по ТФ различается до
total variation 0,41. Поэтому здесь контроль доводится до точного совпадения.

ТОЧНОЕ СОВПАДЕНИЕ ТЕКУЩЕГО ПОЛОЖЕНИЯ
====================================
Ключ страты — `(возраст, d_close в тиках, сторона, ТФ-полоса)`. Внутри ключа обе
группы стоят на ОДНОМ расстоянии от собственной границы, на одной стороне, в
одной полосе нативного ТФ и одного возраста. Отличается только дорога, которой
цена сюда пришла.

Страта без опоры в обеих группах выбрасывается целиком, и потерянное население
публикуется. Координаты не добавляются до исчезновения групп: это последний
уровень контроля в этом цикле.
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
from family2 import load, TF_EDGES, TF_LAB                               # noqa: E402

OUT = ROOT / 'work/081a'
TICK = {'NQ': 0.25, 'ES': 0.25, 'YM': 1.0}
MIN_SIDE = 30                # опора в КАЖДОЙ группе точной страты, объявлено до счёта
AGES = (5, 15, 60)


def exact_strata(q, f, a, inst):
    s = q[(q.age == a) & q.usable].copy()
    fi = s.film.to_numpy()
    s['side'] = f.side.to_numpy()[fi]
    s['tfb'] = pd.cut(f.tf_minutes.to_numpy()[fi], TF_EDGES, labels=TF_LAB).astype(str)
    s['dct'] = np.round(s.d_close.to_numpy() / TICK[inst]).astype(int)
    s['key'] = s.dct.astype(str) + '|' + s.side + '|' + s.tfb
    return s


def compare(s, inst, k=2.0, feat='retraced_fraction'):
    c = COST[inst]
    kept, dropped_rows, dropped_keys = [], 0, 0
    for key, sub in s.groupby('key', observed=True):
        med = sub[feat].median()
        lo, hi = sub[sub[feat] < med], sub[sub[feat] > med]
        if len(lo) < MIN_SIDE or len(hi) < MIN_SIDE:
            dropped_rows += len(sub); dropped_keys += 1; continue
        out = {'key': key, 'n_low': len(lo), 'n_high': len(hi), 'w': min(len(lo), len(hi))}
        for tag, g in (('low', lo), ('high', hi)):
            kind, klo, khi = score_stop(g, k, inst)
            gr = g.gross.to_numpy(np.float64); ma = g.mae_bound.to_numpy(np.float64)
            cert = g.outcome.to_numpy() == 0
            out[tag] = {
                'retraced_p50': round(float(g[feat].median()), 4),
                'd_close_p50': round(float(g.d_close.median()), 4),
                'gross_p50': round(float(g.gross.median()), 3),
                'certified': round(float(cert.mean()), 4),
                'adverse_cert_p50': round(float(g.mae_bound[cert].median()), 3) if cert.any() else None,
                'adverse_cert_p90': round(float(g.mae_bound[cert].quantile(.9)), 3) if cert.any() else None,
                'dur_cert_p50': round(float(g.dur[cert].median()), 1) if cert.any() else None,
                'win': round(float((kind == WIN).mean()), 4),
                'loss': round(float((kind == LOSS).mean()), 4),
                'unknown': round(float((kind == UNKNOWN).mean()), 4),
                'ambiguous': round(float((kind == AMBIG).mean()), 4),
                'lower': round(float(np.where(np.isnan(klo), -ma - c, klo).mean()), 4),
                'upper': round(float(np.where(np.isnan(khi), gr - c, khi).mean()), 4)}
        kept.append(out)
    if not kept:
        return {'strata_kept': 0, 'rows_dropped': dropped_rows, 'note': 'общей поддержки нет'}
    w = np.array([x['w'] for x in kept], float); w /= w.sum()
    fields = ('certified', 'win', 'loss', 'unknown', 'ambiguous', 'lower', 'upper',
              'gross_p50', 'd_close_p50', 'retraced_p50')
    agg = {}
    for tag in ('low', 'high'):
        agg[tag] = {fl: round(float(np.dot(w, [x[tag][fl] for x in kept])), 4) for fl in fields}
        for fl in ('adverse_cert_p50', 'adverse_cert_p90', 'dur_cert_p50'):
            v = [(wi, x[tag][fl]) for wi, x in zip(w, kept) if x[tag][fl] is not None]
            if v:
                ww = np.array([a for a, _ in v]); ww /= ww.sum()
                agg[tag][fl] = round(float(np.dot(ww, [b for _, b in v])), 4)
    delta = {fl: round(agg['high'][fl] - agg['low'][fl], 4)
             for fl in agg['low'] if fl in agg['high']}
    return {'strata_kept': len(kept), 'rows_used': int(sum(x['n_low'] + x['n_high'] for x in kept)),
            'rows_dropped': dropped_rows, 'keys_dropped': dropped_keys,
            'support_retained': round(sum(x['n_low'] + x['n_high'] for x in kept) /
                                      max(len(s), 1), 4),
            'low': agg['low'], 'high': agg['high'], 'delta_high_minus_low': delta,
            'per_stratum': sorted(kept, key=lambda x: -x['w'])[:15]}


if __name__ == '__main__':
    inst = sys.argv[1] if len(sys.argv) > 1 else 'NQ'
    terr = sys.argv[2] if len(sys.argv) > 2 else 'discovery'
    q, f = load(inst, terr)
    res = {}
    for a in AGES:
        s = exact_strata(q, f, a, inst)
        r = compare(s, inst)
        res[f'age_{a}'] = r
        if r.get('strata_kept'):
            d = r['delta_high_minus_low']
            print(f"age {a:>3}: страт {r['strata_kept']:>3} строк {r['rows_used']:>6} "
                  f"опора {r['support_retained']:.3f} | Δd_close {d['d_close_p50']:+.3f} "
                  f"Δcert {d['certified']:+.4f} Δloss {d['loss']:+.4f} Δwin {d['win']:+.4f} "
                  f"Δadv50 {d.get('adverse_cert_p50', float('nan')):+.3f} "
                  f"Δlower {d['lower']:+.4f} Δupper {d['upper']:+.4f}")
        else:
            print(f"age {a:>3}: {r.get('note')}, выброшено {r['rows_dropped']}")
    p = OUT / f'control_{inst}_{terr}.json'
    p.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print(p, 'written')
