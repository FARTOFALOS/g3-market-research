#!/usr/bin/env python3
"""Цена ожидания от ИСХОДНОЙ T0-популяции и совместный профиль двух ветвей.

ЧЕТЫРЕ СЕМАНТИКИ, КОТОРЫЕ ЗДЕСЬ НЕ НАРУШАЮТСЯ
=============================================
1. `unknown` — adversely selected UNRESOLVED tail. Наблюдённый adverse до потери
   наблюдаемости — точная НИЖНЯЯ ГРАНИЦА, конечный исход неизвестен. После
   введения стопа часть станет definite loss (стоп достигнут до промежутка),
   остальные останутся unknown. Доказанным проигрышем эта ветвь не называется.
2. Ожидание не «улучшает качество». Установлено ровно одно: оно резко меняет
   СОСТАВ risk set — быстрые возвраты исчезают, gross у выживших растёт,
   certifiability падает. Это selection effect до доказательства дополнительной
   prefix-информации.
3. `ambiguous` — неоднозначность ПОРЯДКА adverse внутри касательной свечи, не
   неоднозначность исхода сделки. Исход становится неоднозначным только для
   стопа, лежащего в интервале (mae_certain, mae_bound].
4. Клетка «gross ниже расхода, один бар» — экономически немонетизируемая при
   принятой 1-point cost convention и TP на boundary, а не шум.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'work/081a'
COST = {'NQ': 1.00, 'ES': 1.00, 'YM': 4.00}
AGES = (0, 1, 2, 5, 15, 30, 60)
QS = [.10, .25, .50, .75, .90, .99]


def qd(s, r=2):
    return {f'p{int(x*100)}': round(float(s.quantile(x)), r) for x in QS} if len(s) else {}


def wait_table(q, f, inst):
    """Знаменатель — исходная T0-популяция. Каждый возраст: что от неё осталось."""
    N = len(f); nq = f.n_q.to_numpy(); c = COST[inst]
    rows = []
    for a in AGES:
        alive = nq > a
        s = q[q.age == a]
        assert len(s) == int(alive.sum())
        us = s[s.usable]
        cc = us[us.outcome == 0]; uu = us[us.outcome == 1]
        rows.append(dict(
            age=a,
            original_population=N,
            missed_before_age=int(N - alive.sum()),
            missed_share=round(float(1 - alive.mean()), 4),
            alive_decision_points=int(alive.sum()),
            alive_share=round(float(alive.mean()), 4),
            usable_reference=len(us),
            usable_share_of_original=round(len(us) / N, 4),
            usable_share_of_alive=round(len(us) / max(int(alive.sum()), 1), 4),
            remaining_gross=qd(us.gross),
            share_gross_gt_cost=round(float((us.gross > c).mean()), 4) if len(us) else None,
            future_status={'certified': len(cc), 'unknown': len(uu),
                           'certified_share': round(len(cc) / max(len(us), 1), 4)},
            adverse_certified_mae_bound=qd(cc.mae_bound),
            adverse_unknown_lower_bound=qd(uu.mae_bound),
            duration_certified=qd(cc.dur, 0)))
    return rows


def branches(q, f, inst, ages=(0, 5, 30)):
    """Совместный gross × adverse у будущих certified и unknown — раздельно.

    Это ещё НЕ сигнал: различие будущих ветвей становится сигналом только если
    оно видно по префиксу в момент q. Проверка отдельная.
    """
    c = COST[inst]
    gb = [0, c, 2 * c, 4 * c, np.inf]; gl = ['<=1c', '1-2c', '2-4c', '>4c']
    mb = [-1e-9, 0.0, c, 2 * c, 4 * c, np.inf]; ml = ['0', '(0,1c]', '1-2c', '2-4c', '>4c']
    out = {}
    for a in ages:
        s = q[(q.age == a) & q.usable]
        d = {}
        for tag, sel in (('certified', s[s.outcome == 0]), ('unknown', s[s.outcome == 1])):
            if not len(sel):
                continue
            gi = pd.cut(sel.gross, gb, labels=gl)
            key = sel.mae_bound if tag == 'unknown' else sel.mae_certain
            mi = pd.cut(key, mb, labels=ml)
            ct = pd.crosstab(gi, mi, normalize=True)
            d[tag] = {'n': len(sel),
                      'adverse_field': 'mae_observed_until_loss (нижняя граница)'
                      if tag == 'unknown' else 'mae_certain (точный, до касательной)',
                      'cells': {str(i): {str(k): round(float(v), 4) for k, v in r.items()}
                                for i, r in ct.iterrows()},
                      'gross': qd(sel.gross), 'adverse': qd(key)}
        out[f'age_{a}'] = d
    return out


def stop_ambiguity(q, inst, stops=(0.5, 1.0, 2.0, 4.0, 8.0)):
    """Исход неоднозначен только для стопа внутри (mae_certain, mae_bound]."""
    s = q[(q.age == 0) & q.usable & (q.outcome == 0)]
    out = {}
    for st in stops:
        hit_certain = (s.mae_certain >= st).mean()
        hit_bound = (s.mae_bound >= st).mean()
        out[str(st)] = {
            'stop_definitely_hit_before_touch_candle': round(float(hit_certain), 4),
            'stop_hit_only_inside_touch_candle_order_unknown':
                round(float(hit_bound - hit_certain), 4),
            'stop_definitely_not_hit': round(float(1 - hit_bound), 4)}
    return out


def main(inst='NQ', terr='discovery'):
    q = pd.read_parquet(OUT / f'paths/q_{inst}_{terr}.parquet')
    f = pd.read_parquet(OUT / f'paths/films_{inst}_{terr}.parquet')
    res = {'instrument': inst, 'territory': terr, 'cost_points': COST[inst],
           'price_of_waiting_from_original_population': wait_table(q, f, inst),
           'joint_gross_x_adverse_by_future_branch': branches(q, f, inst),
           'outcome_ambiguity_is_a_property_of_the_stop': stop_ambiguity(q, inst)}
    p = OUT / f'gate_a_wait_{inst}_{terr}.json'
    p.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print(p, 'written')
    return res


if __name__ == '__main__':
    main(*(sys.argv[1:] or ['NQ', 'discovery']))
