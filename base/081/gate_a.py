#!/usr/bin/env python3
"""GATE A — пространство выбора, без единого выигрыша.

ЧЕГО ЗДЕСЬ НЕТ
==============
Здесь не считается ни одна сделка. `contact_certified` НЕ является выигрышем
позиции: торговый TP обязан произойти раньше заранее известного policy exit,
иначе исход определяет management rule, а не рынок. Защитное семейство ещё не
объявлено — по AGENTS.md его числа снимаются с наложенных фильмов, а не
назначаются до них. Поэтому GATE A только показывает совместный профиль
возможностей и неблагоприятных продолжений.

ЗНАМЕНАТЕЛИ
===========
Основной знаменатель — ВСЕ доступные T0-возможности. Будущие статусы
(`contact_certified` / `unknown`) раскладываются рядом и никогда не входят в
условие отбора популяции.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'work/081a'
COST = {'NQ': 1.00, 'ES': 1.00, 'YM': 4.00}          # DECLARE_081A §2, сценарий 057
TICK = {'NQ': 0.25, 'ES': 0.25, 'YM': 1.00}
QS = [.10, .25, .50, .75, .90, .99]


def qd(s, r=2):
    return {f'p{int(x*100)}': round(float(s.quantile(x)), r) for x in QS} if len(s) else {}


def population(f, t0, inst):
    """Разложение исходной популяции. Ни одно условие не смотрит в будущее."""
    n = len(f)
    ref = t0.ref_ok.to_numpy(); us = t0.usable.to_numpy()
    d = dict(films=n,
             ref_available=int(ref.sum()), ref_share=round(float(ref.mean()), 4),
             usable=int(us.sum()), usable_share=round(float(us.mean()), 4),
             ref_ok_but_not_outside=int((ref & ~us).sum()))
    # будущие статусы — рядом, не в знаменателе
    st = f.film1_status.to_numpy()
    d['status_all'] = {k: int(v) for k, v in pd.Series(st).value_counts().items()}
    d['status_among_usable'] = {k: int(v) for k, v in
                                pd.Series(st[us]).value_counts().items()}
    d['status_share_among_usable'] = {k: round(float(v), 4) for k, v in
                                      pd.Series(st[us]).value_counts(normalize=True).items()}
    return d


def t0_profile(t0, inst):
    """Профиль первого решения на ВСЕХ пригодных T0, без отбора по исходу."""
    u = t0[t0.usable]
    c = COST[inst]; tk = TICK[inst]
    out = {'n_usable': len(u),
           'gross_points': qd(u.gross), 'gross_ticks': qd(u.gross / tk, 1),
           'd_close_points': qd(u.d_close),
           'share_gross_gt_cost': round(float((u.gross > c).mean()), 4),
           'share_gross_gt_2cost': round(float((u.gross > 2 * c).mean()), 4),
           'share_gross_le_1tick': round(float((u.gross <= tk).mean()), 4)}
    # тот же порог, разложенный по будущему статусу — рядом, не вместо
    out['share_gross_gt_cost_by_outcome'] = {
        'certified': round(float((u.gross[u.outcome == 0] > c).mean()), 4),
        'unknown': round(float((u.gross[u.outcome == 1] > c).mean()), 4)}
    # execution slip: prefix против execution layer
    slip = u.d_close - u.gross
    out['exec_slip_points'] = qd(slip, 3)
    out['exec_slip_ge_1tick'] = round(float((slip.abs() >= tk - 1e-6).mean()), 4)
    out['exec_slip_toward_target'] = round(float((slip < 0).mean()), 4)
    # adverse и длительность публикуются РАЗДЕЛЬНО по статусу (DECLARE §4)
    for tag, sel in (('certified', u[u.outcome == 0]), ('unknown', u[u.outcome == 1])):
        key = 'completed' if tag == 'certified' else 'open_at_loss_of_observability'
        out[f'path_{key}'] = {
            'n': len(sel),
            'mae_certain': qd(sel.mae_certain), 'mae_bound': qd(sel.mae_bound),
            'ambiguous_share': round(float((sel.mae_bound > sel.mae_certain + 1e-9).mean()), 4),
            'dur_bars': qd(sel.dur, 0)}
    if 'path_open_at_loss_of_observability' in out:
        out['path_open_at_loss_of_observability']['note'] = (
            'mae_observed_until_loss — нижняя граница MAE ещё открытой сделки, не её MAE')
    return out


def joint(t0, inst):
    """Совместное распределение (gross, adverse) по сделкам, не медианы порознь."""
    u = t0[(t0.usable) & (t0.outcome == 0)]
    c = COST[inst]
    r_c = u.mae_certain / u.gross
    r_b = u.mae_bound / u.gross
    out = {'n': len(u),
           'note': 'только завершённые пути; у незавершённых MAE неполон',
           'ratio_mae_bound_over_gross': qd(r_b, 3),
           'ratio_mae_certain_over_gross': qd(r_c, 3),
           'share_mae_bound_ge_gross': round(float((u.mae_bound >= u.gross).mean()), 4),
           'share_mae_certain_ge_gross': round(float((u.mae_certain >= u.gross).mean()), 4),
           'share_mae_certain_zero': round(float((u.mae_certain <= 1e-9).mean()), 4)}
    # клетки: строка — оставшийся gross, столбец — adverse, в тех же пунктах
    gb = [0, c, 2 * c, 4 * c, np.inf]
    mb = [-1e-9, 0.0, c, 2 * c, 4 * c, np.inf]
    gi = pd.cut(u.gross, gb, labels=['<=1c', '1-2c', '2-4c', '>4c'])
    mi = pd.cut(u.mae_certain, mb, labels=['0', '(0,1c]', '1-2c', '2-4c', '>4c'])
    ct = pd.crosstab(gi, mi)
    out['cells_gross_x_mae_certain'] = {str(i): {str(k): int(v) for k, v in r.items()}
                                        for i, r in ct.iterrows()}
    out['cells_share'] = {str(i): {str(k): round(float(v), 4) for k, v in r.items()}
                          for i, r in (ct / ct.values.sum()).iterrows()}
    return out


def wait_price(q, f, inst):
    """Цена ожидания на ИСХОДНОЙ популяции возможностей, а не на доживших."""
    N = len(f); nq = f.n_q.to_numpy(); c = COST[inst]
    rows = []
    for a in (0, 1, 2, 3, 5, 8, 15, 30, 60, 120, 240):
        alive = int((nq > a).sum())
        s = q[(q.age >= a) & q.usable]
        first = s.groupby('film', sort=False).head(1)
        r = dict(age=a,
                 alive=alive, alive_share=round(alive / N, 4),
                 contacted_before=N - alive,
                 entered=len(first), entered_share_of_original=round(len(first) / N, 4),
                 status_share={k: round(float(v), 4) for k, v in
                               first.outcome.map({0: 'certified', 1: 'unknown'})
                               .value_counts(normalize=True).items()},
                 gross=qd(first.gross), share_gross_gt_cost=round(float((first.gross > c).mean()), 4)
                 if len(first) else None)
        comp = first[first.outcome == 0]
        r['mae_bound_completed'] = qd(comp.mae_bound)
        r['dur_completed'] = qd(comp.dur, 0)
        rows.append(r)
    return rows


def epoch_drift(q, f, inst):
    """Меняется ли распределение ДВИЖЕНИЯ внутри discovery-территории."""
    t0 = q[(q.age == 0) & q.usable].reset_index(drop=True)
    yr = pd.to_datetime(f.t0_ts_ns, utc=True).dt.year.to_numpy()[t0.film.to_numpy()]
    c = COST[inst]
    g = pd.DataFrame({'year': yr, 'gross': t0.gross.to_numpy(),
                      'zw': f.zone_width.to_numpy()[t0.film.to_numpy()],
                      'px': f.t0_close.to_numpy()[t0.film.to_numpy()]})
    agg = g.groupby('year').agg(n=('gross', 'size'), px_med=('px', 'median'),
                                zw_med=('zw', 'median'), gross_med=('gross', 'median'),
                                gross_p90=('gross', lambda s: s.quantile(.9)))
    agg['share_gt_cost'] = g.groupby('year').gross.apply(lambda s: (s > c).mean())
    return {str(k): {kk: round(float(vv), 4) for kk, vv in v.items()}
            for k, v in agg.iterrows()}


def main(inst='NQ', terr='discovery'):
    q = pd.read_parquet(OUT / f'paths/q_{inst}_{terr}.parquet')
    f = pd.read_parquet(OUT / f'paths/films_{inst}_{terr}.parquet')
    t0 = q[q.age == 0].reset_index(drop=True)
    res = {'instrument': inst, 'territory': terr,
           'cost_convention_points_round_trip': COST[inst],
           'cost_status': 'scenario из 057, не measured execution cost',
           'population': population(f, t0, inst),
           't0_first_decision': t0_profile(t0, inst),
           'joint_gross_vs_adverse': joint(t0, inst),
           'price_of_waiting': wait_price(q, f, inst),
           'movement_distribution_by_year_within_discovery': epoch_drift(q, f, inst)}
    p = OUT / f'gate_a_{inst}_{terr}.json'
    p.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'{p} written')
    return res


if __name__ == '__main__':
    main(*(sys.argv[1:] or ['NQ', 'discovery']))
