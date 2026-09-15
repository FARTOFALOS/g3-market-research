#!/usr/bin/env python3
"""Политики 081A v2 на торговой территории. Пространство поиска не расширяется.

Сетка Семьи 1 та же: D ∈ {0, 1c, 2c, 4c} × age ∈ {0, 1, 2, 5, 15} × k ∈ {1,2,3}.
Три θ ∈ {0,25; 0,50; 0,75}. Критерий отбора тот же. Новых признаков нет.

РЕШЕНИЕ СУЩЕСТВУЕТ ТОЛЬКО ВНУТРИ ТОРГОВОГО ОКНА
===============================================
Минуты вне 03:00 ET → NY close решением не являются: трейдера там нет. Префикс
при этом читается по ВСЕЙ истории фильма от T0, включая внеоконные бары —
дорога цены не перестаёт существовать ночью. Правило лишь не может на неё
ответить до открытия окна.

ЧЕТЫРЕ ЧИСЛА ВМЕСТО ОДНОГО (по требованию ревизии)
==================================================
    determined              исход установлен (TP / стоп / выход по времени)
    unknown_share           доля неустановленных
    optimistic              неустановленным присвоен TP
    pessimistic_scenario    ОБЪЯВЛЕННЫЙ ПЕССИМИСТИЧНЫЙ СЦЕНАРИЙ, НЕ нижняя
                            граница P&L: неустановленной сделке приписан убыток
                            в размере уже наблюдённого хода против неё
`pessimistic_scenario` — критерий ранжирования discovery, объявленный заранее.
Экономической идентифицированной границей он не является и так не подаётся.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from trading import (sessions, bar_session_map, COST, PV, TICK,
                     WIN, LOSS, AMBIG, TIME_EXIT, UNKNOWN, NO_TRADE)   # noqa: E402

OUT = ROOT / 'work/081a'
D_MULT = (0.0, 1.0, 2.0, 4.0)
AGES = (0, 1, 2, 5, 15)
KS = (1.0, 2.0, 3.0)
THETAS = (0.25, 0.50, 0.75)
MIN_FREQ = 0.02            # объявлено в FREEZE_081A §2
MAX_UNK = 0.20


def load(inst, terr, k):
    f = pd.read_parquet(OUT / f'paths/films_{inst}_{terr}.parquet')
    tr = pd.read_parquet(OUT / f'paths/trade_{inst}_{terr}_k{k:g}.parquet')
    px = pd.read_parquet(OUT / f'paths/prefix_{inst}_{terr}.parquet', columns=['run_max'])
    q = pd.read_parquet(OUT / f'paths/q_{inst}_{terr}.parquet',
                        columns=['film', 'age', 'mae_bound'])
    tr['film'] = q.film.to_numpy(); tr['age'] = q.age.to_numpy()
    tr['mae_bound'] = q.mae_bound.to_numpy()
    tr['retraced_fraction'] = 1.0 - tr.d_close.to_numpy() / px.run_max.to_numpy()
    ts = np.load(ROOT / f'data/market/{inst}/close_ts_utc_ns.npy')
    st, cl = sessions(); sess, _ = bar_session_map(ts, st, cl)
    qpos = np.repeat(f.t0_spine_pos.to_numpy(), f.n_q.to_numpy()) + tr.age.to_numpy()
    tr['in_window'] = sess[qpos] >= 0
    return f, tr


def run(f, tr, inst, age_min, d_mult, k, theta=None, cost_mult=1.0):
    c = COST[inst]; N = len(f)
    cond = (tr.in_window.to_numpy() & (tr.age.to_numpy() >= age_min)
            & (tr.d_close.to_numpy() > d_mult * c))
    if theta is not None:
        cond &= (tr.retraced_fraction.to_numpy() >= theta)
    trig = tr[cond].groupby('film', sort=False).head(1)
    never = N - len(trig)
    cancelled = int((trig.kind.to_numpy() == NO_TRADE).sum())
    ent = trig[trig.kind.to_numpy() != NO_TRADE]
    n = len(ent)
    if not n:
        return None
    kind = ent.kind.to_numpy()
    cst = c * cost_mult
    lo = ent.lo.to_numpy(np.float64) - cst
    hi = ent.hi.to_numpy(np.float64) - cst
    det = kind != UNKNOWN
    unk = ~det
    mae = ent.mae_bound.to_numpy(np.float64)
    opt = np.where(unk, ent.gross.to_numpy(np.float64) - cst, hi)
    pess = np.where(unk, -mae - cst, lo)
    sh = lambda v: round(float((kind == v).mean()), 4)
    return {'policy': {'age_min': age_min, 'd_mult': d_mult, 'k': k, 'theta': theta},
            'stream': {'original': N, 'never_triggered': never,
                       'execution_cancelled': cancelled, 'executed': n},
            'freq': round(n / N, 4),
            'outcomes': {'win': sh(WIN), 'loss': sh(LOSS), 'ambiguous': sh(AMBIG),
                         'time_exit': sh(TIME_EXIT), 'unknown': sh(UNKNOWN)},
            'unknown_share': round(float(unk.mean()), 4),
            'determined_mean': round(float(lo[det].mean()), 4),
            'determined_mean_upper': round(float(hi[det].mean()), 4),
            'optimistic_mean': round(float(opt.mean()), 4),
            'pessimistic_scenario_mean': round(float(pess.mean()), 4),
            'gross_determined': round(float((lo[det] + cst).mean()), 4),
            'tp_unconfirmed_share': round(float(
                (ent.tp_conf.to_numpy()[np.isin(kind, (WIN, AMBIG))] == 0).mean()), 4)
            if np.isin(kind, (WIN, AMBIG)).any() else None,
            'dur_median': round(float(ent.dur.median()), 1)}


def sweep(inst='NQ', terr='discovery'):
    rows = []
    for k in KS:
        f, tr = load(inst, terr, k)
        for a in AGES:
            for d in D_MULT:
                r = run(f, tr, inst, a, d, k)
                if r:
                    rows.append(r)
        del tr
    return rows


if __name__ == '__main__':
    inst = sys.argv[1] if len(sys.argv) > 1 else 'NQ'
    terr = sys.argv[2] if len(sys.argv) > 2 else 'discovery'
    rows = sweep(inst, terr)
    p = OUT / f'v2_family1_{inst}_{terr}.json'
    p.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f"{'a':>3}{'D':>3}{'k':>3}{'испол':>8}{'частота':>9}{'unk':>7}{'win':>7}{'loss':>7}"
          f"{'time':>7}{'валовое':>9}{'опред':>8}{'оптим':>8}{'пессим':>8}")
    ok = [r for r in rows if r['freq'] >= MIN_FREQ and r['unknown_share'] <= MAX_UNK]
    for r in sorted(rows, key=lambda x: -x['pessimistic_scenario_mean'])[:14]:
        pp = r['policy']; o = r['outcomes']
        print(f"{pp['age_min']:>3}{pp['d_mult']:>3.0f}{pp['k']:>3.0f}{r['stream']['executed']:>8}"
              f"{r['freq']:>9.4f}{r['unknown_share']:>7.4f}{o['win']:>7.3f}{o['loss']:>7.3f}"
              f"{o['time_exit']:>7.3f}{r['gross_determined']:>9.4f}{r['determined_mean']:>8.4f}"
              f"{r['optimistic_mean']:>8.4f}{r['pessimistic_scenario_mean']:>8.4f}")
    print(f'\nдопустимых по частоте>={MIN_FREQ} и unknown<={MAX_UNK}: {len(ok)} из {len(rows)}')
    best = max(ok, key=lambda x: x['pessimistic_scenario_mean'])
    print('ЛУЧШАЯ по объявленному критерию:', best['policy'],
          '| пессим', best['pessimistic_scenario_mean'], '| опред', best['determined_mean'])
