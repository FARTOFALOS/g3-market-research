#!/usr/bin/env python3
"""СЕМЬЯ 1 — что можно решить, зная только простое текущее положение.

Объявленная заранее сетка, а не feature mining. Два параметра, оба в единицах,
которые уже существуют в постановке:

    D — порог prefix-расстояния `d_close` в кратных расходу круга: {1c, 2c, 4c}
        плюс 0 (без порога). Расход — единственная естественная единица: правило
        спрашивает «останется ли чем оплатить круг».
    a — возраст, с которого разрешён вход: {0, 1, 2, 5, 15}. Это ось WAIT.

Защитное семейство — объявленное в `risk_family.py` (R0, R1 k∈{1,2,3}).

ВОЗВРАТ К ИСХОДНОЙ ПОПУЛЯЦИИ
===========================
Любое правило WAIT считается от ИСХОДНОЙ T0-популяции: сколько возможностей уже
кончилось до его срабатывания, сколько осталось, сколько взято. Сравнение
«вход на T0 на всех» с «поздний вход только на доживших» запрещено.

ГРАНИЦЫ НЕРАЗРЕШЁННОЙ ВЕТВИ
===========================
`unknown` не ноль. Публикуются две границы:
    upper — вся неразрешённая ветвь в итоге доходит до TP
    lower — каждая теряет не меньше уже НАБЛЮДЁННОГО против неё хода
Нижняя граница не является предсказанием: это то, что уже видно на ленте.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from risk_family import score_stop, COST, K_FAMILY, WIN, LOSS, AMBIG, UNKNOWN  # noqa

OUT = ROOT / 'work/081a'
D_MULT = (0.0, 1.0, 2.0, 4.0)
AGES = (0, 1, 2, 5, 15)


def first_entry(q, a, D, c):
    """Первый q с age>=a и d_close>D. Правило смотрит только в префикс."""
    s = q[(q.age >= a) & (q.d_close > D * c)]
    s = s[s.usable]
    return s.groupby('film', sort=False).head(1)


def bounded(rows, k, inst, cost_mult=1.0):
    c = COST[inst] * cost_mult
    kind, lo, hi = score_stop(rows, k, inst, cost_mult)
    g = rows.gross.to_numpy(np.float64)
    madv = rows.mae_bound.to_numpy(np.float64)
    unk = kind == UNKNOWN
    up = np.where(np.isnan(hi), g - c, hi)             # unknown доходит до TP
    dn = np.where(np.isnan(lo), -madv - c, lo)         # unknown теряет наблюдённое
    return dict(n=int(len(rows)),
                share_unknown=round(float(unk.mean()), 4),
                upper=round(float(up.mean()), 4),
                lower=round(float(dn.mean()), 4),
                determined_only=round(float(np.nanmean(lo[np.isin(kind, (WIN, LOSS))])), 4)
                if np.isin(kind, (WIN, LOSS)).any() else None,
                win=round(float((kind == WIN).mean()), 4),
                loss=round(float((kind == LOSS).mean()), 4),
                ambiguous=round(float((kind == AMBIG).mean()), 4))


def sweep(inst='NQ', terr='discovery'):
    q = pd.read_parquet(OUT / f'paths/q_{inst}_{terr}.parquet')
    f = pd.read_parquet(OUT / f'paths/films_{inst}_{terr}.parquet')
    N = len(f); c = COST[inst]
    rows = []
    for a in AGES:
        for D in D_MULT:
            ent = first_entry(q, a, D, c)
            if not len(ent):
                continue
            taken = len(ent)
            missed = int((f.n_q.to_numpy() <= a).sum())
            r = {'age_min': a, 'd_close_gt_cost_mult': D,
                 'original_population': N,
                 'missed_before_age': missed,
                 'taken': taken, 'taken_share_of_original': round(taken / N, 4),
                 'gross_median': round(float(ent.gross.median()), 3)}
            for k, lab in [(None, 'R0')] + [(kk, f'R1k{kk:g}') for kk in K_FAMILY]:
                r[lab] = bounded(ent, k, inst)
                r[lab + '_gross'] = bounded(ent, k, inst, cost_mult=0.0)['determined_only']
            rows.append(r)
    return rows


if __name__ == '__main__':
    inst = sys.argv[1] if len(sys.argv) > 1 else 'NQ'
    terr = sys.argv[2] if len(sys.argv) > 2 else 'discovery'
    res = sweep(inst, terr)
    p = OUT / f'family1_{inst}_{terr}.json'
    p.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print(p, 'written', len(res), 'cells')
