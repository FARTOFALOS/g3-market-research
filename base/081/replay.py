#!/usr/bin/env python3
"""Последовательный replay политики от T0 на ВСЕЙ исходной Film-популяции.

ПОЧЕМУ НЕ СРЕЗ ВЫЖИВШИХ
=======================
Ячейка вроде «age>=5, d_close>=2c» — это не подмножество строк, у которых
условие выполнилось. Это политика: от T0 ждать; если контакт случился раньше —
возможность упущена; на первом разрешённом q проверить доступный префикс; при
выполнении условия — ENTER по объявленной execution policy; иначе ждать дальше.
Знаменатель — исходная популяция фильмов, а не дожившие.

ИСПОЛНЕНИЕ НЕ ВЫБИРАЕТ МИНУТУ
=============================
Условие срабатывания читается ТОЛЬКО по префиксу (`d_close`, `age`). Доступность
и направленная пригодность next-open — результат исполнения, а не признак
решения. Если на сработавшей минуте reference недоступна или open уже не
снаружи границы — это `execution_cancelled`, а не повод ждать дальше:
разрешена одна попытка входа на Film-1 (постановка §14).

ПОТОК, КОТОРЫЙ СОХРАНЯЕТСЯ ЦЕЛИКОМ
==================================
    missed_before_eligible   контакт/конец сертификации раньше разрешённого age
    expired_without_trigger  фильм дожил, условие не выполнилось ни на одном q
    execution_cancelled      условие сработало, исполнить нечем
    win / loss / ambiguous   исполнено, исход определён либо ограничен парой
    unknown                  исполнено, позиция открыта на потере наблюдаемости
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


def first_trigger(q, cond):
    """Первая строка каждого фильма, где выполнено ПРЕФИКСНОЕ условие.

    Строки уже лежат film-major с возрастающим age, поэтому первая по порядку
    внутри фильма — она и есть первая по времени.
    """
    s = q[cond]
    return s.groupby('film', sort=False).head(1)


def replay(q, f, inst, age_min, d_mult, k, cost_mult=1.0):
    """Один прогон одной политики. Возвращает поток и сводку."""
    c = COST[inst]
    N = len(f)
    nq = f.n_q.to_numpy()

    # 1) возможность кончилась раньше, чем политике вообще разрешено смотреть
    missed = int((nq <= age_min).sum())

    # 2) первое срабатывание префиксного условия — исполнение сюда не вмешивается
    cond = (q.age.to_numpy() >= age_min) & (q.d_close.to_numpy() > d_mult * c)
    trig = first_trigger(q, cond)
    triggered_films = set(trig.film.to_numpy().tolist())

    alive = int((nq > age_min).sum())
    expired = alive - len(trig)

    # 3) исполнение на сработавшей минуте, одна попытка
    ok = trig.usable.to_numpy()
    cancelled = int((~ok).sum())
    ent = trig[ok]

    kind, lo, hi = score_stop(ent, k, inst, cost_mult)
    g = ent.gross.to_numpy(np.float64)
    madv = ent.mae_bound.to_numpy(np.float64)
    cst = c * cost_mult
    up = np.where(np.isnan(hi), g - cst, hi)       # неразрешённые доходят до TP
    dn = np.where(np.isnan(lo), -madv - cst, lo)   # теряют не меньше наблюдённого

    n = len(ent)
    stream = {
        'original_population': N,
        'missed_before_eligible': missed,
        'expired_without_trigger': expired,
        'execution_cancelled': cancelled,
        'executed': n,
        'executed_share_of_original': round(n / N, 4),
        'win': int((kind == WIN).sum()), 'loss': int((kind == LOSS).sum()),
        'ambiguous': int((kind == AMBIG).sum()), 'unknown': int((kind == UNKNOWN).sum())}
    assert missed + expired + cancelled + n == N, 'поток не разбивает популяцию'

    det = np.isin(kind, (WIN, LOSS))
    res = {
        'policy': {'age_min': age_min, 'd_close_gt_cost_mult': d_mult,
                   'stop': 'R0' if k is None else f'R1 k={k:g}',
                   'cost_mult': cost_mult},
        'stream': stream,
        'share_unknown_of_executed': round(float((kind == UNKNOWN).mean()), 4) if n else None,
        'mean_points_upper': round(float(up.mean()), 4) if n else None,
        'mean_points_lower': round(float(dn.mean()), 4) if n else None,
        'mean_points_determined_only': round(float(np.nanmean(lo[det])), 4) if det.any() else None,
        'total_points_upper': round(float(up.sum()), 1) if n else None,
        'total_points_lower': round(float(dn.sum()), 1) if n else None,
        'gross_median_at_execution': round(float(ent.gross.median()), 3) if n else None,
        'd_close_median_at_decision': round(float(ent.d_close.median()), 3) if n else None,
    }
    # execution outcome: prefix-условие прошло, а исполнение оказалось ниже расхода
    if n:
        res['executed_but_gross_le_cost'] = round(float((g <= cst).mean()), 4) if cost_mult else None
        res['exec_slip_median'] = round(float(np.median(ent.d_close.to_numpy() - g)), 3)
    return res


def sweep(inst='NQ', terr='discovery', ages=(0, 1, 2, 5, 15), dmults=(0.0, 1.0, 2.0, 4.0),
          ks=(None, 1.0, 2.0, 3.0)):
    q = pd.read_parquet(OUT / f'paths/q_{inst}_{terr}.parquet')
    f = pd.read_parquet(OUT / f'paths/films_{inst}_{terr}.parquet')
    rows = []
    for a in ages:
        for d in dmults:
            for k in ks:
                rows.append(replay(q, f, inst, a, d, k))
    return rows


if __name__ == '__main__':
    inst = sys.argv[1] if len(sys.argv) > 1 else 'NQ'
    terr = sys.argv[2] if len(sys.argv) > 2 else 'discovery'
    res = sweep(inst, terr)
    p = OUT / f'family1_replay_{inst}_{terr}.json'
    p.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print(p, 'written', len(res), 'policies')
