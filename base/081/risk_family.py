#!/usr/bin/env python3
"""Защитное семейство, СНЯТОЕ С НАЛОЖЕНИЯ GATE A, и счёт исхода одной попытки.

ОТКУДА ЧИСЛА (AGENTS.md: «числа правила читаются с наложенных фильмов»)
=======================================================================
Наложение GATE A на NQ/discovery, T0, завершённые пути:
    mae_certain / gross:  p25 0,333   p50 0,000   p75 1,600   p90 5,500
    mae_bound   / gross:  p25 0,333   p50 1,083   p75 3,000   p90 7,500
    dur, баров:           p50 1       p75 4       p90 18      p99 228
Отсюда семейство, и ничего сверх него:

    R0            без стопа — только опорная граница, риск не ограничен
    R1(k)         стоп на k × gross, k ∈ {1, 2, 3}   (p25 / p50 / p75 отношения)
    R2(H)         выход по времени на H баров, H ∈ {18, 60}  (p90 / за p90)

Стоп выражен в долях собственной цели, а не в пунктах: цель и расход живут в
пунктах, но наложение 2006-2018 охватывает пятнадцатикратное изменение уровня
NQ, и порог в пунктах на нём означал бы в 2006 и в 2018 разное. Отношение
`stop / target` от уровня не зависит и напрямую задаёт reward:risk.

ТРИ ИСХОДА, А НЕ ДВА
====================
    win        стоп точно не задет: mae_bound < s        → +gross − cost
    loss       стоп точно задет до касательной свечи:
               mae_certain >= s                          → −s − cost
    ambiguous  s ∈ (mae_certain, mae_bound]: уровень достигнут только внутри
               касательной свечи, порядок внутри минуты неизвестен
               → возвращается парой границ, не средним
    unknown    фильм без сертифицированного контакта, стоп до потери
               наблюдаемости не достигнут → исход не определён

У фильма без сертифицированного контакта стоп, достигнутый ДО промежутка,
делает исход определённым проигрышем: `mae_observed_until_loss >= s` — точное
наблюдение, а не нижняя граница для этого вопроса. Остальные остаются unknown.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

COST = {'NQ': 1.00, 'ES': 1.00, 'YM': 4.00}
K_FAMILY = (1.0, 2.0, 3.0)
H_FAMILY = (18, 60)

WIN, LOSS, AMBIG, UNKNOWN = 0, 1, 2, 3


def score_stop(rows, k, inst, cost_mult=1.0):
    """Исход одной попытки при стопе `k × gross`. Возвращает разметку и границы.

    `rows` — строки слоя путей с колонками gross/mae_certain/mae_bound/outcome.
    Ничего не усредняется между границами неоднозначности.
    """
    c = COST[inst] * cost_mult
    g = rows.gross.to_numpy(np.float64)
    mc = rows.mae_certain.to_numpy(np.float64)
    mb = rows.mae_bound.to_numpy(np.float64)
    cert = rows.outcome.to_numpy() == 0
    s = k * g if k is not None else np.full(g.shape, np.inf)

    kind = np.full(g.shape, UNKNOWN, dtype=np.int8)
    hit_sure = mc >= s
    hit_maybe = (~hit_sure) & (mb >= s)
    kind[cert & hit_sure] = LOSS
    kind[cert & hit_maybe] = AMBIG
    kind[cert & ~hit_sure & ~hit_maybe] = WIN
    # несертифицированный фильм: стоп до промежутка делает проигрыш определённым
    kind[~cert & hit_sure] = LOSS
    # остальные несертифицированные остаются UNKNOWN

    lo = np.full(g.shape, np.nan)          # нижняя граница результата, пункты
    hi = np.full(g.shape, np.nan)
    lo[kind == WIN] = (g - c)[kind == WIN]
    hi[kind == WIN] = (g - c)[kind == WIN]
    lo[kind == LOSS] = (-s - c)[kind == LOSS]
    hi[kind == LOSS] = (-s - c)[kind == LOSS]
    lo[kind == AMBIG] = (-s - c)[kind == AMBIG]
    hi[kind == AMBIG] = (g - c)[kind == AMBIG]
    return kind, lo, hi


def summarise(kind, lo, hi, label):
    n = kind.size
    sh = {k: round(float((kind == v).mean()), 4) for k, v in
          (('win', WIN), ('loss', LOSS), ('ambiguous', AMBIG), ('unknown', UNKNOWN))}
    det = np.isin(kind, (WIN, LOSS))
    out = {'rule': label, 'n': int(n), 'shares': sh,
           'n_determined': int(det.sum()),
           'mean_points_determined_only': round(float(np.nanmean(lo[det])), 4)
           if det.any() else None}
    # границы по ВСЕМ строкам, где исход хотя бы ограничен (win/loss/ambiguous)
    b = np.isin(kind, (WIN, LOSS, AMBIG))
    if b.any():
        out['mean_points_lower_bound'] = round(float(np.nanmean(lo[b])), 4)
        out['mean_points_upper_bound'] = round(float(np.nanmean(hi[b])), 4)
        out['n_bounded'] = int(b.sum())
    out['note_unknown'] = ('unknown не ноль и не проигрыш: позиция открыта '
                           'в момент потери наблюдаемости')
    return out
