#!/usr/bin/env python3
"""Откуда берётся результат К-1: вклад по дням и календарным участкам.

Концентрация прибыли в верхнем проценте сделок — характеристика источника
результата и его хрупкости, а не опровержение возможности преимущества.
Исключение лучших сделок задним числом здесь используется только как
диагностика: будущие выигрыши фильтром не становятся и в правило не входят.

Условие К-1 зафиксировано дословно по итогу перебора прошлого цикла и здесь не
подбирается. Оно намеренно названо нейтрально: `C[k] > max(H[k-4..k-1])` вместе
с «верхняя тень длиннее тела». Второе отношение — сравнение ДЛИН, а не порядка
точек; это записано и в карточке.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from candidate_check import POINT, COST                            # noqa: E402
from minute_edge import load, MIN                                  # noqa: E402
from minute_candidates import parts, CANDIDATES, mask_of, stream   # noqa: E402


def blocks(ts_ns):
    day = (ts_ns // MIN) // 1440
    return day


def report(trades, cost):
    g = np.array([t[1] for t in trades])
    ts = np.array([t[0] for t in trades])
    day = blocks(ts)
    net = g - cost
    days = np.unique(day)
    by_day = np.array([net[day == d].sum() for d in days])
    by_day_g = np.array([g[day == d].sum() for d in days])
    order = np.argsort(-by_day_g)
    total_g = g.sum()
    top = {}
    for m in (1, 5, 10, 25):
        top[f'top{m}_days_share_of_gross'] = float(
            by_day_g[order[:m]].sum() / total_g) if total_g else None
    quarters = ((ts // MIN) // 1440 + 719163) // 91
    qs = np.unique(quarters)
    by_q = {int(q): float(net[quarters == q].mean()) for q in qs}
    return {
        'trades': len(g), 'days': int(len(days)),
        'mean_gross': float(g.mean()),
        'share_days_positive_net': float((by_day > 0).mean()),
        'median_day_net': float(np.median(by_day)),
        'gross_total': float(total_g),
        **top,
        'mean_gross_drop_best_day': float(
            np.delete(g, np.flatnonzero(day == days[order[0]])).mean()),
        'mean_gross_drop_best_5_days': float(
            np.delete(g, np.flatnonzero(np.isin(day, days[order[:5]]))).mean()),
        'quarters_with_positive_mean_net': int(sum(v > 0 for v in by_q.values())),
        'quarters_total': len(by_q),
        'worst_quarter_mean_net': float(min(by_q.values())),
        'best_quarter_mean_net': float(max(by_q.values())),
        'note': 'исключение лучших дней — диагностика хрупкости источника, '
                'а не правило: будущие выигрыши фильтром не становятся'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--instrument', default='NQ')
    ap.add_argument('--candidate', default='K-1')
    ap.add_argument('--hold', type=int, default=3)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    cost = COST[a.instrument]
    out = {'candidate': a.candidate, 'hold': a.hold,
           'condition': CANDIDATES[a.candidate]['condition'],
           'negate': CANDIDATES[a.candidate]['negate'],
           'periods': {}}
    for tag, s0, s1 in (('поиск', '2020-01-01', '2025-11-01'),
                        ('резерв', '2025-11-01', '2026-05-05')):
        m, k, ss = load(a.instrument, s0, s1)
        sel = mask_of(parts(m, k, ss), CANDIDATES[a.candidate])
        tr = stream(m, k, sel, CANDIDATES[a.candidate]['side'], a.instrument,
                    a.hold)
        out['periods'][tag] = report(tr, cost)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1),
                           encoding='utf-8')
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
