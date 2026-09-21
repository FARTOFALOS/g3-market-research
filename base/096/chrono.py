#!/usr/bin/env python3
"""096 chrono — непрерывен ли late-era эффект или собран отдельными годами.

Разложение УЖЕ посчитанного агрегата части III тем же замороженным RULE_V1.
Ни одного нового условия, фильтра или порога не вводится. Вопрос узкий:
late-era ΔY = +0,084 — устойчивый режим или сумма нескольких периодов?

ЗАПРЕТ, объявленный здесь же: если окажется, что эффект несут конкретные годы,
это НЕ основание торговать «в похожие режимы». Post-hoc выбор режима — новая
ветка со своим провенансом (guardrail Q), а не вывод этой.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'work/096'
sys.path.insert(0, str(ROOT / 'base/086')); sys.path.insert(0, str(HERE))
import fork086 as F                                                          # noqa: E402
from part2 import operator                                                   # noqa: E402
from replay import build, delta                                              # noqa: E402

WIN_DAYS, STEP = 250, 25


def main():
    d = build().sort_values('day').reset_index(drop=True)
    late = d[d.year.between(2019, 2025)]
    res = {}

    r = delta(late, 'y_lo', 'y_hi')
    print(f'late-era 2019-2025 целиком: N={len(late)} ΔY [{r[0]:+.4f};{r[1]:+.4f}] '
          f'CI [{r[2]:+.4f};{r[3]:+.4f}]')
    res['late_all'] = {'N': int(len(late)), 'dY': [r[0], r[1]], 'ci': [r[2], r[3]]}

    print('\n=== leave-one-year-out по поздней эпохе ===')
    res['loo'] = []
    for y in range(2019, 2026):
        sub = late[late.year != y]
        rr = delta(sub, 'y_lo', 'y_hi')
        drop = (rr[0] + rr[1]) / 2 - (r[0] + r[1]) / 2
        print(f'  без {y}: N={len(sub):>6} ΔY [{rr[0]:+.4f};{rr[1]:+.4f}] '
              f'CI [{rr[2]:+.4f};{rr[3]:+.4f}]  сдвиг {drop:+.4f}')
        res['loo'].append({'year': y, 'dY': [rr[0], rr[1]], 'ci': [rr[2], rr[3]], 'shift': drop})

    print('\n=== leave-two-out: убрать 2021 и 2022 вместе ===')
    sub = late[~late.year.isin((2021, 2022))]
    rr = delta(sub, 'y_lo', 'y_hi')
    print(f'  N={len(sub):>6} ΔY [{rr[0]:+.4f};{rr[1]:+.4f}] CI [{rr[2]:+.4f};{rr[3]:+.4f}]')
    res['drop_2021_2022'] = {'N': int(len(sub)), 'dY': [rr[0], rr[1]], 'ci': [rr[2], rr[3]]}

    print(f'\n=== скользящее окно {WIN_DAYS} торговых дней, шаг {STEP} (вся лента) ===')
    days = np.sort(d.day.unique())
    res['rolling'] = []
    print(f"  {'центр окна':>12} {'N':>6} {'iso':>6} {'ΔY':>9}")
    for i in range(0, len(days) - WIN_DAYS + 1, STEP):
        win = days[i:i + WIN_DAYS]
        s = d[d.day.isin(win)]
        rr = delta(s, 'y_lo', 'y_hi', B=200)
        if rr is None:
            continue
        mid = (rr[0] + rr[1]) / 2
        c = pd.Timestamp(win[len(win) // 2] * 86400 * 10 ** 9).strftime('%Y-%m')
        bar = '#' * int(abs(mid) * 60)
        print(f'  {c:>12} {len(s):>6} {s.iso.mean():>6.3f} {mid:>+9.4f}  '
              f"{'  ' if mid >= 0 else '-'}{bar}")
        res['rolling'].append({'center': c, 'N': int(len(s)), 'dY_mid': mid})
    (OUT / 'chrono.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print('\n->', OUT / 'chrono.json')


if __name__ == '__main__':
    main()
