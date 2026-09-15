#!/usr/bin/env python3
"""Тот же cross-test без единой настройки — на отложенном материале.

Проверяется ровно то, что на NQ/discovery выглядит остаточным эффектом:
при точном совпадении положения и фиксированном квартиле `prefix_sigma`
упорядочивает ли `retraced_fraction` долю прихода к границе за окно.

Ничего не переопределяется: тот же срез возраста 5, те же окна, та же страта,
тот же `MIN_SIDE`, тот же `cross()`, те же nq. Территории:

    NQ evaluation   2019-2026, другая эпоха того же инструмента
    ES discovery    другой инструмент, та же эпоха
    YM discovery    другой инструмент, та же эпоха

Ожидание записано ДО счёта: если на NQ/discovery найден признак рынка, а не
свойство одной выборки, знак Δ должен сохраниться на h=15 и h=60 и оставаться
близким к нулю на h=5. Противоположное или нулевое поведение понижает статус
находки, и оно так и будет записано.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'work/082'
sys.path.insert(0, str(HERE))
from contrast import load, HS                                          # noqa: E402
from crosstest import cross, fmt                                       # noqa: E402

AGE = 5
TERRITORIES = [('NQ', 'evaluation'), ('ES', 'discovery'), ('YM', 'discovery')]


def main():
    res = {}
    for inst, terr in TERRITORIES:
        print(f"\n########## {inst} / {terr}, возраст {AGE}")
        d0 = load(inst, terr, AGE)
        for h in HS:
            s = d0[d0[f'contact_{h}'] >= 0].copy()
            print(f"  h={h}, n={len(s)}")
            for nq in (2, 4):
                r0 = cross(s, 'retraced_fraction', f'contact_{h}', 'prefix_sigma',
                           refine='prefix_sigma', nq=nq)
                r1 = cross(s, 'prefix_sigma', f'contact_{h}', 'retraced_fraction',
                           refine='retraced_fraction', nq=nq)
                print(f"     {f'retrace | sigma nq={nq}':>26}: {fmt(r0)}")
                print(f"     {f'sigma | retrace nq={nq}':>26}: {fmt(r1)}")
                res[f'{inst}_{terr}_h{h}_retrace_sigmafix{nq}'] = r0
                res[f'{inst}_{terr}_h{h}_sigma_retracefix{nq}'] = r1
            r = cross(s, 'retraced_fraction', f'contact_{h}', 'prefix_sigma')
            res[f'{inst}_{terr}_h{h}_retrace_raw'] = r
            print(f"     {'retrace без контроля':>26}: {fmt(r)}")
    p = OUT / 'holdout.json'
    p.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float),
                 encoding='utf-8')
    print('\n', p, 'written')


if __name__ == '__main__':
    main()
