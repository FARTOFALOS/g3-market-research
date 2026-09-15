#!/usr/bin/env python3
"""083 — свод замороженного directional question по четырём территориям.

Основной estimand — Film-weighted: `δ = (B − M) / N` по всей объявленной
префиксной популяции. Дневная кластеризация нужна ДЛЯ НЕОПРЕДЕЛЁННОСТИ и
зависимости; она не заменяет estimand. Средний день — ДРУГАЯ величина (средний
день вместо среднего Film-среза), и публикуется рядом именно под этим именем.

Вердикт формулируется через precision: half-width самого широкого day-block CI
переводится в шкалу разрешившейся гонки, и называется, какой размер перевеса
данные исключают, а какой остаётся совместимым.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'work/083'
TERR = [('NQ', 'discovery'), ('NQ', 'evaluation'), ('ES', 'discovery'), ('YM', 'discovery')]
A, H = 5, 60


def day_weighted(d, h):
    x = d[d.state_readable]
    o = x[f'outcome_{h}'].to_numpy()
    num = pd.Series((o == 1).astype(int) - (o == 2), index=x.index).groupby(x.t0_day).sum()
    den = x.groupby('t0_day').size()
    return float((num / den).mean()), int(len(den))


def main():
    rows = []
    for inst, terr in TERR:
        b = json.loads((OUT / f'balance_{inst}_{terr}.json').read_text(encoding='utf-8'))
        c = b['ages'][str(A)]['h'][str(H)]
        d = pd.read_parquet(OUT / f'race_{inst}_{terr}_a{A}.parquet')
        dw, ndays = day_weighted(d, H)
        hw = (c['ci'][1] - c['ci'][0]) / 2
        rs = c['resolved_share']
        # перевод δ в шкалу разрешившейся гонки: ratio = 0.5 + δ/(2·resolved_share)
        rows.append(dict(
            terr=f'{inst} {terr}', N=c['n'], days=ndays,
            B=c['boundary_first'], M=c['mirror_first'], Amb=c['same_bar_ambiguous'],
            U=c['unresolved'], L=c['lost_observability'],
            delta=c['delta'], ci=c['ci'], hw=hw,
            amb_lo=c['amb_lo'], amb_hi=c['amb_hi'],
            obs_lo=c['obs_lo'], obs_hi=c['obs_hi'],
            resolved_share=rs, ratio=c['resolved_only_ratio'],
            day_weighted_delta=dw,
            excluded_ratio_above=0.5 + hw / (2 * rs),
            gap_e=c['gap_through']['gap_e_share'], gap_m=c['gap_through']['gap_m_share'],
            north=c['by_side']['north']['delta'], south=c['by_side']['south']['delta'],
            nbars_q=[c['by_n_bars_quartile'][str(k)] for k in range(4)]))
    print(f'ОСНОВНАЯ ЯЧЕЙКА: возраст {A}, бюджет {H} баров\n')
    print(f'{"территория":<16}{"N":>7}{"дней":>6}{"B":>7}{"M":>7}{"A":>5}{"U":>6}{"L":>6}'
          f'{"δ":>9}{"CI 95%":>20}{"ratio":>8}')
    for r in rows:
        print(f'{r["terr"]:<16}{r["N"]:>7}{r["days"]:>6}{r["B"]:>7}{r["M"]:>7}{r["Amb"]:>5}'
              f'{r["U"]:>6}{r["L"]:>6}{r["delta"]:>+9.4f}'
              f'{f"[{r['ci'][0]:+.4f};{r['ci'][1]:+.4f}]":>20}{r["ratio"]:>8.4f}')
    print(f'\n{"территория":<16}{"amb bounds":>22}{"obs bounds":>22}{"разрешилось":>13}'
          f'{"δ сред. дня":>13}')
    for r in rows:
        print(f'{r["terr"]:<16}{f"[{r['amb_lo']:+.4f};{r['amb_hi']:+.4f}]":>22}'
              f'{f"[{r['obs_lo']:+.4f};{r['obs_hi']:+.4f}]":>22}'
              f'{r["resolved_share"]:>13.3f}{r["day_weighted_delta"]:>+13.4f}')
    print(f'\nPRECISION. Полуширина CI и что она исключает в шкале гонки '
          f'(доля boundary среди разрешившихся):')
    for r in rows:
        print(f'  {r["terr"]:<16} ±{r["hw"]:.4f} по δ  →  перевес сильнее '
              f'{r["excluded_ratio_above"]:.3f} : {1-r["excluded_ratio_above"]:.3f} не поддержан; '
              f'наблюдено {r["ratio"]:.4f}')
    print(f'\nСТОРОНЫ (снос дал бы противоположные знаки):')
    for r in rows:
        print(f'  {r["terr"]:<16} north {r["north"]:+.4f}   south {r["south"]:+.4f}')
    print(f'\nГЭП ЧЕРЕЗ УРОВЕНЬ без накрывающего бара (симметрия предиката):')
    for r in rows:
        print(f'  {r["terr"]:<16} через e {r["gap_e"]:.4f}   через m {r["gap_m"]:.4f}')
    print(f'\nСКОРОСТЬ ПРОТИВ НАПРАВЛЕНИЯ по квартилям n_bars = d_close / prefix_sigma:')
    print(f'  {"территория":<16}{"квартиль":>10}{"разрешилось":>13}{"δ":>10}{"CI 95%":>20}')
    for r in rows:
        for k, q in enumerate(r['nbars_q']):
            print(f'  {r["terr"] if k==0 else "":<16}{k:>10}{q["resolved_share"]:>13.3f}'
                  f'{q["delta"]:>+10.4f}{f"[{q['ci'][0]:+.4f};{q['ci'][1]:+.4f}]":>20}')
    (OUT / 'verdict.json').write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
