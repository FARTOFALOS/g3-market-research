#!/usr/bin/env python3
"""Достаточно ли устойчивых медиан, чтобы назвать сдвиг состава малым. Нет.

Медиана — одна квантиль: обусловливание может перестроить хвосты, оставив её на
месте. И сравниваются не маргиналы, а группы ВНУТРИ страт, поэтому значение
имеет совместное распределение внутри страты, а не общая медиана.

Три проверки, от слабой к решающей:

1. КВАНТИЛИ, а не медиана: p10…p90 трёх чтений состояния у попавших в
   `at_risk` и у выброшенных.
2. СОСТАВ ВНУТРИ СТРАТЫ: доля `at_risk` по квартилям `prefix_sigma` внутри
   страты. Если обусловливание бьёт по квартилям ровно, оно не переставляет
   сравниваемые группы.
3. РЕШАЮЩАЯ: `at_risk` почти целиком определяется часами. Значит его можно
   ОБОЙТИ — ограничиться часами, где обусловливания практически нет
   (`at_risk >= 0.95` на самом длинном окне), и пересчитать решающий cross-test
   там. Если остаточный эффект retrace при фиксированном σ на h=15/60 выживает
   на этом «чистом по часам» подмножестве, объяснение наблюдаемостью закрыто.

Внутренний контроль, который уже есть в основном расчёте: сила обусловливания
и место, где retrace выживает, идут В РАЗНЫЕ СТОРОНЫ. Самое слабое
обусловливание — h=5 (at_risk 0,972), и именно там retrace поглощается σ;
самое сильное — h=60 (0,793), и там retrace выживает.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'work/082'
sys.path.insert(0, str(HERE))
from contrast import load, AGES, HS                                    # noqa: E402
from crosstest import cross, fmt                                       # noqa: E402

QS = (.10, .25, .50, .75, .90)


def main(inst='NQ', terr='discovery'):
    res = {}
    for a in AGES:
        d = load(inst, terr, a)
        d = d[d.state_readable]
        A = {}
        for h in HS:
            ar = (d[f'contact_{h}'] >= 0).to_numpy()
            cell = {'at_risk_share': round(float(ar.mean()), 4)}
            # 1. квантили, а не медиана
            for col in ('retraced_fraction', 'prefix_sigma', 'd_close'):
                v = d[col].to_numpy()
                cell[col] = {
                    'at_risk': [round(float(np.quantile(v[ar], q)), 4) for q in QS],
                    'dropped': ([round(float(np.quantile(v[~ar], q)), 4) for q in QS]
                                if (~ar).sum() > 30 else None)}
            # 2. состав внутри страты: доля at_risk по квартилям sigma
            qq = (d.groupby('base_key', observed=True).prefix_sigma
                  .transform(lambda s: pd.qcut(s, 4, labels=False, duplicates='drop')))
            ok = qq.notna()
            cell['at_risk_by_sigma_quartile_within_stratum'] = {
                str(int(k)): round(float(ar[ok.to_numpy()][qq[ok].to_numpy() == k].mean()), 4)
                for k in sorted(qq[ok].unique())}
            A[f'h{h}'] = cell

        # 3. решающая: часы, которых обусловливание почти не касается
        ar60 = (d[f'contact_60'] >= 0)
        byh = ar60.groupby(d.hour_utc).mean()
        clean_hours = sorted(byh[byh >= .95].index.tolist())
        A['clean_hours_at_risk_ge_0.95_at_h60'] = clean_hours
        sub = d[d.hour_utc.isin(clean_hours)]
        A['clean_hours_rows'] = int(len(sub))
        A['clean_hours_share_of_population'] = round(float(len(sub) / max(len(d), 1)), 4)
        print(f"\n########## возраст {a}: часов без обусловливания "
              f"{len(clean_hours)} из 24, строк {len(sub)} "
              f"({len(sub)/max(len(d),1):.3f} популяции)")
        for h in (15, 60):
            s = sub[sub[f'contact_{h}'] >= 0].copy()
            if len(s) < 200:
                print(f"  h{h}: опоры нет ({len(s)})")
                continue
            print(f"  h{h}, отклик contact_{h}, n={len(s)}")
            for nq in (2, 4):
                r0 = cross(s, 'retraced_fraction', f'contact_{h}', 'prefix_sigma',
                           refine='prefix_sigma', nq=nq)
                r1 = cross(s, 'prefix_sigma', f'contact_{h}', 'retraced_fraction',
                           refine='retraced_fraction', nq=nq)
                print(f"     {f'retrace | sigma nq={nq}':>26}: {fmt(r0)}")
                print(f"     {f'sigma | retrace nq={nq}':>26}: {fmt(r1)}")
                A[f'clean_hours_h{h}_retrace_sigmafix{nq}'] = r0
                A[f'clean_hours_h{h}_sigma_retracefix{nq}'] = r1
        res[f'age_{a}'] = A

    p = OUT / f'observability_{inst}_{terr}.json'
    p.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float),
                 encoding='utf-8')
    print('\n', p, 'written')


if __name__ == '__main__':
    main(*(sys.argv[1:] or ['NQ', 'discovery']))
