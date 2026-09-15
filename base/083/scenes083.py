#!/usr/bin/env python3
"""083 — реальные сцены гонки: по одной на каждый исход, свечами.

Сцены не доказывают эффект и не выбираются ради подтверждения среднего.
Их задача — показать, что именно кодируют пять исходов на настоящей ленте.
Выбор детерминированный: первая по позиции `q` строка каждого исхода.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'work/083'
A, H = 5, 60
NAMES = {1: 'boundary_first', 2: 'mirror_first', 3: 'same_bar_ambiguous',
         0: 'unresolved (бюджет исчерпан)', 4: 'lost_observability (лента оборвалась)'}


def main(inst='NQ', terr='discovery'):
    m = ROOT / 'data/market' / inst
    op = np.load(m / 'open.npy'); hi = np.load(m / 'high.npy')
    lo = np.load(m / 'low.npy'); cl = np.load(m / 'close.npy')
    ts = np.load(m / 'close_ts_utc_ns.npy')
    d = pd.read_parquet(OUT / f'race_{inst}_{terr}_a{A}.parquet')
    d = d[d.state_readable].sort_values('q')
    lines = [f'083 — сцены гонки, {inst} {terr}, возраст {A}, бюджет {H} баров',
             'граница e — собственная exit_boundary Film-1; зеркало m = 2·close[q] − e',
             'предикат касания один и тот же: low <= уровень <= high', '']
    for code in (1, 2, 3, 0, 4):
        sub = d[d[f'outcome_{H}'] == code]
        if not len(sub):
            continue
        r = sub.iloc[0]
        q = int(r.q); t0 = q - A
        end = q + (int(r.res_bars) if r.res_bars > 0 else min(int(r.walkable), H))
        lines.append(f'--- {NAMES[code]} --- {r.riz_id}  {r.side}  ТФ {r.tf_minutes}м')
        lines.append(f'    T0 {pd.Timestamp(ts[t0]).strftime("%Y-%m-%d %H:%M")} UTC   '
                     f'e {r.exit_boundary:g}   close[q] {cl[q]:g}   d {r.d_close:g}   '
                     f'm {r.mirror:g}   лента после q {int(r.walkable)} бар')
        for j in range(t0, min(end, q + 12) + 1):
            tag = 'T0' if j == t0 else ('q ' if j == q else f'+{j-q}')
            mark = ''
            if j > q:
                if lo[j] <= r.exit_boundary <= hi[j]:
                    mark += '  << касание e'
                if lo[j] <= r.mirror <= hi[j]:
                    mark += '  << касание m'
            lines.append(f'    {tag:>4} {pd.Timestamp(ts[j]).strftime("%H:%M")} '
                         f'o {op[j]:>9g} h {hi[j]:>9g} l {lo[j]:>9g} c {cl[j]:>9g}{mark}')
        if end > q + 12:
            lines.append(f'    … до +{end-q} без касания обоих уровней')
        lines.append('')
    # отдельная ветвь: гэп через уровень в самом q (вне направленного контраста)
    g = pd.read_parquet(OUT / f'race_{inst}_{terr}_a{A}.parquet')
    g = g[~g.state_readable].sort_values('q')
    if len(g):
        r = g.iloc[0]; q = int(r.q); t0 = q - A
        lines.append(f'--- gap-through в q: d_close <= 0, зеркало не определено --- {r.riz_id}  {r.side}')
        lines.append(f'    e {r.exit_boundary:g}   close[q] {cl[q]:g}   d_close {r.d_close:g}')
        for j in range(t0, q + 1):
            tag = 'T0' if j == t0 else ('q ' if j == q else f'+{j-t0}')
            lines.append(f'    {tag:>4} {pd.Timestamp(ts[j]).strftime("%Y-%m-%d %H:%M")} '
                         f'o {op[j]:>9g} h {hi[j]:>9g} l {lo[j]:>9g} c {cl[j]:>9g}')
        lines.append('    цена по другую сторону границы, ни один бар её не накрыл — '
                     'Film-1 канонически жив, направление к границе в q не определено')
    p = OUT / f'scenes083_{inst}_{terr}.txt'
    p.write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))
    print(p)


if __name__ == '__main__':
    main(*(sys.argv[1:3] or ['NQ', 'discovery']))
