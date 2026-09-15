#!/usr/bin/env python3
"""Установленное различие на реальных Film-1: опора и ближайшие контрпримеры.

Берётся ОДНА ячейка сравнения — точное совпадение положения (`d_close` до тика,
сторона, полоса ТФ) плюс фиксированный квартиль `prefix_sigma`, возраст 5,
окно 15 баров. Внутри неё показываются четыре фильма:

    опора        высокий retrace, дошёл; низкий retrace, не дошёл
    контрпример  высокий retrace, НЕ дошёл; низкий retrace, дошёл

Контрпримеры печатаются не для красоты: различие 0,436 → 0,630 — это сдвиг
долей, а не правило. Внутри ячейки обе ветви есть у обеих групп, и карточка
обязана это показывать.

Отдельно печатается фильм, у которого контакт случился в самом баре q+1: это
наглядная форма границы различимости — остаточного пути с известным порядком
у него нет вовсе.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'work/082'
P081 = ROOT / 'work/081a'
sys.path.insert(0, str(HERE))
from contrast import load, MIN_SIDE                                    # noqa: E402
from identify import attach_offset                                     # noqa: E402

AGE, H, NQ = 5, 15, 4


def render(f, i, m, kind_arr, age, h, tag):
    opn, high, low, close, ts = m
    t0 = int(f.q) - age
    q = int(f.q)
    b = float(f.boundary)
    up = f.side == 'north'
    L = [f"--- {tag}",
         f"riz_id {f.riz_id}  ТФ {f.tf_minutes}  сторона {f.side}  граница {b:.2f}",
         f"  на срезе q = T0+{age}:  d_close {f.d_close:+.2f}   run_max {f.run_max:.2f}   "
         f"retraced_fraction {f.retraced_fraction:.3f}   prefix_sigma {f.prefix_sigma:.3f}",
         f"  исход окна h={h}: " + (
             f"ДОШЁЛ на баре q+{int(f.contact_offset)}"
             if f[f'contact_{h}'] == 1 else "НЕ дошёл за окно"),
         "   поз      время UTC          open     high      low    close   до границы"]
    end = q + h
    for j in range(t0, min(end, close.size - 1) + 1):
        d = (close[j] - b) if up else (b - close[j])
        hit = low[j] <= b <= high[j]
        mark = ''
        if j == t0:
            mark = ' T0'
        elif j == q:
            mark = ' q '
        else:
            mark = '   '
        note = '  <<< КОНТАКТ, Film-1 кончился' if hit and j > t0 else ''
        L.append(f"  {j}{mark} {pd.Timestamp(ts[j], tz='UTC'):%Y-%m-%d %H:%M} "
                 f"{opn[j]:8.2f} {high[j]:8.2f} {low[j]:8.2f} {close[j]:8.2f} "
                 f"{d:+8.2f}{note}")
        if hit and j > t0:
            break
    return '\n'.join(L)


def main(inst='NQ', terr='discovery'):
    mk = ROOT / 'data/market' / inst
    m = (np.load(mk / 'open.npy'), np.load(mk / 'high.npy'), np.load(mk / 'low.npy'),
         np.load(mk / 'close.npy'), np.load(mk / 'close_ts_utc_ns.npy'))
    fl = pd.read_parquet(P081 / f'paths/films_{inst}_{terr}.parquet',
                         columns=['exit_boundary'])

    d = attach_offset(load(inst, terr, AGE), inst, terr)
    d = d[(d[f'contact_{H}'] >= 0) & d.state_readable].copy()
    d['boundary'] = fl.exit_boundary.to_numpy()[d.film.to_numpy()]
    d['_qq'] = (d.groupby('base_key', observed=True).prefix_sigma
                .transform(lambda s: pd.qcut(s, NQ, labels=False, duplicates='drop')))
    d = d[d._qq.notna()]
    d['cell'] = d.base_key + '#' + d._qq.astype(int).astype(str)

    # самая населённая ячейка, где обе ветви есть у обеих групп
    best, bestn = None, -1
    for cell, g in d.groupby('cell', observed=True):
        med = g.retraced_fraction.median()
        lo, hi = g[g.retraced_fraction < med], g[g.retraced_fraction > med]
        if len(lo) < MIN_SIDE or len(hi) < MIN_SIDE:
            continue
        need = [hi[hi[f'contact_{H}'] == 1], lo[lo[f'contact_{H}'] == 0],
                hi[hi[f'contact_{H}'] == 0], lo[lo[f'contact_{H}'] == 1]]
        if any(len(x) == 0 for x in need):
            continue
        if len(g) > bestn:
            best, bestn = (cell, g, med, need), len(g)
    cell, g, med, need = best
    lo, hi = g[g.retraced_fraction < med], g[g.retraced_fraction > med]

    head = [
        f"ЯЧЕЙКА СРАВНЕНИЯ: {cell}",
        f"  точное совпадение положения (d_close в тиках | сторона | полоса ТФ) "
        f"плюс квартиль prefix_sigma",
        f"  возраст среза 5, окно 15 баров, {inst}/{terr}",
        f"  фильмов в ячейке {len(g)}: retrace-низ {len(lo)}, retrace-верх {len(hi)}",
        f"  порог деления retraced_fraction = {med:.3f}",
        f"  доля дошедших за 15 баров: низ {lo[f'contact_{H}'].mean():.4f}  "
        f"верх {hi[f'contact_{H}'].mean():.4f}",
        "",
        "Это сдвиг ДОЛИ, а не правило: обе ветви есть у обеих групп, и ниже",
        "показаны и опора, и ближайшие контрпримеры из ТОЙ ЖЕ ячейки.", ""]

    tags = ['ОПОРА: retrace ВЫСОКИЙ, дошёл за окно',
            'ОПОРА: retrace НИЗКИЙ, не дошёл за окно',
            'КОНТРПРИМЕР: retrace ВЫСОКИЙ, НЕ дошёл',
            'КОНТРПРИМЕР: retrace НИЗКИЙ, дошёл']
    body = []
    for sub, tag in zip(need, tags):
        r = sub.sort_values('retraced_fraction',
                            ascending='НИЗКИЙ' in tag).iloc[0]
        body.append(render(r, None, m, None, AGE, H, tag))

    z = g[g.contact_offset == 1]
    if len(z):
        body.append(render(z.iloc[0], None, m, None, AGE, H,
                           'ГРАНИЦА РАЗЛИЧИМОСТИ: контакт в самом баре q+1 — '
                           'остаточного пути с известным порядком нет вовсе'))

    txt = '\n'.join(head) + '\n\n'.join(body) + '\n'
    p = OUT / f'scenes082_{inst}_{terr}.txt'
    p.write_text(txt, encoding='utf-8')
    print(txt)
    print(p, 'written')


if __name__ == '__main__':
    main(*(sys.argv[1:] or ['NQ', 'discovery']))
