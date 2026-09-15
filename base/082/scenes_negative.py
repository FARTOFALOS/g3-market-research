#!/usr/bin/env python3
"""Живой Film-1, у которого close уже по другую сторону границы. Реальные свечи.

Вопрос: это гэп через уровень, ошибка предиката контакта, допустимое состояние
Film-1 или потеря наблюдаемости. Решается просмотром, а не рассуждением.
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
sys.path.insert(0, str(ROOT / 'base/080'))
from tape import presence_grid, gap_kinds                              # noqa: E402

MIN = 60_000_000_000


def main(inst='NQ', terr='discovery', age=5, n_show=8):
    m = ROOT / 'data/market' / inst
    opn = np.load(m / 'open.npy'); high = np.load(m / 'high.npy')
    low = np.load(m / 'low.npy'); close = np.load(m / 'close.npy')
    ts = np.load(m / 'close_ts_utc_ns.npy')
    last = high.size - 1
    grid, lo_, hi_ = presence_grid()
    kind, _ = gap_kinds(inst, grid, lo_, hi_)

    f = pd.read_parquet(P081 / f'paths/films_{inst}_{terr}.parquet')
    t0 = f.t0_spine_pos.to_numpy().astype(np.int64)
    e = f.exit_boundary.to_numpy().astype(np.float64)
    north = (f.side.to_numpy() == 'north')

    rows = []
    for i in range(len(f)):
        q = t0[i] + age
        if q > last:
            continue
        ok = True
        for j in range(t0[i] + 1, q + 1):
            if kind[j] != 0 or (low[j] <= e[i] <= high[j]):
                ok = False; break
        if not ok:
            continue
        d = (close[q] - e[i]) if north[i] else (e[i] - close[q])
        if d <= 0:
            rows.append((i, q, d))
    rows.sort(key=lambda r: r[2])
    print(f'{inst}/{terr} age {age}: живых срезов с d_close <= 0 — {len(rows)}\n')

    pick = rows[:3] + rows[len(rows) // 2 - 2:len(rows) // 2 + 2] + rows[-3:]
    seen = set()
    out = []
    for i, q, d in pick:
        if i in seen:
            continue
        seen.add(i)
        if len(seen) > n_show:
            break
        b = e[i]; up = north[i]
        L = [f"riz_id {f.riz_id.iloc[i]}  ТФ {f.tf_minutes.iloc[i]}  сторона "
             f"{f.side.iloc[i]}  exit_boundary {b:.2f}",
             f"  T0 поз {t0[i]}  {pd.Timestamp(ts[t0[i]], tz='UTC')}   "
             f"статус {f.film1_status.iloc[i]}",
             f"  срез q = T0+{age}, d_close = {d:+.2f}  "
             f"(цена ушла ЗА границу, но ни один бар её диапазоном не накрыл)",
             "  поз      время UTC          open     high      low    close   "
             "промеж  накрыл границу"]
        for j in range(t0[i], min(q + 6, last) + 1):
            hit = low[j] <= b <= high[j]
            mark = '  <<< КОНТАКТ' if hit and j > t0[i] else ''
            tag = ' T0' if j == t0[i] else (' q ' if j == q else '   ')
            L.append(f"  {j}{tag} {pd.Timestamp(ts[j], tz='UTC'):%Y-%m-%d %H:%M} "
                     f"{opn[j]:8.2f} {high[j]:8.2f} {low[j]:8.2f} {close[j]:8.2f} "
                     f"  {kind[j]:>4}   {'да' if hit else 'нет':>4}{mark}")
        # что случилось с границей между баром q-1 и q
        prev = close[q - 1]
        L.append(f"  переход: close[q-1] {prev:.2f} -> open[q] {opn[q]:.2f}, "
                 f"граница {b:.2f}; диапазон бара q [{low[q]:.2f}, {high[q]:.2f}] "
                 f"{'НЕ содержит' if not (low[q] <= b <= high[q]) else 'содержит'} границу")
        out.append('\n'.join(L))
    txt = ('\n\n'.join(out))
    p = OUT / f'scenes_negative_{inst}_{terr}_a{age}.txt'
    p.write_text(txt, encoding='utf-8')
    print(txt)
    print('\n', p, 'written')


if __name__ == '__main__':
    a = sys.argv[1:]
    main(a[0] if a else 'NQ', a[1] if len(a) > 1 else 'discovery',
         int(a[2]) if len(a) > 2 else 5)
