#!/usr/bin/env python3
"""Уточнение сцены S-13: какой тип взаимодействия с опорой стоит за результатом.

ПРЕДМЕТ
=======
`argH_first[-10..10] = 10` означает только одно: новый максимум ДОСТУПНОГО окна
поставлен на решающей минуте. Будущая вершина в этот момент неизвестна, и
никакого «сноса» отсюда не следует — это гипотеза, оцениваемая по всем
соответствующим случаям.

Но одно и то же начало объединяет два разных взаимодействия с прежней областью,
и различие выражается имеющимися отношениями точек:

- **закрытие вышло за прежний максимум**: `C[10] > maxH[-10..-1]` — приёмка
  снаружи состоялась к концу решающей минуты;
- **закрытие вернулось внутрь**: `C[10] <= maxH[-10..-1]` — максимум обновлён
  тенью, а закрытие осталось в прежнем диапазоне.

Роль различия в сцене понятна заранее и не сводится к возможности перебрать
больше сочетаний. Второе отношение S-13 (`L[-6] < L[-5]`) проверяется отдельно:
само по себе оно не устанавливает ни опорного минимума, ни разворота.

ПЕРИОДЫ
=======
Объявленное разделение S-13 сохранено (выдвижение до 2024-01-01, оценка после).
Дополнительно показана погодовая картина с ИСПРАВЛЕННОЙ календарной атрибуцией.
Все периоды этой истории уже просматривались в 062–067; независимым
подтверждением ни один не является.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from calendar_utils import year_of, date_key                        # noqa: E402
from candidate_check import POINT, COST                             # noqa: E402
from compact_ordinal import build, compact_events, TFS, BACK, MIN   # noqa: E402
from context_release import contexts, ENTRY, CURSOR                 # noqa: E402


def families(corpus):
    """Зафиксированные семейства. Ничего не подбирается по прибыли."""
    ev = compact_events(corpus, CURSOR)
    col = {int(o): i for i, o in enumerate(corpus.ordinals)}
    C, H, L = corpus.ohlc[:, :, 3], corpus.ohlc[:, :, 1], corpus.ohlc[:, :, 2]
    prior_max = np.max(np.stack([H[:, col[o]] for o in range(-BACK, 0)]), axis=0)
    arg = ev[f'argH_first[-{BACK}..{CURSOR}]'].values == CURSOR
    low = L[:, col[-6]] < L[:, col[-5]]
    out_close = C[:, col[CURSOR]] > prior_max
    return {
        'S-13 как есть': arg & low,
        'обновление максимума': arg,
        'обновление + закрытие снаружи': arg & out_close,
        'обновление + закрытие внутри': arg & ~out_close,
        'обновление + закрытие снаружи + низы': arg & out_close & low,
        'обновление + закрытие внутри + низы': arg & ~out_close & low,
        'без распознавания': np.ones(corpus.n, bool)}


def run_stream(corpus, t0, sel, side, hold):
    col = {int(o): i for i, o in enumerate(corpus.ordinals)}
    O = corpus.ohlc[:, :, 0]
    p = POINT['NQ']
    busy, tr = -1, []
    for i in np.argsort(t0):
        if not sel[i]:
            continue
        a, b = col[ENTRY], col[ENTRY + hold]
        if not (np.isfinite(O[i, a]) and np.isfinite(O[i, b])):
            continue
        t_in = int(t0[i]) + ENTRY * MIN
        if t_in <= busy:
            continue
        tr.append((t_in, float(side * (O[i, b] - O[i, a]) * p)))
        busy = int(t0[i]) + (ENTRY + hold) * MIN
    return tr


def block(tr, cost):
    if len(tr) < 2:
        return {'n': len(tr)}
    g = np.array([t[1] for t in tr])
    ts = np.array([t[0] for t in tr])
    yr = year_of(ts)
    sem = float(g.std(ddof=1) / np.sqrt(len(g)))
    return {'n': int(len(g)), 'days': int(len(np.unique(date_key(ts)))),
            'mean_gross': float(g.mean()), 'sem': sem,
            't': float(g.mean() / sem),
            'mean_net_at_15': float(g.mean() - cost),
            'share_positive': float((g > 0).mean()),
            'by_year': {int(y): {'n': int((yr == y).sum()),
                                 'mean': float(g[yr == y].mean())}
                        for y in np.unique(yr)}}


def diff(a, b):
    if a.get('n', 0) < 2 or b.get('n', 0) < 2:
        return None
    d = a['mean_gross'] - b['mean_gross']
    s = float(np.hypot(a['sem'], b['sem']))
    return {'difference': d, 'sem': s, 't': d / s if s else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--instrument', default='NQ')
    ap.add_argument('--start', default='2020-01-01')
    ap.add_argument('--stop', default='2025-11-01')
    ap.add_argument('--split', default='2024-01-01')
    ap.add_argument('--regime-split', default='2023-01-01')
    ap.add_argument('--holds', nargs='+', type=int, default=[1, 5, 15])
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    cost = COST[a.instrument]
    corpus, t0, side_col, coverage = build(a.instrument, a.start, a.stop, TFS)
    fam = families(corpus)
    ctx, slot_id, _ = contexts(t0, 0)
    outside = ctx['вне слотов']
    s1 = np.datetime64(a.split, 'ns').astype('int64')
    s2 = np.datetime64(a.regime_split, 'ns').astype('int64')
    periods = {'выдвижение (до 2024)': t0 < s1,
               'оценка (2024+)': t0 >= s1,
               'до 2023': t0 < s2,
               '2023 и позже': t0 >= s2}
    side = -1
    res = {'implementation_sha256': hashlib.sha256(
               Path(__file__).read_bytes()).hexdigest(),
           'note': 'все случаи вне слотов публикаций; слоты объяснили 4 сделки '
                   'из 401 и здесь исключены как отдельный, измеренный канал',
           'side': 'продажа, сторона зафиксирована в S-13',
           'families': {}, 'period_difference': {}}
    for fname, fmask in fam.items():
        entry = {'films': int((fmask & outside).sum())}
        for pname, pmask in periods.items():
            entry[pname] = {f'hold_{h}': block(
                run_stream(corpus, t0, fmask & outside & pmask, side, h), cost)
                for h in a.holds}
        res['families'][fname] = entry
    for fname in fam:
        e = res['families'][fname]
        res['period_difference'][fname] = {
            'выдвижение против оценки, hold_5':
                diff(e['выдвижение (до 2024)']['hold_5'],
                     e['оценка (2024+)']['hold_5']),
            'до 2023 против 2023+, hold_5':
                diff(e['до 2023']['hold_5'], e['2023 и позже']['hold_5'])}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1,
                                      default=str), encoding='utf-8')
    for fname, e in res['families'].items():
        print(f'== {fname}  фильмов {e["films"]}')
        for pname in periods:
            v = e[pname]['hold_5']
            if v.get('n', 0) < 2:
                continue
            print(f'   {pname:22s} n={v["n"]:5d} дней {v["days"]:4d} '
                  f'вал {v["mean_gross"]:8.2f} ± {v["sem"]:6.2f} t={v["t"]:5.2f} '
                  f'чист15 {v["mean_net_at_15"]:8.2f}')
        d = res['period_difference'][fname]['до 2023 против 2023+, hold_5']
        if d:
            print(f'   различие до/после 2023: {d["difference"]:.1f} ± '
                  f'{d["sem"]:.1f} (t={d["t"]:.2f})')


if __name__ == '__main__':
    main()
