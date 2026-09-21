#!/usr/bin/env python3
"""096 replay — хронологический прогон замороженного RULE_V1 по всей ленте NQ.

СТАТУС, ОБЪЯВЛЕННЫЙ ДО ЗАПУСКА (иначе прогон не имеет смысла):

    Это `historical chronological replication`, НЕ independent validation и НЕ
    proof of transfer. Оператор создан системой, которая уже много месяцев
    работала со всей лентой 2006-2026: слепота исполнителя не делает старый год
    новой информацией, и случайный выбор года постфактум независимости не
    возвращает. Отвечает на вопрос «как эта штука вела себя исторически», а не
    «нашли ли мы переносимое свойство рынка».

    Что здесь НОВОГО: исходы вилки УСЛОВНО НА RULE_V1 на ранней эпохе
    (NQ_development 2006-2018) ни разу не читались — часть II шла на NQ_search
    2020-2025. Из ранней эпохи брались только 60 сцен для разметки, без исходов.
    Территория экспонирована на уровне популяции (086 мерил там общий X), но не
    условно на этом операторе.

    Терминал части II (`UNRESOLVED AT DECLARED PRECISION`) этим прогоном НЕ
    отменяется ни в какую сторону. Положительный ранний результат был бы
    репликацией, а не установлением эффекта.

ЧЕГО ЗДЕСЬ НЕТ: исполнимой экономики. FREEZE_096_PART_II §7 открывает её только
после положительного структурного результата; он не положителен. P&L, просадки и
execution не считаются — это было бы отменой собственной заморозки.

    python -B replay.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'work/096'
sys.path.insert(0, str(ROOT / 'base/086'))
sys.path.insert(0, str(HERE))
import fork086 as F                                                          # noqa: E402
from part2 import operator                                                   # noqa: E402

PERIODS = ('NQ_development', 'NQ_evaluation', 'NQ_holdout')
SEED, NBOOT, L = 20260916, 2000, 5


def build():
    frames = []
    for per in PERIODS:
        g = F.geometry(per)
        ok = g.in_window.to_numpy() & np.isfinite(g.ttc.to_numpy()) & (g.sigma.to_numpy() > 0)
        g = g[ok].reset_index(drop=True)
        code, _ = F.walk(g)
        sig = operator(g, 'NQ')
        unk = np.isin(code, (F.SAME, F.LOST, F.EDGE))
        frames.append(pd.DataFrame({
            'period': per, 'day': g.t0_day.to_numpy(), 'p0': g.p0.to_numpy(),
            'iso': (sig == 0),
            'y_lo': np.where(code == F.WIN, 1.0, 0.0),
            'y_hi': np.where(code == F.WIN, 1.0, np.where(unk, 1.0, 0.0)),
            'year': pd.to_datetime(g.t0_day.to_numpy(), unit='D').year}))
    d = pd.concat(frames, ignore_index=True)
    d['x_lo'] = d.y_lo - d.p0
    d['x_hi'] = d.y_hi - d.p0
    return d


def delta(d, lo, hi, B=NBOOT):
    """identified set + внешний 95% CI: на каждом дневном репликате оба конца."""
    a, b = d.iso.to_numpy(), (~d.iso).to_numpy()
    if a.sum() < 30 or b.sum() < 30:
        return None
    ud, di = np.unique(d.day.to_numpy(), return_inverse=True)
    D = len(ud)
    LO, HI = d[lo].to_numpy(), d[hi].to_numpy()

    def pt(w):
        na, nb = w[a].sum(), w[b].sum()
        if na == 0 or nb == 0:
            return np.nan, np.nan
        return ((w[a] * LO[a]).sum() / na - (w[b] * HI[b]).sum() / nb,
                (w[a] * HI[a]).sum() / na - (w[b] * LO[b]).sum() / nb)

    plo, phi = pt(np.ones(len(d)))
    rng = np.random.default_rng(SEED)
    nb_ = int(np.ceil(D / L))
    st = rng.integers(0, max(D - L + 1, 1), size=(B, nb_))
    idx = (st[:, :, None] + np.arange(L)[None, None, :]).reshape(B, -1)[:, :D]
    reps = np.array([pt(np.bincount(idx[i], minlength=D)[di]) for i in range(B)])
    return plo, phi, float(np.nanquantile(reps[:, 0], .025)), float(np.nanquantile(reps[:, 1], .975))


def row(d, lbl, res):
    r = delta(d, 'y_lo', 'y_hi')
    rx = delta(d, 'x_lo', 'x_hi')
    if r is None:
        print(f'  {lbl:>14}  N={len(d):>6}  — мало наблюдений в плече')
        return
    out = {'label': lbl, 'N': int(len(d)), 'days': int(d.day.nunique()),
           'share_iso': round(float(d.iso.mean()), 4),
           'dY': [round(r[0], 4), round(r[1], 4)], 'dY_ci': [round(r[2], 4), round(r[3], 4)],
           'dX': [round(rx[0], 4), round(rx[1], 4)], 'dX_ci': [round(rx[2], 4), round(rx[3], 4)]}
    res.append(out)
    sign = '+' if rx[0] > 0 else ('-' if rx[1] < 0 else '0')
    excl = '*' if rx[2] * rx[3] > 0 else ' '
    print(f'  {lbl:>14}  N={len(d):>6} дней={d.day.nunique():>4} iso={d.iso.mean():.3f} | '
          f'ΔY [{r[0]:+.3f};{r[1]:+.3f}] | ΔX [{rx[0]:+.3f};{rx[1]:+.3f}] '
          f'CI [{rx[2]:+.3f};{rx[3]:+.3f}] {sign}{excl}')


def main():
    d = build()
    res = []
    print(f'вся лента: N={len(d)} дней={d.day.nunique()} годы {d.year.min()}-{d.year.max()}\n')
    print('=== по эпохам (ранняя НЕ читалась условно на этом операторе) ===')
    for per in PERIODS:
        row(d[d.period == per], per.replace('NQ_', ''), res)
    print('\n=== по годам ===')
    for y, g in d.groupby('year'):
        row(g, str(y), res)
    print('\n  * = внешний 95% CI не включает ноль')
    (OUT / 'replay.json').write_text(json.dumps(
        {'status': 'historical chronological replication, NOT independent validation',
         'note': 'executable economics deliberately not computed: FREEZE_096_PART_II §7 '
                 'opens it only after a positive structural result, which did not occur',
         'rows': res}, ensure_ascii=False, indent=1), encoding='utf-8')
    print('\n->', OUT / 'replay.json')


if __name__ == '__main__':
    main()
