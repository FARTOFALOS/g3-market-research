#!/usr/bin/env python3
"""095 deep-time — углубить ЕДИНСТВЕННЫЙ проявившийся канал (время сессии).

Не расширяем признаки. Углубляем время и отделяем НАПРАВЛЕННОЕ состояние от
перераспределения ВОЛАТИЛЬНОСТИ. Три вопроса (по плану трейдера):
  A. где перекос возникает и насколько локален (15-мин у открытия);
  B. сохраняется ли по годам;
  C. это направленность или просто больше волатильности?
     bias_norm = (MFE_away - MFE_toward)/(MFE_away+MFE_toward) ∈ [-1,1] —
     убирает масштаб: >0 = реальный перекос ПРОЧЬ независимо от размера хода.
     vol = (MFE_away+MFE_toward) — уровень волатильности сцены.

Данные: work/095/dest_*.parquet (per-film, порядок = load_films discovery).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'base/081'))
from paths import load_films  # noqa: E402


def main(inst='NQ', terr='discovery'):
    d = pd.read_parquet(ROOT / f'work/095/dest_{inst}_{terr}.parquet')
    f = load_films(inst, terr)
    et = pd.to_datetime(f.t0_ts_ns.to_numpy(), utc=True).tz_convert('America/New_York')
    d = d.assign(minute=et.hour * 60 + et.minute, year=et.year, hour=et.hour)
    a = d[d.ok == 1].copy()
    s = a.mfe_away + a.mfe_tow
    a['bias'] = np.where(s > 0, (a.mfe_away - a.mfe_tow) / s, np.nan)
    a['vol'] = s
    a['away'] = (a.closed_away == 1).astype(float)

    print(f'=== 095 deep-time {inst}/{terr}: {len(a):,} зажиганий ===')
    print('bias_norm = направленность прочь за вычетом масштаба (0=симметрия)\n')

    print('A. 15-мин корзины 07:00–12:00 ET (локальность у открытия 09:30):')
    print('  время | n | close_away | disp_close_med | bias_norm | vol_med')
    for m0 in range(7 * 60, 12 * 60, 15):
        g = a[(a.minute >= m0) & (a.minute < m0 + 15)]
        if len(g) < 300:
            continue
        hh, mm = divmod(m0, 60)
        print(f'  {hh:02d}:{mm:02d} | {len(g):>5} | {g.away.mean():.3f} | {np.nanmedian(g.disp_close):+6.2f} | {np.nanmean(g.bias):+.3f} | {np.median(g.vol):5.1f}')

    print('\nB. устойчивость по годам (час T0 = 9,10,11 ET, объединённо):')
    win = a[a.hour.isin([9, 10, 11])]
    print('  year | n | close_away | bias_norm | vol_med')
    for y, g in win.groupby('year'):
        print(f'  {y} | {len(g):>5} | {g.away.mean():.3f} | {np.nanmean(g.bias):+.3f} | {np.median(g.vol):5.1f}')
    print(f'  ВСЕ 9-11: close_away={win.away.mean():.3f} bias_norm={np.nanmean(win.bias):+.3f}')
    off = a[a.hour.isin([3, 4, 5, 14])]
    print(f'  оффчас 3-5+14: close_away={off.away.mean():.3f} bias_norm={np.nanmean(off.bias):+.3f} vol_med={np.median(off.vol):.1f}')

    print('\nC. направленность vs волатильность по часам:')
    print('  hour | n | close_away | bias_norm | vol_med')
    for h, g in a.groupby('hour'):
        if len(g) < 500:
            continue
        print(f'   {h:>2} | {len(g):>6} | {g.away.mean():.3f} | {np.nanmean(g.bias):+.3f} | {np.median(g.vol):5.1f}')

    # boundary: одинаковое состояние (положение к b + час) — разное ли будущее?
    print('\nГраница: within (hour, d_close_bin) — насколько close_away отклоняется от 0.5')
    a['pos'] = pd.cut((a.mfe_away * 0 + 1), [0, 2])  # placeholder if no d_close; use zonew scale
    # используем нормированное стартовое положение недоступно тут; показываем hour x vol-tercile
    a['volt'] = pd.qcut(a.vol, 3, labels=['low', 'mid', 'high'])
    for h in [5, 9, 11, 14]:
        g = a[a.hour == h]
        if len(g) < 500:
            continue
        row = ' '.join(f'{vt}:{gg.away.mean():.3f}(n{len(gg)})' for vt, gg in g.groupby('volt', observed=True))
        print(f'  hour {h:>2}: close_away по терцилям волатильности → {row}')


if __name__ == '__main__':
    main(sys.argv[2] if len(sys.argv) > 2 else 'NQ', sys.argv[1] if len(sys.argv) > 1 else 'discovery')
