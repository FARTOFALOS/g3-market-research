#!/usr/bin/env python3
"""095 econ (шаг 1) — экономика RIZ-обусловленного пост-открытийного продолжения.

Сцена (095): свежий RIZ (ТФ>=60), зажигание в первые 120 мин после открытия NY,
вход open[t0+1] в сторону ПРОЧЬ от собственной b (продолжение). Выход —
структурный, не выдуманный: закрытие сессии NY (disp_close уже сложен так, что
+ = продолжение). Costs 1.0 пт круг. Единица — день. Territory: discovery
2006-2018 и НЕтронутая evaluation 2019-2026 (later-epoch, известная экспозиция).

Также sensitivity: структурный стоп X (adverse=mfe_tow) — кривая читается, не
подбирается максимум. Пороги сцены (ТФ>=60, since 0-120) заморожены до счёта
экономики (взяты из карты 095, не из P&L).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'base/081'))
from paths import load_films  # noqa: E402
COST = 1.0


def prep(terr):
    d = pd.read_parquet(ROOT / f'work/095/dest_NQ_{terr}.parquet')
    f = load_films('NQ', terr)
    et = pd.to_datetime(f.t0_ts_ns.to_numpy(), utc=True).tz_convert('America/New_York')
    d = d.assign(since=et.hour.to_numpy() * 60 + et.minute.to_numpy() - 570,
                 tf=f.tf_minutes.to_numpy(), year=et.year.to_numpy(),
                 day=(f.t0_ts_ns.to_numpy() // 86400000000000))
    a = d[(d.ok == 1) & (d.tf >= 60) & (d.since >= 0) & (d.since < 120)].copy()
    a['net'] = a.disp_close - COST                       # удержание до закрытия
    return a


def econ(a, lbl):
    net = a.net.to_numpy()
    dv = a.groupby('day').net.mean()
    dsum = a.groupby('day').net.sum()
    print(f'{lbl}: сделок={len(a):,} дней={dv.size:,}')
    print(f'  per-trade net={net.mean():+.3f}pt median={np.median(net):+.2f}  win(disp>0)={(a.disp_close>0).mean():.3f}')
    print(f'  DAY-UNIT mean={dv.mean():+.3f}pt  median={dv.median():+.2f}  frac_days+={ (dsum>0).mean():.3f}')
    # bootstrap CI на дне
    rng = np.random.default_rng(5); n = dv.size
    bm = np.array([dv.sample(n, replace=True, random_state=int(rng.integers(1e9))).mean() for _ in range(3000)])
    print(f'  day-mean 95% CI=[{np.percentile(bm,2.5):+.3f}, {np.percentile(bm,97.5):+.3f}]')
    return dv.mean()


def stop_curve(a, lbl):
    # структурный стоп X: если adverse(mfe_tow)>=X -> -X, иначе disp_close; выход к close
    print(f'  {lbl} стоп-кривая (adverse=mfe_tow):')
    mt = a.mfe_tow.to_numpy(); dc = a.disp_close.to_numpy(); day = a.day.to_numpy()
    for X in [5, 10, 15, 20, 30, 1e9]:
        pay = np.where(mt >= X, -X, dc) - COST
        dfp = pd.DataFrame({'day': day, 'p': pay})
        dv = dfp.groupby('day').p.mean()
        fp = (dfp.groupby('day').p.sum() > 0).mean()
        lab = 'no-stop' if X > 1e8 else f'{X:g}'
        print(f'     стоп={lab:>7}: per-trade={pay.mean():+.3f} day-mean={dv.mean():+.3f} frac+={fp:.3f}')


if __name__ == '__main__':
    dis = prep('discovery'); ev = prep('evaluation')
    print('=== ШАГ 1: экономика пост-открытийного продолжения (удержание до close NY, cost 1.0) ===\n')
    econ(dis, 'DISCOVERY 2006-2018')
    print()
    econ(ev, 'EVALUATION 2019-2026 (later-epoch, известная экспозиция)')
    print('\n--- стоп-кривые (структурный стоп из adverse, не подбор максимума) ---')
    stop_curve(dis, 'DISCOVERY')
    stop_curve(ev, 'EVALUATION')
    print('\n--- по годам evaluation (day-unit mean, hold-to-close) ---')
    for y, g in ev.groupby('year'):
        dv = g.groupby('day').net.mean()
        fp = (g.groupby('day').net.sum() > 0).mean()
        print(f'  {y}: сделок={len(g):>5} day-mean={dv.mean():+.3f} frac+={fp:.3f}')
