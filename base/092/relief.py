#!/usr/bin/env python3
"""092 — экономический рельеф: агрегация карты, БЕЗ выбора setup.

Показывает структуру conditional executable economics по координатам состояния,
в двух весовых проекциях:
  A) state-occupancy — вес на каждую открытую (riz_id, q). Основная.
  B) scene-balanced  — вес на фильм (среднее по его открытым q). Диагностика:
     не создаётся ли рельеф непропорциональным вкладом нескольких длинных фильмов.
Две проекции не усредняются в один показатель (GO §1).

Деньги считаются ТОЛЬКО по реально открытым позициям (GO §2). unknown —
цензура, из resolved-money исключён, но доля публикуется. Costs применяются
только к открытой позиции; публикуются 0x / 1x / 2x и break-even.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'work/092'
REACHED, TIME_EXIT, UNKNOWN, SEEN_UNCERT, MISSED_TARGET, NO_EXEC, NO_WINDOW = range(7)
COST = {'NQ': 1.00, 'ES': 1.00, 'YM': 4.00}
PV = {'NQ': 20.0, 'ES': 50.0, 'YM': 5.0}
TICK = {'NQ': 0.25, 'ES': 0.25, 'YM': 1.0}


def opened(d):
    return d[d.kind.isin([REACHED, TIME_EXIT, UNKNOWN, SEEN_UNCERT])].copy()


def econ(sub, inst):
    """Экономика подмножества открытых позиций. Возвращает dict чисел."""
    n = len(sub)
    if n == 0:
        return {'n': 0}
    reached = sub.kind == REACHED
    time_x = sub.kind == TIME_EXIT
    unk = sub.kind == UNKNOWN
    resolved = sub[~unk]
    pay = resolved.pay.to_numpy()
    c = COST[inst]
    out = {
        'n_opened': int(n),
        'reach_rate': float(reached.mean()),
        'timeexit_rate': float(time_x.mean()),
        'unknown_rate': float(unk.mean()),
        'tp_unconfirmed_of_reached': float((sub.loc[reached, 'tp_conf'] == 0).mean()) if reached.any() else float('nan'),
        'n_resolved': int(len(resolved)),
        'gross_mean_pts': float(np.nanmean(pay)) if len(pay) else float('nan'),
        'gross_median_pts': float(np.nanmedian(pay)) if len(pay) else float('nan'),
        'net_mean_pts_1x': float(np.nanmean(pay) - c) if len(pay) else float('nan'),
        'net_mean_pts_2x': float(np.nanmean(pay) - 2 * c) if len(pay) else float('nan'),
        'breakeven_cost_pts': float(np.nanmean(pay)) if len(pay) else float('nan'),
        'net_mean_usd_1x': float((np.nanmean(pay) - c) * PV[inst]) if len(pay) else float('nan'),
        'mae_bound_median_pts': float(np.nanmedian(sub.mae_b)),
        'mfe_median_pts': float(np.nanmedian(sub.mfe)),
        'dur_median_bars': float(np.nanmedian(sub.dur)),
    }
    return out


def by_bins(sub, inst, col, bins, labels):
    b = pd.cut(sub[col], bins=bins, labels=labels, right=False)
    rows = []
    for lab, g in sub.groupby(b, observed=True):
        e = econ(g, inst)
        e['bin'] = str(lab)
        rows.append(e)
    return rows


def scene_balanced(sub, inst):
    """Проекция B: сначала свести открытые q к фильму, потом усреднить по фильмам."""
    reached = (sub.kind == REACHED).astype(float)
    unk = (sub.kind == UNKNOWN)
    payr = sub.pay.where(~unk)
    per_film = sub.assign(_r=reached, _pay=payr).groupby('film').agg(
        r=('_r', 'mean'), pay=('_pay', 'mean'), unk=('kind', lambda s: (s == UNKNOWN).mean()))
    c = COST[inst]
    return {
        'n_films': int(len(per_film)),
        'reach_rate_filmmean': float(per_film.r.mean()),
        'gross_mean_pts_filmmean': float(np.nanmean(per_film.pay)),
        'net_mean_pts_1x_filmmean': float(np.nanmean(per_film.pay) - c),
        'unknown_rate_filmmean': float(per_film.unk.mean()),
    }


def main(inst='NQ', terr='discovery'):
    d = pd.read_parquet(OUT / f'map_{inst}_{terr}.parquet')
    full_n = len(d)
    funnel = {int(k): int(v) for k, v in d.kind.value_counts().items()}
    sub = opened(d)
    tk = TICK[inst]
    sub['d_ticks'] = sub.d_close / tk
    sub['birth_ticks'] = sub.birth / tk

    report = {
        'instrument': inst, 'territory': terr,
        'estimand': 'state-occupancy over opened (riz_id,q); участие к b, выход к NY close, стопа нет',
        'q_total': full_n,
        'funnel_counts': {['reached', 'time_exit', 'unknown', 'seen_uncert',
                           'missed_target', 'no_exec', 'no_window'][k]: v
                          for k, v in funnel.items()},
        'opened_total': int(len(sub)),
        'A_state_occupancy_ALL': econ(sub, inst),
        'B_scene_balanced_ALL': scene_balanced(sub, inst),
    }

    # маргинали по трём координатам (state-occupancy)
    report['by_age'] = by_bins(sub, inst, 'age',
        [0, 1, 2, 5, 15, 30, 60, 120, 240, 10**9],
        ['0', '1', '2-4', '5-14', '15-29', '30-59', '60-119', '120-239', '240+'])
    report['by_dclose_ticks'] = by_bins(sub, inst, 'd_ticks',
        [0, 1, 2, 4, 8, 16, 32, 64, 10**9],
        ['0-1c', '1-2c', '2-4c', '4-8c', '8-16c', '16-32c', '32-64c', '64c+'])
    report['by_ttc_min'] = by_bins(sub, inst, 'ttc',
        [0, 30, 60, 120, 240, 480, 720, 10**9],
        ['0-30', '30-60', '60-120', '120-240', '240-480', '480-720', '720+'])
    report['by_birth_ticks'] = by_bins(sub, inst, 'birth_ticks',
        [0, 2, 4, 8, 16, 32, 64, 10**9],
        ['0-2c', '2-4c', '4-8c', '8-16c', '16-32c', '32-64c', '64c+'])

    # 2D рельеф d_close x ttc (основной интересующий срез: близко к b + запас времени)
    dcut = pd.cut(sub.d_ticks, [0, 2, 4, 8, 16, 32, 10**9],
                  labels=['0-2c', '2-4c', '4-8c', '8-16c', '16-32c', '32c+'], right=False)
    tcut = pd.cut(sub.ttc, [0, 60, 120, 240, 480, 10**9],
                  labels=['0-60', '60-120', '120-240', '240-480', '480+'], right=False)
    grid = []
    for (dl, tl), g in sub.groupby([dcut, tcut], observed=True):
        e = econ(g, inst)
        e['d_close'] = str(dl); e['ttc'] = str(tl)
        grid.append(e)
    report['grid_dclose_x_ttc'] = grid

    (OUT / f'relief_{inst}_{terr}.json').write_text(json.dumps(report, indent=1, ensure_ascii=False))

    # --- печать компактно ---
    def line(e, key='bin'):
        return (f"  {e.get(key,''):>10} | n={e['n_opened']:>8} reach={e['reach_rate']:.3f} "
                f"unk={e['unknown_rate']:.3f} | gross={e['gross_mean_pts']:+.3f} "
                f"net1x={e['net_mean_pts_1x']:+.3f}pt (${e['net_mean_usd_1x']:+.1f}) "
                f"| MAEb_med={e['mae_bound_median_pts']:.2f} dur_med={e['dur_median_bars']:.0f}")
    A = report['A_state_occupancy_ALL']; B = report['B_scene_balanced_ALL']
    print(f"\n=== {inst}/{terr} ЭКОНОМИЧЕСКИЙ РЕЛЬЕФ (стопа нет) ===")
    print(f"q всего {full_n:,} | открыто {len(sub):,} | воронка вне сделки "
          f"{full_n-len(sub):,}")
    print(f"\n[A state-occupancy, все открытые]")
    print(f"  reach={A['reach_rate']:.4f} time_exit={A['timeexit_rate']:.4f} unknown={A['unknown_rate']:.4f}")
    print(f"  gross_mean={A['gross_mean_pts']:+.4f}pt  net@1x={A['net_mean_pts_1x']:+.4f}pt (${A['net_mean_usd_1x']:+.2f})  net@2x={A['net_mean_pts_2x']:+.4f}pt")
    print(f"  break-even cost={A['breakeven_cost_pts']:.4f}pt  tp_unconfirmed(of reached)={A['tp_unconfirmed_of_reached']:.4f}")
    print(f"  MAE_bound med={A['mae_bound_median_pts']:.2f}pt  MFE med={A['mfe_median_pts']:.2f}pt  dur med={A['dur_median_bars']:.0f} bars")
    print(f"\n[B scene-balanced, вес=фильм] reach={B['reach_rate_filmmean']:.4f} "
          f"gross={B['gross_mean_pts_filmmean']:+.4f}pt net@1x={B['net_mean_pts_1x_filmmean']:+.4f}pt "
          f"unk={B['unknown_rate_filmmean']:.4f}  (фильмов {B['n_films']:,})")
    for name in ['by_age', 'by_dclose_ticks', 'by_ttc_min', 'by_birth_ticks']:
        print(f"\n[{name}]")
        for e in report[name]:
            print(line(e))
    print(f"\n[grid d_close x ttc]  (reach / net@1x pt / n)")
    dord = ['0-2c', '2-4c', '4-8c', '8-16c', '16-32c', '32c+']
    tord = ['0-60', '60-120', '120-240', '240-480', '480+']
    gmap = {(e['d_close'], e['ttc']): e for e in grid}
    hdr = ' ' * 8 + ''.join(f'{t:>22}' for t in tord)
    print(hdr)
    for dl in dord:
        cells = []
        for tl in tord:
            e = gmap.get((dl, tl))
            if e and e['n_opened'] > 0:
                cells.append(f"{e['reach_rate']:.2f}/{e['net_mean_pts_1x']:+.2f}/{e['n_opened']//1000}k")
            else:
                cells.append('-')
        print(f'{dl:>8}' + ''.join(f'{c:>22}' for c in cells))


if __name__ == '__main__':
    terr = sys.argv[1] if len(sys.argv) > 1 else 'discovery'
    for inst in (sys.argv[2:] or ['NQ']):
        main(inst, terr)
