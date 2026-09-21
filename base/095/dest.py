#!/usr/bin/env python3
"""095 — БЕЗУСЛОВНАЯ карта достижения: пусть база покажет, куда цена доходит чаще.

Слово трейдера: не гонять своё предположение (вход+стоп) по базе, а наложить все
фильмы трафаретом и дать базе показать, куда она доходит чаще всего. Здесь нет
сделки, нет стопа, нет выбранного направления.

Складываем все зажигания в ОБЩИЙ кадр: `+` = прочь от собственной `b`
(continuation), `−` = к `b` (reversion). Отсчёт от первой доступной цены
open[t0+1]. Смотрим распределение смещения по горизонтам (мин) и максимальные
достижения в каждую сторону до закрытия NY. Раскладка в пунктах и в долях ширины
зоны |far−b|. Цензура: реальный гэп ленты или конец сессии (без переката и без
загляда за F — F не используется вовсе).

geometry (проверено 080A): north — цена НАД зоной, b=exit=zone_top; away=ВВЕРХ.
south — под зоной; away=ВНИЗ.
"""
from __future__ import annotations
import os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit, prange

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = Path(os.environ.get('G3_095_OUT', str(ROOT / 'work/095')))
sys.path.insert(0, str(ROOT / 'base/081'))
sys.path.insert(0, str(ROOT / 'base/080'))
from trading import sessions, bar_session_map                          # noqa: E402
from tape import presence_grid, gap_kinds                              # noqa: E402
from paths import load_films                                           # noqa: E402

HOR = np.array([5, 15, 30, 60, 120, 240], np.int64)


@njit(parallel=True, cache=True)
def _dest(t0s, es, north, high, low, opn, close, sess, last_bar, kind_gap, last,
          hor, disp, mfe_away, mfe_tow, closed_away, ok, dclose_close, reach_bars):
    nh = hor.size
    for i in prange(t0s.size):
        t0 = t0s[i]; up = north[i]
        j = t0 + 1
        if j > last or kind_gap[j] != 0 or sess[j] < 0:
            ok[i] = 0; continue
        sj = sess[j]; o = opn[j]; lb = last_bar[sj]
        ok[i] = 1
        fav = 0.0; tow = 0.0
        # смещение на фиксированных горизонтах (если бар доступен и без гэпа до него)
        gap_hit = False
        # маркеры горизонтов
        for hk in range(nh):
            disp[i, hk] = np.nan
        end = lb
        cur = -1
        for b in range(j, lb + 1):
            if kind_gap[b] != 0:
                end = b - 1; gap_hit = True; break
            if up:
                f = high[b] - o; t = o - low[b]; d = close[b] - o
            else:
                f = o - low[b]; t = high[b] - o; d = o - close[b]
            if f > fav:
                fav = f
            if t > tow:
                tow = t
            off = b - j
            for hk in range(nh):
                if off == hor[hk]:
                    disp[i, hk] = d
            cur = b
        mfe_away[i] = fav; mfe_tow[i] = tow
        # смещение на закрытии (последний наблюдённый бар)
        if cur >= 0:
            if up:
                dclose_close[i] = close[cur] - o
            else:
                dclose_close[i] = o - close[cur]
            closed_away[i] = 1 if dclose_close[i] > 0 else 0
            reach_bars[i] = cur - j


def build(inst='NQ', terr='discovery'):
    m = ROOT / 'data/market' / inst
    opn = np.load(m / 'open.npy'); high = np.load(m / 'high.npy')
    low = np.load(m / 'low.npy'); close = np.load(m / 'close.npy')
    ts = np.load(m / 'close_ts_utc_ns.npy')
    grid, glo, ghi = presence_grid(); kind_gap, _ = gap_kinds(inst, grid, glo, ghi)
    st, cl = sessions(); cl = cl.astype(np.int64)
    sess, last_bar = bar_session_map(ts, st, cl)
    f = load_films(inst, terr)
    t0 = f.t0_spine_pos.to_numpy().astype(np.int64)
    e = f.exit_boundary.to_numpy().astype(np.float64)
    north = (f.side.to_numpy() == 'north')
    zonew = (f.zone_top - f.zone_bottom).to_numpy().astype(np.float64)
    n = len(f); last = opn.size - 1
    disp = np.full((n, HOR.size), np.nan, np.float32)
    mfe_away = np.zeros(n, np.float32); mfe_tow = np.zeros(n, np.float32)
    closed_away = np.full(n, -1, np.int8); ok = np.zeros(n, np.int8)
    dclose = np.full(n, np.nan, np.float32); reach = np.full(n, -1, np.int32)
    t = time.time()
    _dest(t0, e, north, high, low, opn, close, sess, last_bar, kind_gap, last,
          HOR, disp, mfe_away, mfe_tow, closed_away, ok, dclose, reach)
    secs = round(time.time() - t, 2)
    d = pd.DataFrame({'ok': ok, 'north': north, 'zonew': zonew.astype(np.float32),
        'mfe_away': mfe_away, 'mfe_tow': mfe_tow, 'closed_away': closed_away,
        'disp_close': dclose, 'reach_bars': reach, 'et_hour': pd.to_datetime(
            f.t0_ts_ns.to_numpy(), utc=True).tz_convert('America/New_York').hour.astype(np.int8)})
    for hk, h in enumerate(HOR):
        d[f'disp_{h}'] = disp[:, hk]
    d.attrs['seconds'] = secs
    return d


def q(a, p):
    a = a[~np.isnan(a)]
    return np.quantile(a, p) if len(a) else np.nan


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    terr = sys.argv[1] if len(sys.argv) > 1 else 'discovery'
    inst = sys.argv[2] if len(sys.argv) > 2 else 'NQ'
    d = build(inst, terr)
    d.to_parquet(OUT / f'dest_{inst}_{terr}.parquet', index=False)
    a = d[d.ok == 1].copy()
    print(f'=== 095 БЕЗУСЛОВНАЯ огибающая от зажигания ({inst} discovery) ===')
    print(f'зажиганий с исполнимым входом: {len(a):,}  (+ = ПРОЧЬ от b, − = К b)')
    print('\nСМЕЩЕНИЕ (пункты) по горизонтам — квантили, база наложена:')
    print(' гор,мин |   p10 |   p25 | median|   p75 |   p90 | доля>0(away) | n')
    for h in HOR:
        c = a[f'disp_{h}'].to_numpy()
        cc = c[~np.isnan(c)]
        pos = (cc > 0).mean() if len(cc) else np.nan
        print(f'  {h:>5} | {q(c,.1):+6.2f}|{q(c,.25):+6.2f}|{np.nanmedian(c):+6.2f}|{q(c,.75):+6.2f}|{q(c,.9):+6.2f}| {pos:.3f} | {len(cc)}')
    print('\nМАКС ДОСТИЖЕНИЕ до закрытия NY (пункты): прочь vs к b')
    print(f'  MFE away: median={np.median(a.mfe_away):.2f} p75={q(a.mfe_away.to_numpy(),.75):.2f} p90={q(a.mfe_away.to_numpy(),.9):.2f}')
    print(f'  MFE toward b: median={np.median(a.mfe_tow):.2f} p75={q(a.mfe_tow.to_numpy(),.75):.2f} p90={q(a.mfe_tow.to_numpy(),.9):.2f}')
    print(f'  закрытие сессии на стороне AWAY: {(a.closed_away==1).mean():.3f}  disp_close median={np.nanmedian(a.disp_close):+.2f}')
    print('\nВ ДОЛЯХ ШИРИНЫ ЗОНЫ |far-b| (нормировка на масштаб RIZ):')
    zw = a.zonew.to_numpy()
    for lbl, arr in [('MFE away/zone', a.mfe_away.to_numpy()/zw), ('MFE tow/zone', a.mfe_tow.to_numpy()/zw),
                     ('disp_close/zone', a.disp_close.to_numpy()/zw)]:
        print(f'  {lbl:>16}: median={np.nanmedian(arr):+.2f} p25={q(arr,.25):+.2f} p75={q(arr,.75):+.2f}')
    print('\nдоля away на закрытии по часу T0 (ET):')
    for h,g in a.groupby('et_hour'):
        if len(g)<500: continue
        print(f'  {h:>2}:00 n={len(g):>6} away_close={ (g.closed_away==1).mean():.3f} disp_close_med={np.nanmedian(g.disp_close):+.2f} mfe_away_med={np.median(g.mfe_away):.1f} mfe_tow_med={np.median(g.mfe_tow):.1f}')
