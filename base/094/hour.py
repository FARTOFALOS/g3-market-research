#!/usr/bin/env python3
"""094 — продолжение зажигания прочь от зоны, стоп = бар T0, срез по часу T0 (ET).

Новая сцена (инициатива агента; METHOD §K — новый вопрос ценнее переработки
исключённого возврата). 081 §8 явно НЕ входил в continuation/пост-контактный вход;
086 нашёл, что на этой вилке благоприятна сторона ПРОДОЛЖЕНИЯ (прочь от b), но
перенос не установлен. Здесь другой оператор той же сильной стороны:

Событие: T0 (зажигание RIZ), prefix, заморожено 080A.
Геометрия (проверена): north — цена НАД зоной, exit_boundary=zone_top; продолжение
= ВВЕРХ (long), прочь от зоны; adverse — вниз к b. south — зеркально (short вниз).
Стоп СТРУКТУРНЫЙ: противоположный край бара зажигания T0 (north: low[t0]-тик).
Цель: не выдумана — выход к закрытию NY (structural time exit), стоп — бар T0.
Координата: час T0 по America/New_York (session time; S-07 показал сигнал времени).

Вход = open[t0+1] (next-open, как 081). Окно 03:00 ET → закрытие XNYS; вне окна
или неисполнимо — не сделка (воронка отдельно). unknown при потере наблюдаемости.
Смотрим по часу T0: reach-stop, NY-close drift, MFE(продолжение)/MAE, экономику.
Пороги не вводятся — сначала рельеф по часу.
"""
from __future__ import annotations
import os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit, prange

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = Path(os.environ.get('G3_094_OUT', str(ROOT / 'work/094')))
sys.path.insert(0, str(ROOT / 'base/081'))
sys.path.insert(0, str(ROOT / 'base/080'))
from trading import sessions, bar_session_map                          # noqa: E402
from tape import presence_grid, gap_kinds                              # noqa: E402
from paths import load_films                                           # noqa: E402

TICK = {'NQ': 0.25, 'ES': 0.25, 'YM': 1.0}
COST = {'NQ': 1.00, 'ES': 1.00, 'YM': 4.00}
PV = {'NQ': 20.0, 'ES': 50.0, 'YM': 5.0}
STOPPED, TIME_EXIT, UNKNOWN, NO_EXEC = 0, 1, 2, 3


@njit(parallel=True, cache=True)
def _walk(t0s, Fs, es, north, high, low, opn, close, sess, last_bar, kind_gap, tick, last,
          o_kind, o_stop, o_pay, o_mfe, o_mae, o_dur, close_ns, ts, o_ttc):
    # НЕТ утечки: удержание до закрытия NY по РЕАЛЬНЫМ барам; F (граница фильма)
    # как граница выхода НЕ используется; цензура только на настоящем гэпе ленты.
    for i in prange(t0s.size):
        t0 = t0s[i]; up = north[i]  # up=north -> продолжение ВВЕРХ (long)
        j = t0 + 1
        if j > last or kind_gap[j] != 0 or sess[j] < 0:
            o_kind[i] = NO_EXEC; continue
        sj = sess[j]; o = opn[j]
        stop = (low[t0] - tick) if up else (high[t0] + tick)  # структурный стоп = бар T0
        sd = (o - stop) if up else (stop - o)
        if sd <= 0.0:
            o_kind[i] = NO_EXEC; continue
        o_ttc[i] = (close_ns[sj] - ts[j]) / 6.0e10
        lb = last_bar[sj]
        o_stop[i] = sd
        res = UNKNOWN; pay = np.nan; dur = 0; run_fav = 0.0; run_adv = 0.0
        for b in range(j, lb + 1):
            if kind_gap[b] != 0:            # реальный разрыв ленты — исход неизвестен
                res = UNKNOWN; dur = b - 1 - t0; break
            if up:
                fav = high[b] - o; adv = o - low[b]
                open_through = opn[b] <= stop; stop_touch = low[b] <= stop
            else:
                fav = o - low[b]; adv = high[b] - o
                open_through = opn[b] >= stop; stop_touch = high[b] >= stop
            if fav > run_fav:
                run_fav = fav
            if adv > run_adv:
                run_adv = adv
            if open_through:
                fill = opn[b]
                pay = (fill - o) if up else (o - fill)
                res = STOPPED; dur = b - t0; break
            if stop_touch:
                pay = -sd; res = STOPPED; dur = b - t0; break
            if b == lb:
                px = close[lb]
                pay = (px - o) if up else (o - px)
                res = TIME_EXIT; dur = lb - t0
        o_kind[i] = res; o_pay[i] = pay; o_mfe[i] = run_fav; o_mae[i] = run_adv; o_dur[i] = dur


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
    F = f.certified_fresh_until_pos_strict.to_numpy().astype(np.int64)
    e = f.exit_boundary.to_numpy().astype(np.float64)
    north = (f.side.to_numpy() == 'north')
    n = len(f)
    z = lambda dt, v=0: np.full(n, v, dtype=dt)
    o = dict(kind=z(np.int8, NO_EXEC), stop=z(np.float32, np.nan), pay=z(np.float32, np.nan),
             mfe=z(np.float32, np.nan), mae=z(np.float32, np.nan), dur=z(np.int32),
             ttc=z(np.float32, np.nan))
    last = opn.size - 1
    t = time.time()
    _walk(t0, F, e, north, high, low, opn, close, sess, last_bar, kind_gap, TICK[inst], last,
          o['kind'], o['stop'], o['pay'], o['mfe'], o['mae'], o['dur'], cl, ts, o['ttc'])
    secs = round(time.time() - t, 2)
    et = pd.to_datetime(f.t0_ts_ns.to_numpy(), utc=True).tz_convert('America/New_York')
    r = pd.DataFrame({'north': north, 'kind': o['kind'], 'stop_dist': o['stop'], 'pay': o['pay'],
        'mfe': o['mfe'], 'mae': o['mae'], 'dur': o['dur'], 'ttc': o['ttc'],
        'et_hour': et.hour.astype(np.int8), 'day': (f.t0_ts_ns.to_numpy() // 86400000000000),
        'tf': f.tf_minutes.to_numpy().astype(np.int32)})
    r.attrs['seconds'] = secs
    return r


def report(r, inst):
    C = COST[inst]
    tr = r[r.kind.isin([STOPPED, TIME_EXIT])].copy(); tr['net'] = tr.pay - C
    print(f'=== 094 продолжение зажигания, стоп=бар T0 ({inst} discovery) ===')
    print(f'население={len(r):,}  сделок={len(tr):,}  no_exec={(r.kind==NO_EXEC).sum():,}  unknown={(r.kind==UNKNOWN).sum():,}')
    print(f'ОБЩЕЕ: net/сделку={tr.net.mean():+.3f}  median={tr.net.median():+.2f}  stop_med={tr.stop_dist.median():.2f}  MFE_med={tr.mfe.median():.2f}  MAE_med={tr.mae.median():.2f}  stop-rate={(tr.kind==STOPPED).mean():.3f}')
    dv = tr.groupby('day').net.mean()
    fdp = (tr.groupby('day').net.sum() > 0).mean()
    print(f'day-level(all): mean={dv.mean():+.3f}  frac_days+={fdp:.3f}')
    print('\nпо часу T0 (ET):  hour | n | net/сделку | median | day-mean | frac_days+ | stop% | MFE_med')
    for h, g in tr.groupby('et_hour'):
        if len(g) < 200: continue
        dvh = g.groupby('day').net.mean()
        fdph = (g.groupby('day').net.sum() > 0).mean()
        print(f'   {h:>2}:00 | {len(g):>6} | {g.net.mean():+.3f} | {g.net.median():+.2f} | {dvh.mean():+.3f} | {fdph:.3f} | {(g.kind==STOPPED).mean():.3f} | {g.mfe.median():.1f}')


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    terr = sys.argv[1] if len(sys.argv) > 1 else 'discovery'
    inst = sys.argv[2] if len(sys.argv) > 2 else 'NQ'
    r = build(inst, terr)
    r.to_parquet(OUT / f'hour_{inst}_{terr}.parquet', index=False)
    print(f'walk {r.attrs["seconds"]}s\n')
    report(r, inst)
