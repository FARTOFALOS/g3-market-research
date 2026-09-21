#!/usr/bin/env python3
"""095 control — RIZ-специфичность vs общий opening-drive.

Вопрос трейдера (2): пост-открытийный перекос на продолжение — свойство RIZ или
всей ленты? Два контроля:
  A. north/south симметрия RIZ-эффекта в сложенном кадре. Если перекос на
     продолжение положителен и для north (away=вверх), и для south (away=вниз) —
     это настоящее продолжение, а НЕ направленный дрейф индекса (тот дал бы
     противоположные знаки в сложенном кадре).
  B. Плацебо-контроль: обычные минуты БЕЗ свежего RIZ, направление задаёт
     недавнее движение (моментум sign(close[b]-close[b-K])) — аналог «прочь от
     зоны». Сопоставлены по «минутам от открытия». Если общий моментум после
     открытия продолжается так же — RIZ лишь отбирает сцену; если RIZ-away
     продолжается сильнее — RIZ добавляет структуру.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit, prange

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'base/081'))
sys.path.insert(0, str(ROOT / 'base/080'))
from trading import sessions, bar_session_map                          # noqa: E402
from tape import presence_grid, gap_kinds                              # noqa: E402
from paths import load_films                                           # noqa: E402


@njit(parallel=True, cache=True)
def _ctrl(anchors, up, opn, high, low, close, sess, last_bar, kind_gap, last,
          o_close_cont, o_bias, o_since):
    for i in prange(anchors.size):
        t = anchors[i]; u = up[i]; j = t + 1
        if j > last or kind_gap[j] != 0 or sess[j] < 0:
            o_close_cont[i] = -1; continue
        sj = sess[j]; o = opn[j]; lb = last_bar[sj]
        fav = 0.0; tow = 0.0; cur = -1
        for b in range(j, lb + 1):
            if kind_gap[b] != 0:
                break
            if u:
                f = high[b] - o; tw = o - low[b]
            else:
                f = o - low[b]; tw = high[b] - o
            if f > fav:
                fav = f
            if tw > tow:
                tow = tw
            cur = b
        if cur < 0:
            o_close_cont[i] = -1; continue
        dc = (close[cur] - o) if u else (o - close[cur])
        o_close_cont[i] = 1 if dc > 0 else 0
        s = fav + tow
        o_bias[i] = (fav - tow) / s if s > 0 else np.nan


def since_open_bucket(minute):
    so = minute - 570
    return so


def main(inst='NQ', terr='discovery', K=5):
    m = ROOT / 'data/market' / inst
    opn = np.load(m / 'open.npy'); high = np.load(m / 'high.npy')
    low = np.load(m / 'low.npy'); close = np.load(m / 'close.npy')
    ts = np.load(m / 'close_ts_utc_ns.npy')
    grid, glo, ghi = presence_grid(); kind_gap, _ = gap_kinds(inst, grid, glo, ghi)
    st, cl = sessions(); cl = cl.astype(np.int64)
    sess, last_bar = bar_session_map(ts, st, cl)
    last = opn.size - 1
    et = pd.to_datetime(ts, utc=True).tz_convert('America/New_York')
    minute = (et.hour * 60 + et.minute).to_numpy()
    year = et.year.to_numpy()

    # ---- A. RIZ north/south симметрия (из dest) ----
    d = pd.read_parquet(ROOT / f'work/095/dest_{inst}_{terr}.parquet')
    f = load_films(inst, terr)
    etf = pd.to_datetime(f.t0_ts_ns.to_numpy(), utc=True).tz_convert('America/New_York')
    d = d.assign(since=etf.hour.to_numpy() * 60 + etf.minute.to_numpy() - 570, tf=f.tf_minutes.to_numpy())
    a = d[(d.ok == 1) & (d.tf >= 60)].copy()
    win = a[(a.since >= 0) & (a.since < 120)]           # пост-открытие 0-120 мин (пик)
    print(f'=== A. RIZ north/south симметрия, ТФ60+, 0-120 мин после открытия (n={len(win)}) ===')
    for lab, g in [('north (away=вверх)', win[win.north]), ('south (away=вниз)', win[~win.north])]:
        cont = (g.closed_away == 1).mean()
        print(f'  {lab:>20}: n={len(g):>6} close_continuation={cont:.3f}')
    print('  (оба >0.5 => продолжение, не дрейф индекса)')

    # ---- B. плацебо-контроль на обычных минутах ----
    # пул: бары в окне 0-120 мин после открытия, исполнимые, без гэпа, с историей K
    inwin = (minute - 570 >= 0) & (minute - 570 < 120) & (sess >= 0)
    idx = np.nonzero(inwin)[0]
    idx = idx[(idx > K + 1) & (idx < last - 1)]
    rng = np.random.default_rng(3)
    nsamp = min(200000, idx.size)
    anchors = rng.choice(idx, size=nsamp, replace=False)
    # моментум-направление
    up = (close[anchors] > close[anchors - K])
    o_cc = np.full(nsamp, -1, np.int8); o_bias = np.full(nsamp, np.nan, np.float32)
    o_since = np.zeros(nsamp, np.int32)
    t0 = time.time()
    _ctrl(anchors.astype(np.int64), up, opn, high, low, close, sess, last_bar, kind_gap, last,
          o_cc, o_bias, o_since)
    ok = o_cc >= 0
    print(f'\n=== B. плацебо (обычные минуты, момент K={K}), 0-120 мин после открытия ===')
    print(f'  анкеров={ok.sum():,} (walk {round(time.time()-t0,1)}s)')
    print(f'  close_continuation (моментум-сторона) = {(o_cc[ok]==1).mean():.3f}  bias_norm={np.nanmean(o_bias[ok]):+.3f}')
    print(f'  RIZ (ТФ60+, 0-120): close_continuation(away) = {(win.closed_away==1).mean():.3f}  bias_norm={np.nanmean(((win.mfe_away-win.mfe_tow)/(win.mfe_away+win.mfe_tow)).replace([np.inf,-np.inf],np.nan)):+.3f}')
    # до открытия контроль тоже
    pre = (minute - 570 >= -120) & (minute - 570 < 0) & (sess < 0)  # до окна NY нет sess; используем pre-open как есть
    print('\n  ИТОГ: если плацебо ~ RIZ — эффект общий (opening-drive); если RIZ заметно выше — RIZ добавляет структуру.')


if __name__ == '__main__':
    main(sys.argv[2] if len(sys.argv) > 2 else 'NQ', sys.argv[1] if len(sys.argv) > 1 else 'discovery')
