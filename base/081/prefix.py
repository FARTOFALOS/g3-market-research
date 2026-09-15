#!/usr/bin/env python3
"""СЕМЬЯ 2 — развитие префикса от T0 до q. Только закрытые бары.

КАЖДОЕ ОТНОШЕНИЕ ИМЕЕТ ТОЧНЫЙ МОМЕНТ ПОЯВЛЕНИЯ
==============================================
Все величины ниже вычисляются одним прямым проходом по фильму и зависят только
от баров `t0 … q`. При изменении будущих свечей ни одно значение и ни один
timestamp не меняется. Будущий абсолютный экстремум, подтверждённый будущим
разворотом, и любая hindsight-конструкция здесь запрещены как определение.

ЧТО ЧИТАЕТСЯ
============
    run_max        максимум `d_close` по префиксу — докуда цена уходила от границы
    retrace        d_close / run_max — насколько уже вернулась (1 = на максимуме)
    bars_since_max сколько баров назад поставлен префиксный экстремум
    n_new_max      сколько раз префиксный экстремум обновлялся (темп удаления)
    consec_away    подряд идущих баров увеличения d_close на конец q
    body_ratio     |close−open| / (high−low) на баре q — тело против теней
    close_in_range (close−low)/(high−low) на баре q, развёрнуто по стороне:
                   1 = закрытие на краю, обращённом ОТ границы
    away_frac      доля баров префикса, на которых d_close вырос

`retrace` — это и есть «прекращение удаления» и «возвращение», записанные
непрерывным отношением, а не событием. Постановка прямо разрешает непрерывное
отношение, threshold и состояние, уже существующее на T0.
"""
from __future__ import annotations
import os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit, prange

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = Path(os.environ.get('G3_081A_OUT', str(ROOT / 'work/081a')))
sys.path.insert(0, str(HERE))
import paths as P                                                      # noqa: E402


@njit(parallel=True, cache=True)
def _prefix(t0s, Fs, es, north, off, opn, high, low, close,
            run_max, retrace, bars_since_max, n_new_max, consec_away,
            body_ratio, close_in_range, away_frac):
    for i in prange(t0s.size):
        t0 = t0s[i]; F = Fs[i]; e = es[i]; up = north[i]; base = off[i]
        rmax = -np.inf; rmax_at = t0; nmax = 0; consec = 0; nup = 0
        prev = np.nan
        for q in range(t0, F + 1):
            k = q - t0
            pos = base + k
            dc = (close[q] - e) if up else (e - close[q])
            if dc > rmax:
                rmax = dc; rmax_at = q; nmax += 1
            if k > 0 and dc > prev:
                consec += 1; nup += 1
            elif k > 0:
                consec = 0
            prev = dc
            run_max[pos] = rmax
            retrace[pos] = (dc / rmax) if rmax > 0 else np.nan
            bars_since_max[pos] = q - rmax_at
            n_new_max[pos] = nmax
            consec_away[pos] = consec
            rng = high[q] - low[q]
            body_ratio[pos] = (abs(close[q] - opn[q]) / rng) if rng > 0 else np.nan
            if rng > 0:
                cir = (close[q] - low[q]) / rng
                close_in_range[pos] = cir if up else (1.0 - cir)
            else:
                close_in_range[pos] = np.nan
            away_frac[pos] = (nup / k) if k > 0 else np.nan


def build(inst='NQ', terr='discovery'):
    m = ROOT / 'data/market' / inst
    opn = np.load(m / 'open.npy'); high = np.load(m / 'high.npy')
    low = np.load(m / 'low.npy'); close = np.load(m / 'close.npy')
    f = pd.read_parquet(OUT / f'paths/films_{inst}_{terr}.parquet')
    t0 = f.t0_spine_pos.to_numpy().astype(np.int64)
    F = f.certified_fresh_until_pos_strict.to_numpy().astype(np.int64)
    e = f.exit_boundary.to_numpy().astype(np.float64)
    north = (f.side.to_numpy() == 'north')
    nq = f.n_q.to_numpy().astype(np.int64)
    off = np.concatenate(([0], np.cumsum(nq)))[:-1]
    total = int(nq.sum())

    z = lambda dt: np.empty(total, dtype=dt)
    run_max, retrace = z(np.float32), z(np.float32)
    bars_since_max, n_new_max, consec_away = z(np.int32), z(np.int32), z(np.int32)
    body_ratio, close_in_range, away_frac = z(np.float32), z(np.float32), z(np.float32)
    t = time.time()
    _prefix(t0, F, e, north, off, opn, high, low, close,
            run_max, retrace, bars_since_max, n_new_max, consec_away,
            body_ratio, close_in_range, away_frac)
    d = pd.DataFrame({'run_max': run_max, 'retrace': retrace,
                      'bars_since_max': bars_since_max, 'n_new_max': n_new_max,
                      'consec_away': consec_away, 'body_ratio': body_ratio,
                      'close_in_range': close_in_range, 'away_frac': away_frac})
    d.attrs['seconds'] = round(time.time() - t, 2)
    return d


if __name__ == '__main__':
    inst = sys.argv[1] if len(sys.argv) > 1 else 'NQ'
    terr = sys.argv[2] if len(sys.argv) > 2 else 'discovery'
    d = build(inst, terr)
    p = OUT / f'paths/prefix_{inst}_{terr}.parquet'
    d.to_parquet(p, index=False, compression='zstd')
    print(f'{inst}/{terr}: {len(d)} rows, {d.attrs["seconds"]}s, {p.stat().st_size/1e6:.1f} MB')
