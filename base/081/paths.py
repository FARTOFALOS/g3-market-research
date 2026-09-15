#!/usr/bin/env python3
"""Оставшаяся траектория Film-1 как suffix-структура, а не 10 млн независимых строк.

СМЫСЛОВОЙ ОБЪЕКТ
================
Первичный future-object — оставшаяся OHLC-траектория от бара `q+1` до
сертифицированного контакта либо до потери наблюдаемости. `gross`, `adverse`,
`duration` — её чтения, а не сама она.

Все продолжения одного Film-1 вложены друг в друга: путь от `q+1` есть хвост
пути от `q`. Поэтому решётка решений считается одним обратным проходом по
фильму, а не перебором пар `(q, будущее)`.

ДВА РАССТОЯНИЯ, КОТОРЫЕ НЕЛЬЗЯ СМЕШИВАТЬ (DECLARE_081A §4.1)
============================================================
`d_close`  — `close[q]` относительно `e`. Prefix. Доступно на закрытии q,
             разрешено в правиле решения.
`gross`    — `open[q+1]` относительно `e`. Execution layer. Существует только
             после того, как намерение ENTER уже возникло; признаком решения
             не является.

ADVERSE — ПАРА, А НЕ ЧИСЛО (FREEZE_080A §8)
===========================================
`mae_certain` — ход против позиции по барам `q+1 … c-1`, строго до касательной
                свечи. При `c == q+1` равен 0: порядок внутри минуты неизвестен.
`mae_bound`   — то же включая касательную свечу.
У фильма без сертифицированного контакта касательной свечи нет, и величина по
барам `q+1 … F` — это `mae_observed_until_loss`, НИЖНЯЯ ГРАНИЦА MAE ещё открытой
сделки, а не её MAE. Она возвращается в тех же колонках, но строки помечены
`outcome = 1 (unknown)` и в одном распределении с завершёнными не смешиваются.
"""
from __future__ import annotations
import os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit, prange

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
INDEX = Path(os.environ.get('G3_080A_INDEX', str(ROOT / 'work/080a/index')))
OUT = Path(os.environ.get('G3_081A_OUT', str(ROOT / 'work/081a')))
sys.path.insert(0, str(ROOT / 'base/080'))
from tape import presence_grid, gap_kinds                              # noqa: E402

MIN = 60_000_000_000
NS_DAY = 86_400_000_000_000
#: календарный раздел объявлен в DECLARE_081A §1 до первого совместного будущего
SPLIT_NS = int(pd.Timestamp('2019-01-01', tz='UTC').value)
EMBARGO_NS = int(pd.Timestamp('2018-12-25', tz='UTC').value)

OUTCOME_TP, OUTCOME_UNKNOWN = 0, 1


@njit(parallel=True, cache=True)
def _walk(t0s, ends, Fs, cert, es, north, off,
          opn, high, low, close, kind, last,
          d_close, gross, mae_c, mae_b, dur, ref_ok, age, film, outcome):
    """Один обратный проход по каждому фильму. Запись только в свой срез.

    q идёт от F вниз к t0; бар входа `j = q+1`. Путь — бары `[q+1 … end]`.
    Единственный случай пустого пути — `q = F = end` у несертифицированного
    фильма: следующий бар лежит уже за промежутком, reference не определена.
    Строка при этом всё равно выпускается: решение существовало, исполнения нет.
    """
    for i in prange(t0s.size):
        t0 = t0s[i]; F = Fs[i]; end = ends[i]; e = es[i]
        up = north[i]                      # north: short, против позиции — вверх
        base = off[i]; is_cert = cert[i]
        NEG = -np.inf if up else np.inf
        run_all = NEG                      # экстремум по [q+2 … end]
        run_excl = NEG                     # экстремум по [q+2 … end-1]
        for q in range(F, t0 - 1, -1):
            j = q + 1
            pos = base + (q - t0)
            film[pos] = i
            age[pos] = q - t0
            d_close[pos] = (close[q] - e) if up else (e - close[q])
            outcome[pos] = OUTCOME_TP if is_cert else OUTCOME_UNKNOWN
            if j <= end:
                o = opn[j]
                gross[pos] = (o - e) if up else (e - o)
                ref_ok[pos] = (j <= last) and (kind[j] == 0)
                dur[pos] = end - q
                if up:
                    hi_all = high[j] if high[j] > run_all else run_all
                    hi_excl = run_excl
                    if j != end and high[j] > hi_excl:
                        hi_excl = high[j]
                    mae_b[pos] = hi_all - o
                    mae_c[pos] = (hi_excl - o) if hi_excl > -np.inf else 0.0
                else:
                    lo_all = low[j] if low[j] < run_all else run_all
                    lo_excl = run_excl
                    if j != end and low[j] < lo_excl:
                        lo_excl = low[j]
                    mae_b[pos] = o - lo_all
                    mae_c[pos] = (o - lo_excl) if lo_excl < np.inf else 0.0
                if mae_c[pos] < 0.0:
                    mae_c[pos] = 0.0
                if mae_b[pos] < 0.0:
                    mae_b[pos] = 0.0
                if not is_cert:
                    # касательной свечи нет: окно точное, но сделка ещё открыта
                    mae_c[pos] = mae_b[pos]
                # сдвиг бегущих экстремумов на бар j для следующего (меньшего) q
                if up:
                    if high[j] > run_all:
                        run_all = high[j]
                    if j != end and high[j] > run_excl:
                        run_excl = high[j]
                else:
                    if low[j] < run_all:
                        run_all = low[j]
                    if j != end and low[j] < run_excl:
                        run_excl = low[j]
            else:
                # q = F = end у несертифицированного фильма: пути нет
                gross[pos] = np.nan
                mae_c[pos] = np.nan
                mae_b[pos] = np.nan
                dur[pos] = 0
                ref_ok[pos] = False


def load_films(inst, territory='discovery'):
    f = pd.read_parquet(INDEX / f'film1_{inst}.parquet', columns=[
        'riz_id', 'tf_minutes', 'side', 'zone_top', 'zone_bottom', 'exit_boundary',
        't0_spine_pos', 't0_ts_ns', 't0_close', 't0_session_id',
        'first_observed_contact_pos', 'film1_status',
        'certified_fresh_until_pos_strict', 'q_certified_strict'])
    ts = f.t0_ts_ns.to_numpy()
    if territory == 'discovery':
        f = f[ts < EMBARGO_NS]
    elif territory == 'evaluation':
        f = f[ts >= SPLIT_NS]
    elif territory == 'all':
        pass
    else:
        raise ValueError(territory)
    return f.reset_index(drop=True)


def build(inst, territory='discovery', grid=None, lo=None, hi=None):
    m = ROOT / 'data/market' / inst
    opn = np.load(m / 'open.npy'); high = np.load(m / 'high.npy')
    low = np.load(m / 'low.npy'); close = np.load(m / 'close.npy')
    tsb = np.load(m / 'close_ts_utc_ns.npy')
    last = opn.size - 1
    kind, _ = gap_kinds(inst, grid, lo, hi)

    f = load_films(inst, territory)
    t0 = f.t0_spine_pos.to_numpy().astype(np.int64)
    F = f.certified_fresh_until_pos_strict.to_numpy().astype(np.int64)
    c = f.first_observed_contact_pos.to_numpy().astype(np.int64)
    cert = f.film1_status.to_numpy() == 'contact_certified'
    end = np.where(cert, c, F)
    e = f.exit_boundary.to_numpy().astype(np.float64)
    north = (f.side.to_numpy() == 'north')

    nq = (F - t0 + 1).astype(np.int64)
    assert (nq == f.q_certified_strict.to_numpy()).all(), 'сетка решений разошлась с 080A'
    off = np.concatenate(([0], np.cumsum(nq)))
    total = int(off[-1])

    z = lambda dt: np.empty(total, dtype=dt)
    d_close, gross = z(np.float32), z(np.float32)
    mae_c, mae_b = z(np.float32), z(np.float32)
    dur, age, film = z(np.int32), z(np.int32), z(np.int32)
    ref_ok = z(np.bool_); outcome = z(np.int8)

    t = time.time()
    _walk(t0, end, F, cert, e, north, off[:-1],
          opn, high, low, close, kind, last,
          d_close, gross, mae_c, mae_b, dur, ref_ok, age, film, outcome)
    secs = time.time() - t

    q = pd.DataFrame({
        'film': film, 'age': age, 'outcome': outcome, 'ref_ok': ref_ok,
        'd_close': d_close, 'gross': gross,
        'mae_certain': mae_c, 'mae_bound': mae_b, 'dur': dur})
    #: направленная пригодность — отдельный от доступности вопрос (FREEZE_080A §7)
    q['usable'] = q.ref_ok & (q.gross > 0)
    q.attrs['walk_seconds'] = round(secs, 2)

    f = f.assign(
        cert=cert, end_pos=end, n_q=nq,
        zone_width=(f.zone_top - f.zone_bottom).to_numpy(),
        t0_day=(f.t0_ts_ns.to_numpy() // NS_DAY).astype(np.int32),
        end_ts_ns=tsb[end])
    return f, q


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    grid, lo, hi = presence_grid()
    terr = sys.argv[1] if len(sys.argv) > 1 else 'discovery'
    for inst in (sys.argv[2:] or ['NQ']):
        f, q = build(inst, terr, grid, lo, hi)
        d = OUT / 'paths'; d.mkdir(exist_ok=True)
        f.to_parquet(d / f'films_{inst}_{terr}.parquet', index=False)
        q.to_parquet(d / f'q_{inst}_{terr}.parquet', index=False, compression='zstd')
        print(f'{inst}/{terr}: {len(f)} films, {len(q)} q, walk {q.attrs["walk_seconds"]}s, '
              f'{(d / f"q_{inst}_{terr}.parquet").stat().st_size/1e6:.1f} MB', flush=True)
