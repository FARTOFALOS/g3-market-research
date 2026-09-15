#!/usr/bin/env python3
"""Торговая территория 081A v2: окно 03:00 ET → NY close, выход к закрытию дня.

ЧТО ИЗМЕНИЛОСЬ ОТНОСИТЕЛЬНО v1
==============================
v1 считал все часы. Мандат проекта (`setups/README.md`) требует сделок Лондона
и Нью-Йорка без входов в Азию и закрытой позиции к закрытию Нью-Йорка. Поэтому
торговый кандидат пересчитывается здесь, а всечасовые находки GATE A/B
остаются описательным наблюдением рынка, не торговой конструкцией.

Окно заморожено до пересчёта: **03:00 America/New_York → официальное закрытие
NYSE**, исторический календарь XNYS с сокращёнными днями. Календарь берётся
готовым из `setups/S-04/calendar.parquet` (пакет `exchange_calendars` 4.13.2,
конвенция «NYSE cash close, including historical early closes»); новый источник
не заводится и окно не подбирается.

СЕМАНТИКА ИСПОЛНЕНИЯ — ЗАМОРОЖЕНА ОДИН РАЗ
==========================================
Основная условная модель TP: **касание границы считается исполнением**.
Обязательная консервативная чувствительность: TP подтверждён, только если цена
прошла границу не менее чем на один тик. Если Film-1 кончается касанием, а
прохода на тик не наблюдалось, **продолжение за пределы Film-1 не читается**, и
консервативное исполнение TP помечается `unconfirmed`.

Стоп: наблюдаемое касание уровня стопа срабатывает. Если непрерывно
наблюдаемый бар ОТКРЫЛСЯ за стопом, берётся худший open. Если нужный участок
пути лежит внутри неизвестного промежутка — исход остаётся `unknown`.

Оба барьера в одном баре — порядок внутри минуты неизвестен: возвращается пара
границ, среднее не берётся.

ВЫХОД ПО ВРЕМЕНИ
================
Позиция, не закрытая TP или стопом, закрывается по `close` последнего
наблюдаемого бара сессии на закрытии NYSE. Это заранее определённый торговый
выход, а не знание из будущего о промежутке ленты.
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

TICK = {'NQ': 0.25, 'ES': 0.25, 'YM': 1.0}
COST = {'NQ': 1.00, 'ES': 1.00, 'YM': 4.00}
PV = {'NQ': 20.0, 'ES': 50.0, 'YM': 5.0}
CAL = ROOT / 'setups/S-04/calendar.parquet'
WINDOW_START_ET_MIN = 3 * 60                      # 03:00 America/New_York

# исходы
WIN, LOSS, AMBIG, TIME_EXIT, UNKNOWN, NO_TRADE = 0, 1, 2, 3, 4, 5


def sessions():
    """Заморожённый XNYS: (start 03:00 ET, официальный close) в UTC-нс."""
    s = pd.read_parquet(CAL)
    close_ns = s.close_ns.to_numpy().astype(np.int64)
    d = pd.to_datetime(s.date)
    start = (pd.DatetimeIndex(d).tz_localize('America/New_York')
             + pd.Timedelta(minutes=WINDOW_START_ET_MIN))
    start_ns = start.tz_convert('UTC').as_unit('ns').asi8.astype(np.int64)
    ok = start_ns < close_ns
    return start_ns[ok], close_ns[ok]


def bar_session_map(ts, start_ns, close_ns):
    """`sess[b]` — индекс сессии бара b либо -1; `last_bar[i]` — бар закрытия i."""
    i = np.searchsorted(close_ns, ts, side='left')
    i = np.clip(i, 0, len(close_ns) - 1)
    inside = (ts >= start_ns[i]) & (ts <= close_ns[i])
    sess = np.where(inside, i, -1).astype(np.int64)
    # последний наблюдаемый бар каждой сессии
    last_bar = np.searchsorted(ts, close_ns, side='right') - 1
    return sess, last_bar


@njit(parallel=True, cache=True)
def _trades(t0s, Fs, ends, cert, es, north, off, nq,
            opn, high, low, close, sess, last_bar, kind_gap, tick, kmult,
            out_kind, out_lo, out_hi, out_dur, out_gross, out_dclose,
            out_tp_conf, out_entry_ts):
    for i in prange(t0s.size):
        t0 = t0s[i]; F = Fs[i]; end_f = ends[i]; e = es[i]; up = north[i]
        base = off[i]; is_cert = cert[i]
        for a in range(nq[i]):
            q = t0 + a
            pos = base + a
            out_dclose[pos] = (close[q] - e) if up else (e - close[q])
            j = q + 1
            # ПРИГОДНОСТЬ ПРОВЕРЯЕТСЯ ПО БАРУ ВХОДА, а не по минуте решения:
            # сигнал на последней минуте, чей next open лежит уже за закрытием
            # XNYS, отменяется; в дни, когда XNYS закрыта, входов нет вовсе.
            if j > F or kind_gap[j] != 0:
                out_kind[pos] = NO_TRADE
                continue
            sj = sess[j]
            if sj < 0:
                out_kind[pos] = NO_TRADE
                continue
            lb = last_bar[sj]
            o = opn[j]
            g = (o - e) if up else (e - o)
            if g <= 0.0:
                out_kind[pos] = NO_TRADE          # open уже не снаружи
                continue
            out_gross[pos] = g
            out_entry_ts[pos] = j
            s_dist = kmult * g
            stop = (o + s_dist) if up else (o - s_dist)
            # предел наблюдения сделки: закрытие своей сессии либо конец Film-1.
            # `F` здесь НЕ ограничивает: F = c-1 у сертифицированного фильма, и
            # ограничение им отрезало бы саму касательную свечу, где и стоит TP.
            limit = lb if lb < end_f else end_f
            res = UNKNOWN; lo = np.nan; hi = np.nan; dur = 0
            tp_conf = 0
            for b in range(j, limit + 1):
                # стоп: открытие за стопом — худший open
                if up:
                    open_through = opn[b] >= stop
                    stop_touch = high[b] >= stop
                    tp_touch = low[b] <= e
                    tp_through = low[b] <= e - tick
                else:
                    open_through = opn[b] <= stop
                    stop_touch = low[b] <= stop
                    tp_touch = high[b] >= e
                    tp_through = high[b] >= e + tick
                if open_through:
                    fill = opn[b]
                    loss = (fill - o) if up else (o - fill)
                    res = LOSS; lo = -loss; hi = -loss; dur = b - q
                    break
                if stop_touch and tp_touch:
                    res = AMBIG; lo = -s_dist; hi = g; dur = b - q
                    tp_conf = 1 if tp_through else 0
                    break
                if stop_touch:
                    res = LOSS; lo = -s_dist; hi = -s_dist; dur = b - q
                    break
                if tp_touch:
                    res = WIN; lo = g; hi = g; dur = b - q
                    tp_conf = 1 if tp_through else 0
                    break
            if res == UNKNOWN:
                # барьеры не сработали до предела наблюдения
                if limit == lb:
                    px = close[lb]                # заранее известный выход к NY close
                    r = (o - px) if up else (px - o)
                    res = TIME_EXIT; lo = r; hi = r; dur = lb - q
                else:
                    dur = limit - q               # сертификат кончился раньше NY close
            out_kind[pos] = res; out_lo[pos] = lo; out_hi[pos] = hi
            out_dur[pos] = dur; out_tp_conf[pos] = tp_conf


def build(inst, terr, kmult=1.0):
    m = ROOT / 'data/market' / inst
    opn = np.load(m / 'open.npy'); high = np.load(m / 'high.npy')
    low = np.load(m / 'low.npy'); close = np.load(m / 'close.npy')
    ts = np.load(m / 'close_ts_utc_ns.npy')
    sys.path.insert(0, str(ROOT / 'base/080'))
    from tape import presence_grid, gap_kinds
    grid, glo, ghi = presence_grid()
    kind_gap, _ = gap_kinds(inst, grid, glo, ghi)

    st, cl = sessions()
    sess, last_bar = bar_session_map(ts, st, cl)

    f = pd.read_parquet(OUT / f'paths/films_{inst}_{terr}.parquet')
    t0 = f.t0_spine_pos.to_numpy().astype(np.int64)
    F = f.certified_fresh_until_pos_strict.to_numpy().astype(np.int64)
    end = f.end_pos.to_numpy().astype(np.int64)
    cert = f.cert.to_numpy()
    e = f.exit_boundary.to_numpy().astype(np.float64)
    north = (f.side.to_numpy() == 'north')
    nq = f.n_q.to_numpy().astype(np.int64)
    off = np.concatenate(([0], np.cumsum(nq)))[:-1]
    n = int(nq.sum())

    z = lambda dt, v=0: np.full(n, v, dtype=dt)
    out = dict(kind=z(np.int8, NO_TRADE), lo=z(np.float32, np.nan), hi=z(np.float32, np.nan),
               dur=z(np.int32), gross=z(np.float32, np.nan), dclose=z(np.float32, np.nan),
               tp_conf=z(np.int8), entry=z(np.int64, -1))
    t = time.time()
    _trades(t0, F, end, cert, e, north, off, nq,
            opn, high, low, close, sess, last_bar, kind_gap, TICK[inst], kmult,
            out['kind'], out['lo'], out['hi'], out['dur'], out['gross'],
            out['dclose'], out['tp_conf'], out['entry'])
    d = pd.DataFrame(out).rename(columns={'dclose': 'd_close'})
    d.attrs['seconds'] = round(time.time() - t, 2)
    return f, d


if __name__ == '__main__':
    terr = sys.argv[1] if len(sys.argv) > 1 else 'discovery'
    for inst in (sys.argv[2:] or ['NQ']):
        for k in (1.0, 2.0, 3.0):
            f, d = build(inst, terr, k)
            p = OUT / f'paths/trade_{inst}_{terr}_k{k:g}.parquet'
            d.to_parquet(p, index=False, compression='zstd')
            tradable = (d.kind != NO_TRADE).sum()
            print(f'{inst}/{terr}/k={k:g}: {len(d)} q, торгуемых {tradable} '
                  f'({tradable/len(d):.4f}), {d.attrs["seconds"]}s, {p.stat().st_size/1e6:.1f} MB',
                  flush=True)
