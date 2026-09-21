#!/usr/bin/env python3
"""092 — экономический рельеф участия в живом Film-1 к собственной `b` до NY close.

ЧТО ЭТО
=======
Первая discovery-карта ветки 092. НЕ выбирает setup и НЕ добавляет признаки по
результату. Её задача — восстановить структуру conditional executable economics
ещё живых состояний Film-1 по уже известным координатам (SEMANTIC FREEZE §7):

    d_close (текущее расстояние до собственной b), time_to_close (минут до
    закрытия NY у сессии входа), age (возраст Film-1 в барах), birth geometry
    (расстояние на T0), side, tf — контроль.

ОБЪЕКТ И РЕЗОЛЮЦИЯ (наследуются из 080A/081 без переопределения)
================================================================
Один RIZ -> его Film-1: T0 -> первый честный контакт собственной exit boundary.
Единица (riz_id, q), q от t0 до certified_fresh_until_pos_strict. Решение на
close(q), вход `open[q+1]` (next-open reference). Окно 03:00 America/New_York ->
официальное закрытие XNYS; выход к close последнего наблюдаемого бара сессии.

ОТЛИЧИЕ ОТ 081 trading.py: СТОПА НЕТ.
Стоп — параметр будущей action-policy, а не свойство карты рельефа. Здесь
участие ведётся к собственной `b`; если `b` не достигнута к закрытию сессии —
выход по close (TIME_EXIT); если наблюдаемость/фильм кончились раньше закрытия —
исход UNKNOWN (цензура, не ноль). Ход против позиции (MAE) и в пользу (MFE)
пишутся как excursions рядом, а не превращаются в стоп.

OPPORTUNITY FUNNEL != TRADE ECONOMICS (GO §2)
=============================================
Позиция открывается только если `open[q+1]` доступен, сертифицирован, лежит в
окне и собственная `b` ещё впереди по направлению. Иначе — код воронки
(NO_EXEC / NO_WINDOW / MISSED_TARGET). Эти случаи считаются в воронке, но НЕ
входят в expectancy реально открытых сделок как P&L=0. Costs — только на
открытую позицию.

КАСАНИЕ vs ПРОХОД НА ТИК
=======================
Основная модель: касание собственной `b` = достижение. Консервативная
чувствительность: `tp_conf=1`, только если цена прошла границу не менее чем на
тик на баре касания. Обе конвенции публикуются в агрегации; сам проход по ленте
от этого не меняется.
"""
from __future__ import annotations
import os, sys, time, json
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit, prange

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = Path(os.environ.get('G3_092_OUT', str(ROOT / 'work/092')))
sys.path.insert(0, str(ROOT / 'base/081'))
sys.path.insert(0, str(ROOT / 'base/080'))
from trading import sessions, bar_session_map                          # noqa: E402
from tape import presence_grid, gap_kinds                              # noqa: E402
from paths import load_films                                           # noqa: E402

TICK = {'NQ': 0.25, 'ES': 0.25, 'YM': 1.0}
COST = {'NQ': 1.00, 'ES': 1.00, 'YM': 4.00}
PV = {'NQ': 20.0, 'ES': 50.0, 'YM': 5.0}

# исходы открытых позиций и коды воронки
REACHED, TIME_EXIT, UNKNOWN = 0, 1, 2
SEEN_UNCERT = 3           # геометрическое касание у несертифицированного фильма
MISSED_TARGET, NO_EXEC, NO_WINDOW = 4, 5, 6


@njit(parallel=True, cache=True)
def _walk(t0s, Fs, ends, cert, es, north, off, nq,
          opn, high, low, close, sess, last_bar, kind_gap, tick,
          out_kind, out_pay, out_gross, out_dclose, out_dur,
          out_mae_c, out_mae_b, out_mfe, out_tp_conf, out_entry, out_reach_bar):
    for i in prange(t0s.size):
        t0 = t0s[i]; F = Fs[i]; end_f = ends[i]; e = es[i]; up = north[i]
        base = off[i]; is_cert = cert[i]
        for a in range(nq[i]):
            q = t0 + a
            pos = base + a
            out_dclose[pos] = (close[q] - e) if up else (e - close[q])
            j = q + 1
            # --- воронка возможностей (приоритет: exec > window > target) ---
            if j > F or kind_gap[j] != 0:
                out_kind[pos] = NO_EXEC
                continue
            sj = sess[j]
            if sj < 0:
                out_kind[pos] = NO_WINDOW
                continue
            o = opn[j]
            g = (o - e) if up else (e - o)
            if g <= 0.0:
                out_kind[pos] = MISSED_TARGET     # open уже на/за b
                continue
            # --- открытая позиция: участие к b, выход к NY close ---
            lb = last_bar[sj]
            limit = lb if lb < end_f else end_f
            out_gross[pos] = g
            out_entry[pos] = j
            run_adv = 0.0          # MAE, включая бар резолюции (bound)
            run_adv_pre = 0.0      # MAE, исключая бар резолюции (certain)
            run_fav = 0.0          # MFE в пользу к b
            res = UNKNOWN; pay = np.nan; dur = 0; tp_conf = 0; rbar = -1
            for b in range(j, limit + 1):
                if up:
                    adv = high[b] - o          # ход против short — вверх
                    fav = o - low[b]           # в пользу — вниз к b
                    tp_touch = low[b] <= e
                    tp_through = low[b] <= e - tick
                else:
                    adv = o - low[b]
                    fav = high[b] - o
                    tp_touch = high[b] >= e
                    tp_through = high[b] >= e + tick
                if adv > run_adv:
                    run_adv = adv
                if fav > run_fav:
                    run_fav = fav
                if tp_touch:
                    res = REACHED if is_cert else SEEN_UNCERT
                    pay = g; dur = b - q; rbar = b
                    tp_conf = 1 if tp_through else 0
                    break
                # бар не коснулся: его adverse входит и в certain (до касания)
                if adv > run_adv_pre:
                    run_adv_pre = adv
            if res == UNKNOWN:
                if limit == lb:
                    px = close[lb]
                    pay = (o - px) if up else (px - o)
                    res = TIME_EXIT; dur = lb - q; rbar = lb
                    run_adv_pre = run_adv          # весь путь наблюдён
                else:
                    dur = limit - q                # сертификат кончился до NY close
                    run_adv_pre = run_adv
            else:
                # на баре касания adverse уже учтён в run_adv (bound), но не в pre
                pass
            out_kind[pos] = res
            out_pay[pos] = pay
            out_dur[pos] = dur
            out_mae_b[pos] = run_adv
            out_mae_c[pos] = run_adv_pre
            out_mfe[pos] = run_fav
            out_tp_conf[pos] = tp_conf
            out_reach_bar[pos] = rbar


def build(inst='NQ', terr='discovery'):
    m = ROOT / 'data/market' / inst
    opn = np.load(m / 'open.npy'); high = np.load(m / 'high.npy')
    low = np.load(m / 'low.npy'); close = np.load(m / 'close.npy')
    ts = np.load(m / 'close_ts_utc_ns.npy')
    grid, glo, ghi = presence_grid()
    kind_gap, _ = gap_kinds(inst, grid, glo, ghi)
    st, cl = sessions()
    sess, last_bar = bar_session_map(ts, st, cl)

    f = load_films(inst, terr)
    t0 = f.t0_spine_pos.to_numpy().astype(np.int64)
    F = f.certified_fresh_until_pos_strict.to_numpy().astype(np.int64)
    c = f.first_observed_contact_pos.to_numpy().astype(np.int64)
    cert = (f.film1_status.to_numpy() == 'contact_certified')
    end = np.where(cert, c, F)
    e = f.exit_boundary.to_numpy().astype(np.float64)
    north = (f.side.to_numpy() == 'north')
    nq = (F - t0 + 1).astype(np.int64)
    assert (nq == f.q_certified_strict.to_numpy()).all(), 'сетка решений разошлась с 080A'
    off = np.concatenate(([0], np.cumsum(nq)))
    total = int(off[-1])

    z = lambda dt, v=0: np.full(total, v, dtype=dt)
    out = dict(kind=z(np.int8, NO_EXEC), pay=z(np.float32, np.nan),
               gross=z(np.float32, np.nan), dclose=z(np.float32, np.nan),
               dur=z(np.int32), mae_c=z(np.float32, np.nan), mae_b=z(np.float32, np.nan),
               mfe=z(np.float32, np.nan), tp_conf=z(np.int8), entry=z(np.int64, -1),
               rbar=z(np.int64, -1))
    t = time.time()
    _walk(t0, F, end, cert, e, north, off[:-1], nq,
          opn, high, low, close, sess, last_bar, kind_gap, TICK[inst],
          out['kind'], out['pay'], out['gross'], out['dclose'], out['dur'],
          out['mae_c'], out['mae_b'], out['mfe'], out['tp_conf'], out['entry'], out['rbar'])
    secs = round(time.time() - t, 2)

    # координаты состояния, известные на close(q)
    film_ix = np.repeat(np.arange(len(f)), nq)
    age = np.concatenate([np.arange(n) for n in nq]).astype(np.int32)
    tf = f.tf_minutes.to_numpy().astype(np.int32)[film_ix]
    side_north = north[film_ix]
    zone_w = (f.zone_top - f.zone_bottom).to_numpy().astype(np.float64)[film_ix]
    # birth geometry: signed distance close(T0) -> e, то же в тиках/нормировке
    t0_close = f.t0_close.to_numpy().astype(np.float64)
    birth = np.where(north, t0_close - e, e - t0_close)[film_ix].astype(np.float32)

    d = pd.DataFrame({
        'film': film_ix.astype(np.int32), 'age': age, 'kind': out['kind'],
        'd_close': out['dclose'], 'gross': out['gross'], 'pay': out['pay'],
        'dur': out['dur'], 'mae_c': out['mae_c'], 'mae_b': out['mae_b'],
        'mfe': out['mfe'], 'tp_conf': out['tp_conf'],
        'birth': birth, 'zone_w': zone_w.astype(np.float32),
        'tf': tf, 'north': side_north,
    })
    # time_to_close (мин) — только для открытых позиций, у которых есть entry-бар
    ent = out['entry']
    opened = ent >= 0
    ttc = np.full(total, np.nan, dtype=np.float32)
    sess_ent = sess[np.where(opened, ent, 0)]
    close_ns_ent = cl[np.clip(sess_ent, 0, len(cl) - 1)]
    q_ix = np.where(opened, ent - 1, 0)
    ttc_val = (close_ns_ent - ts[q_ix]) / 6.0e10
    ttc[opened] = ttc_val[opened].astype(np.float32)
    d['ttc'] = ttc
    d.attrs['walk_seconds'] = secs
    return f, d, secs


KIND_NAMES = {REACHED: 'reached', TIME_EXIT: 'time_exit', UNKNOWN: 'unknown',
              SEEN_UNCERT: 'seen_uncert', MISSED_TARGET: 'missed_target',
              NO_EXEC: 'no_exec', NO_WINDOW: 'no_window'}


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    terr = sys.argv[1] if len(sys.argv) > 1 else 'discovery'
    for inst in (sys.argv[2:] or ['NQ']):
        f, d, secs = build(inst, terr)
        p = OUT / f'map_{inst}_{terr}.parquet'
        d.to_parquet(p, index=False, compression='zstd')
        vc = d.kind.map(KIND_NAMES).value_counts().to_dict()
        print(f'{inst}/{terr}: {len(f)} films, {len(d)} q, walk {secs}s, '
              f'{p.stat().st_size/1e6:.1f} MB', flush=True)
        print('  funnel:', vc, flush=True)
