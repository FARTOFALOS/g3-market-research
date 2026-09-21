#!/usr/bin/env python3
"""092 — frozen action-policy v1: одно действие на Film-1 при первом появлении региона.

ЭТО ДРУГОЙ ESTIMAND, ЧЕМ КАРТА.
Карта (map.py/relief.py) — state-occupancy: все открытые (riz_id, q). Её числа
(+7.70 scene-balanced, +3.45 day-level и т. п.) НЕЛЬЗЯ сравнивать с результатом
здесь. Здесь 182 тыс. cursor-строк схлопываются в одну возможность на Film-1 —
проверяется, было ли найдено торговое состояние или лишь хорошая поверхность.

FREEZE v1 (ровно то, из чего возник регион; признаки НЕ добавляются)
====================================================================
Инструмент: NQ. Объект: canonical live Film-1 (080A/081, без переопределения).
Recognition: первый закрытый q, на котором ОДНОВРЕМЕННО
    d_close(q) = |close[q] - e| в направлении к b  >= 8.0 пункта NQ
    time_to_NY_close(q)                            >= 480 минут
q читается на закрытии; time_to_NY_close(q) = (офиц. закрытие XNYS сессии q) -
ts(q). Окно/календарь — замороженные 081 (03:00 ET -> закрытие XNYS).
Action: вход open[q+1] в направлении собственной b, если на входе b ещё впереди
и вход executable. Максимум ОДНО действие на riz_id (первое eligible распознание).
TP: собственная b (касание = достижение; проход на тик — консервативная метка).
Fallback: выход по close последнего наблюдаемого бара сессии (NY-close exit).
Cost: 1.0 пункт круг (рядом 0x/2x, break-even). Censor/gap/pre-entry — по
замороженной семантике map.py: no-trade не превращается в trade P&L.

Если первый state-eligible q неисполним (gap / вне окна / open уже за b), сцена
теряет возможность (учитывается в воронке), а НЕ откладывается на лучший вход.
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

# FREEZE v1 пороги региона
D_MIN_PTS = 8.0
TTC_MIN_MIN = 480.0

# исходы
REACHED, TIME_EXIT, UNKNOWN = 0, 1, 2
NO_ELIGIBLE, NO_EXEC, NO_WINDOW, MISSED_TARGET = 3, 4, 5, 6
NAMES = {REACHED: 'reached', TIME_EXIT: 'time_exit', UNKNOWN: 'unknown',
         NO_ELIGIBLE: 'no_eligible', NO_EXEC: 'no_exec', NO_WINDOW: 'no_window',
         MISSED_TARGET: 'missed_target'}


@njit(parallel=True, cache=True)
def _policy(t0s, Fs, ends, cert, es, north, nq,
            opn, high, low, close, ts, sess, close_ns, last_bar, kind_gap, tick,
            o_kind, o_pay, o_gross, o_dclose, o_ttc, o_dur, o_mae_b, o_mfe,
            o_tp_conf, o_qbar, o_entrybar):
    for i in prange(t0s.size):
        t0 = t0s[i]; F = Fs[i]; end_f = ends[i]; e = es[i]; up = north[i]
        is_cert = cert[i]
        chosen = -1; ttc_ch = np.nan; dcl_ch = np.nan
        # --- первый q, где выполнено состояние региона ---
        for a in range(nq[i]):
            q = t0 + a
            dcl = (close[q] - e) if up else (e - close[q])
            if dcl < D_MIN_PTS:
                continue
            sq = sess[q]
            if sq < 0:
                continue
            ttc = (close_ns[sq] - ts[q]) / 6.0e10
            if ttc < TTC_MIN_MIN:
                continue
            chosen = q; ttc_ch = ttc; dcl_ch = dcl
            break
        o_qbar[i] = chosen
        if chosen < 0:
            o_kind[i] = NO_ELIGIBLE
            continue
        o_dclose[i] = dcl_ch; o_ttc[i] = ttc_ch
        q = chosen; j = q + 1
        # --- исполнимость первого eligible входа ---
        if j > F or kind_gap[j] != 0:
            o_kind[i] = NO_EXEC; continue
        sj = sess[j]
        if sj < 0:
            o_kind[i] = NO_WINDOW; continue
        o = opn[j]
        g = (o - e) if up else (e - o)
        if g <= 0.0:
            o_kind[i] = MISSED_TARGET; continue
        # --- участие к b, выход к NY close, без стопа ---
        lb = last_bar[sj]
        limit = lb if lb < end_f else end_f
        o_gross[i] = g; o_entrybar[i] = j
        run_adv = 0.0; run_fav = 0.0
        res = UNKNOWN; pay = np.nan; dur = 0; tp_conf = 0
        for b in range(j, limit + 1):
            if up:
                adv = high[b] - o; fav = o - low[b]
                tp_touch = low[b] <= e; tp_through = low[b] <= e - tick
            else:
                adv = o - low[b]; fav = high[b] - o
                tp_touch = high[b] >= e; tp_through = high[b] >= e + tick
            if adv > run_adv:
                run_adv = adv
            if fav > run_fav:
                run_fav = fav
            if tp_touch:
                res = REACHED; pay = g; dur = b - q
                tp_conf = 1 if tp_through else 0
                break
        if res == UNKNOWN:
            if limit == lb:
                px = close[lb]
                pay = (o - px) if up else (px - o)
                res = TIME_EXIT; dur = lb - q
            else:
                dur = limit - q
        o_kind[i] = res; o_pay[i] = pay; o_dur[i] = dur
        o_mae_b[i] = run_adv; o_mfe[i] = run_fav; o_tp_conf[i] = tp_conf


def build(inst='NQ', terr='discovery'):
    m = ROOT / 'data/market' / inst
    opn = np.load(m / 'open.npy'); high = np.load(m / 'high.npy')
    low = np.load(m / 'low.npy'); close = np.load(m / 'close.npy')
    ts = np.load(m / 'close_ts_utc_ns.npy')
    grid, glo, ghi = presence_grid()
    kind_gap, _ = gap_kinds(inst, grid, glo, ghi)
    st, cl = sessions()
    cl = cl.astype(np.int64)
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
    n = len(f)

    z = lambda dt, v=0: np.full(n, v, dtype=dt)
    o = dict(kind=z(np.int8, NO_ELIGIBLE), pay=z(np.float32, np.nan), gross=z(np.float32, np.nan),
             dclose=z(np.float32, np.nan), ttc=z(np.float32, np.nan), dur=z(np.int32),
             mae_b=z(np.float32, np.nan), mfe=z(np.float32, np.nan), tp_conf=z(np.int8),
             qbar=z(np.int64, -1), entrybar=z(np.int64, -1))
    t = time.time()
    _policy(t0, F, end, cert, e, north, nq,
            opn, high, low, close, ts, sess, cl, last_bar, kind_gap, TICK[inst],
            o['kind'], o['pay'], o['gross'], o['dclose'], o['ttc'], o['dur'],
            o['mae_b'], o['mfe'], o['tp_conf'], o['qbar'], o['entrybar'])
    secs = round(time.time() - t, 2)

    day = np.where(o['entrybar'] >= 0, ts[np.clip(o['entrybar'], 0, len(ts) - 1)] // 86400000000000, -1)
    r = pd.DataFrame({
        'riz_id': f.riz_id.to_numpy(), 'tf': f.tf_minutes.to_numpy().astype(np.int32),
        'north': north, 'kind': o['kind'], 'd_close': o['dclose'], 'ttc': o['ttc'],
        'gross': o['gross'], 'pay': o['pay'], 'dur': o['dur'], 'mae_b': o['mae_b'],
        'mfe': o['mfe'], 'tp_conf': o['tp_conf'], 'day': day.astype(np.int64),
        't0_ns': f.t0_ts_ns.to_numpy()})
    r.attrs['seconds'] = secs
    return r


def report(r, inst, terr):
    C = COST[inst]
    vc = {NAMES[k]: int(v) for k, v in r.kind.value_counts().items()}
    n_films = len(r)
    trades = r[r.kind.isin([REACHED, TIME_EXIT, UNKNOWN])].copy()
    resolved = trades[trades.kind != UNKNOWN].copy()
    resolved['net'] = resolved.pay - C
    reach = (trades.kind == REACHED)
    out = {
        'freeze': {'d_min_pts': D_MIN_PTS, 'ttc_min_min': TTC_MIN_MIN,
                   'one_action_per_riz': True, 'cost_pts': C,
                   'estimand': 'frozen action-policy v1; one trade per Film-1 at first eligible recognition'},
        'instrument': inst, 'territory': terr,
        'n_films_population': n_films,
        'funnel': vc,
        'n_trades_opened': int(len(trades)),
        'n_resolved': int(len(resolved)),
        'unknown_rate_of_trades': float((trades.kind == UNKNOWN).mean()) if len(trades) else float('nan'),
        'reach_rate': float(reach.mean()) if len(trades) else float('nan'),
        'per_trade_net_mean_pts': float(resolved.net.mean()),
        'per_trade_net_median_pts': float(resolved.net.median()),
        'per_trade_gross_mean_pts': float(resolved.pay.mean()),
        'per_trade_net_mean_usd_1x': float(resolved.net.mean() * PV[inst]),
        'net_mean_2x': float(resolved.pay.mean() - 2 * C),
        'breakeven_cost_pts': float(resolved.pay.mean()),
        'tp_unconfirmed_of_reached': float((trades.loc[reach, 'tp_conf'] == 0).mean()) if reach.any() else float('nan'),
        'mae_bound_median_pts': float(trades.mae_b.median()),
        'mfe_median_pts': float(trades.mfe.median()),
        'dur_median_bars': float(trades.dur.median()),
        'target_hit_share': float((resolved.kind == REACHED).mean()),
        'close_exit_share': float((resolved.kind == TIME_EXIT).mean()),
    }
    # day-level (единица — день; здесь и так один trade на фильм, дни независимее)
    dg = resolved.groupby('day').net
    dv = dg.mean()
    out['day_level'] = {
        'n_days': int(dv.size),
        'trades_per_day_median': float(resolved.groupby('day').size().median()),
        'day_net_mean': float(dv.mean()), 'day_net_median': float(dv.median()),
        'frac_days_positive': float((dg.sum() > 0).mean()),
        'leave_top1_out': float(dv.sort_values(ascending=False).iloc[1:].mean()),
        'leave_top3_out': float(dv.sort_values(ascending=False).iloc[3:].mean()),
    }
    # годовое распределение
    yr = resolved.copy(); yr['year'] = pd.to_datetime(yr.t0_ns).dt.year
    ann = yr.groupby('year').net.agg(['size', 'mean', 'median'])
    out['annual'] = {int(y): {'n': int(r0['size']), 'net_mean': float(r0['mean']),
                              'net_median': float(r0['median'])} for y, r0 in ann.iterrows()}
    return out


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    terr = sys.argv[1] if len(sys.argv) > 1 else 'discovery'
    for inst in (sys.argv[2:] or ['NQ']):
        r = build(inst, terr)
        r.to_parquet(OUT / f'policy_v1_{inst}_{terr}.parquet', index=False)
        rep = report(r, inst, terr)
        (OUT / f'policy_v1_{inst}_{terr}.json').write_text(json.dumps(rep, indent=1, ensure_ascii=False))
        print(f"\n=== FREEZE v1 action-policy {inst}/{terr}  (walk {r.attrs['seconds']}s) ===")
        print(f"population films={rep['n_films_population']:,}  funnel={rep['funnel']}")
        print(f"trades opened={rep['n_trades_opened']:,}  resolved={rep['n_resolved']:,}  unknown_rate={rep['unknown_rate_of_trades']:.4f}")
        print(f"reach={rep['reach_rate']:.4f}  target_hit={rep['target_hit_share']:.4f}  close_exit={rep['close_exit_share']:.4f}")
        print(f"per-trade net mean={rep['per_trade_net_mean_pts']:+.4f}pt (${rep['per_trade_net_mean_usd_1x']:+.2f})  median={rep['per_trade_net_median_pts']:+.4f}pt")
        print(f"  gross mean={rep['per_trade_gross_mean_pts']:+.4f}  break-even={rep['breakeven_cost_pts']:.4f}  net@2x={rep['net_mean_2x']:+.4f}  tp_unconf={rep['tp_unconfirmed_of_reached']:.4f}")
        print(f"  MAE_bound med={rep['mae_bound_median_pts']:.2f}  MFE med={rep['mfe_median_pts']:.2f}  dur med={rep['dur_median_bars']:.0f} bars")
        dl = rep['day_level']
        print(f"day-level: days={dl['n_days']} trades/day med={dl['trades_per_day_median']:.0f}  day_net mean={dl['day_net_mean']:+.3f} median={dl['day_net_median']:+.3f}")
        print(f"  frac days+={dl['frac_days_positive']:.3f}  leave-top1={dl['leave_top1_out']:+.3f}  leave-top3={dl['leave_top3_out']:+.3f}")
        print("annual net mean:", {y: round(v['net_mean'], 1) for y, v in rep['annual'].items()})
