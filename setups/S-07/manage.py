"""S-07 v3: ведение позиции, ранние признаки провала, риск и размер.

Всё считается на префиксе: состояние на минуте m использует только минуты по m.
Единица — свеча (медиана истинного диапазона последних 30 закрытых минут),
зафиксированная в минуту решения и не пересчитываемая внутри сделки.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import run as base

ROOT, HERE, OUT, MIN = base.ROOT, base.HERE, base.OUT, base.MIN
MINUTE = 3
HOLD = 120
EPOCHS = [('2006-2012', '2006', '2012'), ('2013-2019', '2013', '2019'), ('2020-2026', '2020', '2026')]


def trajectories(ins='NQ', minute=MINUTE, hold=HOLD):
    """Матрицы close/high/low по минутам позиции плюс всё, что известно на входе."""
    m = base.market(ins)
    u = base.unit(m)
    cal = pd.read_parquet(ROOT / 'setups/S-04/calendar.parquet')
    ts = m['close_ts_utc_ns']
    open_pos = np.searchsorted(ts, cal.open_ns.to_numpy())
    p = open_pos + minute
    ok = (p + hold + 1 < len(ts)) & (ts[np.minimum(p, len(ts) - 1)] == cal.open_ns.to_numpy() + minute * MIN)
    idx = np.where(ok)[0]
    p = p[ok].astype(np.int64)
    b = p + 1
    # непрерывная минутная сетка на всём удержании и вход внутри сессии
    span = ts[b[:, None] + np.arange(hold)] - ts[b][:, None]
    good = (span == np.arange(hold) * MIN).all(axis=1)
    p, b, idx = p[good], b[good], idx[good]
    win_hi = np.maximum.reduce([m['high'][p - j] for j in range(base.WINDOW)])
    win_lo = np.minimum.reduce([m['low'][p - j] for j in range(base.WINDOW)])
    mid = (win_hi + win_lo) / 2
    side = np.where(m['close'][p] > mid, 1., -1.)
    entry = m['open'][b]
    unit = u[p]
    keep = np.isfinite(unit) & (unit > 0)
    p, b, idx, mid, side, entry, unit = [x[keep] for x in (p, b, idx, mid, side, entry, unit)]
    j = b[:, None] + np.arange(hold)
    close = m['close'][j]
    high = m['high'][j]
    low = m['low'][j]
    op = m['open'][j]
    # закрытие сессии обрезает удержание
    session_close = cal.close_ns.to_numpy()[idx][:, None]
    live = ts[j] < session_close
    s = side[:, None]
    pnl = s * (close - entry[:, None]) / unit[:, None]
    adverse = s * (np.where(s > 0, low, high) - entry[:, None]) / unit[:, None]   # минутный экстремум против
    favour = s * (np.where(s > 0, high, low) - entry[:, None]) / unit[:, None]    # минутный экстремум за
    mae = np.minimum.accumulate(adverse, axis=1)
    mfe = np.maximum.accumulate(favour, axis=1)
    against = (s * (close - op)) < 0
    back = (s * (close - mid[:, None])) < 0        # цена вернулась за уровень решения
    return dict(dates=cal.date.to_numpy()[idx], side=side, entry=entry, unit=unit, mid=mid,
                pnl=pnl, mae=mae, mfe=mfe, adverse=adverse, favour=favour,
                against=against, back=back, live=live,
                point=base.POINT[ins], cost=base.COST[ins], instrument=ins)


def run_length(against):
    """Сколько последних свечей подряд закрылись против позиции."""
    n = against.shape[1]
    out = np.zeros_like(against, dtype=np.int32)
    out[:, 0] = against[:, 0]
    for k in range(1, n):
        out[:, k] = np.where(against[:, k], out[:, k - 1] + 1, 0)
    return out


def diagnostics(t, recent_only=True):
    """Что говорит состояние на минуте m об оставшемся пути."""
    year = pd.Series(t['dates']).str.slice(0, 4).to_numpy()
    sel = year >= '2020' if recent_only else np.ones(len(year), bool)
    pnl, live = t['pnl'][sel], t['live'][sel]
    final = np.where(live, pnl, np.nan)
    last = np.take_along_axis(np.where(live, pnl, -np.inf), np.argmax(np.where(live, np.arange(pnl.shape[1]), -1), axis=1)[:, None], 1).ravel()
    runs = run_length(t['against'][sel])
    back = t['back'][sel]
    rows = []
    for m in [5, 10, 15, 20, 30, 45, 60, 90]:
        k = m - 1
        alive = live[:, k]
        cur = pnl[:, k]
        rest = last - cur
        base_rest = float(np.nanmean(rest[alive]))
        rows.append(dict(minute=m, kind='всё', n=int(alive.sum()), current=round(float(np.nanmean(cur[alive])), 3),
                         rest=round(base_rest, 3), win_rest=round(float(np.nanmean(rest[alive] > 0)), 3)))
        for label, mask in [('в минусе', cur < 0), ('в плюсе', cur > 0),
                            ('за уровнем решения', back[:, k]),
                            ('три свечи против подряд', runs[:, k] >= 3),
                            ('минус больше свечи', cur < -1)]:
            g = alive & mask
            if g.sum() < 30:
                continue
            rows.append(dict(minute=m, kind=label, n=int(g.sum()),
                             current=round(float(np.nanmean(cur[g])), 3),
                             rest=round(float(np.nanmean(rest[g])), 3),
                             win_rest=round(float(np.nanmean(rest[g] > 0)), 3)))
    return pd.DataFrame(rows)


def manage(t, rule):
    """Исполнение правила ведения. Возвращает P&L в свечах на сделку."""
    pnl, mae, mfe, live, back = t['pnl'], t['mae'], t['mfe'], t['live'], t['back']
    adverse = t['adverse']
    runs = run_length(t['against'])
    n, h = pnl.shape
    out = np.full(n, np.nan)
    for i in range(n):
        stop = -rule.get('stop', np.inf)
        moved = False
        result = None
        for k in range(h):
            if not live[i, k]:
                break
            # стоп проверяется по минутному экстремуму этой минуты, перенос — на её закрытии
            if adverse[i, k] <= stop:
                result = stop
                break
            if rule.get('breakeven') and not moved and mfe[i, k] >= rule['breakeven']:
                moved = True
                stop = max(stop, 0.)
            if rule.get('trail') and mfe[i, k] - pnl[i, k] >= rule['trail'] and mfe[i, k] >= rule.get('trail_arm', 0):
                result = pnl[i, k]
                break
            if rule.get('time_check') and k + 1 == rule['time_check'] and pnl[i, k] < rule.get('time_floor', 0):
                result = pnl[i, k]
                break
            if rule.get('back_out') and k + 1 >= rule.get('back_after', 1) and back[i, k]:
                result = pnl[i, k]
                break
            if rule.get('run_out') and k + 1 >= rule.get('run_after', 1) and runs[i, k] >= rule['run_out']:
                result = pnl[i, k]
                break
            result = pnl[i, k]
        out[i] = result if result is not None else np.nan
    return out


def evaluate(t, rules):
    year = pd.Series(t['dates']).str.slice(0, 4).to_numpy()
    rows = []
    for name, rule in rules.items():
        candles = manage(t, rule)
        usd = candles * t['unit'] * t['point'] - t['cost']
        for epoch, a, b in EPOCHS:
            sel = (year >= a) & (year <= b)
            g = usd[sel]
            if not np.isfinite(g).any():
                continue
            daily = pd.Series(g, index=t['dates'][sel]).groupby(level=0).sum().sort_index()
            eq = np.r_[0, np.cumsum(daily.to_numpy())]
            drop = max(1, int(len(daily) * .05))
            rows.append(dict(rule=name, epoch=epoch, n=int(np.isfinite(g).sum()),
                             candles=round(float(np.nanmean(candles[sel])), 3),
                             mean=round(float(np.nanmean(g)), 1), median=round(float(np.nanmedian(g)), 1),
                             win=round(float(np.nanmean(g > 0)), 3), total=round(float(daily.sum()), 0),
                             dd=round(float((np.maximum.accumulate(eq) - eq).max()), 0),
                             wo_best5=round(float(daily.sum() - daily.nlargest(drop).sum()), 0),
                             worst=round(float(np.nanmin(g)), 0)))
    return pd.DataFrame(rows)


RULES = {
    'держать 120 минут': {},
    'безубыток после +1 свечи': {'breakeven': 1.},
    'безубыток после +2 свечей': {'breakeven': 2.},
    'безубыток после +3 свечей': {'breakeven': 3.},
    'стоп 6 свечей': {'stop': 6.},
    'стоп 6 + безубыток после +3': {'stop': 6., 'breakeven': 3.},
    'выход, если через 15 мин в минусе': {'time_check': 15},
    'выход, если через 30 мин в минусе': {'time_check': 30},
    'выход, если через 30 мин ниже -1 свечи': {'time_check': 30, 'time_floor': -1.},
    'выход при возврате за уровень решения': {'back_out': True, 'back_after': 5},
    'выход после 3 свечей против подряд': {'run_out': 3},
    'выход после 4 свечей против подряд': {'run_out': 4},
    'трейлинг: откат 3 свечи от максимума': {'trail': 3., 'trail_arm': 3.},
    'трейлинг: откат 4 свечи от максимума': {'trail': 4., 'trail_arm': 4.},
    'выход, если через 45 мин в минусе': {'time_check': 45},
    'три свечи против после 20-й минуты': {'run_out': 3, 'run_after': 20},
    'три свечи против после 30-й минуты': {'run_out': 3, 'run_after': 30},
    'возврат за уровень после 30-й минуты': {'back_out': True, 'back_after': 30},
    'минус на 15-й + возврат за уровень после 30-й': {'time_check': 15, 'back_out': True, 'back_after': 30},
    'минус на 15-й + три против после 30-й': {'time_check': 15, 'run_out': 3, 'run_after': 30},
}


def sizing(t, risk_dollars=500., stop_candles=6.):
    """Размер позиции в свечах: столько контрактов, чтобы стоп стоил фиксированно."""
    year = pd.Series(t['dates']).str.slice(0, 4).to_numpy()
    candles = manage(t, {})
    per_contract = t['unit'] * t['point']
    size = np.maximum(1, np.floor(risk_dollars / (stop_candles * per_contract)))
    fixed = candles * per_contract - t['cost']
    scaled = candles * per_contract * size - t['cost'] * size
    rows = []
    for label, series in [('один контракт', fixed), (f'риск ${risk_dollars:.0f} на {stop_candles:.0f} свечей', scaled)]:
        for epoch, a, b in EPOCHS:
            sel = (year >= a) & (year <= b)
            daily = pd.Series(series[sel], index=t['dates'][sel]).groupby(level=0).sum().sort_index()
            eq = np.r_[0, np.cumsum(daily.to_numpy())]
            rows.append(dict(sizing=label, epoch=epoch, contracts_median=float(np.median(size[sel])),
                             total=round(float(daily.sum()), 0), mean=round(float(daily.mean()), 1),
                             dd=round(float((np.maximum.accumulate(eq) - eq).max()), 0),
                             worst_day=round(float(daily.min()), 0),
                             dd_over_mean=round(float((np.maximum.accumulate(eq) - eq).max() / daily.mean()), 1)))
    return pd.DataFrame(rows)


def main():
    t = trajectories('NQ')
    pd.set_option('display.width', 250)
    diag = diagnostics(t)
    diag.to_csv(HERE / 'diagnostics_v3.csv', index=False)
    print('=== что говорит состояние на минуте m об оставшемся пути (NQ, 2020-2026, свечи) ===')
    print(diag.to_string(index=False), flush=True)
    ev = evaluate(t, RULES)
    ev.to_csv(HERE / 'management_v3.csv', index=False)
    print()
    print('=== правила ведения, 2020-2026 ===')
    print(ev.loc[ev.epoch == '2020-2026'].sort_values('total', ascending=False).to_string(index=False), flush=True)
    print()
    print('=== те же правила в свечах по эпохам ===')
    print(ev.pivot_table(index='rule', columns='epoch', values='candles').round(3).to_string(), flush=True)
    sz = sizing(t)
    sz.to_csv(HERE / 'sizing_v3.csv', index=False)
    print()
    print('=== размер позиции ===')
    print(sz.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
