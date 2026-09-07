"""S-07: где на самом деле стоит стоп и как вести. Только сохранённая история."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\Admin\Claude\g3-market-research")
sys.path.insert(0, str(ROOT / 'setups/S-07'))
import run as base
import manage as M

MIN, HOLD, MINUTE, EPOCHS = M.MIN, M.HOLD, M.MINUTE, M.EPOCHS


def traj(ins='NQ'):
    """Как M.trajectories, но дополнительно отдаёт границы предрыночного окна."""
    m = base.market(ins)
    u = base.unit(m)
    cal = pd.read_parquet(ROOT / 'setups/S-04/calendar.parquet')
    ts = m['close_ts_utc_ns']
    open_pos = np.searchsorted(ts, cal.open_ns.to_numpy())
    p = open_pos + MINUTE
    ok = (p + HOLD + 1 < len(ts)) & (ts[np.minimum(p, len(ts) - 1)] == cal.open_ns.to_numpy() + MINUTE * MIN)
    idx = np.where(ok)[0]
    p = p[ok].astype(np.int64)
    b = p + 1
    span = ts[b[:, None] + np.arange(HOLD)] - ts[b][:, None]
    good = (span == np.arange(HOLD) * MIN).all(axis=1)
    p, b, idx = p[good], b[good], idx[good]
    hi = np.maximum.reduce([m['high'][p - j] for j in range(base.WINDOW)])
    lo = np.minimum.reduce([m['low'][p - j] for j in range(base.WINDOW)])
    mid = (hi + lo) / 2
    side = np.where(m['close'][p] > mid, 1., -1.)
    entry = m['open'][b]
    unit = u[p]
    dclose = m['close'][p]
    keep = np.isfinite(unit) & (unit > 0)
    p, b, idx, mid, side, entry, unit, hi, lo, dclose = [
        x[keep] for x in (p, b, idx, mid, side, entry, unit, hi, lo, dclose)]
    j = b[:, None] + np.arange(HOLD)
    close, high, low = m['close'][j], m['high'][j], m['low'][j]
    live = ts[j] < cal.close_ns.to_numpy()[idx][:, None]
    s = side[:, None]
    pnl = s * (close - entry[:, None]) / unit[:, None]
    adverse = s * (np.where(s > 0, low, high) - entry[:, None]) / unit[:, None]
    return dict(dates=cal.date.to_numpy()[idx], side=side, entry=entry, unit=unit, mid=mid,
                pre_hi=hi, pre_lo=lo, decide_close=dclose, pnl=pnl, adverse=adverse,
                mae=np.minimum.accumulate(adverse, axis=1), live=live,
                point=base.POINT[ins], cost=base.COST[ins])


t = traj()
year = pd.Series(t['dates']).str.slice(0, 4).to_numpy()
per = t['unit'] * t['point']
n, h = t['pnl'].shape
last_i = np.array([int(np.max(np.where(t['live'][i])[0])) for i in range(n)])
final = t['pnl'][np.arange(n), last_i]
mae_full = t['mae'][np.arange(n), last_i]
disp = (t['decide_close'] - t['mid']) * t['side'] / t['unit']
halfrange = (t['pre_hi'] - t['pre_lo']) / 2 / t['unit']


def money(candles, sel):
    usd = candles * per - t['cost']
    d = pd.Series(usd[sel], index=t['dates'][sel]).groupby(level=0).sum().sort_index()
    eq = np.r_[0, np.cumsum(d.to_numpy())]
    dd = float((np.maximum.accumulate(eq) - eq).max())
    return dict(total=round(float(d.sum())), dd=round(dd),
                rr=round(float(d.sum()) / dd, 2) if dd else None,
                med=round(float(d.median())), worst=round(float(d.min())))


def simulate(level, time_check=None):
    """level — уровень стопа в свечах от входа (отрицательный), массив на сделку."""
    out = np.full(n, np.nan)
    pnl, adverse, live = t['pnl'], t['adverse'], t['live']
    for i in range(n):
        res = None
        lv = level[i]
        for k in range(h):
            if not live[i, k]:
                res = pnl[i, k - 1]
                break
            if adverse[i, k] <= lv:
                res = lv
                break
            if time_check and k + 1 == time_check and pnl[i, k] < 0:
                res = pnl[i, k]
                break
        out[i] = pnl[i, last_i[i]] if res is None else res
    return out


NEG_INF = np.full(n, -1e9)

print('=== 1. Куда сделка ходит против себя (MAE в свечах, вся история 2006-2026) ===')
for lab, sel in [('все', np.ones(n, bool)), ('итог > 0', final > 0), ('итог < 0', final <= 0)]:
    q = np.percentile(mae_full[sel], [50, 75, 90, 95, 99])
    print(f'  {lab:<9} n={int(sel.sum()):>5}  медиана {q[0]:6.2f}  75% {q[1]:6.2f}  90% {q[2]:6.2f}  95% {q[3]:6.2f}  99% {q[4]:6.2f}')
print('  стоп на уровне X свечей:')
for x in (4, 6, 8, 10, 12, 16, 20, 24):
    killed = float((mae_full[final > 0] <= -x).mean())
    hit_l = float((mae_full[final <= 0] <= -x).mean())
    print(f'    {x:>3}: убил бы {killed:6.1%} будущих победителей, задел {hit_l:6.1%} проигравших')

print()
print('=== 2. Стоп в свечах поверх выхода на 15-й минуте, по эпохам ===')
rows = []
for x in (None, 24, 20, 16, 12, 10, 8, 6):
    lvl = NEG_INF if x is None else np.full(n, -float(x))
    c = simulate(lvl, time_check=15)
    r = {'стоп': 'нет' if x is None else f'{x} свечей'}
    for ep, a, b_ in EPOCHS:
        s = money(c, (year >= a) & (year <= b_))
        r[ep] = f"{s['total']:>9,} / {s['rr']}"
    r['худший день 2020+'] = money(c, year >= '2020')['worst']
    rows.append(r)
print(pd.DataFrame(rows).to_string(index=False))
print('  формат: итог $ / (итог к просадка)')

print()
print('=== 3. Структурный стоп вместо стопа в свечах ===')
opp = np.where(t['side'] > 0, t['pre_lo'], t['pre_hi'])
struct = (opp - t['entry']) * t['side'] / t['unit']
midlvl = (t['mid'] - t['entry']) * t['side'] / t['unit']
for lab, lv in [('противоположный край предрынка', struct),
                ('уровень решения mid', midlvl),
                ('mid минус полдиапазона', midlvl - halfrange * 0.5)]:
    lv = np.minimum(lv, -0.25)
    print(f'  {lab}: медиана {np.median(lv):6.2f} свечи, 10% {np.percentile(lv, 10):6.2f}, 90% {np.percentile(lv, 90):6.2f}')
    c = simulate(lv, time_check=15)
    for ep, a, b_ in EPOCHS:
        print('       ', ep, money(c, (year >= a) & (year <= b_)))

print()
print('=== 4. Момент выхода из минуса: плато или подгонка (стоп 16 свечей) ===')
rows = []
for mn in (5, 10, 15, 20, 25, 30, 45, 60, None):
    c = simulate(np.full(n, -16.), time_check=mn)
    r = {'выход из минуса': 'нет' if mn is None else f'{mn} мин'}
    for ep, a, b_ in EPOCHS:
        s = money(c, (year >= a) & (year <= b_))
        r[ep] = f"{s['total']:>9,} / {s['rr']}"
    rows.append(r)
print(pd.DataFrame(rows).to_string(index=False))

print()
print('=== 5. Узнавание: сильнее ли сигнал, когда close дальше от mid (удержание 120) ===')
for ep, a, b_ in EPOCHS:
    sel = (year >= a) & (year <= b_)
    q = pd.qcut(disp[sel], 5, labels=['1 ближе', '2', '3', '4', '5 дальше'])
    g = pd.DataFrame({'q': q, 'candles': final[sel]}).groupby('q', observed=True).candles
    print(f'  {ep}: ' + '  '.join(f'{k} {v:+.2f}' for k, v in g.mean().items()))

print()
print('=== 6. Ширина свечи: где явление слабеет (удержание 120) ===')
for ep, a, b_ in EPOCHS:
    sel = (year >= a) & (year <= b_)
    q = pd.qcut(t['unit'][sel].round(4), 5, labels=False, duplicates='drop')
    g = pd.DataFrame({'q': q, 'candles': final[sel]}).groupby('q', observed=True).candles
    print(f'  {ep}: ' + '  '.join(f'{k} {v:+.2f}' for k, v in g.mean().items()))
