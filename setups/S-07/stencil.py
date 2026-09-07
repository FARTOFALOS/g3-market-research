"""S-07: сделка как фильм. Свинги, ноги, перелои — и стоп как место, а не порог.

Здесь не считаются распределения просадки. Каждый день раскладывается на
структуру в свечах: где первый противоположный экстремум, на какой свече он
подтверждается, сносят ли его дальше, сколько свечей в ногах. Потом все дни
кладутся друг на друга по одному якорю — минуте входа — и смотрится, что
повторяется по месту.

Свинг считается подтверждённым по k свечам с каждой стороны; k берётся 1, 2 и 3
сразу, чтобы видеть, устойчива ли картина к тому, как назван свинг. Подтверждение
приходит на k свечей позже самой точки, и структурный стоп это учитывает:
до первого подтверждения структурного уровня нет.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'setups/S-07'))
import run as base          # noqa: E402
import manage as M          # noqa: E402

HERE = Path(__file__).resolve().parent
MIN, HOLD, MINUTE, EPOCHS = M.MIN, M.HOLD, M.MINUTE, M.EPOCHS
KS = (1, 2, 3)


def films(ins='NQ'):
    """Минутный фильм каждой сделки в свечах: путь close, худшая и лучшая точка минуты."""
    m = base.market(ins)
    u = base.unit(m)
    cal = pd.read_parquet(ROOT / 'setups/S-04/calendar.parquet')
    ts = m['close_ts_utc_ns']
    p = np.searchsorted(ts, cal.open_ns.to_numpy()) + MINUTE
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
    keep = np.isfinite(unit) & (unit > 0)
    p, b, idx, mid, side, entry, unit, hi, lo = [x[keep] for x in (p, b, idx, mid, side, entry, unit, hi, lo)]
    j = b[:, None] + np.arange(HOLD)
    s = side[:, None]
    e, un = entry[:, None], unit[:, None]
    path = s * (m['close'][j] - e) / un                                   # где закрылась минута
    worst = s * (np.where(s > 0, m['low'][j], m['high'][j]) - e) / un      # худшая точка минуты
    best = s * (np.where(s > 0, m['high'][j], m['low'][j]) - e) / un       # лучшая точка минуты
    live = ts[j] < cal.close_ns.to_numpy()[idx][:, None]
    return dict(dates=cal.date.to_numpy()[idx], side=side, entry=entry, unit=unit,
                mid_from_entry=(mid - entry) * side / unit,
                far_edge=(np.where(side > 0, lo, hi) - entry) * side / unit,
                path=path, worst=worst, best=best, live=live,
                point=base.POINT[ins], cost=base.COST[ins])


def swings(worst, best, live, k):
    """Подтверждённые свинги одного фильма.

    Возвращает два списка (индекс точки, уровень, индекс подтверждения):
    низы по худшей точке минуты и верхи по лучшей.
    """
    n = int(live.sum())
    lows, highs = [], []
    for i in range(k, n - k):
        w = worst[i - k:i + k + 1]
        if worst[i] == w.min() and (worst[i] < w[:k]).all() and (worst[i] <= w[k + 1:]).all():
            lows.append((i, float(worst[i]), i + k))
        h = best[i - k:i + k + 1]
        if best[i] == h.max() and (best[i] > h[:k]).all() and (best[i] >= h[k + 1:]).all():
            highs.append((i, float(best[i]), i + k))
    return lows, highs


def structure(f, k):
    """Структура каждого дня: первый свинг-низ, его снос, ноги."""
    n = len(f['dates'])
    rows = []
    for i in range(n):
        live = f['live'][i]
        m = int(live.sum())
        if m < 3 * k + 2:
            continue
        w, b = f['worst'][i][:m], f['best'][i][:m]
        lows, highs = swings(w, b, live, k)
        final = float(f['path'][i][m - 1])
        rec = dict(date=str(f['dates'][i]), win=final > 0, final=final,
                   n_low=len(lows), n_high=len(highs), minutes=m)
        if lows:
            j, lvl, conf = lows[0]
            rec.update(first_low_at=j, first_low_depth=lvl, first_low_confirmed_at=conf)
            after = w[conf + 1:]
            broke = np.where(after <= lvl)[0]
            rec['broke_first_low'] = bool(len(broke))
            rec['broke_at'] = int(conf + 1 + broke[0]) if len(broke) else None
        # ноги: чередование подтверждённых свингов по времени точки
        pts = sorted([(j, 'L') for j, _, _ in lows] + [(j, 'H') for j, _, _ in highs])
        legs, prev = [], None
        for j, kind in pts:
            if prev is not None and kind != prev[1]:
                legs.append(j - prev[0])
            if prev is None or kind != prev[1]:
                prev = (j, kind)
        rec['legs'] = len(legs)
        rec['leg_candles_med'] = float(np.median(legs)) if legs else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def structural_stop(f, k, buffer_candles=0.25, hard=32.):
    """Стоп за последним ПОДТВЕРЖДЁННЫМ свинг-низом. До подтверждения — только жёсткий предел."""
    n = len(f['dates'])
    out = np.full(n, np.nan)
    for i in range(n):
        live = f['live'][i]
        m = int(live.sum())
        w, b, path = f['worst'][i][:m], f['best'][i][:m], f['path'][i][:m]
        lows, _ = swings(w, b, live, k)
        conf = {c: lvl for _, lvl, c in lows}
        level = -hard
        res = None
        for step in range(m):
            if w[step] <= level:
                res = level
                break
            if step in conf:
                level = max(-hard, min(conf[step] - buffer_candles, -buffer_candles))
        out[i] = path[m - 1] if res is None else res
    return out


def money(candles, f, sel):
    usd = candles * f['unit'] * f['point'] - f['cost']
    d = pd.Series(usd[sel], index=f['dates'][sel]).groupby(level=0).sum().sort_index()
    eq = np.r_[0, np.cumsum(d.to_numpy())]
    dd = float((np.maximum.accumulate(eq) - eq).max())
    return dict(total=round(float(d.sum())), dd=round(dd),
                rr=round(float(d.sum()) / dd, 2) if dd else None,
                med=round(float(d.median())), worst=round(float(d.min())))


def flat_stop(f, level, hard_only=True):
    n = len(f['dates'])
    out = np.full(n, np.nan)
    for i in range(n):
        m = int(f['live'][i].sum())
        w, path = f['worst'][i][:m], f['path'][i][:m]
        hit = np.where(w <= -level)[0]
        out[i] = -level if len(hit) else path[m - 1]
    return out


def main():
    f = films('NQ')
    year = pd.Series(f['dates']).str.slice(0, 4).to_numpy()
    report = {}

    print('=== ТРАФАРЕТ: где стоит первый противоположный экстремум ===')
    for k in KS:
        st = structure(f, k)
        st['year'] = st.date.str.slice(0, 4)
        has = st.dropna(subset=['first_low_at'])
        print(f'\n  свинг подтверждается по {k} свечам с каждой стороны, дней {len(st)}')
        for lab, sel in [('дни в плюс', has.win), ('дни в минус', ~has.win)]:
            g = has.loc[sel]
            print(f'    {lab:<11} n={len(g):>4}  '
                  f'первый низ на свече: медиана {g.first_low_at.median():4.0f}, '
                  f'25/75 {g.first_low_at.quantile(.25):.0f}/{g.first_low_at.quantile(.75):.0f}  |  '
                  f'глубина: медиана {g.first_low_depth.median():6.2f}, '
                  f'25/75 {g.first_low_depth.quantile(.25):.2f}/{g.first_low_depth.quantile(.75):.2f}')
        print(f'    снос первого низа:  в плюс {has.loc[has.win].broke_first_low.mean():.1%}, '
              f'в минус {has.loc[~has.win].broke_first_low.mean():.1%}')
        br = has.loc[has.broke_first_low.fillna(False)]
        print(f'    после сноса день закрылся в плюсе: {br.win.mean():.1%} (n={len(br)})')
        nb = has.loc[~has.broke_first_low.fillna(True)]
        print(f'    без сноса закрылся в плюсе:        {nb.win.mean():.1%} (n={len(nb)})')
        print(f'    ноги: медиана {st.leg_candles_med.median():.1f} свечей, '
              f'свингов за 120 минут медиана {st.legs.median():.0f}')
        report[f'k{k}'] = dict(
            n=len(st),
            first_low_at_median_win=float(has.loc[has.win].first_low_at.median()),
            first_low_at_median_loss=float(has.loc[~has.win].first_low_at.median()),
            first_low_depth_median_win=round(float(has.loc[has.win].first_low_depth.median()), 2),
            first_low_depth_median_loss=round(float(has.loc[~has.win].first_low_depth.median()), 2),
            broke_share_win=round(float(has.loc[has.win].broke_first_low.mean()), 3),
            broke_share_loss=round(float(has.loc[~has.win].broke_first_low.mean()), 3),
            win_after_break=round(float(br.win.mean()), 3),
            win_without_break=round(float(nb.win.mean()), 3),
            leg_candles_median=float(st.leg_candles_med.median()),
            legs_median=float(st.legs.median()))

    print('\n=== СТОП КАК МЕСТО: за подтверждённым свингом против плоского порога ===')
    rows = []
    for k in KS:
        c = structural_stop(f, k)
        r = {'стоп': f'за свингом k={k}'}
        for ep, a, b in EPOCHS:
            s = money(c, f, (year >= a) & (year <= b))
            r[ep] = f"{s['total']:>9,} / {s['rr']}"
        s = money(c, f, year >= '2020')
        r['медиана'] = s['med']
        r['худший день'] = s['worst']
        rows.append(r)
    for lvl in (16., 20., 24., 32.):
        c = flat_stop(f, lvl)
        r = {'стоп': f'плоский {lvl:.0f} свечей'}
        for ep, a, b in EPOCHS:
            s = money(c, f, (year >= a) & (year <= b))
            r[ep] = f"{s['total']:>9,} / {s['rr']}"
        s = money(c, f, year >= '2020')
        r['медиана'] = s['med']
        r['худший день'] = s['worst']
        rows.append(r)
    grid = pd.DataFrame(rows)
    print(grid.to_string(index=False))
    grid.to_csv(HERE / 'stencil_v1.csv', index=False, lineterminator='\n')
    (HERE / 'stencil_v1.json').write_text(
        json.dumps({'what': 'S-07: структура фильма сделки и стоп за подтверждённым свингом',
                    'caveat': 'разбор на уже просмотренной истории; независимым подтверждением не является',
                    'structure': report}, ensure_ascii=False, indent=1),
        encoding='utf-8', newline='\n')


if __name__ == '__main__':
    main()
