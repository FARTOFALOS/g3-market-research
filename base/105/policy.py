"""Политика дня и проверка требований контракта для одной ветви (или композита — сигналы нескольких ветвей вместе).

Одна позиция одновременно (сигнал при занятой позиции пропускается), постоянный размер, дневной бюджет 60 пт от начала
дня; предел следующей попытки = min(20, остаток бюджета): если плановая потеря идеи (риск + 1,25) больше предела —
защитный уровень придвигается до предела (решение политики, влияние входит в результат); предел < 1 пт — пропуск.
Территория требований — допустимые даты calendar_nq.csv 2020-01-01…2025-12-31; 2026 — отдельно (диагностика).
"""
import math, numpy as np, pandas as pd
from trade import simulate, COST, SLIP, LIMIT, DAY_LIMIT

ELIG = ('regular', 'short')


def run_policy(T, trades):
    """trades — результат simulate (все сигналы). Возвращает (сделки политики, дни)."""
    t = trades[trades.xtype > 0].sort_values(['date', 'irec'])
    taken = []
    for D, g in t.groupby('date', sort=True):
        pnl = 0.0; busy = -1
        for r in g.itertuples():
            if r.irec + 1 <= busy or not r.fits:
                continue
            rem = DAY_LIMIT + pnl
            lim = min(LIMIT, rem)
            row = r._asdict()
            if r.plan_loss > lim:
                newrisk = lim - COST - SLIP
                if newrisk < 1.0:
                    continue
                sig = pd.DataFrame([dict(date=D, irec=r.irec, side=r.side, stop=r.entry - r.side * newrisk,
                                         target=r.target, iend=getattr(r, 'iend', -1) if hasattr(r, 'iend') else -1)])
                x = simulate(T, sig).iloc[0]
                for k in ('exit', 'xtype', 'jexit', 'net', 'risk', 'plan_loss'):
                    row[k] = x[k]
                row['cut_by_budget'] = True
            pnl += row['net']; busy = row['jexit']
            row['day_pnl_after'] = pnl
            taken.append(row)
    tk = pd.DataFrame(taken)
    return tk


def requirements(T, tk, label, y0=20200101, y1=20251231):
    cal = pd.read_csv('calendar_nq.csv')
    terr = cal[(cal.date >= y0) & (cal.date <= y1)]
    elig = terr[terr.status.isin(ELIG) | ((terr.status == 'special') & terr['last'].notna())]
    unknown = terr[terr.status == 'unknown']
    x = tk[(tk.date >= y0) & (tk.date <= y1)]
    day = x.groupby('date').agg(net=('net', 'sum'), n=('net', 'size'), worst_cum=('day_pnl_after', 'min'))
    total = day.net.sum()
    k = math.ceil(0.05 * len(day)) if len(day) else 0
    top = day.net.nlargest(k).sum() if k else np.nan
    res = dict(branch=label, elig_days=len(elig), unknown_days=len(unknown), trade_days=len(day),
               daily_cover=round(len(day) / max(len(elig), 1), 3),
               trades=len(x), total_net=round(total, 1), per_trade=round(x.net.mean(), 2) if len(x) else np.nan,
               per_day=round(day.net.mean(), 2) if len(day) else np.nan,
               t_day=round(day.net.mean() / day.net.std() * np.sqrt(len(day)), 2) if len(day) > 2 else np.nan,
               days_ge0=round((day.net >= 0).mean(), 3) if len(day) else np.nan,
               worst_day=round(day.net.min(), 1) if len(day) else np.nan,
               day_limit_breach=int((day.worst_cum < -DAY_LIMIT).sum()),
               trade_over20=int((x.net < -LIMIT).sum()), worst_trade=round(x.net.min(), 2) if len(x) else np.nan,
               top5pct_share=round(top / total, 2) if total > 0 else np.nan,
               budget_cut=int(x.get('cut_by_budget', pd.Series(dtype=bool)).fillna(False).sum()) if len(x) else 0)
    yrs = x.assign(y=x.date // 10000).groupby('y').net.sum().round(0).to_dict()
    res['by_year'] = yrs
    return res, day
