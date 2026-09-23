"""CL1 (LOG.md, объявлено до счёта): ближайшая политика из двух живых ветвей — заморожена как есть, не цель.
Ветви (правила без изменений): S07 — 09:34 по σ S-07 v1, защита 18,5, выход 11:33 (ежедневная возможность);
PV2_V7m — первый пивот отката после первой минуты выхода дня за норму (10:00–15:30), вход на подтверждении, отмена —
пивот ∓ тик, время 120. Правила дня: одна позиция; бюджет 60; предел = min(20, остаток); «замок»: при плюсе дня предел
= min(20, плюс), попытка с пределом < 3 пропускается. Территория требований 2020-01-01…2025-12-31 (2026 не считается —
слово трейдера). Печатает таблицу требований, годы, распределение дня, какие дни проваливаются.
"""
import math, numpy as np, pandas as pd
from trade import Tape, simulate
from composite import s07_signals, run, req

pd.set_option('display.width', 250)
T = Tape()
cols = ['date', 'irec', 'side', 'stop', 'target', 'iend', 'branch', 'entry', 'exit', 'xtype', 'jexit', 'risk', 'plan_loss', 'net', 'fits']
s07 = simulate(T, s07_signals(T)); s07['branch'] = 'S07'
pv = pd.read_csv('PV2_V7m_120.csv'); pv['branch'] = 'PV2_V7m'; pv['iend'] = -1
lib = pd.concat([s07[cols], pv[cols]], ignore_index=True)
tk = run(T, lib, lock=True, min_lim=3.0)
tk.to_csv('CL1_policy.csv', index=False)
r, day = req(tk)
print('REQUIREMENTS 2020-25:', r)
print(tk.groupby('branch').net.agg(['size', 'mean', 'sum']).round(2).to_string())
day['yr'] = (day.index // 10000)
print(day.groupby('yr').agg(days=('net', 'size'), total=('net', 'sum'), ge0=('net', lambda x: (x >= 0).mean())).round(2).to_string())
print('day net quantiles', day.net.quantile([0, .05, .25, .5, .75, .95, 1]).round(1).to_dict())
k = math.ceil(0.05 * len(day)); top = day.net.nlargest(k)
print('top 5% days:', k, 'sum', round(top.sum(), 1), 'min of top', round(top.min(), 1))
