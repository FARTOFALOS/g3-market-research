"""DS1 (LOG.md, объявлено до счёта): форма дня — сборка политики из живых ветвей.
Первая попытка дня: 09:34 по σ S-07 v1, цель +X (X = 2, 3, 5, 8 пт; без цели — выход 11:33), защита 18,5.
Отыгрыш (только если день ниже нуля): PV2 V7m (первый пивот после выхода за норму, отмена — пивот, время 120),
PV2 S07 (первый пивот по σ, время 120), V7m, V2 — как посчитаны, без изменений.
Правила дня: одна позиция; бюджет 60; предел = min(20, остаток); «замок»: при плюсе дня предел = min(20, плюс),
попытка с пределом < 3 пт пропускается. Территория 2020–25; все требования + какие дни проваливаются.
"""
import math, numpy as np, pandas as pd
from trade import simulate, TICK
from composite import s07_signals, run, req
from trade import Tape
T = Tape()
import composite

pd.set_option('display.width', 250)
parts = []
for nm, f in [('PV2_V7m', 'PV2_V7m_120.csv'), ('PV2_S07', 'PV2_S07_120.csv'), ('V7m', 'V7m_trades.csv'), ('v2', 'v2_trades.csv')]:
    s = pd.read_csv(f); s['branch'] = nm
    if 'iend' not in s:
        s['iend'] = -1
    parts.append(s)
rec = pd.concat(parts, ignore_index=True)
cols = ['date', 'irec', 'side', 'stop', 'target', 'iend', 'branch', 'entry', 'exit', 'xtype', 'jexit', 'risk', 'plan_loss', 'net', 'fits']
base = s07_signals(T, cap=18.5)
out = []
for X in (12.0, 15.0, 20.0, 30.0):
    b = base.copy()
    b['target'] = np.nan if X is None else b.stop + b.side * (18.5 + X)
    s = simulate(T, b); s['branch'] = f'S07x{X}'; s['iend'] = b.iend.values
    lib = pd.concat([s[cols], rec[cols]], ignore_index=True)
    tk = run(T, lib, lock=True, min_lim=3.0)
    r, day = req(tk)
    first = tk.groupby('date').head(1)
    r.update(X=X, first_win=round((first.net > 0).mean(), 3), first_mean=round(first.net.mean(), 2))
    out.append(r)
    print(X, tk.groupby('branch').net.agg(['size', 'mean', 'sum']).round(2).to_dict('index'))
print(pd.DataFrame(out).to_string(index=False))
