"""Вопрос 8: сторона S-07 v1, защитный предел 17 пт, повторный вход, выход 11:33."""
import numpy as np, pandas as pd
from tape import load_minutes
COST, SLIP, CAP, BUDGET, MAXATT = 0.75, 0.5, 17.0, 60.0, 3
def run(inst, y0, y1, cap=CAP):
    df = load_minutes(f'{y0-1}-12-28', '2026-07-11' if y1 == 2026 else f'{y1}-12-31', inst)
    out = []
    for d, g in df.groupby('date'):
        if d // 10000 < y0: continue
        g = g.set_index('mod')
        need = list(range(543, 694))   # 09:03 … 11:33
        if not all(m in g.index for m in need): out.append(dict(date=d, status='gap')); continue
        pre = g.loc[543:572]           # 30 закрытых минут, последняя — 09:33 (открытие 09:32→ close 09:33)
        # минута с открытием 09:32 закрывается в 09:33; решение на её close
        H, L = pre.h.max(), pre.l.min(); mid = (H + L) / 2
        side = 1 if g.at[572, 'c'] > mid else -1
        pnl = 0.0; att = 0; k = 573; loss = 0.0
        while att < MAXATT and k <= 692:
            rem = BUDGET - loss
            lim = min(cap, rem - COST - SLIP)
            if lim <= 1: break
            e = g.at[k, 'o']; att += 1; stop = e - side * lim; exited = False
            for j in range(k, 693):
                o_, h_, l_ = g.at[j, 'o'], g.at[j, 'h'], g.at[j, 'l']
                if j > k and side * (o_ - stop) <= 0:           # открытие за пределом
                    r = side * (o_ - e) - COST; exited = True
                elif side * ((l_ if side > 0 else h_) - stop) <= 0:
                    r = -lim - SLIP - COST; exited = True
                if exited:
                    pnl += r; loss = max(loss, -pnl) if pnl < 0 else loss; k = j + 1; break
            if not exited:
                pnl += side * (g.at[692, 'c'] - e) - COST; break
        out.append(dict(date=d, status='ok', side=side, attempts=att, pnl=pnl,
                        hold=side * (g.at[692, 'c'] - g.at[573, 'o']) - COST))
    return pd.DataFrame(out)
def summ(x, label):
    x = x.dropna(); tr = x[x != 0]; k = int(np.ceil(0.05 * len(tr)))
    top = tr.sort_values(ascending=False).iloc[:k].sum()
    return dict(label=label, days=len(x), mean=round(x.mean(), 2), t=round(x.mean() / x.std() * np.sqrt(len(x)), 2),
                share_ge0=round((x >= 0).mean(), 3), worst=round(x.min(), 1), top5_over_net=round(top / tr.sum(), 2) if tr.sum() > 0 else None)
rows = []
for inst, y0, y1 in [('NQ', 2020, 2026), ('NQ', 2013, 2019), ('ES', 2020, 2026)]:
    r = run(inst, y0, y1); r.to_csv(f's07cap_{inst}_{y0}.csv', index=False); ok = r[r.status == 'ok']
    rows.append(summ(ok.pnl, f'{inst} {y0}-{y1 % 100} cap17+reentry'))
    rows.append(summ(ok.hold, f'{inst} {y0}-{y1 % 100} hold120 no cap'))
    if inst == 'NQ' and y0 == 2020:
        ok = ok.assign(yr=ok.date // 10000)
        print(ok.groupby('yr').agg(days=('pnl', 'size'), cap=('pnl', 'mean'), hold=('hold', 'mean'), att=('attempts', 'mean')).round(2).to_string())
pd.set_option('display.width', 200); print(pd.DataFrame(rows).to_string(index=False))
