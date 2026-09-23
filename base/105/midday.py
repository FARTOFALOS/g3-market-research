"""MD1 (LOG.md, объявлено до счёта): «новый минимум дня в полдень → отскок» — одиночная клетка S1 (2020–25: +8,5 пт за
30 мин, t 4,1; в 2006–19 ноль). Сцена: первая минута часа H (H = 11, 12, 13, 14 — соседние часы как контроль места) с
low ниже минимума дня с 02:00. Узнавание — её закрытие; действие — long на open следующей; отмена — low этой минуты −
тик (новый минимум ниже — «ниже дна» продолжилось); остаток — отскок, время 30. Зеркально — новый максимум (short).
Показать по годам 2013–2025 и эпохам: n (дни), net, struct, ход 30 мин без отмены; по часам — место или шум.
"""
import numpy as np, pandas as pd
from trade import Tape, simulate, TICK, epoch

pd.set_option('display.width', 250)
T = Tape()
rows = []
for D in sorted(T.days):
    if D > 20251231:
        break
    ix = T.days[D]; m = T.mod[ix]
    if len(ix) < 600 or m[0] != 120:
        continue
    hh = np.maximum.accumulate(T.h[ix]); ll = np.minimum.accumulate(T.l[ix])
    for H in (11, 12, 13, 14):
        for sd in (1, -1):
            sel = np.nonzero((m >= H * 60) & (m < H * 60 + 60))[0]
            for q in sel:
                if q == 0:
                    continue
                if (sd > 0 and T.l[ix[q]] < ll[q - 1]) or (sd < 0 and T.h[ix[q]] > hh[q - 1]):
                    i = ix[q]
                    rows.append(dict(date=D, irec=i, side=sd, stop=(T.l[i] - TICK) if sd > 0 else (T.h[i] + TICK), target=np.nan,
                                     hour=H, fwd30=sd * (T.c[min(i + 30, ix[-1])] - T.o[i + 1]) if i + 1 <= ix[-1] else np.nan))
                    break
S = pd.DataFrame(rows)
s = simulate(T, S, tmax=30)
s = s[s.xtype > 0].copy()
s['struct'] = s.net + 0.75 + 0.5 * (s.xtype == 1)
s['yr'] = s.date // 10000
s['kind'] = np.where(s.side > 0, 'newLow->long', 'newHigh->short')
print(s.groupby(['kind', 'hour', 'epoch']).agg(n=('net', 'size'), fwd30=('fwd30', 'mean'), net=('net', 'mean'), struct=('struct', 'mean'),
      risk=('risk', 'median')).round(2).unstack('epoch').to_string())
print('\nпо годам, 12:00 new low -> long:')
x = s[(s.kind == 'newLow->long') & (s.hour == 12)]
print(x.groupby('yr').agg(n=('net', 'size'), fwd30=('fwd30', 'mean'), net=('net', 'mean'), win=('net', lambda v: (v > 0).mean())).round(2).to_string())
s.to_csv('MD1_trades.csv', index=False)
