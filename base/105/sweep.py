"""SW1 (LOG.md, объявлено до счёта): снятие ликвидности старшего уровня → слом структуры телом на минутке → вход в
обратную сторону (линза трейдера 2026-09-23).
Уровни (известны до события): PD — H/L вчерашней сессии 09:30–закрытие; H1 — H/L предыдущего часового бара (часы
по :00); H4 — H/L предыдущего 4-часового бара (блоки от 18:00 ET: 18, 22, 02, 06, 10, 14); AS — H/L 18:00–01:59;
LN — H/L 02:00–09:29 (для NY). Снятие — первая минута в окне, чей high выше уровня (low ниже). Структура — последний
подтверждённый пивот против (для снятия максимума — последний пивот-минимум: low[p] < low[p−1], ≤ low[p+1], ≤ low[p+2],
подтверждён к минуте снятия). Слом — первая минута в 30 мин после снятия, закрывшаяся телом за этим пивотом (close <
пивот-минимум), при этом X = экстремум с момента снятия. Вход open следующей; отмена — X ± тик (новый экстремум за
снятием = идея разворота кончилась); цель — противоположный край 60 минут до снятия (противоположная ликвидность);
время 60. Окна (по минуте слома): LON 02:00–04:59, NYAM 09:30–10:59, NYPM 13:30–15:44; прочее — отдельно.
Эпохи 2006–12 / 2013–19 / 2020–25; 2026 в поиске не показывается (слово трейдера: год на потом).
"""
import numpy as np, pandas as pd
from trade import Tape, simulate, TICK

pd.set_option('display.width', 260); pd.set_option('display.max_rows', 300)
T = Tape()
o, h, l, c, mod, dt = T.o, T.h, T.l, T.c, T.mod, T.date
n = len(o)
# подтверждённые пивоты: цена последнего подтверждённого пивота к каждой минуте (в пределах даты)
pl = np.zeros(n, bool); ph = np.zeros(n, bool)
idx = np.arange(1, n - 2)
same = (dt[idx - 1] == dt[idx]) & (dt[idx + 2] == dt[idx])
pl[idx] = same & (l[idx] < l[idx - 1]) & (l[idx] <= l[idx + 1]) & (l[idx] <= l[idx + 2])
ph[idx] = same & (h[idx] > h[idx - 1]) & (h[idx] >= h[idx + 1]) & (h[idx] >= h[idx + 2])
lastPL = np.full(n, np.nan); lastPH = np.full(n, np.nan)
cp = np.nonzero(pl)[0]; lastPL[cp + 2] = l[cp]
cp = np.nonzero(ph)[0]; lastPH[cp + 2] = h[cp]
df = pd.DataFrame(dict(d=dt, a=lastPL, b=lastPH))
lastPL = df.groupby('d').a.ffill().to_numpy(); lastPH = df.groupby('d').b.ffill().to_numpy()

# часовые и 4-часовые бары по всей ленте (ключ — абсолютный час)
ts_et = pd.to_datetime(T.o * 0 + 0)  # заглушка не нужна
et = (T.date.astype(np.int64) * 10000 + (mod // 60) * 100)  # yyyymmddHH00 по дате и часу открытия
hk = pd.Series(et)
H1 = pd.DataFrame(dict(k=et, h=h, l=l)).groupby('k').agg(h=('h', 'max'), l=('l', 'min'))
blk = ((mod - 1080) % 1440) // 240          # 4-часовые блоки от 18:00
# блок 18:00–21:59 относится к торговой дате следующего дня; для ключа берём (дата, блок) и сортировку по времени
k4 = pd.Series(list(zip(dt, blk)))
first_idx = {}
rows = []
dates = sorted(D for D in T.days if D <= 20251231)
H1d = H1.to_dict('index')
prev_ny = None
for D in dates:
    ix = T.days[D]; m = mod[ix]
    Dp = int((pd.Timestamp(str(D)) - pd.Timedelta(days=1)).strftime('%Y%m%d'))
    lv = []
    if prev_ny is not None:
        lv.append(('PD', prev_ny[0], prev_ny[1], 120, 945))
    if Dp in T.date_start:
        a0 = T.date_start[Dp]; a1 = T.date_start[D]
        pa = np.arange(a0, a1); pa = pa[mod[pa] >= 1080]
        da = np.arange(a1, ix[0]); da = da[mod[da] < 120]
        asia = np.concatenate([pa, da])
        if len(asia) >= 400 and mod[asia[0]] == 1080:
            lv.append(('AS', h[asia].max(), l[asia].min(), 120, 945))
    lon = ix[(m >= 120) & (m <= 569)]
    if len(lon) >= 440:
        lv.append(('LN', h[lon].max(), l[lon].min(), 570, 945))
    # H1: для каждого часа 03..15 — уровни предыдущего часа, окно — этот час
    for hr in range(3, 16):
        kp = D * 10000 + (hr - 1) * 100
        if kp in H1d:
            lv.append(('H1', H1d[kp]['h'], H1d[kp]['l'], hr * 60, hr * 60 + 59))
    # H4: блоки 02–06, 06–10, 10–14, 14–16 — уровни предыдущего блока
    for b0 in (360, 600, 840):
        prv = ix[(m >= b0 - 240) & (m < b0)]
        if len(prv) >= 200:
            lv.append(('H4', h[prv].max(), l[prv].min(), b0, min(b0 + 239, 945)))
    for nm, LH, LL, w0, w1 in lv:
        sc = ix[(m >= w0) & (m <= w1)]
        for sd in (-1, 1):          # sd = сторона сделки: −1 — после снятия максимума
            hit = np.nonzero(h[sc] > LH)[0] if sd < 0 else np.nonzero(l[sc] < LL)[0]
            if not len(hit):
                continue
            s0 = sc[hit[0]]
            struct = lastPL[s0] if sd < 0 else lastPH[s0]
            if not np.isfinite(struct):
                continue
            pre = np.arange(max(s0 - 60, ix[0]), s0)
            if len(pre) < 30:
                continue
            tgt = l[pre].min() if sd < 0 else h[pre].max()
            X = h[s0] if sd < 0 else l[s0]
            for j in range(s0, min(s0 + 31, ix[-1])):
                if dt[j] != D or mod[j] - mod[s0] != j - s0:
                    break
                X = max(X, h[j]) if sd < 0 else min(X, l[j])
                if (sd < 0 and c[j] < struct) or (sd > 0 and c[j] > struct):
                    rows.append(dict(date=D, irec=j, side=sd, stop=X - sd * TICK, target=tgt, lvl=nm, mbrk=int(mod[j])))
                    break
    ny = ix[m >= 570]
    prev_ny = (h[ny].max(), l[ny].min()) if len(ny) >= 300 else None
S = pd.DataFrame(rows)
S['win'] = np.select([S.mbrk < 300, (S.mbrk >= 570) & (S.mbrk < 660), (S.mbrk >= 810) & (S.mbrk < 945)], ['LON', 'NYAM', 'NYPM'], 'other')
s = simulate(T, S, tmax=60)
s = s[(s.xtype > 0) & (s.epoch != '2026')].copy()
s.to_parquet('SW1_trades.parquet')
f = s[s.fits].copy()
f['struct'] = f.net + 0.75 + 0.5 * (f.xtype == 1)
f['p0'] = f.risk / (f.risk + f.side * (f.target - f.entry))
f['res'] = f.xtype.isin([1, 2]); f['hit'] = (f.xtype == 2).astype(float)
print('fits share by epoch', s.groupby('epoch').fits.mean().round(2).to_dict())
g = f.groupby(['lvl', 'win', 'epoch']).apply(lambda x: pd.Series(dict(
    n=len(x), dY=(x[x.res].hit - x[x.res].p0).mean(), struct=x.struct.mean(), net=x.net.mean(),
    t=x.net.mean() / x.net.std() * np.sqrt(len(x)), risk=x.risk.median())), include_groups=False)
print(g.round(2).unstack('epoch').to_string())
