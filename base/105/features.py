"""MZ1 — мозаика косвенных признаков (слово трейдера 2026-09-23). Набор признаков на каждой 5-й минуте 02:00–15:25,
NQ 2006-01…2025-12 (2026 не входит). Только префикс: всё известно на закрытии минуты t; вход open t+1.
Единица масштаба u — медиана (h − l) 30 закрытых минут (разрешена контрактом). Исходы: y15, y30 = (close[t+m] −
open[t+1]) / u и в пунктах.
"""
import numpy as np, pandas as pd
from trade import Tape
from v0_s18 import bands

T = Tape()
o, h, l, c, mod, dt, u = T.o, T.h, T.l, T.c, T.mod, T.date, T.u
info, UB, LB = bands(T)
n = len(o)
# часовые бары (по дате и часу)
hourkey = dt * 100 + mod // 60
H1 = pd.DataFrame(dict(k=hourkey, h=h, l=l)).groupby('k').agg(h=('h', 'max'), l=('l', 'min'))
H1h = H1.h.to_dict(); H1l = H1.l.to_dict()
pl = np.zeros(n, bool); ph = np.zeros(n, bool)
ii = np.arange(1, n - 2)
same = (dt[ii - 1] == dt[ii]) & (dt[ii + 2] == dt[ii])
pl[ii] = same & (l[ii] < l[ii - 1]) & (l[ii] <= l[ii + 1]) & (l[ii] <= l[ii + 2])
ph[ii] = same & (h[ii] > h[ii - 1]) & (h[ii] >= h[ii + 1]) & (h[ii] >= h[ii + 2])

rows = []
prev_ny = None
for D in sorted(T.days):
    if D > 20251231:
        break
    ix = T.days[D]; m = mod[ix]
    ny = ix[m >= 570]
    Dp = int((pd.Timestamp(str(D)) - pd.Timedelta(days=1)).strftime('%Y%m%d'))
    asia = np.array([], dtype=np.int64)
    if Dp in T.date_start:
        a0 = T.date_start[Dp]; a1 = T.date_start[D]
        pa = np.arange(a0, a1); pa = pa[mod[pa] >= 1080]
        da = np.arange(a1, ix[0]); da = da[mod[da] < 120]
        asia = np.concatenate([pa, da])
    ok_asia = len(asia) >= 400 and mod[asia[0]] == 1080
    AH, AL = (h[asia].max(), l[asia].min()) if ok_asia else (np.nan, np.nan)
    if len(ix) < 400 or mod[ix[0]] != 120:
        prev_ny = (h[ny].max(), l[ny].min(), c[ny[-1]]) if len(ny) >= 300 else None
        continue
    O02 = o[ix[0]]
    i930 = T.at(D, 570); O930 = o[i930] if i930 >= 0 else np.nan
    lon = ix[(m >= 120) & (m <= 569)]
    LH, LL = (h[lon].max(), l[lon].min()) if len(lon) >= 440 else (np.nan, np.nan)
    PH, PLo, PC = prev_ny if prev_ny is not None else (np.nan, np.nan, np.nan)
    hh = np.maximum.accumulate(h[ix]); ll = np.minimum.accumulate(l[ix])
    # время с последнего обновления экстремума дня
    newH = np.r_[True, h[ix][1:] > hh[:-1]]; newL = np.r_[True, l[ix][1:] < ll[:-1]]
    tH = np.maximum.accumulate(np.where(newH, m, 0)); tL = np.maximum.accumulate(np.where(newL, m, 0))
    swPH = np.maximum.accumulate(h[ix] > PH) if np.isfinite(PH) else np.zeros(len(ix), bool)
    swPL = np.maximum.accumulate(l[ix] < PLo) if np.isfinite(PLo) else np.zeros(len(ix), bool)
    swAH = np.maximum.accumulate(h[ix] > AH) if ok_asia else np.zeros(len(ix), bool)
    swAL = np.maximum.accumulate(l[ix] < AL) if ok_asia else np.zeros(len(ix), bool)
    # последний подтверждённый пивот (к минуте)
    cPL = np.full(len(ix), np.nan); cPH = np.full(len(ix), np.nan)
    for q in range(len(ix)):
        pass
    pos = np.arange(len(ix))
    for q in range(5, len(ix) - 31, 5):
        t = ix[q]
        if m[q] > 925 or ix[q + 30] - t != 30 or m[q + 30] - m[q] != 30 or not np.isfinite(u[t]) or u[t] <= 0:
            continue
        ut = u[t]
        if t - 120 < 0 or dt[t - 120] != D and mod[t] >= 240:
            pass
        def ret(k):
            j = t - k
            return (c[t] - c[j]) / ut if j >= 0 else np.nan
        hr = m[q] // 60
        pk = D * 100 + hr - 1
        ph1, pl1 = H1h.get(pk, np.nan), H1l.get(pk, np.nan)
        rng = hh[q] - ll[q]
        # последние пивоты до t (подтверждены к t: индекс пивота ≤ t−2)
        seg = np.arange(max(ix[0], t - 60), t - 1)
        lp = seg[pl[seg]]; hp = seg[ph[seg]]
        r = dict(date=D, t=int(t), mod=int(m[q]), dow=pd.Timestamp(str(D)).dayofweek, u=ut,
                 r1=ret(1), r5=ret(5), r15=ret(15), r30=ret(30), r60=ret(60), r120=ret(120),
                 d02=(c[t] - O02) / ut, d930=((c[t] - O930) / ut) if (m[q] >= 570 and np.isfinite(O930)) else 0.0,
                 dPC=(c[t] - PC) / ut, dPH=(c[t] - PH) / ut, dPL=(c[t] - PLo) / ut,
                 dAH=(c[t] - AH) / ut, dAL=(c[t] - AL) / ut,
                 dLH=((c[t] - LH) / ut) if m[q] >= 570 else 0.0, dLL=((c[t] - LL) / ut) if m[q] >= 570 else 0.0,
                 dH1h=(c[t] - ph1) / ut, dH1l=(c[t] - pl1) / ut,
                 dHH=(c[t] - hh[q]) / ut, dLLd=(c[t] - ll[q]) / ut, pos=(c[t] - ll[q]) / rng if rng > 0 else 0.5,
                 rng_u=rng / ut, sinceH=m[q] - tH[q], sinceL=m[q] - tL[q],
                 swPH=int(swPH[q]), swPL=int(swPL[q]), swAH=int(swAH[q]), swAL=int(swAL[q]),
                 dUB=((c[t] - UB[t]) / ut) if np.isfinite(UB[t]) else 0.0, dLB=((c[t] - LB[t]) / ut) if np.isfinite(LB[t]) else 0.0,
                 dPiv_l=((c[t] - l[lp[-1]]) / ut) if len(lp) else 0.0, dPiv_h=((c[t] - h[hp[-1]]) / ut) if len(hp) else 0.0,
                 body1=(c[t] - o[t]) / ut, rng1=(h[t] - l[t]) / ut, cpos1=((c[t] - l[t]) / (h[t] - l[t])) if h[t] > l[t] else 0.5,
                 y15=(c[t + 15] - o[t + 1]) / ut, y30=(c[t + 30] - o[t + 1]) / ut,
                 p15=c[t + 15] - o[t + 1], p30=c[t + 30] - o[t + 1])
        rows.append(r)
    prev_ny = (h[ny].max(), l[ny].min(), c[ny[-1]]) if len(ny) >= 300 else None
X = pd.DataFrame(rows)
X.to_parquet('mz1_features.parquet')
print(X.shape, X.date.min(), X.date.max())
print(X.isna().mean().round(3)[X.isna().mean() > 0].to_dict())
