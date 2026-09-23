"""Пакет v0-2 (LOG.md): день за своей нормой (сцена S-18), отмена — граница нормы. Объявлено до счёта."""
import numpy as np, pandas as pd
from numba import njit
from trade import Tape, simulate, report, TICK, SLIP, COST, LIMIT, epoch

pd.set_option('display.width', 260); pd.set_option('display.max_columns', 40)
MARKS = list(range(30, 361, 30))


def sessions(T):
    """матрица сессий: для каждой даты с баром 09:30 — индексы баров минут k = 1..390 (или −1)."""
    rows = []
    for D, ix in T.days.items():
        m = T.mod[ix]
        ny = ix[m >= 570]
        if len(ny) == 0 or T.mod[ny[0]] != 570:
            continue
        K = np.full(391, -1, np.int64)
        k = T.mod[ny] - 570 + 1
        ok = (k >= 1) & (k <= 390)
        K[k[ok]] = ny[ok]
        rows.append((D, K, int(ny[-1])))
    return rows


def bands(T, look=14):
    S = sessions(T)
    n = len(S)
    Cm = np.full((n, 391), np.nan)
    O = np.full(n, np.nan); last = np.full(n, np.nan)
    for s, (D, K, il) in enumerate(S):
        v = K >= 0
        Cm[s, v] = T.c[K[v]]
        O[s] = T.o[K[1]]; last[s] = T.c[il]
    move = np.abs(Cm / O[:, None] - 1)
    UBbar = np.full(len(T.o), np.nan); LBbar = np.full(len(T.o), np.nan)
    info = []
    for s in range(look, n):
        hist = move[s - look:s]
        cnt = np.isfinite(hist).sum(0)
        with np.errstate(invalid='ignore', all='ignore'):
            norm = np.where(cnt >= 10, np.nanmean(hist, 0), np.nan)
        D, K, il = S[s]
        prev = last[s - 1]
        up = max(O[s], prev) * (1 + norm); dn = min(O[s], prev) * (1 - norm)
        v = K >= 0
        UBbar[K[v]] = up[v]; LBbar[K[v]] = dn[v]
        info.append((D, K, up, dn))
    return info, UBbar, LBbar


def signals(T, info):
    rows = []
    for D, K, up, dn in info:
        zp = 0
        for k in MARKS:
            i = K[k]
            if i < 0 or not np.isfinite(up[k]):
                zp = None
                break
            p = T.c[i]
            z = 1 if p > up[k] else (-1 if p < dn[k] else 0)
            kind = 'F' if (z != 0 and z != zp) else ('S' if z != 0 else ('R' if zp != 0 else 'I'))
            if z != 0:
                b = up[k] if z > 0 else dn[k]
                rows.append(dict(date=D, irec=int(i), side=z, stop=b - z * TICK, target=np.nan, kind=kind, k=k,
                                 beyond=z * (p - b)))
            zp = z
    return pd.DataFrame(rows)


@njit(cache=True)
def _trail(o, h, l, c, mod, dt, irec, side, stop0, ilast, UB, LB, slip, tick):
    n = len(irec)
    px = np.full(n, np.nan); typ = np.zeros(n, np.int8); jx = np.full(n, -1); mae = np.full(n, np.nan); mfe = np.full(n, np.nan)
    for k in range(n):
        i = irec[k]
        if i + 1 >= len(o) or dt[i + 1] != dt[i] or mod[i + 1] != mod[i] + 1 or i + 1 > ilast[k]:
            typ[k] = -1; continue
        j = i + 1; s = side[k]; e = o[j]; st = stop0[k]
        if s * (e - st) <= 0:
            typ[k] = -2; continue
        mf = 0.0; ma = 0.0
        while True:
            if s > 0:
                if l[j] <= st:
                    ma = max(ma, e - st); px[k] = min(o[j], st) - slip; typ[k] = 1; break
                mf = max(mf, h[j] - e); ma = max(ma, e - l[j])
                if UB[j] == UB[j]:
                    st = max(st, UB[j] - tick)
            else:
                if h[j] >= st:
                    ma = max(ma, st - e); px[k] = max(o[j], st) + slip; typ[k] = 1; break
                mf = max(mf, e - l[j]); ma = max(ma, h[j] - e)
                if LB[j] == LB[j]:
                    st = min(st, LB[j] + tick)
            if j >= ilast[k]:
                px[k] = c[j]; typ[k] = 3; break
            if j + 1 >= len(o) or dt[j + 1] != dt[j] or mod[j + 1] != mod[j] + 1:
                px[k] = c[j]; typ[k] = 4; break
            j += 1
        jx[k] = j; mae[k] = ma; mfe[k] = mf
    return px, typ, jx, mae, mfe


def simulate_trail(T, sig, UB, LB, tmax=120):
    s = simulate(T, sig, tmax)       # вход, риск и допуск считаются так же; выход пересчитываем
    dayend = np.array([T.days[d][-1] for d in s.date])
    ilast = np.minimum(s.irec.to_numpy() + tmax, dayend).astype(np.int64)
    px, typ, jx, mae, mfe = _trail(T.o, T.h, T.l, T.c, T.mod, T.date, s.irec.to_numpy().astype(np.int64),
                                   s.side.to_numpy().astype(np.int64), s.stop.to_numpy().astype(float), ilast, UB, LB, SLIP, TICK)
    s['exit'] = px; s['xtype'] = typ; s['jexit'] = jx; s['mae'] = mae; s['mfe'] = mfe
    s['net'] = s.side * (s.exit - s.entry) - COST
    s['hold'] = s.jexit - s.irec
    s['fits'] = (s.xtype > 0) & (s.plan_loss <= LIMIT)
    return s


if __name__ == '__main__':
    T = Tape()
    info, UB, LB = bands(T)
    sig = signals(T, info)
    print(sig.groupby([sig.date.map(epoch), 'kind']).size().unstack().to_string())
    allr = []
    for kind, lab in [('S', 'V7S'), ('F', 'V7F')]:
        s = simulate(T, sig[sig.kind == kind])
        s.to_csv(f'{lab}_trades.csv', index=False)
        allr += report(s, lab)
    s = simulate_trail(T, sig[sig.kind == 'S'], UB, LB)
    s.to_csv('V7T_trades.csv', index=False)
    allr += report(s, 'V7T')
    r = pd.DataFrame(allr)
    r.to_csv('v0_s18.csv', index=False)
    print(r.to_string(index=False))
