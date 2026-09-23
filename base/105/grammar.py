"""G1 (LOG.md): грамматика конкретных версий — 9 триггеров × 4 окна. Объявлено до счёта."""
import numpy as np, pandas as pd
from numba import njit
from trade import Tape, simulate, TICK

pd.set_option('display.width', 260); pd.set_option('display.max_rows', 400)


@njit(cache=True)
def contig(mod, dt, a, b):
    return dt[a] == dt[b] and mod[b] - mod[a] == b - a


@njit(cache=True)
def box_kernel(o, h, l, c, mod, dt, u, inwin, N, tick):
    n = len(o)
    T_ = []; I_ = []; S_ = []; P_ = []
    for i in range(N + 1, n - 31):
        if not inwin[i] or not contig(mod, dt, i - N, i):
            continue
        H2 = h[i - N]; L2 = l[i - N]
        for j in range(i - N, i):
            H2 = max(H2, h[j]); L2 = min(L2, l[j])
        if not (c[i - 1] <= H2 and c[i - 1] >= L2):
            continue
        for s in (1, -1):
            if (s > 0 and c[i] > H2) or (s < 0 and c[i] < L2):
                lvl = H2 if s > 0 else L2
                comp = (H2 - L2) < 4 * u[i - 1]
                st = (l[i] - tick) if s > 0 else (h[i] + tick)
                T_.append(1); I_.append(i); S_.append(s); P_.append(st)
                if comp:
                    T_.append(2); I_.append(i); S_.append(s); P_.append(st)
                X = h[i] if s > 0 else l[i]
                for j in range(i + 1, i + 6):
                    if not contig(mod, dt, i, j) or not inwin[j]:
                        break
                    X = max(X, h[j]) if s > 0 else min(X, l[j])
                    if (s > 0 and c[j] < lvl) or (s < 0 and c[j] > lvl):
                        T_.append(3); I_.append(j); S_.append(-s); P_.append((X + tick) if s > 0 else (X - tick))
                        break
                for j in range(i + 1, i + 31):
                    if not contig(mod, dt, i, j) or not inwin[j]:
                        break
                    if (s > 0 and l[j] <= lvl) or (s < 0 and h[j] >= lvl):
                        if (s > 0 and c[j] > lvl) or (s < 0 and c[j] < lvl):
                            T_.append(4); I_.append(j); S_.append(s); P_.append((l[j] - tick) if s > 0 else (h[j] + tick))
                        break
    return np.array(T_), np.array(I_), np.array(S_), np.array(P_)


@njit(cache=True)
def five_kernel(o, h, l, c, mod, dt, inwin, tick):
    n = len(o)
    cnt = 0
    for i in range(4, n):
        if mod[i] % 5 == 4 and contig(mod, dt, i - 4, i):
            cnt += 1
    E = np.empty(cnt, np.int64); k = 0
    for i in range(4, n):
        if mod[i] % 5 == 4 and contig(mod, dt, i - 4, i):
            E[k] = i; k += 1
    m = len(E)
    O5 = np.empty(m); H5 = np.empty(m); L5 = np.empty(m); C5 = np.empty(m)
    for k in range(m):
        i = E[k]
        O5[k] = o[i - 4]; C5[k] = c[i]
        hh = h[i - 4]; ll = l[i - 4]
        for j in range(i - 3, i + 1):
            hh = max(hh, h[j]); ll = min(ll, l[j])
        H5[k] = hh; L5[k] = ll
    T_ = []; I_ = []; S_ = []; P_ = []
    for k in range(3, m):
        i = E[k]
        if not inwin[i]:
            continue
        ok1 = E[k] - E[k - 1] == 5 and dt[E[k]] == dt[E[k - 1]]
        ok2 = ok1 and E[k - 1] - E[k - 2] == 5 and dt[E[k - 1]] == dt[E[k - 2]]
        ok3 = ok2 and E[k - 2] - E[k - 3] == 5 and dt[E[k - 2]] == dt[E[k - 3]]
        if ok2 and H5[k - 1] <= H5[k - 2] and L5[k - 1] >= L5[k - 2]:
            md = 1 if C5[k - 2] > O5[k - 2] else -1
            if md > 0 and C5[k] > H5[k - 1]:
                T_.append(5); I_.append(i); S_.append(1); P_.append(L5[k - 1] - tick)
            elif md < 0 and C5[k] < L5[k - 1]:
                T_.append(5); I_.append(i); S_.append(-1); P_.append(H5[k - 1] + tick)
        if ok1 and H5[k] > H5[k - 1] and L5[k] < L5[k - 1]:
            r = H5[k] - L5[k]
            if C5[k] >= H5[k] - 0.25 * r:
                T_.append(6); I_.append(i); S_.append(1); P_.append(L5[k] - tick)
            elif C5[k] <= L5[k] + 0.25 * r:
                T_.append(6); I_.append(i); S_.append(-1); P_.append(H5[k] + tick)
        if ok3:
            if C5[k] > C5[k - 1] and C5[k - 1] > C5[k - 2] and C5[k - 2] > C5[k - 3]:
                T_.append(7); I_.append(i); S_.append(1); P_.append(L5[k] - tick)
                T_.append(8); I_.append(i); S_.append(-1); P_.append(H5[k] + tick)
            elif C5[k] < C5[k - 1] and C5[k - 1] < C5[k - 2] and C5[k - 2] < C5[k - 3]:
                T_.append(7); I_.append(i); S_.append(-1); P_.append(H5[k] + tick)
                T_.append(8); I_.append(i); S_.append(1); P_.append(L5[k] - tick)
    return np.array(T_), np.array(I_), np.array(S_), np.array(P_)


@njit(cache=True)
def double_kernel(o, h, l, c, mod, dt, u, inwin, starts, stops, tick):
    T_ = []; I_ = []; S_ = []; P_ = []
    for d in range(len(starts)):
        a = starts[d]; b = stops[d]
        HH = h[a]; tH = mod[a]; LL = l[a]; tL = mod[a]
        for i in range(a + 1, b + 1):
            if h[i] > HH:
                HH = h[i]; tH = mod[i]
                continue
            if l[i] < LL:
                LL = l[i]; tL = mod[i]
                continue
            if not inwin[i]:
                continue
            if mod[i] - tH >= 15 and h[i] >= HH - 0.5 * u[i - 1] and c[i] < o[i]:
                T_.append(9); I_.append(i); S_.append(-1); P_.append(HH + tick)
            if mod[i] - tL >= 15 and l[i] <= LL + 0.5 * u[i - 1] and c[i] > o[i]:
                T_.append(9); I_.append(i); S_.append(1); P_.append(LL - tick)
    return np.array(T_), np.array(I_), np.array(S_), np.array(P_)


NAMES = {1: 'T1 box-break', 2: 'T2 squeeze-break', 3: 'T3 failed-break', 4: 'T4 retest', 5: 'T5 inside5', 6: 'T6 engulf5',
         7: 'T7c three5-cont', 8: 'T7f three5-fade', 9: 'T8 double-test'}


def window(m):
    return np.select([m < 570, m < 660, m < 840], ['pre', 'open', 'mid'], 'late')


def build(T, y1=20251231):
    n = len(T.o)
    inwin = np.zeros(n, bool)
    for D, ix in T.days.items():
        if D <= y1:
            inwin[ix[T.mod[ix] <= 930]] = True
    u = np.nan_to_num(T.u, nan=1e9)
    parts = [box_kernel(T.o, T.h, T.l, T.c, T.mod, T.date, u, inwin, 30, TICK),
             five_kernel(T.o, T.h, T.l, T.c, T.mod, T.date, inwin, TICK)]
    ds = sorted(D for D in T.days if D <= y1)
    starts = np.array([T.days[D][0] for D in ds]); stops = np.array([T.days[D][-1] for D in ds])
    parts.append(double_kernel(T.o, T.h, T.l, T.c, T.mod, T.date, u, inwin, starts, stops, TICK))
    S = pd.concat([pd.DataFrame(dict(trig=a, irec=b, side=c_, stop=d)) for a, b, c_, d in parts], ignore_index=True)
    S['date'] = T.date[S.irec]; S['target'] = np.nan
    S['win'] = window(T.mod[S.irec])
    return S


if __name__ == '__main__':
    T = Tape()
    S = build(T)
    print('signals', len(S), S.groupby('trig').size().to_dict(), flush=True)
    s = simulate(T, S, tmax=60)
    s.to_parquet('g1_trades.parquet')
    f = s[s.fits & (s.epoch != 'other')].copy()
    f['struct'] = f.net + 0.75 + 0.5 * (f.xtype == 1)
    d = f.groupby(['trig', 'win', 'epoch', 'date']).agg(net=('net', 'sum'), st=('struct', 'sum'), k=('net', 'size')).reset_index()
    agg = d.groupby(['trig', 'win', 'epoch']).agg(n=('k', 'sum'), days=('k', 'size'), net_sum=('net', 'sum'), st_sum=('st', 'sum'),
                                                   net_dsd=('net', 'std'), st_dsd=('st', 'std')).reset_index()
    agg['net'] = agg.net_sum / agg.n; agg['struct'] = agg.st_sum / agg.n
    agg['t_net'] = (agg.net_sum / agg.days) / agg.net_dsd * np.sqrt(agg.days)
    agg['t_st'] = (agg.st_sum / agg.days) / agg.st_dsd * np.sqrt(agg.days)
    W = agg.pivot_table(index=['trig', 'win'], columns='epoch', values=['n', 'net', 'struct', 't_net', 't_st'])
    W.columns = [f'{a}_{b}' for a, b in W.columns]
    W = W.reset_index(); W['trig'] = W.trig.map(NAMES)
    cols = ['trig', 'win', 'n_2020-25', 'struct_2006-12', 'struct_2013-19', 'struct_2020-25', 't_st_2006-12', 't_st_2013-19',
            't_st_2020-25', 'net_2020-25', 't_net_2020-25']
    W.to_csv('g1_versions.csv', index=False)
    print(W[cols].round(2).to_string(index=False))
    fits = s[s.xtype > 0].groupby(['trig', 'epoch']).fits.mean().unstack().round(2)
    print(fits.to_string())
