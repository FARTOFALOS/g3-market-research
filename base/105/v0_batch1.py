"""Пакет v0-1 (LOG.md, «Режим торговая конструкция»): шесть грубых конструкций, объявлены до счёта."""
import sys, numpy as np, pandas as pd
from trade import Tape, simulate, report, TICK, epoch

pd.set_option('display.width', 260); pd.set_option('display.max_columns', 40)


def seg(T, ix, m0, m1):
    m = T.mod[ix]
    return ix[(m >= m0) & (m <= m1)]


def or15(T, ix):
    m = T.mod[ix]
    k0 = np.searchsorted(m, 570)
    if k0 + 15 > len(ix) or m[k0] != 570 or m[k0 + 14] != 584:
        return None
    b = ix[k0:k0 + 15]
    return T.h[b].max(), T.l[b].min()


def v1(T):
    rows = []
    for D, ix in T.days.items():
        r = or15(T, ix)
        if r is None:
            continue
        ORH, ORL = r
        sc = seg(T, ix, 585, 719)
        for side in (1, -1):
            br = np.nonzero(T.l[sc] < ORL)[0] if side > 0 else np.nonzero(T.h[sc] > ORH)[0]
            if not len(br):
                continue
            a = br[0]; X = np.inf if side > 0 else -np.inf
            for b in range(a, len(sc)):
                i = sc[b]
                if T.mod[i] - T.mod[sc[a]] >= 15:
                    break
                X = min(X, T.l[i]) if side > 0 else max(X, T.h[i])
                if (side > 0 and T.c[i] > ORL) or (side < 0 and T.c[i] < ORH):
                    rows.append(dict(date=D, irec=i, side=side, stop=X - side * TICK, target=ORH if side > 0 else ORL))
                    break
    return pd.DataFrame(rows)


def v2(T):
    rows = []
    for D, ix in T.days.items():
        r = or15(T, ix)
        if r is None:
            continue
        ORH, ORL = r
        sc = seg(T, ix, 585, 719)
        for side in (1, -1):
            br = np.nonzero(T.c[sc] > ORH)[0] if side > 0 else np.nonzero(T.c[sc] < ORL)[0]
            if not len(br):
                continue
            b0 = sc[br[0]]; u = T.u[b0]
            HB = T.h[b0] if side > 0 else T.l[b0]
            pulled = False; PL = None
            for i in sc[br[0] + 1:]:
                if not pulled:
                    HB = max(HB, T.h[i]) if side > 0 else min(HB, T.l[i])
                    if (side > 0 and T.l[i] <= HB - u) or (side < 0 and T.h[i] >= HB + u):
                        pulled = True; PL = T.l[i] if side > 0 else T.h[i]
                else:
                    PL = min(PL, T.l[i]) if side > 0 else max(PL, T.h[i])
                if pulled:
                    if (side > 0 and PL < ORL) or (side < 0 and PL > ORH):
                        break
                    if (side > 0 and T.c[i] > HB) or (side < 0 and T.c[i] < HB):
                        rows.append(dict(date=D, irec=i, side=side, stop=PL - side * TICK, target=np.nan))
                        break
    return pd.DataFrame(rows)


def s07_side(T, D, ix):
    m = T.mod[ix]
    k = np.searchsorted(m, 544)
    if k + 30 > len(ix) or m[k] != 544 or m[k + 29] != 573:
        return 0, -1
    b = ix[k:k + 30]
    mid = (T.h[b].max() + T.l[b].min()) / 2
    return (1 if T.c[b[-1]] > mid else -1), int(b[-1])


def v3(T):
    rows = []
    for D, ix in T.days.items():
        sg, i33 = s07_side(T, D, ix)
        if sg == 0:
            continue
        i1133 = T.at(D, 693)
        if i1133 < 0:
            continue
        m = T.mod[ix]
        pre = ix[(m >= 570) & (m <= 573)]
        HH = T.h[pre].max() if sg > 0 else T.l[pre].min()
        pulled = False; PL = None; PLbar = -1
        for i in seg(T, ix, 574, 692):
            if not pulled:
                HH = max(HH, T.h[i]) if sg > 0 else min(HH, T.l[i])
                if (sg > 0 and T.l[i] <= HH - T.u[i]) or (sg < 0 and T.h[i] >= HH + T.u[i]):
                    pulled = True; PL = T.l[i] if sg > 0 else T.h[i]; PLbar = i
                continue
            if (sg > 0 and T.l[i] < PL) or (sg < 0 and T.h[i] > PL):
                PL = T.l[i] if sg > 0 else T.h[i]; PLbar = i; continue
            if (sg > 0 and T.c[i] > T.h[PLbar]) or (sg < 0 and T.c[i] < T.l[PLbar]):
                rows.append(dict(date=D, irec=i, side=sg, stop=PL - sg * TICK, target=np.nan, iend=i1133))
                pulled = False
                HH = max(HH, T.h[i]) if sg > 0 else min(HH, T.l[i])
    return pd.DataFrame(rows)


def v4(T):
    o, h, l, c, u, mod, dt = T.o, T.h, T.l, T.c, T.u, T.mod, T.date
    rng = h - l
    n = len(o)
    up = np.zeros(n, bool); up[1:] = True
    cand = np.nonzero((rng[1:-1] >= 3 * u[:-2]) & (np.abs(c[1:-1] - o[1:-1]) >= 0.7 * rng[1:-1]))[0] + 1
    rows = []
    for i in cand:
        if not (120 <= mod[i] and i + 7 < n):
            continue
        D = int(dt[i])
        if D not in T.days or mod[i] >= T.close_mod[i] - 2:
            continue
        j1 = i + 1
        if dt[j1] != D or mod[j1] != mod[i] + 1 or not (h[j1] <= h[i] and l[j1] >= l[i]):
            continue
        sd = 1 if c[i] > o[i] else -1
        ext = l[j1] if sd > 0 else h[j1]
        for j in range(j1 + 1, j1 + 6):
            if dt[j] != D or mod[j] != mod[j - 1] + 1 or mod[j] >= T.close_mod[j]:
                break
            ext = min(ext, l[j]) if sd > 0 else max(ext, h[j])
            if (sd > 0 and c[j] < l[j1]) or (sd < 0 and c[j] > h[j1]):
                break
            if (sd > 0 and c[j] > h[j1]) or (sd < 0 and c[j] < l[j1]):
                tg = h[i] + rng[i] if sd > 0 else l[i] - rng[i]
                rows.append(dict(date=D, irec=j, side=sd, stop=ext - sd * TICK, target=tg))
                break
    return pd.DataFrame(rows)


def v5(T):
    rows = []
    for D, ix in T.days.items():
        ny = seg(T, ix, 570, 959)
        if len(ny) < 60 or T.mod[ny[0]] != 570:
            continue
        for side in (-1, 1):   # side −1: вынос вверх провалился → short
            ext = T.h[ny[0]] if side < 0 else T.l[ny[0]]; t_ext = T.mod[ny[0]]
            k = 1
            while k < len(ny):
                i = ny[k]
                broke = (T.h[i] > ext) if side < 0 else (T.l[i] < ext)
                if broke and 630 <= T.mod[i] <= 900 and T.mod[i] - t_ext >= 15:
                    LV = ext; X = T.h[i] if side < 0 else T.l[i]; done = False
                    for kk in range(k, min(k + 5, len(ny))):
                        ii = ny[kk]
                        if T.mod[ii] - T.mod[i] >= 5:
                            break
                        X = max(X, T.h[ii]) if side < 0 else min(X, T.l[ii])
                        if (side < 0 and T.c[ii] < LV) or (side > 0 and T.c[ii] > LV):
                            seen = ny[:kk + 1]
                            mid = (T.h[seen].max() + T.l[seen].min()) / 2
                            rows.append(dict(date=D, irec=ii, side=side, stop=X - side * TICK, target=mid))
                            done = True
                            break
                    if done:
                        break
                if broke:
                    ext = T.h[i] if side < 0 else T.l[i]; t_ext = T.mod[i]
                k += 1
    return pd.DataFrame(rows)


def v6(T):
    rows = []
    dates = sorted(T.days)
    for D in dates:
        ix = T.days[D]
        dd = pd.Timestamp(str(D)); P = int((dd - pd.Timedelta(days=1)).strftime('%Y%m%d'))
        if P not in T.date_start:
            continue
        p0 = T.date_start[P]
        pe = T.date_start[D]
        pi = np.arange(p0, pe); pi = pi[T.mod[pi] >= 1080]
        di = np.arange(pe, ix[-1] + 1); di = di[T.mod[di] < 570]
        ni = np.concatenate([pi, di])
        if len(ni) < 600 or T.mod[ni[0]] != 1080 or T.at(D, 570) < 0:
            continue
        ONH, ONL = T.h[ni].max(), T.l[ni].min(); mid = (ONH + ONL) / 2
        sc = seg(T, ix, 570, 719)
        for side in (-1, 1):
            br = np.nonzero(T.h[sc] > ONH)[0] if side < 0 else np.nonzero(T.l[sc] < ONL)[0]
            if not len(br):
                continue
            a = br[0]; X = -np.inf if side < 0 else np.inf
            for b in range(a, len(sc)):
                i = sc[b]
                if T.mod[i] - T.mod[sc[a]] >= 15:
                    break
                X = max(X, T.h[i]) if side < 0 else min(X, T.l[i])
                if (side < 0 and T.c[i] < ONH) or (side > 0 and T.c[i] > ONL):
                    rows.append(dict(date=D, irec=i, side=side, stop=X - side * TICK, target=mid))
                    break
    return pd.DataFrame(rows)


def film(T, s):
    """рентген: у стопнутых — дошла бы цена до цели / чем кончился бы исходный горизонт; MFE до стопа в долях риска."""
    f = s[s.fits & (s.xtype == 1)]
    after = []
    for r in f.itertuples():
        ix = T.days[r.date]; last = min(r.irec + 120, ix[-1])
        if hasattr(r, 'iend') and r.iend == r.iend and r.iend > 0:
            last = min(last, int(r.iend))
        path = np.arange(r.jexit, last + 1)
        if r.target == r.target:
            hit = (T.h[path] >= r.target).any() if r.side > 0 else (T.l[path] <= r.target).any()
        else:
            hit = np.nan
        after.append((hit, r.side * (T.c[last] - r.entry) - 0.75))
    a = pd.DataFrame(after, columns=['tgt_after', 'hold_net'], index=f.index)
    return a


if __name__ == '__main__':
    which = sys.argv[1:] or ['v1', 'v2', 'v3', 'v4', 'v5', 'v6']
    T = Tape()
    allr = []
    for v in which:
        sig = globals()[v](T)
        s = simulate(T, sig, tmax=60 if v == 'v4' else 120)
        s.to_csv(f'{v}_trades.csv', index=False)
        a = film(T, s)
        s = s.join(a)

        def extra(f):
            st = f[f.xtype == 1]; w = f[f.net > 0]
            return dict(stop_mfe_R=round((st.mfe / st.risk).median(), 2),
                        stop_then_tgt=round(st.tgt_after.astype(float).mean(), 2) if st.tgt_after.notna().any() else np.nan,
                        stop_hold_net=round(st.hold_net.mean(), 2), win_mae_R=round((w.mae / w.risk).median(), 2))
        allr += report(s, v, extra)
        dead = s.xtype.value_counts().to_dict()
        print(v, 'xtype counts', dead, flush=True)
    r = pd.DataFrame(allr)
    r.to_csv('v0_batch1.csv', index=False)
    print(r.to_string(index=False))
