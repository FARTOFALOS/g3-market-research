"""PV1 (LOG.md): толчок → первый пивот → продолжение; FILM, трафарет отношений подтверждающих свечей, действие,
контроль якоря (пивоты без толчка). Объявлено до счёта."""
import numpy as np, pandas as pd
from numba import njit
from trade import Tape, simulate, TICK, epoch

pd.set_option('display.width', 260); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 200)


@njit(cache=True)
def films(o, h, l, c, mod, dt, u, inwin, push):
    """push=1: якорь — толчок (close > max high 15 предыдущих); push=0: контроль — любой пивот без условия толчка,
    нога = максимум 15 минут до пивота. Возвращает строки (T0, p, cidx, side, LH, pivot)."""
    n = len(o)
    R = []
    for t in range(16, n - 40):
        if not inwin[t] or dt[t - 15] != dt[t] or mod[t] - mod[t - 15] != 15:
            continue
        for s in (1, -1):
            if push == 1:
                mx = h[t - 15]; mn = l[t - 15]
                for j in range(t - 15, t):
                    mx = max(mx, h[j]); mn = min(mn, l[j])
                if not ((s > 0 and c[t] > mx) or (s < 0 and c[t] < mn)):
                    continue
            else:
                # контроль: каждая 15-я минута как «якорь» без толчка
                if mod[t] % 15 != 0:
                    continue
            LH = h[t] if s > 0 else l[t]
            base = l[t] if s > 0 else h[t]
            found = -1
            for p in range(t + 1, t + 31):
                if p + 2 >= n or dt[p + 2] != dt[t] or mod[p + 2] - mod[t] != p + 2 - t or not inwin[p + 2]:
                    break
                if (s > 0 and l[p] < base) or (s < 0 and h[p] > base):
                    break
                if s > 0:
                    ok = l[p] < l[p - 1] and l[p] <= l[p + 1] and l[p] <= l[p + 2] and LH - l[p] >= u[t]
                else:
                    ok = h[p] > h[p - 1] and h[p] >= h[p + 1] and h[p] >= h[p + 2] and h[p] - LH >= u[t]
                if ok:
                    # нога до пивота (включая p+1, p+2 — они уже видны к c)
                    for j in range(t, p + 3):
                        LH = max(LH, h[j]) if s > 0 else min(LH, l[j])
                    found = p
                    break
                LH = max(LH, h[p]) if s > 0 else min(LH, l[p])
            if found < 0:
                continue
            p = found
            R.append((t, p, p + 2, s, LH, l[p] if s > 0 else h[p]))
    return R


def relations(T, p, sd):
    """узор подтверждающих свечей p+1, p+2 (в системе стороны): для каждой — close > high пред. (A), close > close
    пред. (B), тело по стороне (C)."""
    o, h, l, c = T.o, T.h, T.l, T.c
    out = []
    for k in (1, 2):
        j = p + k
        if sd > 0:
            A = c[j] > h[j - 1]; B = c[j] > c[j - 1]; C = c[j] > o[j]
        else:
            A = c[j] < l[j - 1]; B = c[j] < c[j - 1]; C = c[j] < o[j]
        out.append(('A' if A else 'B' if B else 'b') + ('+' if C else '-'))
    return np.array([a + '|' + b for a, b in zip(*out)]) if isinstance(out[0], np.ndarray) else out


def build(T, push):
    n = len(T.o)
    inwin = np.zeros(n, bool)
    for D, ix in T.days.items():
        if D <= 20251231:
            inwin[ix[T.mod[ix] <= 950]] = True
    u = np.nan_to_num(T.u, nan=1e9)
    R = np.array(films(T.o, T.h, T.l, T.c, T.mod, T.date, u, inwin, push))
    F = pd.DataFrame(R, columns=['t0', 'p', 'cidx', 'side', 'LH', 'piv'])
    for k in ['t0', 'p', 'cidx', 'side']:
        F[k] = F[k].astype(np.int64)
    F['date'] = T.date[F.cidx]; F['epoch'] = F.date.map(epoch); F['win'] = np.where(T.mod[F.cidx] < 570, 'pre', 'NY')
    # узор
    o, h, l, c = T.o, T.h, T.l, T.c
    pat = []
    for k in (1, 2):
        j = F.p.to_numpy() + k; s = F.side.to_numpy()
        A = np.where(s > 0, c[j] > h[j - 1], c[j] < l[j - 1]); B = np.where(s > 0, c[j] > c[j - 1], c[j] < c[j - 1])
        C = np.where(s > 0, c[j] > o[j], c[j] < o[j])
        pat.append(np.where(A, 'A', np.where(B, 'B', 'b')) + np.where(C, '+', '-'))
    F['pat'] = pd.Series(pat[0]) + '|' + pd.Series(pat[1])
    e = o[F.cidx + 1]
    F['entry'] = e
    F['p0'] = np.clip((F.side * (e - F.piv)) / (F.side * (F.LH - F.piv)), 0, 1)
    return F


def outcome(T, F, horizon=60):
    """FILM: коснулось ли LH+тик раньше пивота−тик (в горизонте), с цензурой."""
    y = np.full(len(F), np.nan)
    for q, r in enumerate(F.itertuples()):
        i = r.cidx + 1; end = min(i + horizon, T.days[r.date][-1] + 1)
        hh = T.h[i:end]; ll = T.l[i:end]
        if r.side > 0:
            a = np.nonzero(hh >= r.LH + TICK)[0]; b = np.nonzero(ll <= r.piv - TICK)[0]
        else:
            a = np.nonzero(ll <= r.LH - TICK)[0]; b = np.nonzero(hh >= r.piv + TICK)[0]
        ta = a[0] if len(a) else 10**6; tb = b[0] if len(b) else 10**6
        if ta < tb: y[q] = 1
        elif tb < ta: y[q] = 0
        elif ta == tb and ta < 10**6: y[q] = 0
    F['y'] = y
    return F


if __name__ == '__main__':
    T = Tape()
    res = {}
    for push, nm in [(1, 'push'), (0, 'ctrl')]:
        F = build(T, push)
        F = F[(F.p0 > 0) & (F.p0 < 1)]
        F = outcome(T, F)
        # действие: цель LH, отмена пивот, время 5 и 15
        for tm in (5, 15):
            sig = pd.DataFrame(dict(date=F.date, irec=F.cidx, side=F.side, stop=F.piv - F.side * TICK, target=F.LH))
            s = simulate(T, sig, tmax=tm)
            F[f'net{tm}'] = s.net.to_numpy(); F[f'fits{tm}'] = s.fits.to_numpy(); F[f'x{tm}'] = s.xtype.to_numpy()
            F[f'struct{tm}'] = F[f'net{tm}'] + 0.75 + 0.5 * (F[f'x{tm}'] == 1)
        F.to_parquet(f'pv1_{nm}.parquet')
        res[nm] = F
        g = F[F.fits5 & (F.epoch != '2026')]
        print(nm, 'films', len(F))
        print(g.groupby(['win', 'epoch']).agg(n=('y', 'size'), Y=('y', 'mean'), p0=('p0', 'mean'),
              risk=('entry', lambda x: 0), net5=('net5', 'mean'), st5=('struct5', 'mean'), net15=('net15', 'mean'),
              st15=('struct15', 'mean')).drop(columns='risk').round(3).to_string())
    # трафарет: узор подтверждения — толчок против контроля, NY и pre, по эпохам
    P = res['push']; C = res['ctrl']
    rows = []
    for nm, F in [('push', P), ('ctrl', C)]:
        g = F[F.fits5 & (F.epoch != '2026')]
        for (w, pat, ep), x in g.groupby(['win', 'pat', 'epoch']):
            rows.append(dict(anchor=nm, win=w, pat=pat, ep=ep, n=len(x), dY=(x.y - x.p0).mean(), st5=x.struct5.mean(),
                             net5=x.net5.mean(), t5=x.net5.mean() / x.net5.std() * np.sqrt(len(x)) if len(x) > 2 else np.nan))
    S = pd.DataFrame(rows)
    W = S.pivot_table(index=['win', 'pat', 'anchor'], columns='ep', values=['n', 'dY', 'st5', 'net5']).round(3)
    print(W.to_string())
    S.to_csv('pv1_stencil.csv', index=False)
