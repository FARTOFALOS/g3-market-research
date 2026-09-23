"""OB-1 шаг (b): зонд различимости ценности действия S07 в 09:33 по префиксу. Гребневая по корзинам + категории;
цели: net S07, срабатывание защиты, отрицательный день F0. Обучение 2006–2017, выбор λ на 2018–2019, 2020–2025 —
основная территория (не чистый OOS). Затем — допуск по прогнозу: ценность действия и доля отрицательных дней в
оставленных днях против всех (по территориям)."""
import numpy as np, pandas as pd

pd.set_option('display.width', 250)
X = pd.read_parquet('obs1_dp1.parquet')
NUM = ['u7', 'u_pre', 'u7_rel', 'dM_pt', 'dM_u', 'mv4', 'rng4', 'gap', 'lon_rng', 'asia_rng', 'y_rng', 'y_dir', 'y_loc']
CAT = ['open_loc_sg', 'dow', 'sg']
X['dayneg'] = (X.day_F0 < 0).astype(float)
tr = X[X.date <= 20171231]; va = X[(X.date >= 20180101) & (X.date <= 20191231)]; te = X[X.date >= 20200101]
edges = {f: np.unique(np.nanquantile(tr[f], [0.2, 0.4, 0.6, 0.8])) for f in NUM}


def design(D):
    B = []
    for f in NUM:
        v = D[f].to_numpy(); b = np.searchsorted(edges[f], np.nan_to_num(v, nan=np.nanmedian(tr[f])))
        Z = np.zeros((len(D), len(edges[f]) + 1)); Z[np.arange(len(D)), b] = 1; B.append(Z)
    for f in CAT:
        levels = sorted(X[f].astype(str).unique())
        Z = np.zeros((len(D), len(levels)))
        for k, lv in enumerate(levels):
            Z[:, k] = (D[f].astype(str) == lv).to_numpy()
        B.append(Z)
    return np.hstack(B)


def ridge(D, y, lam):
    Z = design(D); return np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ y)


out = {}
for target in ['s07_net', 's07_stop', 'dayneg']:
    best = None
    for lam in [1, 10, 100, 1000]:
        w = ridge(tr, tr[target].to_numpy(), lam)
        c = np.corrcoef(design(va) @ w, va[target])[0, 1]
        if best is None or c > best[0]:
            best = (c, lam, w)
    c, lam, w = best
    pte = design(te) @ w
    out[target] = pte
    print(f'{target}: λ {lam}, valid corr {c:.3f}, main 2020-25 corr {np.corrcoef(pte, te[target])[0, 1]:.3f}')
te = te.assign(p_net=out['s07_net'], p_stop=out['s07_stop'], p_neg=out['dayneg'])
va_pred = {}
print('\nдопуск по прогнозу ценности (2020–25): оставить дни с прогнозом выше квантиля q')
for q in [0, 0.25, 0.5, 0.75]:
    thr = te.p_net.quantile(q)
    k = te[te.p_net >= thr]
    print(f'q {q}: дней {len(k)}, S07 net {k.s07_net.mean():.2f}, защита {k.s07_stop.mean():.2f}, день F0 < 0 {k.dayneg.mean():.3f}, '
          f'итог F0 {k.day_F0.sum():.0f}')
print('\nпо прогнозу отрицательного дня (оставить дни с низким риском):')
for q in [1, 0.75, 0.5, 0.25]:
    thr = te.p_neg.quantile(q)
    k = te[te.p_neg <= thr]
    print(f'q {q}: дней {len(k)}, S07 net {k.s07_net.mean():.2f}, день F0 < 0 {k.dayneg.mean():.3f}, итог F0 {k.day_F0.sum():.0f}')
# вклад групп признаков в прогноз ценности
w = ridge(tr, tr['s07_net'].to_numpy(), 100); Z = design(te); off = 0
for f in NUM + CAT:
    n = (len(edges[f]) + 1) if f in NUM else len(sorted(X[f].astype(str).unique()))
    part = Z[:, off:off + n] @ w[off:off + n]; off += n
    print(f, 'var', round(part.var(), 3), 'weights', np.round(w[off - n:off], 2))
