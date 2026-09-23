"""MZ1 модель (LOG.md): зеркально-симметричная гребневая регрессия по корзинам признаков × сессия.
Обучение 2006–2017, выбор λ на 2018–2019 (корреляция прогноза с y30), проверка 2020–2025 один раз."""
import numpy as np, pandas as pd

pd.set_option('display.width', 250)
X = pd.read_parquet('mz1_features.parquet')
X['sess'] = np.select([X['mod'] < 570, X['mod'] < 660], [0, 1], 2)
PAIRS = [('dHH', 'dLLd'), ('dPH', 'dPL'), ('dAH', 'dAL'), ('dLH', 'dLL'), ('dH1h', 'dH1l'), ('dUB', 'dLB'),
         ('dPiv_h', 'dPiv_l'), ('sinceH', 'sinceL'), ('swPH', 'swPL'), ('swAH', 'swAL')]
FLIP = ['r1', 'r5', 'r15', 'r30', 'r60', 'r120', 'd02', 'd930', 'dPC', 'body1']
ONE_MINUS = ['pos', 'cpos1']
SIGNLESS_PAIR = {'sinceH', 'sinceL', 'swPH', 'swPL', 'swAH', 'swAL'}
NEUTRAL = ['rng_u', 'rng1', 'u']
for a in ['dPC', 'dPH', 'dPL', 'dAH', 'dAL', 'dLH', 'dLL']:
    X[a + '_na'] = X[a].isna().astype(float)
X = X.fillna(0.0)
CLIP = 20.0
for k in FLIP + [p for pr in PAIRS for p in pr if p not in SIGNLESS_PAIR] + ['y15', 'y30']:
    X[k] = X[k].clip(-CLIP, CLIP)
X['sinceH'] = np.log1p(X.sinceH); X['sinceL'] = np.log1p(X.sinceL)
X['u_rel'] = X.u / X.groupby('date').u.transform('first')


def mirror(D):
    M = D.copy()
    for k in FLIP:
        M[k] = -D[k]
    for a, b in PAIRS:
        if a in SIGNLESS_PAIR:
            M[a], M[b] = D[b], D[a]
        else:
            M[a], M[b] = -D[b], -D[a]
    for k in ONE_MINUS:
        M[k] = 1 - D[k]
    M['y15'] = -D.y15; M['y30'] = -D.y30; M['p15'] = -D.p15; M['p30'] = -D.p30
    return M


FEATS = FLIP + [p for pr in PAIRS for p in pr] + ONE_MINUS + ['rng_u', 'rng1', 'u_rel']
NB = 8
train = X[X.date <= 20171231]; valid = X[(X.date >= 20180101) & (X.date <= 20191231)]; test = X[X.date >= 20200101]
TR = pd.concat([train, mirror(train)], ignore_index=True)
edges = {}
for f in FEATS:
    v = TR[f].to_numpy()
    q = np.unique(np.quantile(v, np.linspace(0, 1, NB + 1)[1:-1]))
    edges[f] = q


def design(D):
    cols = []
    sess = D.sess.to_numpy()
    tb = np.clip((D['mod'].to_numpy() - 120) // 60, 0, 13)
    blocks = []
    for f in FEATS:
        b = np.searchsorted(edges[f], D[f].to_numpy())
        nb = len(edges[f]) + 1
        for s_ in range(3):
            Z = np.zeros((len(D), nb), np.float32)
            sel = sess == s_
            Z[np.nonzero(sel)[0], b[sel]] = 1.0
            blocks.append(Z)
    Zt = np.zeros((len(D), 14), np.float32); Zt[np.arange(len(D)), tb] = 1.0
    blocks.append(Zt)
    return np.hstack(blocks)


def fit(D, lam):
    y = D.y30.to_numpy().astype(np.float64)
    p = None; XtX = None; Xty = None
    for a in range(0, len(D), 100000):
        Z = design(D.iloc[a:a + 100000]).astype(np.float64)
        if XtX is None:
            XtX = np.zeros((Z.shape[1], Z.shape[1])); Xty = np.zeros(Z.shape[1])
        XtX += Z.T @ Z; Xty += Z.T @ y[a:a + 100000]
    return np.linalg.solve(XtX + lam * np.eye(len(Xty)), Xty)


def predict(D, w):
    out = np.empty(len(D))
    for a in range(0, len(D), 100000):
        out[a:a + 100000] = design(D.iloc[a:a + 100000]) @ w
    return out


def evaluate(D, pred, label):
    D = D.assign(pred=pred)
    D['dec'] = pd.qcut(D.pred.rank(method='first'), 10, labels=False)
    g = D.groupby('dec').agg(n=('y30', 'size'), pred=('pred', 'mean'), y30=('y30', 'mean'), p30=('p30', 'mean'), p15=('p15', 'mean'))
    cor = np.corrcoef(D.pred, D.y30)[0, 1]
    print(label, 'corr', round(cor, 4)); print(g.round(3).to_string())
    return cor


if __name__ == '__main__':
    best = None
    for lam in [1e2, 1e3, 1e4, 1e5]:
        w = fit(TR, lam)
        cv = np.corrcoef(predict(valid, w), valid.y30)[0, 1]
        print('lambda', lam, 'valid corr', round(cv, 4), flush=True)
        if best is None or cv > best[0]:
            best = (cv, lam, w)
    cv, lam, w = best
    print('chosen lambda', lam)
    evaluate(valid, predict(valid, w), 'VALID 2018-19')
    np.save('mz1_w.npy', w)
    # проверка 2020–2025 — один раз
    pt = predict(test, w)
    evaluate(test, pt, 'TEST 2020-25')
    for s_, nm in [(0, 'pre'), (1, 'NYAM'), (2, 'NYPM')]:
        m_ = test.sess.to_numpy() == s_
        print(nm, 'test corr', round(np.corrcoef(pt[m_], test.y30.to_numpy()[m_])[0, 1], 4))
    test.assign(pred=pt)[['date', 't', 'mod', 'sess', 'pred', 'y30', 'p30', 'p15', 'u']].to_parquet('mz1_test_pred.parquet')
