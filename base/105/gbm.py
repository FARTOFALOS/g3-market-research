"""MZ2 (LOG.md, объявлено до счёта): рентген взаимодействий — градиентный бустинг деревьев глубины 3 на корзинах тех же
признаков MZ1 (+ сессия, час), зеркально-симметричное обучение, квадратичная потеря на y30 (обрезка ±20u).
Обучение 2006–2017, число деревьев выбирается по 2018–2019, 2020–2025 — основная территория (не чистый OOS).
Смысл — верхняя оценка различимости с взаимодействиями; сетапом модель не является (слово архитектора)."""
import numpy as np, pandas as pd
from numba import njit
import mz1_model as M

pd.set_option('display.width', 250)
FE = M.FEATS + ['sess', 'hourb']


def binned(D):
    cols = []
    for f in M.FEATS:
        cols.append(np.searchsorted(M.edges[f], D[f].to_numpy()).astype(np.int8))
    cols.append(D.sess.to_numpy().astype(np.int8))
    cols.append(np.clip((D['mod'].to_numpy() - 120) // 60, 0, 13).astype(np.int8))
    return np.stack(cols, 1)


@njit(cache=True)
def best_split(B, g, idx, nf, nb):
    best = (-1.0, -1, -1)
    G = g[idx].sum(); N = len(idx)
    for f in range(nf):
        sg = np.zeros(nb); cn = np.zeros(nb)
        for i in idx:
            b = B[i, f]; sg[b] += g[i]; cn[b] += 1
        gl = 0.0; nl = 0.0
        for b in range(nb - 1):
            gl += sg[b]; nl += cn[b]
            nr = N - nl
            if nl < 2000 or nr < 2000:
                continue
            gain = gl * gl / nl + (G - gl) ** 2 / nr - G * G / N
            if gain > best[0]:
                best = (gain, f, b)
    return best


def fit_tree(B, g, depth=3):
    nodes = [np.arange(len(g))]
    rules = [[]]
    leaves = []
    for d in range(depth):
        nn, nr = [], []
        for idx, rl in zip(nodes, rules):
            gain, f, b = best_split(B, g, idx, B.shape[1], 16)
            if f < 0:
                leaves.append((rl, g[idx].mean())); continue
            L = idx[B[idx, f] <= b]; R = idx[B[idx, f] > b]
            nn += [L, R]; nr += [rl + [(f, b, 0)], rl + [(f, b, 1)]]
        nodes, rules = nn, nr
    for idx, rl in zip(nodes, rules):
        leaves.append((rl, g[idx].mean()))
    return leaves


def apply_tree(B, leaves):
    out = np.zeros(len(B))
    for rl, v in leaves:
        m = np.ones(len(B), bool)
        for f, b, side in rl:
            m &= (B[:, f] <= b) if side == 0 else (B[:, f] > b)
        out[m] = v
    return out


if __name__ == '__main__':
    TR = M.TR; va = M.valid; te = M.test
    Btr = binned(TR); Bva = binned(va); Bte = binned(te)
    y = TR.y30.to_numpy(); yva = va.y30.to_numpy()
    F = np.zeros(len(y)); Fva = np.zeros(len(yva)); Fte = np.zeros(len(te))
    lr = 0.05; best = (-1, 0); trees = []
    for k in range(300):
        leaves = fit_tree(Btr, y - F)
        trees.append(leaves)
        F += lr * apply_tree(Btr, leaves); Fva += lr * apply_tree(Bva, leaves); Fte += lr * apply_tree(Bte, leaves)
        if (k + 1) % 25 == 0:
            cv = np.corrcoef(Fva, yva)[0, 1]
            print('trees', k + 1, 'train corr', round(np.corrcoef(F, y)[0, 1], 4), 'valid corr', round(cv, 4), flush=True)
            if cv > best[0]:
                best = (cv, k + 1, Fte.copy())
    cv, nt, pte = best
    np.save('mz2_pred_main.npy', pte); np.save('mz2_pred_valid.npy', Fva)
    print('chosen trees', nt, 'valid corr', round(cv, 4))
    M.evaluate(te, pte, 'MAIN 2020-25 (GBM)')
    for s_, nm in [(0, 'pre'), (1, 'NYAM'), (2, 'NYPM')]:
        m_ = te.sess.to_numpy() == s_
        print(nm, 'corr', round(np.corrcoef(pte[m_], te.y30.to_numpy()[m_])[0, 1], 4))
    # первые деревья — какие взаимодействия
    for leaves in trees[:3]:
        print([([(FE[f], b, s) for f, b, s in rl], round(v, 3)) for rl, v in leaves])
