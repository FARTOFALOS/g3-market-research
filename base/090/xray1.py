#!/usr/bin/env python3
"""090 train-only X-ray, steps 1-4 (FREEZE_090; trader amendment).

1. adequate continuous R0 reading  P(CONT*|dpos,psig)
2. incremental value of declared K  (side, ord)   -> refinement check
3. locate regions where CONT* stays systematically heterogeneous at comparable R0+K
4. support / estimator adequacy / counterexample checks

Model residual is only an X-ray LOCATOR, not a proven market residual.
No wide feature matrix, no candidate ranking by gain. TEST untouched.
Estimator: cross-fitted (by scene) kNN in standardized log-state space; k varied
for adequacy. Resolved cursors only (cont in {0,1}); 65 censored dropped from fit.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
ROOT = Path('C:/Users/Admin/Claude/g3-market-research')

R = pd.read_parquet(ROOT / 'work/090/cursors_split.parquet')
tr = R[(R.fold == 'train') & (R.cont >= 0)].copy()
print(f"TRAIN resolved cursors={len(tr):,} scenes={tr.scene.nunique():,}  base P(CONT*)={tr.cont.mean():.4f}")

y = tr.cont.to_numpy(float)
scene = tr.scene.to_numpy()
fold = (scene % 5)

def zfeat(df, cols):
    X = np.column_stack([df[c].to_numpy(float) for c in cols])
    mu = X.mean(0); sd = X.std(0); sd[sd == 0] = 1
    return (X - mu) / sd

tr['ldpos'] = np.log(np.clip(tr.dpos, 1e-6, None))
tr['lpsig'] = np.log(np.clip(tr.psig, 1e-6, None))
tr['side_n'] = (tr.j.map(lambda _: 0))  # placeholder; side added below
# side sign from original scene table
sc = pd.read_parquet(ROOT / 'work/090/scenes.parquet').set_index('scene')
tr['side_n'] = tr.scene.map(sc.side).eq('north').astype(float).to_numpy()
tr['lord'] = np.log1p(tr['ord'].to_numpy(float))

def crossfit_knn(feat_cols, k=500):
    """Return out-of-fold P(CONT*) predictions; neighbors drawn only from other folds."""
    X = zfeat(tr, feat_cols)
    pred = np.full(len(tr), np.nan)
    for f in range(5):
        te_m = (fold == f); trn_m = ~te_m
        tree = cKDTree(X[trn_m])
        d, idx = tree.query(X[te_m], k=k, workers=-1)
        yy = y[trn_m][idx]
        pred[te_m] = yy.mean(1)
    return np.clip(pred, 1e-6, 1 - 1e-6)

def logloss(p):
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
def brier(p):
    return np.mean((p - y) ** 2)

# ---- step 1: R0 adequacy (k sweep) ----
print("\n[1] R0 = P(CONT*|dpos,psig), cross-fit kNN, k-sweep (adequacy):")
base_const = np.full(len(tr), y.mean())
print(f"    const baseline           logloss={logloss(base_const):.5f} brier={brier(base_const):.5f}")
preds = {}
for k in [200, 500, 1000, 2000]:
    p = crossfit_knn(['ldpos', 'lpsig'], k)
    preds[k] = p
    print(f"    R0 k={k:<5}             logloss={logloss(p):.5f} brier={brier(p):.5f}")
pR0 = preds[1000]

# ---- step 2: incremental value of declared K ----
print("\n[2] R0+K vs R0 (K = side, ord), k=1000:")
pRK = crossfit_knn(['ldpos', 'lpsig', 'side_n', 'lord'], 1000)
print(f"    R0     logloss={logloss(pR0):.5f} brier={brier(pR0):.5f}")
print(f"    R0+K   logloss={logloss(pRK):.5f} brier={brier(pRK):.5f}")
print(f"    delta logloss (R0 -> R0+K) = {logloss(pR0)-logloss(pRK):+.5f}  "
      f"(positive => K refines the reading)")

# ---- step 3: locate heterogeneity at comparable R0+K ----
# calibration of R0+K
print("\n[3] R0+K calibration (are matched-prediction cells internally homogeneous?):")
tr['pRK'] = pRK
q = pd.qcut(tr.pRK, 12, duplicates='drop')
cal = tr.groupby(q, observed=True).agg(n=('cont', 'size'), n_act=('scene', 'nunique'),
                                       p_obs=('cont', 'mean'), p_pred=('pRK', 'mean'))
with pd.option_context('display.width', 160):
    print(cal.round(4).to_string())

# where is the model both uncertain and well-supported (locator for the contrast)?
print("\n[3b] uncertainty x support map over (dpos decile) x (psig tertile), side pooled:")
tr['dq'] = pd.qcut(tr.dpos, 10, duplicates='drop', labels=False)
tr['sq'] = pd.qcut(tr.psig, 3, duplicates='drop', labels=False)
g = tr.groupby(['dq', 'sq'], observed=True).agg(n=('cont', 'size'), n_act=('scene', 'nunique'),
                                                p_obs=('cont', 'mean'), p_pred=('pRK', 'mean'),
                                                dpos_med=('dpos', 'median'), psig_med=('psig', 'median'))
g['resid'] = g.p_obs - g.p_pred
with pd.option_context('display.width', 200, 'display.max_rows', 40):
    print(g.round(4).to_string())

tr.to_parquet(ROOT / 'work/090/train_xray.parquet')
print(f"\nsaved {ROOT/'work/090/train_xray.parquet'}")
