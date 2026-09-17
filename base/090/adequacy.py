#!/usr/bin/env python3
"""090 adequacy of R0 in the near-boundary corner (train only).

Question (trader): is the located dq0/sq2 deficit a stable market residual, or an
under-reading of R0 by a coarse estimator? Test with boundary/interaction-capable
continuous estimators of the SAME R0=(dpos,psig); do NOT pick the estimator that
preserves the residual.

  A. local-linear logit (LOESS deg-1) in standardized (ldpos,lpsig): edge-bias
     corrected, reads interaction+boundary locally.
  B. parametric logistic with interaction+boundary basis (IRLS).
Both cross-fit by scene. Then: corner calibration, continuous psig-slice at fixed
dpos (no quantile bins), and corner support/composition (hidden mix?).

If a reasonable R0 reading removes the corner miss -> baseline-reading, no C.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path('C:/Users/Admin/Claude/g3-market-research')
R = pd.read_parquet(ROOT / 'work/090/cursors_split.parquet')
sc = pd.read_parquet(ROOT / 'work/090/scenes.parquet').set_index('scene')
tr = R[(R.fold == 'train') & (R.cont >= 0)].copy()
tr['side'] = tr.scene.map(sc.side)
tr['ldpos'] = np.log(np.clip(tr.dpos, 1e-6, None))
tr['lpsig'] = np.log(np.clip(tr.psig, 1e-6, None))
tr['q_date'] = pd.to_datetime(tr.q_ts_ns, utc=True)
y = tr.cont.to_numpy(float); scene = tr.scene.to_numpy(); fold = scene % 5
u = tr.ldpos.to_numpy(); s = tr.lpsig.to_numpy()
uu = (u - u.mean()) / u.std(); ss = (s - s.mean()) / s.std()

corner = (tr.dpos <= 1.5) & (tr.psig >= 2.5)
cm = corner.to_numpy()
print(f"corner n={cm.sum()}  observed P={y[cm].mean():.4f}")

# ---------- B. parametric logistic, interaction + boundary basis ----------
def basis(uu, ss, dpos, psig):
    inv = 1.0 / (1.0 + dpos)                      # boundary-sensitive
    return np.column_stack([np.ones_like(uu), uu, ss, uu*uu, ss*ss, uu*ss,
                            inv, inv*ss, psig, dpos*psig*1e-3])
def irls(X, yv, iters=50):
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        eta = np.clip(X @ b, -30, 30); p = 1/(1+np.exp(-eta))
        W = np.clip(p*(1-p), 1e-6, None)
        z = eta + (yv - p)/W
        Xt = X * W[:, None]
        b_new = np.linalg.solve(Xt.T @ X + 1e-6*np.eye(X.shape[1]), Xt.T @ z)
        if np.max(np.abs(b_new - b)) < 1e-8: b = b_new; break
        b = b_new
    return b
Xall = basis(uu, ss, tr.dpos.to_numpy(), tr.psig.to_numpy())
predB = np.full(len(tr), np.nan)
for f in range(5):
    m = fold == f; b = irls(Xall[~m], y[~m]); eta = np.clip(Xall[m] @ b, -30, 30); predB[m] = 1/(1+np.exp(-eta))
print(f"[B] parametric logit interaction: corner pred={predB[cm].mean():.4f}  resid={y[cm].mean()-predB[cm].mean():+.4f}"
      f"   overall logloss={-np.mean(y*np.log(np.clip(predB,1e-6,1-1e-6))+(1-y)*np.log(np.clip(1-predB,1e-6,1-1e-6))):.5f}")

# ---------- A. local-linear logit (LOESS deg-1), corner query points, cross-fit ----------
def loess_pred(qi, hu, hs):
    """predict P at query indices qi using deg-1 local linear on y, neighbors from other folds."""
    out = np.full(len(qi), np.nan)
    for ii, i in enumerate(qi):
        f = fold[i]; m = fold != f
        du = (uu[m] - uu[i]); ds = (ss[m] - ss[i])
        w = np.exp(-0.5*((du/hu)**2 + (ds/hs)**2))
        keep = w > 1e-4
        if keep.sum() < 50:
            out[ii] = np.nan; continue
        du, ds, w, yy = du[keep], ds[keep], w[keep], y[m][keep]
        X = np.column_stack([np.ones_like(du), du, ds])
        XtW = X.T * w
        try:
            beta = np.linalg.solve(XtW @ X + 1e-8*np.eye(3), XtW @ yy)
            out[ii] = np.clip(beta[0], 0, 1)
        except Exception:
            out[ii] = np.nan
    return out
qi = np.where(cm)[0]
for (hu, hs) in [(0.35, 0.35), (0.5, 0.5)]:
    pa = loess_pred(qi, hu, hs)
    ok = ~np.isnan(pa)
    print(f"[A] local-linear h=({hu},{hs}): corner n_ok={ok.sum()} pred={pa[ok].mean():.4f}  "
          f"resid={y[cm][ok].mean()-pa[ok].mean():+.4f}  (eff.support per fit >=50)")

# ---------- continuous psig-slice at fixed dpos, no quantile bins ----------
print("\ncontinuous slice: dpos in [0.9,1.3]; P(CONT*) vs psig by local-linear on psig only:")
sl = (tr.dpos >= 0.9) & (tr.dpos <= 1.3)
ps = tr.psig.to_numpy()[sl.to_numpy()]; ys = y[sl.to_numpy()]
grid = np.array([1.0,1.5,2.0,2.5,3.0,3.5,4.0,5.0])
for g in grid:
    w = np.exp(-0.5*((ps-g)/0.6)**2); W = w.sum()
    ph = (w*ys).sum()/W
    print(f"   psig~{g:>4.1f}  P={ph:.3f}  eff_n={W:6.0f}")

# ---------- corner support / composition (hidden mix?) ----------
print("\ncorner composition (is the low rate a hidden sub-population?):")
cc = tr[cm]
print(f"  distinct scenes={cc.scene.nunique()} of {len(cc)} cursors; distinct t0={cc.scene.map(sc.t0_spine_pos).nunique()}")
print(f"  side:   north P={cc[cc.side=='north'].cont.mean():.3f} (n={(cc.side=='north').sum()})  "
      f"south P={cc[cc.side=='south'].cont.mean():.3f} (n={(cc.side=='south').sum()})")
print(f"  ord0 P={cc[cc['ord']==0].cont.mean():.3f} (n={(cc['ord']==0).sum()})  "
      f"ord>=1 P={cc[cc['ord']>=1].cont.mean():.3f} (n={(cc['ord']>=1).sum()})")
cc = cc.copy(); cc['yr'] = cc.q_date.dt.year
yr = cc.groupby('yr').cont.agg(['size','mean']).round(3)
print("  by year:"); print(yr.to_string())
