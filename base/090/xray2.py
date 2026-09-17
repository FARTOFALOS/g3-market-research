#!/usr/bin/env python3
"""090 X-ray step 4 + locate contrast (train only).

Locator from xray1: near-boundary (low dpos) x high psig corner is where R0+K is
most uncertain and worst-calibrated. Before reading films:
  (a) adequacy: is the corner residual a kNN edge artifact? -> refit R0 with small k
      and read the DIRECT fine-grid local rate; if a maximally-local R0 reading
      already predicts the low corner rate, that part is baseline-reading, not a
      hidden channel. What matters for a channel: does CONT* stay HETEROGENEOUS
      inside a tight (dpos,psig) cell?
  (b) support in the corner.
  (c) emit matched contrast scenes (same tight dpos,psig; opposite CONT*) to read.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
ROOT = Path('C:/Users/Admin/Claude/g3-market-research')
R = pd.read_parquet(ROOT / 'work/090/cursors_split.parquet')
sc = pd.read_parquet(ROOT / 'work/090/scenes.parquet').set_index('scene')
tr = R[(R.fold == 'train') & (R.cont >= 0)].copy()
tr['side'] = tr.scene.map(sc.side)
tr['t0'] = tr.scene.map(sc.t0_spine_pos)
tr['e'] = tr.scene.map(sc.exit_boundary)
tr['endp'] = tr.scene.map(sc.end_pos)
y = tr.cont.to_numpy(float); scene = tr.scene.to_numpy(); fold = scene % 5

# (a) adequacy: can a maximally-LOCAL R0 (small k) read the corner? If small-k fits
# the low rate, then the k=1000 miss was under-bending (baseline reading), and the
# CHANNEL question is whether residual heterogeneity survives a well-read R0.
tr['ldpos'] = np.log(np.clip(tr.dpos, 1e-6, None))
tr['lpsig'] = np.log(np.clip(tr.psig, 1e-6, None))
def z(cols):
    X = np.column_stack([tr[c].to_numpy(float) for c in cols]); m=X.mean(0); s=X.std(0); s[s==0]=1
    return (X-m)/s
def cf(cols, k):
    X=z(cols); p=np.full(len(tr),np.nan)
    for f in range(5):
        m=(fold==f); tree=cKDTree(X[~m]); d,i=tree.query(X[m],k=k,workers=-1); p[m]=y[~m][i].mean(1)
    return np.clip(p,1e-6,1-1e-6)
for k in [50, 100, 1000]:
    tr[f'p{k}'] = cf(['ldpos','lpsig'], k)

corner = tr[(tr.dpos <= 1.5) & (tr.psig >= 2.5)]
print(f"CORNER dpos<=1.5 & psig>=2.5: cursors={len(corner):,} scenes={corner.scene.nunique():,}")
print(f"  observed P(CONT*)={corner.cont.mean():.4f}")
print(f"  R0 pred: k=50 {corner.p50.mean():.4f}  k=100 {corner.p100.mean():.4f}  k=1000 {corner.p1000.mean():.4f}")
print("  -> if small-k pred ~= observed, the coarse-cell miss was under-bending (baseline reading),")
print("     and R0 can read the corner LEVEL; channel question = residual HETEROGENEITY inside tight cells.")

# (b) fine grid inside corner: is the low rate smooth in (dpos,psig) or a genuine mix?
print("\nfine grid inside/around corner  P(CONT*) by (dpos bucket) x (psig bucket):")
db = pd.cut(tr.dpos, [0,1.1,1.6,2.1,3.1], right=True)
sb = pd.cut(tr.psig, [0,1.5,2.5,3.5,5,100], right=True)
tab = tr.groupby([db, sb], observed=True).agg(n=('cont','size'), p=('cont','mean')).round(3)
with pd.option_context('display.width',160,'display.max_rows',40): print(tab.to_string())

# (c) heterogeneity check inside the tightest well-supported corner cell:
# dpos in [1,1.2], psig in [2.5,4]. Within it, is CONT* over-dispersed vs binomial
# when grouped by SCENE-level identity (latent scene heterogeneity)? Report split by
# a purely descriptive read: does the outcome look like one coin or a mixture?
cell = tr[(tr.dpos>=1.0)&(tr.dpos<=1.3)&(tr.psig>=2.5)&(tr.psig<=4.5)].copy()
print(f"\nTIGHT CELL dpos[1,1.3] psig[2.5,4.5]: cursors={len(cell):,} scenes={cell.scene.nunique():,} "
      f"P(CONT*)={cell.cont.mean():.4f}")

# emit matched contrast scenes to read as full films
c1 = cell[cell.cont==1].drop_duplicates('scene').head(6)
c0 = cell[cell.cont==0].drop_duplicates('scene').head(6)
def emit(df, lab):
    print(f"\n--- {lab} contrast scenes (t0, side, e, endp, cursor_ord, dpos, psig) ---")
    for _,r in df.iterrows():
        print(f"  t0={int(r.t0)} {r.side:5s} e={r.e:.2f} endp={int(r.endp)} ord={int(r['ord'])} "
              f"dpos={r.dpos:.2f} psig={r.psig:.2f} scene={int(r.scene)}")
emit(c1,'CONT*=1 (fired another event before contact)')
emit(c0,'CONT*=0 (hit physical terminal next)')
cell.to_parquet(ROOT/'work/090/contrast_cell.parquet')
