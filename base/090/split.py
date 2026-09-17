#!/usr/bin/env python3
"""090 leakage-safe time split (FREEZE_090 sec.7).

Test membership = q calendar time only (no future-length filtering).
Train protected by purge/embargo: a scene enters TRAIN only if its full tape
[t0, end_pos] ends strictly before T_split - embargo. Scenes straddling the
boundary (end_ts >= T_split - embargo) are dropped from train entirely, so no
scene appears in both folds. Max scene duration measured 0.88 day -> embargo 7d
is strictly safe.

Frozen split parameters:
  T_SPLIT = 2016-01-01 UTC        (calendar, ~last 30% of cursors -> test)
  EMBARGO = 7 calendar days
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path('C:/Users/Admin/Claude/g3-market-research')

T_SPLIT = pd.Timestamp('2016-01-01', tz='UTC')
EMBARGO = pd.Timedelta(days=7)

R = pd.read_parquet(ROOT / 'work/090/cursors.parquet')
sc = pd.read_parquet(ROOT / 'work/090/scenes.parquet')
R['q_date'] = pd.to_datetime(R.q_ts_ns, utc=True)
sc['end_date'] = pd.to_datetime(sc.end_ts_ns, utc=True)
scene_end = sc.set_index('scene').end_date
R['scene_end'] = R.scene.map(scene_end)

train_cut = T_SPLIT - EMBARGO
R['fold'] = 'drop'
R.loc[R.scene_end < train_cut, 'fold'] = 'train'          # whole tape safely before embargo
R.loc[R.q_date >= T_SPLIT, 'fold'] = 'test'               # membership by q-time only

# guard: a scene must not appear in both folds
both = R.groupby('scene').fold.nunique()
assert not ((R.groupby('scene').fold.agg(lambda s: ('train' in set(s)) and ('test' in set(s)))).any()), \
    "scene leaked across folds"

tr = R[R.fold == 'train']; te = R[R.fold == 'test']; dr = R[R.fold == 'drop']
print(f"T_split={T_SPLIT.date()}  embargo={EMBARGO.days}d  train_cut={train_cut.date()}")
print(f"TRAIN cursors={len(tr):,}  scenes={tr.scene.nunique():,}  "
      f"P(CONT*=1|resolved)={tr[tr.cont>=0].cont.mean():.4f}  censored={int((tr.cont<0).sum())}")
print(f"TEST  cursors={len(te):,}  scenes={te.scene.nunique():,}  "
      f"P(CONT*=1|resolved)={te[te.cont>=0].cont.mean():.4f}  censored={int((te.cont<0).sum())}")
print(f"DROP (embargo/gap) cursors={len(dr):,}  scenes={dr.scene.nunique():,}")
print(f"TRAIN q-date: {tr.q_date.min()} .. {tr.q_date.max()}")
print(f"TEST  q-date: {te.q_date.min()} .. {te.q_date.max()}")
assert set(tr.scene) & set(te.scene) == set(), "scene overlap"
print("scene-disjoint train/test: OK")

R.to_parquet(ROOT / 'work/090/cursors_split.parquet')
print(f"saved {ROOT/'work/090/cursors_split.parquet'}")
