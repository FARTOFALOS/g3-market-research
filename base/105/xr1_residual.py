"""XR1 — residual distinguishability X-ray of the whole tape (declared in base/105/LOG.md before counting).

Points: the MZ1 table (work/nq-manual/mz1_features.parquet; every 5th minute 02:00-15:25, NQ 2006-2025, prefix only).
Outcomes from o(t+1), in u, clipped at +-20u: signed move 30/60/120, |move| 60, trendiness 60 (|net|/range), skew 60
(best - worst move). Two boosted-tree readers per outcome (depth 3, 200 trees, step 0.05, 16 quantile bins from the
training years; numbers fixed in the declaration): K (known coordinates) and K + R (the rest of MZ1). Walk-forward:
2006-12 -> 2013-19 and 2006-19 -> 2020-25. Measure: out-of-sample R2(K+R) - R2(K), day-clustered t.

Run from repository root: python -B base/105/xr1_residual.py
"""
import json, math, os, sys
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "setups/library"))
OUT = HERE / "xr1_out"
NB, TREES, LR, DEPTH, MINLEAF, CLIP, COST = 16, 200, 0.05, 3, 200, 20.0, 0.75
K_FEATS = ["hh", "logu", "rng_u", "pos", "dUB", "dLB", "d930", "dPC", "in07", "s18pos"]
R_FEATS = ["r1", "r5", "r15", "r30", "r60", "r120", "d02", "dPH", "dPL", "dAH", "dAL", "dLH", "dLL", "dH1h", "dH1l",
           "dHH", "dLLd", "sinceH", "sinceL", "swPH", "swPL", "swAH", "swAL", "dPiv_l", "dPiv_h", "body1", "rng1", "cpos1", "dow"]
OUTCOMES = ["y30", "y60", "y120", "a60", "tr60", "sk60"]


def outcomes_and_state(X):
    os.chdir(HERE)
    from trade import Tape
    T = Tape()
    t = X.t.to_numpy().astype(np.int64)
    assert (T.date[t] == X.date.to_numpy()).all() and (T.mod[t] == X["mod"].to_numpy()).all(), "MZ1 index does not match the tape"
    o, h, l, c, dt, md = T.o, T.h, T.l, T.c, T.date, T.mod
    u = X.u.to_numpy()
    e = o[np.minimum(t + 1, len(o) - 1)]
    for m in (30, 60, 120):
        k = np.minimum(t + m, len(o) - 1)
        ok = (dt[k] == dt[t]) & (md[k] - md[t] == m) & (u > 0)
        X[f"y{m}"] = np.where(ok, np.clip((c[k] - e) / np.where(u > 0, u, 1), -CLIP, CLIP), np.nan)
        X[f"p{m}"] = np.where(ok, c[k] - e, np.nan)
        if m == 60:
            hi = np.array([h[i + 1:i + 61].max() if ok[j] else np.nan for j, i in enumerate(t)])
            lo = np.array([l[i + 1:i + 61].min() if ok[j] else np.nan for j, i in enumerate(t)])
            X["a60"] = np.where(ok, np.clip(np.abs(c[k] - e) / np.where(u > 0, u, 1), 0, CLIP), np.nan)
            rng = hi - lo
            X["tr60"] = np.where(ok & (rng > 0), np.abs(c[k] - e) / np.where(rng > 0, rng, 1), np.nan)
            X["sk60"] = np.where(ok, np.clip(((hi - e) - (e - lo)) / np.where(u > 0, u, 1), -CLIP, CLIP), np.nan)
    # B1 state at the close of bar t
    import joint as J
    import payoff_map as PM
    cal = J.S17.history_calendar(); root = ROOT / "data/market/NQ"
    ev = J.s07_s17(root, cal).set_index("date")
    block = J.block_from(J.s07_s17(root, cal), "s07_exit_min", cal)
    cal2, K, O, C, s18 = PM.s18_positions(cal, root.parent, block)
    dates = pd.to_datetime(cal2.date); key = (dates.dt.year * 10000 + dates.dt.month * 100 + dates.dt.day).to_numpy()
    in07 = {}; legs = {}
    for s, D in enumerate(key):
        d = dates[s]
        if d in ev.index and isinstance(ev.status.get(d), str) and not ev.status[d].startswith("no_entry") and np.isfinite(ev.s07_exit_min.get(d, np.nan)):
            in07[int(D)] = int(3 + ev.s07_exit_min[d])
        if s in s18: legs[int(D)] = s18[s][1]
    date = X.date.to_numpy(); mod = X["mod"].to_numpy()
    kt = mod - 570 + 1
    X["in07"] = [1.0 if (D in in07 and 4 <= k < in07[D]) else 0.0 for D, k in zip(date, kt)]   # still open after bar k closes
    s18pos = np.zeros(len(X))
    for j, (D, k) in enumerate(zip(date, kt)):
        for ke, kx, pos, ep, xp in legs.get(D, []):
            if ke <= k < kx: s18pos[j] = pos; break
    X["s18pos"] = s18pos
    X["hh"] = (X["mod"] - 120) // 30
    X["logu"] = np.log(np.where(X.u > 0, X.u, np.nan))
    return X


def bins_from(train, cols):
    return {f: np.unique(np.nanquantile(train[f].to_numpy(float), np.linspace(0, 1, NB + 1)[1:-1])) for f in cols}


def binned(D, cols, edges):
    B = np.empty((len(D), len(cols)), np.int16)
    for j, f in enumerate(cols):
        v = D[f].to_numpy(float); b = np.searchsorted(edges[f], v).astype(np.int16); b[~np.isfinite(v)] = NB
        B[:, j] = b
    return B


@njit(cache=True)
def grow(B, r, depth, minleaf, nbins):
    n, p = B.shape
    nint = 2 ** depth - 1
    feat = -np.ones(nint, np.int64); thr = np.zeros(nint, np.int64); leaf = np.zeros(2 ** depth)
    node = np.zeros(n, np.int64)                       # current node id per row (heap indexing)
    for d in range(depth):
        first = 2 ** d - 1
        for nd in range(first, first + 2 ** d):
            G = 0.0; N = 0
            sg = np.zeros((p, nbins)); cn = np.zeros((p, nbins))
            for i in range(n):
                if node[i] == nd:
                    G += r[i]; N += 1
                    for f in range(p):
                        sg[f, B[i, f]] += r[i]; cn[f, B[i, f]] += 1
            if N < 2 * minleaf: continue
            best = 0.0; bf = -1; bt = 0
            base = G * G / N
            for f in range(p):
                gl = 0.0; nl = 0.0
                for b in range(nbins - 1):
                    gl += sg[f, b]; nl += cn[f, b]
                    nr = N - nl
                    if nl < minleaf or nr < minleaf: continue
                    gain = gl * gl / nl + (G - gl) * (G - gl) / nr - base
                    if gain > best: best = gain; bf = f; bt = b
            feat[nd] = bf; thr[nd] = bt
        for i in range(n):
            nd = node[i]
            if nd < first or nd >= first + 2 ** d: continue
            f = feat[nd]
            node[i] = 2 * nd + 1 if (f < 0 or B[i, f] <= thr[nd]) else 2 * nd + 2
    first = nint
    s = np.zeros(2 ** depth); c = np.zeros(2 ** depth)
    for i in range(n):
        s[node[i] - first] += r[i]; c[node[i] - first] += 1
    for j in range(2 ** depth):
        leaf[j] = s[j] / c[j] if c[j] > 0 else 0.0
    return feat, thr, leaf


@njit(cache=True)
def apply(B, feat, thr, leaf, depth):
    n = B.shape[0]; out = np.zeros(n); nint = 2 ** depth - 1
    for i in range(n):
        nd = 0
        for d in range(depth):
            f = feat[nd]
            nd = 2 * nd + 1 if (f < 0 or B[i, f] <= thr[nd]) else 2 * nd + 2
        out[i] = leaf[nd - nint]
    return out


def gbm(Btr, y, Bte):
    F = np.full(len(y), y.mean()); P = np.full(Bte.shape[0], y.mean())
    for _ in range(TREES):
        feat, thr, leaf = grow(Btr, y - F, DEPTH, MINLEAF, NB + 1)
        F += LR * apply(Btr, feat, thr, leaf, DEPTH); P += LR * apply(Bte, feat, thr, leaf, DEPTH)
    return P


def main():
    OUT.mkdir(exist_ok=True)
    X = pd.read_parquet(ROOT / "work/nq-manual/mz1_features.parquet")
    X = outcomes_and_state(X)
    res = dict(declaration="base/105/LOG.md XR1", points=len(X), splits={})
    for name, tr_hi, te_lo, te_hi in (("2013-19", 20121231, 20130101, 20191231), ("2020-25", 20191231, 20200101, 20251231)):
        tr = X[X.date <= tr_hi]; te = X[(X.date >= te_lo) & (X.date <= te_hi)]
        edges = bins_from(tr, K_FEATS + R_FEATS)
        BK_tr, BK_te = binned(tr, K_FEATS, edges), binned(te, K_FEATS, edges)
        BA_tr, BA_te = binned(tr, K_FEATS + R_FEATS, edges), binned(te, K_FEATS + R_FEATS, edges)
        res["splits"][name] = {}
        for yv in OUTCOMES:
            mtr = tr[yv].notna().to_numpy(); mte = te[yv].notna().to_numpy()
            y = tr[yv].to_numpy(float)[mtr]; yt = te[yv].to_numpy(float)[mte]
            pk = gbm(BK_tr[mtr], y, BK_te[mte]); pa = gbm(BA_tr[mtr], y, BA_te[mte])
            v = ((yt - yt.mean()) ** 2).mean()
            r2k = 1 - ((yt - pk) ** 2).mean() / v; r2a = 1 - ((yt - pa) ** 2).mean() / v
            dd = (yt - pk) ** 2 - (yt - pa) ** 2
            days = te.date.to_numpy()[mte]
            s = pd.Series(dd).groupby(days).sum(); nd = pd.Series(1, index=range(len(dd))).groupby(days).sum()
            m = dd.mean(); se = math.sqrt(((s - nd * m) ** 2).sum()) / len(dd)
            row = dict(n=int(len(yt)), r2_K=r2k, r2_KR=r2a, gain=r2a - r2k, t_gain=m / se if se > 0 else None)
            if yv in ("y30", "y60", "y120"):
                pts = te[f"p{yv[1:]}"].to_numpy(float)[mte]
                for lab, pr in (("K", pk), ("KR", pa)):
                    top = np.abs(pr) >= np.quantile(np.abs(pr), 0.9)
                    row[f"top10_{lab}_pts_net"] = float((np.sign(pr[top]) * pts[top]).mean() - COST)
            res["splits"][name][yv] = row
            print(name, yv, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    live = [yv for yv in OUTCOMES if all(res["splits"][s][yv]["gain"] > 0 and (res["splits"][s][yv]["t_gain"] or 0) >= 3 for s in ("2013-19", "2020-25"))]
    res["residual_information"] = live
    (OUT / "summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("residual information (both test epochs, t >= 3):", live)


if __name__ == "__main__":
    main()
