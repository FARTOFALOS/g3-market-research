"""SV1 — survival of the "inside the norm" state after 12:00: information beyond band geometry (declared in base/105/LOG.md).

Population = CW1 (full NQ sessions 2006-2025, close at the S-18 mark 12:00 inside the S-18 v0 band).
Outcome at h = 30/60/120: first S-18 mark after 12:00 (12:30..14:00) with close outside the band -> up / down / none.
Models (multinomial logistic, Newton, tiny ridge 1e-6 for numerical stability), walk-forward by year, test 2013..2025:
  geometry = distances from c(12:00) to the upper and lower band edges, in day norms;
  state    = geometry + range 09:30-12:00 / 20-session median, position of c(12:00) in today's range,
             overnight move G and morning move M in norms, minutes since the last new session extreme, its side.
Measure: out-of-sample per-day log-loss(geometry) - log-loss(state); month-block t per test epoch.

Run from repository root: python -B base/105/sv1_survival.py
"""
import json, math, sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "setups/library")); sys.path.insert(0, str(HERE))
import joint as J  # noqa: E402
from tape import load_minutes  # noqa: E402

S18 = J.S18
OUT = HERE / "sv1_out"
K12, HS, MARKS = 150, (30, 60, 120), (180, 210, 240, 270)
GEO = ["d_up", "d_dn"]
STATE = GEO + ["rho", "q", "Gn", "Mn", "age_ext", "side_ext"]


def build():
    cal = J.S17.history_calendar()
    cal, K, O, C = S18.sessions("NQ", cal, ROOT / "data/market")
    S = len(cal); op = O[:, 1]
    cl = np.array([C[s, min(K[s], S18.KMAX)] for s in range(S)])
    move = np.abs(C / op[:, None] - 1.0)
    tape = load_minutes("2005-11-01", "2026-01-01", "NQ")
    am = tape[(tape["mod"] >= 570) & (tape["mod"] < 720)]
    info = {}
    for d, g in am.groupby("date"):
        if len(g) != 150: continue
        h, l = g.h.to_numpy(), g.l.to_numpy()
        last_hi = int(np.flatnonzero(h == np.maximum.accumulate(h))[-1]); last_lo = int(np.flatnonzero(l == np.minimum.accumulate(l))[-1])
        side = 1 if last_hi >= last_lo else -1
        info[int(d)] = (h.max(), l.min(), 149 - max(last_hi, last_lo), side)
    dates = pd.to_datetime(cal.date); D = (dates.dt.year * 10000 + dates.dt.month * 100 + dates.dt.day).to_numpy()
    amr = np.array([info[D[s]][0] - info[D[s]][1] if D[s] in info else np.nan for s in range(S)])
    rows = []
    for s in range(14, S):
        if not (20060101 <= D[s] <= 20251231) or K[s] != 390 or D[s] not in info: continue
        prev = cl[s - 1]
        hist = move[s - 14:s]; cnt = np.isfinite(hist).sum(0)
        with np.errstate(invalid="ignore"):
            norm = np.where(cnt >= 10, np.nanmean(hist, 0), np.nan)
        if not (np.isfinite(op[s]) and np.isfinite(prev) and np.isfinite(norm[K12])
                and all(np.isfinite(C[s, k]) for k in (30, 60, 90, 120, 150))): continue
        up = max(op[s], prev) * (1 + norm); dn = min(op[s], prev) * (1 - norm)
        c12 = C[s, K12]
        if not (dn[K12] <= c12 <= up[K12]): continue
        npts = norm[K12] * op[s]
        hi, lo, age, side_ext = info[D[s]]
        prior = amr[max(0, s - 20):s]; prior = prior[np.isfinite(prior)]
        r = dict(date=int(D[s]), d_up=(up[K12] - c12) / npts, d_dn=(c12 - dn[K12]) / npts,
                 rho=amr[s] / np.median(prior) if len(prior) >= 10 else np.nan,
                 q=0.5 if hi <= lo else (c12 - lo) / (hi - lo), Gn=(op[s] - prev) / npts, Mn=(c12 - op[s]) / npts,
                 age_ext=float(age), side_ext=float(side_ext))
        ex = 0
        for k in MARKS:
            if not (np.isfinite(C[s, k]) and np.isfinite(norm[k])): ex = None; break
            if C[s, k] > up[k] or C[s, k] < dn[k]:
                ex = (k, 1 if C[s, k] > up[k] else 2); break
        for h in HS:
            if ex is None: r[f"y{h}"] = np.nan
            elif ex == 0 or ex[0] > K12 + h: r[f"y{h}"] = 0      # none
            else: r[f"y{h}"] = ex[1]                               # 1 up, 2 down
        rows.append(r)
    P = pd.DataFrame(rows).dropna(subset=STATE).reset_index(drop=True)
    P["year"] = P.date // 10000; P["month"] = P.date // 100
    return P


def fit(X, y, iters=50, ridge=1e-6):
    """Multinomial logistic regression by Newton's method; class 0 is the reference."""
    Z = np.c_[np.ones(len(X)), X]; n, p = Z.shape; K = 3
    B = np.zeros((p, K - 1)); Y = np.eye(K)[y.astype(int)][:, 1:]
    for _ in range(iters):
        eta = Z @ B; m = eta.max(1, keepdims=True).clip(min=0)
        E = np.exp(eta - m); den = np.exp(-m) + E.sum(1, keepdims=True); Pk = E / den
        g = (Z.T @ (Y - Pk)).ravel(order="F") - ridge * B.ravel(order="F")
        H = np.zeros((p * (K - 1), p * (K - 1)))
        for a in range(K - 1):
            for b in range(K - 1):
                w = Pk[:, a] * ((a == b) - Pk[:, b])
                H[a * p:(a + 1) * p, b * p:(b + 1) * p] = -(Z * w[:, None]).T @ Z
        H -= ridge * np.eye(p * (K - 1))
        step = np.linalg.solve(H, g)
        B = B - step.reshape((p, K - 1), order="F")
        if np.abs(step).max() < 1e-8: break
    return B


def predict(B, X):
    Z = np.c_[np.ones(len(X)), X]; eta = Z @ B
    E = np.c_[np.zeros(len(X)), eta]; E = np.exp(E - E.max(1, keepdims=True))
    return E / E.sum(1, keepdims=True)


def month_t(x, months):
    x = np.asarray(x, float); n = len(x)
    s = pd.Series(x).groupby(np.asarray(months)).sum().to_numpy(); k = len(s)
    se = math.sqrt(k / (k - 1) * ((s - s.mean()) ** 2).sum()) / n
    return float(x.mean()), float(se), float(x.mean() / se) if se > 0 else float("nan")


def main():
    OUT.mkdir(exist_ok=True)
    P = build()
    res = dict(declaration="base/105/LOG.md SV1", days=len(P), outcomes={}, epochs={}, reliability={})
    preds = []
    for h in HS:
        y = P[f"y{h}"]; ok = y.notna()
        res["outcomes"][h] = {str(int(k)): int(v) for k, v in y[ok].value_counts().sort_index().items()}
        for Y in range(2013, 2026):
            tr = ok & (P.year < Y); te = ok & (P.year == Y)
            if te.sum() == 0: continue
            mu, sd = P.loc[tr, STATE].mean(), P.loc[tr, STATE].std().replace(0, 1)
            Xs_tr, Xs_te = ((P.loc[tr, STATE] - mu) / sd).to_numpy(), ((P.loc[te, STATE] - mu) / sd).to_numpy()
            gi = [STATE.index(c) for c in GEO]
            pg = predict(fit(Xs_tr[:, gi], y[tr].to_numpy()), Xs_te[:, gi])
            ps = predict(fit(Xs_tr, y[tr].to_numpy()), Xs_te)
            yt = y[te].to_numpy().astype(int)
            lg = -np.log(np.clip(pg[np.arange(len(yt)), yt], 1e-12, 1)); ls = -np.log(np.clip(ps[np.arange(len(yt)), yt], 1e-12, 1))
            f = pd.DataFrame(dict(date=P.loc[te, "date"].to_numpy(), month=P.loc[te, "month"].to_numpy(), h=h, y=yt,
                                  gain=lg - ls, pg_up=pg[:, 1], pg_dn=pg[:, 2], ps_up=ps[:, 1], ps_dn=ps[:, 2]))
            preds.append(f)
    Q = pd.concat(preds, ignore_index=True)
    Q["epoch"] = np.where(Q.date < 20200101, "2013-19", "2020-25")
    passed = False
    for (ep, h), g in Q.groupby(["epoch", "h"]):
        m, se, t = month_t(g.gain, g.month)
        res["epochs"][f"{ep}|{h}"] = dict(n=len(g), gain_per_day=m, se=se, t=t,
                                          base_rate={str(k): float((g.y == k).mean()) for k in (0, 1, 2)})
    for h in HS:
        if all(res["epochs"][f"{ep}|{h}"]["t"] >= 2 for ep in ("2013-19", "2020-25")): passed = True
        g = Q[Q.h == h].copy()
        for name, col, cls in (("up_state", "ps_up", 1), ("down_state", "ps_dn", 2), ("up_geometry", "pg_up", 1)):
            g["dec"] = pd.qcut(g[col], 10, labels=False, duplicates="drop")
            res["reliability"][f"{h}|{name}"] = g.groupby("dec").apply(
                lambda x: dict(pred=round(float(x[col].mean()), 3), obs=round(float((x.y == cls).mean()), 3), n=len(x)), include_groups=False).tolist()
    res["passed_stage1"] = passed
    Q.to_csv(OUT / "predictions.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("days", len(P), "outcomes (0 none, 1 up, 2 down):", res["outcomes"])
    for k, v in res["epochs"].items():
        print(f"{k:12s} n {v['n']:5d} gain/day {v['gain_per_day']:+.4f} se {v['se']:.4f} t {v['t']:+.2f} base {v['base_rate']}")
    for h in HS:
        r = res["reliability"][f"{h}|up_state"]
        print(h, "up_state deciles pred->obs:", [(d["pred"], d["obs"]) for d in r])
    print("stage 1 passed:", passed)


if __name__ == "__main__":
    main()
