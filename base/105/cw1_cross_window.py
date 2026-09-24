"""CW1 — overnight x NY-morning relation -> NY afternoon, on days inside the S-18 norm (declared in base/105/LOG.md).

Population: full NQ sessions 2006-2025 whose close at the S-18 mark 12:00 (k=150) lies inside the S-18 v0 band.
Classes from G = open09:30 - previous RTH close and M = close(12:00) - open09:30 with threshold 0.25 * norm (points):
small / aligned / cancelled / survived. Side s = sign(M). Entry open of the 12:00 bar (k=151).
Money r_h = s*(C[150+h] - O[151]), h = 30/60/120, points and norm units; stopped version exits at the first
S-18 mark after 12:00 with price outside the band (next open). Competing outcomes: first mark 12:30..14:00 outside
the band by 12:00+h, with / against the morning side, or inside through h.
Control: epoch x tercile(range 09:30-12:00 / its 20-session median) x tercile(position of c12 in today's range, by s);
excess of a class = y - mean(y of the two other tested classes in the same cell). Month-block errors.
Null: class labels permuted within cells (2020-25), 1 000 times, max |t| over the 9 money cells.

Run from repository root: python -B base/105/cw1_cross_window.py [NQ|ES]
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
INST = sys.argv[1] if len(sys.argv) > 1 else "NQ"          # ES run = declared development (4) check
OUT = HERE / "cw1_out" / ("" if INST == "NQ" else INST)
COST_PTS = {"NQ": 0.75, "ES": 0.60}                          # ES: $30 per round turn / $50 per point (S-18)
K12, HS, MARKS = 150, (30, 60, 120), (180, 210, 240, 270)
THR, COST, NPERM, SEED = 0.25, 0.75, 1000, 20260924
TESTED = ("aligned", "cancelled", "survived")
EPOCHS = [("2006-12", 20060101, 20121231), ("2013-19", 20130101, 20191231), ("2020-25", 20200101, 20251231)]


def month_t(x, months):
    x = np.asarray(x, float); n = len(x)
    if n < 3: return float("nan"), float("nan")
    s = pd.Series(x).groupby(np.asarray(months)).sum().to_numpy(); k = len(s)
    if k < 2: return float("nan"), float("nan")
    se = math.sqrt(k / (k - 1) * ((s - s.mean()) ** 2).sum()) / n
    return float(x.mean() / se) if se > 0 else float("nan"), float(se)


def build(inst=None, cal=None, root=None, d_lo=20060101, d_hi=20251231):
    """Defaults reproduce the CW1 run; the frozen check (FREEZE_CW1.md) passes another calendar, root and window."""
    inst = inst or INST
    cal = J.S17.history_calendar() if cal is None else cal
    cal, K, O, C = S18.sessions(inst, cal, ROOT / "data/market" if root is None else root)
    S = len(cal); op = O[:, 1]
    cl = np.array([C[s, min(K[s], S18.KMAX)] for s in range(S)])
    move = np.abs(C / op[:, None] - 1.0)
    tape = load_minutes("2005-11-01", "2026-07-11", inst)
    am = tape[(tape["mod"] >= 570) & (tape["mod"] < 720)].groupby("date").agg(hi=("h", "max"), lo=("l", "min"), n=("h", "size"))
    dates = pd.to_datetime(cal.date); D = (dates.dt.year * 10000 + dates.dt.month * 100 + dates.dt.day).to_numpy()
    amr = np.full(S, np.nan)
    for s in range(S):
        if D[s] in am.index and am.at[D[s], "n"] == 150: amr[s] = am.at[D[s], "hi"] - am.at[D[s], "lo"]
    rows, short = [], 0
    for s in range(14, S):
        if not (d_lo <= D[s] <= d_hi): continue
        if K[s] != 390: short += 1; continue
        prev = cl[s - 1]
        hist = move[s - 14:s]; cnt = np.isfinite(hist).sum(0)
        with np.errstate(invalid="ignore"):
            norm = np.where(cnt >= int(np.ceil(0.7 * 14)), np.nanmean(hist, 0), np.nan)
        need = [C[s, k] for k in (30, 60, 90, 120, 150)]
        if not (np.isfinite(op[s]) and np.isfinite(prev) and np.isfinite(norm[K12]) and all(np.isfinite(need))
                and np.isfinite(amr[s]) and np.isfinite(O[s, K12 + 1])):
            continue
        up = max(op[s], prev) * (1 + norm); dn = min(op[s], prev) * (1 - norm)
        c12 = C[s, K12]
        if not (dn[K12] <= c12 <= up[K12]): continue
        npts = norm[K12] * op[s]; G = op[s] - prev; M = c12 - op[s]; thr = THR * npts
        if abs(G) < thr or abs(M) < thr: cls = "small"
        elif np.sign(G) == np.sign(M): cls = "aligned"
        elif abs(M) >= abs(G): cls = "cancelled"
        else: cls = "survived"
        side = 1.0 if M > 0 else -1.0
        hi, lo = am.at[D[s], "hi"], am.at[D[s], "lo"]
        q = 0.5 if hi <= lo else ((c12 - lo) / (hi - lo) if side > 0 else (hi - c12) / (hi - lo))
        prior = amr[max(0, s - 20):s]; prior = prior[np.isfinite(prior)]
        rho = amr[s] / np.median(prior) if len(prior) >= 10 else np.nan
        e = O[s, K12 + 1]
        r = dict(date=int(D[s]), cls=cls, side=side, G=G, M=M, norm_pts=npts, rho=rho, q=q)
        exit_k, exit_dir = None, 0
        for k in MARKS:
            if not (np.isfinite(C[s, k]) and np.isfinite(norm[k])): break
            if C[s, k] > up[k] or C[s, k] < dn[k]:
                exit_k, exit_dir = k, (1 if C[s, k] > up[k] else -1); break
        for h in HS:
            kk = K12 + h
            y = side * (C[s, kk] - e) if np.isfinite(C[s, kk]) else np.nan
            if exit_k is not None and exit_k <= kk and np.isfinite(O[s, exit_k + 1]):
                ys = side * (O[s, exit_k + 1] - e)
                r[f"exit_with_{h}"] = float(exit_dir == side); r[f"exit_against_{h}"] = float(exit_dir == -side)
            else:
                ys = y
                r[f"exit_with_{h}"] = 0.0; r[f"exit_against_{h}"] = 0.0
            r[f"r{h}"] = y; r[f"rn{h}"] = y / npts; r[f"rs{h}"] = ys
        rows.append(r)
    P = pd.DataFrame(rows)
    P["epoch"] = pd.cut(P.date, [0, 20121231, 20191231, 20251231, 20991231], labels=[e[0] for e in EPOCHS] + ["later"]).astype(str)
    P["month"] = P.date // 100
    for ep in P.epoch.unique():
        m = P.epoch == ep
        P.loc[m, "rt"] = pd.qcut(P.loc[m, "rho"], 3, labels=False, duplicates="drop")
        P.loc[m, "qt"] = pd.qcut(P.loc[m, "q"], 3, labels=False, duplicates="drop")
    P["cell"] = P.epoch + "|" + P.rt.astype(str) + "|" + P.qt.astype(str)
    return P, short


def excess(T, col, labels):
    """Per tested day: y - mean(y of the other two tested classes in the same cell)."""
    x = np.full(len(T), np.nan)
    y = T[col].to_numpy(float); cells = T.cell.to_numpy(); lab = np.asarray(labels)
    for c in np.unique(cells):
        idx = np.flatnonzero(cells == c)
        for k in TESTED:
            me = idx[lab[idx] == k]; other = idx[(lab[idx] != k) & np.isfinite(y[idx])]
            if len(me) and len(other): x[me] = y[me] - y[other].mean()
    return x


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    P, short = build()
    T = P[P.cls.isin(TESTED) & P.rt.notna() & P.qt.notna()].reset_index(drop=True)
    res = dict(declaration="base/105/LOG.md CW1", population=len(P), short_sessions_skipped=short,
               classes=P.groupby(["epoch", "cls"]).size().unstack(fill_value=0).to_dict("index"), cells={})
    cols = [f"r{h}" for h in HS] + [f"rn{h}" for h in HS] + [f"rs{h}" for h in HS] + \
           [f"exit_with_{h}" for h in HS] + [f"exit_against_{h}" for h in HS]
    X = {c: excess(T, c, T.cls.to_numpy()) for c in cols}
    for ep, _, _ in EPOCHS:
        for k in TESTED:
            m = ((T.epoch == ep) & (T.cls == k)).to_numpy()
            for h in HS:
                key = f"{ep}|{k}|{h}"
                d = {}
                for c in (f"r{h}", f"rn{h}", f"rs{h}", f"exit_with_{h}", f"exit_against_{h}"):
                    v = X[c][m]; ok = np.isfinite(v)
                    t, se = month_t(v[ok], T.month.to_numpy()[m][ok])
                    d[c] = dict(n=int(ok.sum()), raw=float(np.nanmean(T[c].to_numpy(float)[m])), excess=float(np.mean(v[ok])) if ok.any() else None,
                                t=t, se=se, unknown=int((~ok).sum()))
                res["cells"][key] = d
    # permutation null on 2020-25, 9 money cells (points, unstopped)
    rng = np.random.default_rng(SEED)
    Tm = T[T.epoch == "2020-25"].reset_index(drop=True)
    months = Tm.month.to_numpy(); cells = Tm.cell.to_numpy(); lab0 = Tm.cls.to_numpy()
    def max_t(lab):
        best = 0.0
        for h in HS:
            x = excess(Tm, f"r{h}", lab)
            for k in TESTED:
                m = (lab == k) & np.isfinite(x)
                t, _ = month_t(x[m], months[m])
                if np.isfinite(t): best = max(best, abs(t))
        return best
    null = []
    for _ in range(NPERM):
        lab = lab0.copy()
        for c in np.unique(cells):
            i = np.flatnonzero(cells == c); lab[i] = rng.permutation(lab[i])
        null.append(max_t(lab))
    res["null_max_abs_t_2020_25"] = dict(p95=float(np.quantile(null, .95)), p50=float(np.median(null)), observed=max_t(lab0))
    live = []
    for k in TESTED:
        for h in HS:
            e = [res["cells"][f"{ep}|{k}|{h}"] for ep, _, _ in EPOCHS]
            ex = [c[f"r{h}"]["excess"] for c in e]; exn = [c[f"rn{h}"]["excess"] for c in e]; exs = [c[f"rs{h}"]["excess"] for c in e]
            main = e[2]
            same = all(v is not None and np.sign(v) == np.sign(ex[0]) and v != 0 for v in ex)
            # "alive when stopped" = stopped excess has the unstopped sign in all three epochs (fixed before the run)
            ok = (same and abs(main[f"r{h}"]["t"]) >= 2
                  and abs(main[f"r{h}"]["t"]) > res["null_max_abs_t_2020_25"]["p95"]
                  and np.sign(main[f"r{h}"]["raw"]) == np.sign(ex[2]) and abs(main[f"r{h}"]["raw"]) > COST_PTS[INST]
                  and all(np.sign(a) == np.sign(b) for a, b in zip(ex, exn))
                  and all(v is not None and np.sign(v) == np.sign(ex[2]) for v in exs))
            if ok: live.append(f"{k}|{h}")
    res["money_cells_passing"] = live
    (OUT / "summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    T.to_csv(OUT / "days.csv", index=False)
    print("population", len(P), "tested", len(T), "short skipped", short)
    print(pd.crosstab(P.epoch, P.cls))
    print("null max|t| 2020-25: p50 %.2f p95 %.2f observed %.2f" % (res["null_max_abs_t_2020_25"]["p50"], res["null_max_abs_t_2020_25"]["p95"], res["null_max_abs_t_2020_25"]["observed"]))
    print(f"{'epoch|class|h':24s} {'n':>4s} {'raw':>7s} {'exc':>7s} {'t':>5s} {'exc_n':>6s} {'exc_stop':>8s} {'t_st':>5s} | {'with':>5s} {'exc':>6s} {'t':>5s} | {'agst':>5s} {'exc':>6s} {'t':>5s}")
    for key, d in res["cells"].items():
        h = key.split("|")[2]
        a, n_, st, w, g = d[f"r{h}"], d[f"rn{h}"], d[f"rs{h}"], d[f"exit_with_{h}"], d[f"exit_against_{h}"]
        print(f"{key:24s} {a['n']:4d} {a['raw']:7.2f} {a['excess']:7.2f} {a['t']:5.1f} {n_['excess']:6.3f} {st['excess']:8.2f} {st['t']:5.1f} | "
              f"{w['raw']:5.2f} {w['excess']:6.3f} {w['t']:5.1f} | {g['raw']:5.2f} {g['excess']:6.3f} {g['t']:5.1f}")
    print("money cells passing:", live)


if __name__ == "__main__":
    main()
