"""TR1 — transition to two-sided acceptance after a new session extreme (declared in base/105/LOG.md).

Event: bar t (09:31..15:00 bar-open) makes a new session high (s=+1) or low (s=-1) since 09:30.
Chain: bars after t without a new same-side extreme. Recognition p = t+k is the first k <= 30 where
closes of bars t..p changed side of the current centre m = (H + L)/2 at least twice (x >= 2).
Residual from open(p+1): r_h = s*(close(p+h) - open(p+1)), h = 1,2,3,5; MFE/MAE over 5 bars.
Control: same chains at the same age k with x < 2, matched by side x k x half-hour x depth tercile
x position tercile within the same epoch.  Month-block standard errors.

Run from repository root: python -B base/105/tr1_transition.py
Heavy per-state rows go to work/105-tr1/ (not in git); the summary goes to base/105/tr1_out/.
"""
import json, math, os
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT, HEAVY = HERE / "tr1_out", ROOT / "work/105-tr1"
H_LIST = (1, 2, 3, 5)
COST = 0.75
KMAX, LAST_EVENT_MOD = 30, 900


@njit(cache=True)
def day_states(o, h, l, c, u, last_j):
    nb = len(o)
    cap = 2 * nb * KMAX
    kind = np.empty(cap, np.int8); side = np.empty(cap, np.int8); jj = np.empty(cap, np.int32)
    kk = np.empty(cap, np.int32); depth = np.empty(cap); q = np.empty(cap)
    res = np.empty((cap, 6)); c30 = np.empty(cap, np.int8)
    n = 0
    for s in (1, -1):
        for j in range(1, min(last_j, nb - 7) + 1):
            # new session extreme on side s at bar j?
            new = True
            for i in range(j):
                if (s > 0 and h[i] >= h[j]) or (s < 0 and l[i] <= l[j]):
                    new = False; break
            if not new: continue
            ext = h[j] if s > 0 else l[j]
            jn = -1
            for i in range(j + 1, nb):
                if (s > 0 and h[i] > ext) or (s < 0 and l[i] < ext):
                    jn = i; break
            lc = (jn - j - 1) if jn >= 0 else (nb - 1 - j)
            cont = 1 if (jn >= 0 and jn - j <= 30) else 0
            opp = l[j] if s > 0 else h[j]
            for k in range(1, min(KMAX, lc) + 1):
                p = j + k
                if p + 5 > nb - 1: break
                if s > 0: opp = min(opp, l[p])
                else: opp = max(opp, h[p])
                m = 0.5 * (ext + opp)
                x = 0; prev = 0
                for i in range(j, p + 1):
                    d = c[i] - m
                    sg = 1 if d > 0 else (-1 if d < 0 else 0)
                    if sg != 0:
                        if prev != 0 and sg != prev: x += 1
                        prev = sg
                if not np.isfinite(u[p]) or u[p] <= 0: continue
                rng = abs(ext - opp)
                kind[n] = 1 if x >= 2 else 0
                side[n] = s; jj[n] = j; kk[n] = k
                depth[n] = rng / u[p]
                q[n] = (0.5 if rng == 0 else (c[p] - opp) / (ext - opp))
                e = o[p + 1]
                res[n, 0] = s * (c[p + 1] - e); res[n, 1] = s * (c[p + 2] - e)
                res[n, 2] = s * (c[p + 3] - e); res[n, 3] = s * (c[p + 5] - e)
                if s > 0:
                    res[n, 4] = h[p + 1:p + 6].max() - e; res[n, 5] = e - l[p + 1:p + 6].min()
                else:
                    res[n, 4] = e - l[p + 1:p + 6].min(); res[n, 5] = h[p + 1:p + 6].max() - e
                c30[n] = cont
                n += 1
                if x >= 2: break          # first honest recognition only
    return kind[:n], side[:n], jj[:n], kk[:n], depth[:n], q[:n], res[:n], c30[:n]


def month_mean_se(x, months):
    x = np.asarray(x, float); n = len(x)
    if n < 3: return dict(n=n, mean=float(np.mean(x)) if n else None, se=None, t=None)
    s = pd.Series(x).groupby(np.asarray(months)).sum().to_numpy(); k = len(s)
    se = math.sqrt(k / (k - 1) * ((s - s.mean()) ** 2).sum()) / n if k > 1 else float("nan")
    return dict(n=n, mean=float(x.mean()), median=float(np.median(x)), se=float(se), t=float(x.mean() / se) if se > 0 else None)


def main():
    os.chdir(HERE); OUT.mkdir(exist_ok=True); HEAVY.mkdir(parents=True, exist_ok=True)
    from trade import Tape, epoch
    T = Tape(end="2026-01-01")
    frames, skipped = [], 0
    for D in sorted(T.days):
        if not (20060101 <= D <= 20251231): continue
        ix = T.days[D]; md = T.mod[ix]; cm = int(T.close_mod[ix[0]])
        sx = ix[(md >= 570) & (md < cm)]
        if cm <= 600 or len(sx) != cm - 570 or T.mod[sx[0]] != 570:
            skipped += 1; continue
        last_j = min(LAST_EVENT_MOD, cm - 7) - 570
        kind, side, j, k, depth, q, res, c30 = day_states(T.o[sx], T.h[sx], T.l[sx], T.c[sx], T.u[sx], last_j)
        if len(kind) == 0: continue
        f = pd.DataFrame(dict(date=D, kind=kind, side=side, j=j, k=k, depth=depth, q=q, c30=c30,
                              r1=res[:, 0], r2=res[:, 1], r3=res[:, 2], r5=res[:, 3], mfe=res[:, 4], mae=res[:, 5]))
        f["pmod"] = 570 + f.j + f.k
        frames.append(f)
    S = pd.concat(frames, ignore_index=True)
    S["epoch"] = S.date.map(epoch); S["month"] = S.date // 100; S["year"] = S.date // 10000
    S["bucket"] = (S.pmod - 570) // 30
    S["dt"] = pd.qcut(S.depth, 3, labels=False, duplicates="drop")
    S["qt"] = pd.qcut(S.q.clip(0, 1), 3, labels=False, duplicates="drop")
    cell = ["epoch", "side", "k", "bucket", "dt", "qt"]
    rcols = ["r1", "r2", "r3", "r5", "mfe", "mae", "c30"]
    ctrl = S[S.kind == 0].groupby(cell)[rcols].mean().add_prefix("ctrl_")
    B = S[S.kind == 1].join(ctrl, on=cell)
    unmatched = int(B.ctrl_r1.isna().sum()); B = B[B.ctrl_r1.notna()].copy()
    for c in rcols: B[f"ex_{c}"] = B[c] - B[f"ctrl_{c}"]
    S[S.kind == 1].to_parquet(HEAVY / "bilateral_states.parquet")

    years = {"2006-12": 7, "2013-19": 7, "2020-25": 6}
    table = {}
    for ep, g in B.groupby("epoch"):
        row = dict(bilateral_per_year=len(g) / years[ep], days=int(g.date.nunique()),
                   median_k=float(g.k.median()), c30_bilateral=float(g.c30.mean()), c30_control=float(g.ctrl_c30.mean()))
        for c in ("r1", "r2", "r3", "r5", "mfe", "mae"):
            row[c] = month_mean_se(g[c], g.month)
            row[f"excess_{c}"] = month_mean_se(g[f"ex_{c}"], g.month)
        row["by_side"] = {int(sd): {c: float(x[c].mean()) for c in ("r1", "r3", "r5", "ex_r1", "ex_r3", "ex_r5")}
                          for sd, x in g.groupby("side")}
        table[ep] = row
    base = {ep: {c: float(g[c].mean()) for c in ("r1", "r2", "r3", "r5")} for ep, g in S[S.kind == 0].groupby("epoch")}

    live = []
    for c in ("r1", "r2", "r3", "r5"):
        for pre in ("", "excess_"):
            ms = [table[ep][pre + c]["mean"] for ep in years]
            if all(m > COST for m in ms) or all(m < -COST for m in ms): live.append(pre + c)
    verdict = "family lives: first trading version in this pass" if live else "family closed in one pass"
    summary = dict(declaration="base/105/LOG.md TR1", states_total=len(S), bilateral=int((S.kind == 1).sum()),
                   control_states=int((S.kind == 0).sum()), unmatched_bilateral=unmatched, skipped_days=skipped,
                   cost_points=COST, table=table, control_all_ages_mean=base, live_cells=live, verdict=verdict)
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"states {len(S)}  bilateral {summary['bilateral']}  unmatched {unmatched}  skipped days {skipped}")
    print(f"{'epoch':8s} {'per yr':>7s} {'c30 B/ctl':>10s} | " + " | ".join(f"r{h} raw  exc" for h in H_LIST) + " | mfe  mae")
    for ep in years:
        r = table[ep]
        cells = " | ".join(f"{r['r'+str(h)]['mean']:+.2f} {r['excess_r'+str(h)]['mean']:+.2f}" for h in H_LIST)
        print(f"{ep:8s} {r['bilateral_per_year']:7.0f} {r['c30_bilateral']:.2f}/{r['c30_control']:.2f} | {cells} | "
              f"{r['mfe']['mean']:.2f} {r['mae']['mean']:.2f}")
    for ep in years:
        r = table[ep]
        print(ep, "t(raw r3)", round(r["r3"]["t"], 2), "t(excess r3)", round(r["excess_r3"]["t"], 2),
              "median r3", round(r["r3"]["median"], 2), "se r3", round(r["r3"]["se"], 3))
    print("verdict:", verdict, live)


if __name__ == "__main__":
    main()
