"""RV1 — premarket price reversion as a library payoff source (declared in base/105/LOG.md before counting).

Decision minute t (bar t closed), 02:30..09:14 ET. P = c(t) - c(t-L); F = c(t+H) - o(t+1); L, H in {15, 30}; exit
no later than the 09:29 bar; units u = median range of the 30 previous bars; windows must be consecutive minutes.
1. Residual: slope of F/u on P/u within cells (half-hour x u tercile x tercile of position in the night range since
   18:00 of the previous day), day-clustered t, within-cell permutation null (200).
3. Money: fade P/u in the top / bottom tercile of its cell; entry o(t+1), exit close(t+H); one position; cost 0.75 pt;
   then B1 + RV1 on 2020-25 (no clock overlap).

Run from repository root: python -B base/105/rv1_premarket.py
"""
import json, math, sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from tape import load_minutes  # noqa: E402

OUT = HERE / "rv1_out"
LS, HS = (15, 30), (15, 30)
COST, PT, NPERM, SEED = 0.75, 20.0, 200, 20260924
MINNS = 60_000_000_000


def epoch(d):
    return "2006-12" if d <= 20121231 else ("2013-19" if d <= 20191231 else "2020-25")


def build():
    df = load_minutes("2005-12-01", "2026-01-01", "NQ").reset_index(drop=True)
    o, h, l, c, ts, mod, date = (df[k].to_numpy() for k in ("o", "h", "l", "c", "ts", "mod", "date"))
    u = pd.Series(h - l).rolling(30).median().shift(1).to_numpy()
    rows = []
    days = np.unique(date[(date >= 20060101) & (date <= 20251231)])
    et = pd.to_datetime(ts - MINNS, utc=True).tz_convert("America/New_York")   # bar open, ET
    for D in days:
        d0 = pd.Timestamp(str(D)).tz_localize("America/New_York")
        start = (d0 - pd.Timedelta(hours=6)).value      # 18:00 of the previous calendar day (bar open)
        end = (d0 + pd.Timedelta(hours=9, minutes=30)).value
        a = np.searchsorted(et.asi8, start); b = np.searchsorted(et.asi8, end)
        if b - a < 200: continue
        idx = np.arange(a, b)
        runH = np.maximum.accumulate(h[idx]); runL = np.minimum.accumulate(l[idx])
        for j, i in enumerate(idx):
            m = mod[i]
            if date[i] != D or m < 150 or m > 554 or not np.isfinite(u[i]) or u[i] <= 0: continue
            r = dict(date=int(D), mod=int(m), u=float(u[i]), q=float((c[i] - runL[j]) / (runH[j] - runL[j])) if runH[j] > runL[j] else 0.5)
            for L in LS:
                k = i - L
                r[f"P{L}"] = c[i] - c[k] if j - L >= 0 and ts[i] - ts[k] == L * MINNS else np.nan
            for H in HS:
                k = i + H
                ok = k < len(c) and ts[k] - ts[i] == H * MINNS and date[k] == D and mod[k] <= 569
                r[f"F{H}"] = c[k] - o[i + 1] if ok else np.nan
                r[f"x{H}"] = float(c[k]) if ok else np.nan
                r[f"e{H}"] = float(o[i + 1]) if ok else np.nan
            rows.append(r)
    R = pd.DataFrame(rows)
    R["epoch"] = R.date.map(epoch)
    R["bucket"] = (R["mod"] - 120) // 30
    for ep in R.epoch.unique():
        m = R.epoch == ep
        R.loc[m, "ut"] = pd.qcut(R.loc[m, "u"], 3, labels=False, duplicates="drop")
        R.loc[m, "qt"] = pd.qcut(R.loc[m, "q"], 3, labels=False, duplicates="drop")
    R["cell"] = R.epoch + "|" + R.bucket.astype(str) + "|" + R.ut.astype(str) + "|" + R.qt.astype(str)
    return R


def slope(x, y, cells, day):
    xc = x - pd.Series(x).groupby(cells).transform("mean").to_numpy()
    yc = y - pd.Series(y).groupby(cells).transform("mean").to_numpy()
    sxx = (xc * xc).sum(); b = (xc * yc).sum() / sxx
    s = pd.Series(xc * (yc - b * xc)).groupby(day).sum().to_numpy()
    se = math.sqrt((s * s).sum()) / sxx
    return float(b), float(b / se)


def main():
    OUT.mkdir(exist_ok=True)
    R = build(); rng = np.random.default_rng(SEED)
    res = dict(declaration="base/105/LOG.md RV1", rows=len(R), stage1={}, stage3={})
    for L in LS:
        for H in HS:
            for ep, g in R.groupby("epoch"):
                g = g[np.isfinite(g[f"P{L}"]) & np.isfinite(g[f"F{H}"])]
                x = (g[f"P{L}"] / g.u).to_numpy(); y = (g[f"F{H}"] / g.u).to_numpy()
                cells = g.cell.to_numpy(); day = g.date.to_numpy()
                b, t = slope(x, y, cells, day)
                codes = pd.factorize(cells)[0]; base = np.argsort(codes, kind="stable"); null = []
                for _ in range(NPERM):
                    p = np.lexsort((rng.random(len(x)), codes)); xp = np.empty_like(x); xp[p] = x[base]
                    null.append(slope(xp, y, cells, day)[0])
                res["stage1"][f"L{L}|H{H}|{ep}"] = dict(n=len(g), days=int(len(np.unique(day))), beta=b, t=t,
                                                        null_q005=float(np.quantile(null, .005)), null_q995=float(np.quantile(null, .995)))
    # money: fade the top / bottom tercile of P/u within its cell, one position, no overlap
    daily = {}
    for L in LS:
        for H in HS:
            g = R[np.isfinite(R[f"P{L}"]) & np.isfinite(R[f"F{H}"])].copy()
            g["z"] = g[f"P{L}"] / g.u
            lo = g.groupby("cell").z.transform(lambda v: v.quantile(1 / 3)); hi = g.groupby("cell").z.transform(lambda v: v.quantile(2 / 3))
            g["side"] = np.where(g.z >= hi, -1, np.where(g.z <= lo, 1, 0))
            g = g.sort_values(["date", "mod"])
            trades = []
            for D, d in g[g.side != 0].groupby("date"):
                free = -1
                for r in d.itertuples():
                    if r.mod <= free: continue
                    net = r.side * (getattr(r, f"x{H}") - getattr(r, f"e{H}")) - COST
                    trades.append((D, r.mod, r.side, net)); free = r.mod + H
            T = pd.DataFrame(trades, columns=["date", "mod", "side", "net"]); T["epoch"] = T.date.map(epoch)
            dd = T.groupby("date").net.sum()
            key = f"L{L}|H{H}"; daily[key] = dd
            res["stage3"][key] = {}
            for ep, te in T.groupby("epoch"):
                per_day = te.groupby("date").net.sum()
                res["stage3"][key][ep] = dict(trades=len(te), days=len(per_day), net_per_trade=float(te.net.mean()),
                                              gross_per_trade=float(te.net.mean() + COST), share_win=float((te.net > 0).mean()),
                                              pts_per_day=float(per_day.mean()),
                                              t_day=float(per_day.mean() / per_day.std(ddof=1) * math.sqrt(len(per_day))))
    passed1 = [f"L{L}|H{H}" for L in LS for H in HS if all(
        (lambda s: s["beta"] < 0 and abs(s["t"]) >= 3 and not (s["null_q005"] <= s["beta"] <= s["null_q995"]))(res["stage1"][f"L{L}|H{H}|{ep}"])
        for ep in ("2013-19", "2020-25"))]
    passed3 = [k for k, v in res["stage3"].items() if all(v.get(ep, {}).get("net_per_trade", -1) > 0 and v.get(ep, {}).get("t_day", 0) >= 2 for ep in ("2013-19", "2020-25"))]
    res["stage1_passed"] = passed1; res["stage3_passed"] = passed3
    # library increment for passing money cells
    j = pd.read_csv(ROOT / "setups/library/joint_daily_NQ.csv", parse_dates=["date"]).set_index("date").B1
    j = j.loc["2020-01-02":"2025-12-31"].dropna()
    def curve(x):
        x = np.asarray(x, float); k = max(1, math.ceil(0.05 * (x != 0).sum())); tot = x.sum(); cum = np.cumsum(x)
        return dict(usd_per_session=round(float(x.mean()), 1), share_days_ge_0=round(float((x >= 0).mean()), 3),
                    top5_share=round(float(np.sort(x[x != 0])[::-1][:k].sum() / tot), 3) if tot > 0 else None,
                    max_drawdown=round(float((cum - np.maximum.accumulate(cum)).min()), 0))
    res["library"] = {}
    for key, dd in daily.items():
        rv = pd.Series(dd.to_numpy() * PT, index=pd.to_datetime(dd.index.astype(str))).reindex(j.index, fill_value=0.0)
        res["library"][key] = dict(B1=curve(j), B1_plus_RV1=curve(j + rv), rv1_usd_per_session=round(float(rv.mean()), 1),
                                   corr_with_B1=round(float(rv.corr(j)), 3))
    (OUT / "summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("rows", len(R))
    for k, s in res["stage1"].items():
        print(f"{k:22s} n {s['n']:7d} beta {s['beta']:+.4f} t {s['t']:+6.1f} null99 [{s['null_q005']:+.4f},{s['null_q995']:+.4f}]")
    for k, v in res["stage3"].items():
        for ep, s in v.items():
            print(f"{k:8s} {ep}: trades {s['trades']:6d} gross/trade {s['gross_per_trade']:+.3f} net/trade {s['net_per_trade']:+.3f} win {s['share_win']:.3f} pts/day {s['pts_per_day']:+.2f} t {s['t_day']:+.1f}")
    for k, v in res["library"].items():
        print(k, "B1", v["B1"], "B1+RV1", v["B1_plus_RV1"], "rv1 $/s", v["rv1_usd_per_session"], "corr", v["corr_with_B1"])
    print("stage 1 passed:", passed1, "| money passed:", passed3)


if __name__ == "__main__":
    main()
