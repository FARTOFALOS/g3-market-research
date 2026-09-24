"""RP1 — nature of the B1 premium: predictable instances or pay for constant exposure (declared in base/105/LOG.md).

B1 daily from the frozen engines (setups/library/joint.build) over NQ 2006-01 .. 2026-05-04; the 2020+ part must equal
joint_daily_NQ.csv to the cent. (a) frequency of big days (top 5 % of the epoch) by pre-09:30 state; (b) S-07 payoff
split by exit reason; (c) losing streaks vs within-epoch shuffles. Nothing here changes B1 (FREEZE_B1 section 5).

Run from repository root: python -B base/105/rp1_b1_premium.py
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

OUT = HERE / "rp1_out"
SEED, NPERM = 20260924, 1000


def epoch(d):
    return "2006-12" if d < pd.Timestamp("2013-01-01") else ("2013-19" if d < pd.Timestamp("2020-01-01") else "2020-26")


def wilson(k, n):
    if n == 0: return (np.nan, np.nan)
    p = k / n; z = 1.96; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


def main():
    OUT.mkdir(exist_ok=True)
    cal = J.S17.history_calendar(); root = ROOT / "data/market/NQ"
    d, e, _ = J.build(root, cal, pd.Timestamp("2006-01-03"), pd.Timestamp("2026-05-04"))
    ref = pd.read_csv(ROOT / "setups/library/joint_daily_NQ.csv", parse_dates=["date"]).set_index("date").B1
    chk = d.set_index("date").B1.reindex(ref.index)
    if not np.allclose(chk.dropna(), ref[chk.notna()], atol=0.01) or (chk.isna() != ref.isna()).any():
        raise SystemExit("B1 over 2020+ does not reproduce joint_daily_NQ.csv")
    d = d[["date", "s07", "s18_after_s07", "B1"]].copy()
    d["epoch"] = d.date.map(epoch)
    # pre-09:30 state
    tape = load_minutes("2005-01-01", "2026-05-05", "NQ")
    rth = tape[(tape["mod"] >= 570) & (tape["mod"] < 960)].groupby("date").agg(h=("h", "max"), l=("l", "min"), o=("o", "first"), c=("c", "last"))
    rth.index = pd.to_datetime(rth.index.astype(str)); rth["rng"] = rth.h - rth.l
    rel = rth.rng.rolling(20).mean().shift(1) / rth.rng.rolling(250).median().shift(1)
    gap = (rth.o - rth.c.shift(1)).abs()
    gapn = gap / gap.rolling(250).median().shift(1)
    d = d.set_index("date"); d["vol_regime"] = rel.reindex(d.index); d["gap_rel"] = gapn.reindex(d.index)
    known = d.B1.notna()
    d["big"] = False
    for ep, g in d[known].groupby("epoch"):
        d.loc[g.index, "big"] = g.B1 > g.B1.quantile(0.95)
    # prefix covariates built from B1 up to yesterday (big uses the epoch threshold -> descriptive clustering)
    since, streak, prev_sign = [], [], []
    last_big, run = None, 0
    prev = np.nan
    for i, (dt, r) in enumerate(d.iterrows()):
        since.append(np.nan if last_big is None else i - last_big); streak.append(run); prev_sign.append(np.sign(prev) if np.isfinite(prev) else np.nan)
        if np.isfinite(r.B1):
            if r.big: last_big = i
            run = run + 1 if r.B1 < 0 else 0
            prev = r.B1
    d["since_big"] = since; d["loss_streak"] = streak; d["prev_sign"] = prev_sign
    K = d[known].copy()
    K["since_b"] = pd.cut(K.since_big, [0, 1, 5, 20, 1e9], labels=["1", "2-5", "6-20", "21+"])
    K["streak_b"] = pd.cut(K.loss_streak, [-1, 0, 2, 5, 1e9], labels=["0", "1-2", "3-5", "6+"])
    res = dict(declaration="base/105/LOG.md RP1", sessions=int(known.sum()), regression="2020+ equals joint_daily_NQ.csv", hazard={}, s07_payoff={}, streaks={})
    for ep, g in K.groupby("epoch"):
        g = g.copy()
        g["vol_b"] = pd.qcut(g.vol_regime, 3, labels=["low", "mid", "high"])
        g["gap_b"] = pd.qcut(g.gap_rel, 3, labels=["small", "mid", "large"])
        out = {}
        for var in ("since_b", "streak_b", "prev_sign", "vol_b", "gap_b"):
            for lev, x in g.groupby(var, observed=True):
                k, n = int(x.big.sum()), len(x)
                lo, hi = wilson(k, n)
                out[f"{var}={lev}"] = dict(n=n, big_rate=round(k / n, 4), ci95=[round(lo, 4), round(hi, 4)],
                                           mean_B1=round(float(x.B1.mean()), 1))
        res["hazard"][ep] = out
    # (b) S-07 payoff by exit reason
    e["date"] = pd.to_datetime(e.date); e = e[e.status.isin(["ok", "entered_hole"]) & e.s07_usd.notna()].copy()
    e["epoch"] = e.date.map(epoch)
    e["part"] = np.where(e.s07_reason.str.startswith("минус на 15"), "cut at minute 15",
                np.where(e.s07_reason.str.startswith("катастроф"), "catastrophic limit", "held (120 min / close)"))
    for ep, g in e.groupby("epoch"):
        tot = g.s07_usd.sum()
        res["s07_payoff"][ep] = {p: dict(share_trades=round(len(x) / len(g), 3), mean_usd=round(float(x.s07_usd.mean()), 1),
                                         sum_usd=round(float(x.s07_usd.sum()), 0)) for p, x in g.groupby("part")}
        res["s07_payoff"][ep]["total_usd"] = round(float(tot), 0); res["s07_payoff"][ep]["trades"] = len(g)
    # (c) losing streaks vs shuffles
    rng = np.random.default_rng(SEED)
    def streaks(sig):
        out, run = [], 0
        for v in sig:
            if v < 0: run += 1
            elif run: out.append(run); run = 0
        if run: out.append(run)
        return np.array(out)
    for ep, g in K.groupby("epoch"):
        s = g.B1.to_numpy(); st = streaks(s)
        null_mean, null_max, null6 = [], [], []
        for _ in range(NPERM):
            p = streaks(rng.permutation(s)); null_mean.append(p.mean()); null_max.append(p.max()); null6.append((p >= 6).mean())
        res["streaks"][ep] = dict(share_losing_days=round(float((s < 0).mean()), 3), mean_streak=round(float(st.mean()), 2),
                                  max_streak=int(st.max()), share_streaks_ge6=round(float((st >= 6).mean()), 4),
                                  null_mean_streak_p95=round(float(np.quantile(null_mean, .95)), 2), null_max_p95=int(np.quantile(null_max, .95)),
                                  null_share_ge6_p95=round(float(np.quantile(null6, .95)), 4))
    # summary per epoch of B1 itself
    res["b1_by_epoch"] = {ep: dict(sessions=len(g), usd_per_session=round(float(g.B1.mean()), 1), t=round(float(g.B1.mean() / g.B1.std() * math.sqrt(len(g))), 2),
                                   s07=round(float(g.s07.mean()), 1), s18=round(float(g.s18_after_s07.mean()), 1)) for ep, g in K.groupby("epoch")}
    (OUT / "summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res["b1_by_epoch"], ensure_ascii=False))
    for ep, h in res["hazard"].items():
        print("==", ep)
        for k, v in h.items(): print(f"   {k:22s} n {v['n']:5d} big {v['big_rate']:.3f} ci {v['ci95']} mean B1 {v['mean_B1']:8.1f}")
    print(json.dumps(res["s07_payoff"], ensure_ascii=False, indent=1))
    print(json.dumps(res["streaks"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
