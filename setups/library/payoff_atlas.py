"""Payoff atlas: every positive or near-positive historical candidate of G3 against the frozen library B1.

For each candidate X (daily $ per NQ contract, NQ 2020-01-02 .. 2025-12-31, days where both X and B1 are known):
  standalone: $/session, t, share of days > 0, share of total from its best 5 % traded days, traded days;
  relation to B1: daily correlation, overlap of best-5 % days, X on days B1 <= 0 and on B1's worst half;
  naive library B1 + X: $/session, best-5 % share of total, share of days >= 0, max drawdown, worst day, vs B1 alone.
Naive = daily sum; one-position conflicts are NOT resolved here (clock overlap is listed; a real replay is the
next step for candidates worth it). Sources are the saved series of each card; nothing is re-optimized.

    python -B setups/library/payoff_atlas.py
Output: setups/library/payoff_atlas.json
"""
import json, math
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LO, HI = pd.Timestamp("2020-01-02"), pd.Timestamp("2025-12-31")
PT = 20.0


def load():
    j = pd.read_csv(HERE / "joint_daily_NQ.csv", parse_dates=["date"]).set_index("date")
    S = {"B1": j.B1, "S-07": j.s07, "S-17": j.s17, "S-18 (alone)": j.s18}
    s08 = pd.read_csv(ROOT / "setups/S-08/daily.csv", parse_dates=["date"]).set_index("date")
    S["S-08 v1"] = s08.v1
    tr = pd.read_csv(ROOT / "setups/S-09/trades.csv")
    tr["date"] = pd.to_datetime(tr.day.str[:10])
    days09 = pd.read_csv(ROOT / "setups/S-09/daily.csv"); days09 = pd.to_datetime(days09.day.str[:10])
    for door in sorted(tr.door.unique()):
        s = tr[tr.door == door].groupby("date").net.sum().reindex(days09, fill_value=0.0)
        S[f"S-09 door {door}"] = s
    s20 = pd.read_csv(ROOT / "setups/S-20/daily.csv", parse_dates=["date"]).set_index("date")
    S["S-20 v1 (pts x20)"] = s20.net.where(s20.status == "known") * PT
    f0 = pd.read_csv(ROOT / "base/105/f0_ab_probe_out/f0_ab_days.csv")
    f0 = f0[f0.B_status == "resolved"]
    S["F0 pullback PV2 (world B, pts x20)"] = pd.Series(f0.RB.to_numpy() * PT, index=pd.to_datetime(f0.date.astype(str)))
    return S


CLOCK = {"S-07": "09:34 → до 11:34", "S-17": "09:34 → 30 мин (та же сделка, что S-07)", "S-18 (alone)": "отметки 10:00–15:30, до закрытия",
         "S-08 v1": "09:31–10:00, 60 мин", "S-09 door utro_0933": "09:33, до 120 мин (сторона как у S-07)",
         "S-20 v1 (pts x20)": "после утреннего экстремума, первый ретест", "F0 pullback PV2 (world B, pts x20)": "обычно 10:02–10:22, до 120 мин"}


def curve(x):
    x = np.asarray(x, float); k = max(1, math.ceil(0.05 * (x != 0).sum())); tot = x.sum()
    cum = np.cumsum(x); dd = float((cum - np.maximum.accumulate(cum)).min())
    top = np.sort(x[x != 0])[::-1][:k].sum()
    return dict(usd_per_session=float(x.mean()), t=float(x.mean() / x.std(ddof=1) * math.sqrt(len(x))) if x.std() > 0 else None,
                share_days_ge_0=float((x >= 0).mean()), top5_share_of_total=float(top / tot) if tot > 0 else None,
                max_drawdown=dd, worst_day=float(x.min()), traded_days=int((x != 0).sum()))


def main():
    S = load(); b1 = S["B1"]
    out = dict(territory=[str(LO.date()), str(HI.date())], note="naive daily sums; one-position conflicts not resolved", candidates={})
    for name, x in S.items():
        if name == "B1": continue
        d = pd.concat([b1.rename("b1"), x.rename("x")], axis=1).loc[LO:HI].dropna()
        if len(d) < 50: continue
        k = max(1, math.ceil(0.05 * len(d)))
        best_x = set(d.x.nlargest(k).index); best_b = set(d.b1.nlargest(k).index)
        worst_half = d.b1 <= d.b1.median()
        out["candidates"][name] = dict(
            days=len(d), clock=CLOCK.get(name, "13:38 (дневная дверь)" if "door" in name else ""),
            standalone=curve(d.x), corr_with_B1=float(d.x.corr(d.b1)),
            best5_overlap_with_B1=float(len(best_x & best_b) / k),
            x_on_B1_le0=float(d.x[d.b1 <= 0].mean()), x_on_B1_worst_half=float(d.x[worst_half].mean()),
            x_on_B1_best_half=float(d.x[~worst_half].mean()),
            B1_alone=curve(d.b1), B1_plus_x=curve(d.b1 + d.x))
    (HERE / "payoff_atlas.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{'candidate':38s} {'days':>5s} {'$/s':>7s} {'t':>5s} {'top5':>6s} {'corr':>6s} {'ovl5':>5s} {'x|B1<=0':>8s} {'x|B1 worst½':>11s} | {'B1 $/s':>7s} {'B1+x':>7s} {'top5 B1':>7s} {'top5 +x':>7s} {'dd B1':>8s} {'dd +x':>8s}")
    for name, r in out["candidates"].items():
        s, a, c = r["standalone"], r["B1_alone"], r["B1_plus_x"]
        f = lambda v: "   n/a" if v is None else f"{v:6.2f}"
        print(f"{name:38s} {r['days']:5d} {s['usd_per_session']:7.1f} {s['t'] or 0:5.1f} {f(s['top5_share_of_total'])} {r['corr_with_B1']:6.2f} {r['best5_overlap_with_B1']:5.2f} "
              f"{r['x_on_B1_le0']:8.1f} {r['x_on_B1_worst_half']:11.1f} | {a['usd_per_session']:7.1f} {c['usd_per_session']:7.1f} {f(a['top5_share_of_total'])}  {f(c['top5_share_of_total'])} {a['max_drawdown']:8.0f} {c['max_drawdown']:8.0f}")


if __name__ == "__main__":
    main()
