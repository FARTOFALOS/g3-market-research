"""Contract ceiling of a perfect A/B chooser on the saved paired replay, no reader.

Run from repository root: python -B base/105/f0_ab_ceiling.py
Reads f0_ab_days.csv and f0_ab_events.csv only. The oracle uses the future and
is not a trading rule; it bounds what any morning A/B selector can reach on the
contract's day-level requirements (research/NQ_MANUAL_TRADING_RESEARCH_PLAN.md §2).
"flat" adds the trivial third action "no trade that day" for comparison.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "f0_ab_probe_out"


def contract_view(r, traded):
    k = math.ceil(0.05 * traded.sum())
    total = r.sum()
    top = np.sort(r[traded])[::-1][:k].sum()
    return dict(days=len(r), mean_per_day=float(r.mean()), share_days_ge_0=float(np.mean(r >= 0)),
                days_with_trade=int(traded.sum()),
                top5pct_traded_days_share_of_net=float(top / total) if total > 0 else None)


def ceiling(g):
    ra, rb = g.RA.to_numpy(float), g.RB.to_numpy(float)
    na, nb = g.nA.to_numpy() > 0, g.nB.to_numpy() > 0
    pick_a = ra >= rb
    out = dict(A=contract_view(ra, na), B=contract_view(rb, nb),
               oracle_AB=contract_view(np.where(pick_a, ra, rb), np.where(pick_a, na, nb)))
    best = np.maximum(ra, rb)
    flat = best < 0
    out["oracle_AB_flat"] = contract_view(np.where(flat, 0.0, best), np.where(flat, False, np.where(pick_a, na, nb)))
    f = g[flat]
    out["both_lose_days"] = dict(
        days=len(f), share=float(len(f) / len(g)), mean_best_of_AB=float(best[flat].mean()) if len(f) else None,
        share_morning_A_full_stop=float((f.morning_A == -19.75).mean()) if len(f) else None,
        share_B_traded=float((f.nB > 0).mean()) if len(f) else None,
        mean_downstream_A=float(f.downstream_A.mean()) if len(f) else None,
        mean_downstream_B=float(f.downstream_B.mean()) if len(f) else None,
        by_year=f.date.astype(str).str[:4].value_counts().sort_index().to_dict())
    return out


def main():
    days_path, events_path = OUT / "f0_ab_days.csv", OUT / "f0_ab_events.csv"
    days = pd.read_csv(days_path)
    events = pd.read_csv(events_path, low_memory=False)
    s = events[events.event == "settlement"]
    pair = days[(days.p0_status == "yes") & (days.A_status == "resolved") & (days.B_status == "resolved")].copy()
    for p in ("A", "B"):
        pair[f"n{p}"] = pair.date.map(s[s.policy == p].groupby("date").size()).fillna(0)
    result = dict(
        source={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (days_path, events_path)},
        tie_rule="A when RA == RB; the day-level figures do not depend on it here",
        note="oracle uses the future; not a trading rule; conditional on P_AB",
        territory_2020_2025=ceiling(pair[pair.date >= 20200101]),
        all_history=ceiling(pair))
    (OUT / "oracle_ceiling.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["territory_2020_2025"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
