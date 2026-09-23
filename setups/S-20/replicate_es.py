"""Same frozen S-20 v1 action on ES, with $20 roundtrip cost (0.4 ES point)."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
import run as s20
xc = s20.xc

ES_COST_POINTS = 0.4


def main():
    a, manifest = s20.load_market("ES")
    cal = xc.get_calendar("XNYS", start=s20.BEGIN, end=s20.END)
    rows = []
    for r in cal.schedule.itertuples():
        row = s20.one_day(a, r.Index.strftime("%Y-%m-%d"),
                          pd.Timestamp(r.open).value, pd.Timestamp(r.close).value)
        if row["status"] == "known" and row["trades"]:
            row["net"] = row["gross"] - ES_COST_POINTS
        rows.append(row)
    days = pd.DataFrame(rows)
    out = ROOT / "setups/S-20"
    days.to_csv(out / "daily_ES.csv", index=False)
    known = days.status.eq("known")
    t = days[known & days.trades.eq(1)]
    net = days.net.fillna(0).to_numpy()
    mask = known.to_numpy(float)
    by_year = days.loc[known].assign(year=lambda z: z.date.str[:4]).groupby("year").agg(
        sessions=("net", "size"), mean=("net", "mean"), trades=("trades", "sum")
    ).reset_index().to_dict("records")
    result = {
        "rule": "S-20-v1-first-retest-opening-hour-body-cross",
        "instrument": "ES", "cost_points": ES_COST_POINTS, "cost_usd": 20,
        "contract_multiplier_usd_per_point": 50,
        "contract_specs": "https://www.cmegroup.com/markets/equities/sp/e-mini-sandp500.contractSpecs.html",
        "corpus_id": manifest["corpus_id"],
        "period": [s20.BEGIN, s20.END],
        "runtime_versions": {"numpy": np.__version__, "pandas": pd.__version__,
                             "exchange_calendars": xc.__version__},
        "sessions": len(days), "known_sessions": int(known.sum()),
        "unknown_sessions": int((~known).sum()),
        "reasons": days.reason.value_counts().to_dict(),
        "known_entries": len(t),
        "known_net_sum_points": float(days.loc[known, "net"].sum()),
        "known_net_per_session_points": float(days.loc[known, "net"].mean()),
        "known_net_per_session_usd": float(days.loc[known, "net"].mean()*50),
        "ci95_block20_conditional_points": s20.block_ci(net, mask),
        "known_trade_mean_net_points": float(t.net.mean()) if len(t) else None,
        "known_trade_median_net_points": float(t.net.median()) if len(t) else None,
        "known_trade_worst_net_points": float(t.net.min()) if len(t) else None,
        "known_trade_win_rate": float((t.net > 0).mean()) if len(t) else None,
        "top5_net_sum_points": float(t.nlargest(5, "net").net.sum()),
        "by_year_conditional": by_year,
        "frozen_rule_code_sha256": hashlib.sha256((out / "run.py").read_bytes()).hexdigest(),
        "interpretation_boundary": "Correlated cross-instrument replication; no independent temporal validation."
    }
    (out / "result_ES.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
