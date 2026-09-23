"""Independent checks of saved S-20 v1 fills and first exit conditions."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
import run as s20
MINUTE = s20.MINUTE


def check(a, days, saved, cost):
    assert days.date.is_unique
    assert len(days) == 1508
    t = days[(days.status == "known") & (days.trades == 1)]
    for r in t.itertuples():
        q, en, ex = int(r.event_pos), int(r.entry_pos), int(r.exit_pos)
        d, level = int(r.direction), float(r.level)
        assert en == q + 1 and a["ts"][en] - a["ts"][q] == MINUTE
        assert d * (a["open"][q] - level) < 0 < d * (a["close"][q] - level)
        assert r.entry_price == float(a["open"][en])
        assert np.all(np.diff(a["ts"][en:ex+1]) == MINUTE)
        if r.reason == "closed_back_inside":
            k = ex - 1
            assert d * (a["close"][k] - level) <= 0
            assert ex == k + 1 and r.exit_price == float(a["open"][ex])
            assert np.all(d * (a["close"][en:k] - level) > 0)
        else:
            assert r.reason == "session_close"
            assert r.exit_price == float(a["close"][ex])
            assert np.all(d * (a["close"][en:ex] - level) > 0)
        assert abs(r.gross - d * (r.exit_price - r.entry_price)) < 1e-9
        assert abs(r.net - (r.gross - cost)) < 1e-9
    assert saved["known_entries"] == len(t)
    assert abs(days.loc[days.status == "known", "net"].sum()
               - saved["known_net_sum_points"]) < 1e-9
    return {"known_fills_and_exits_checked": len(t), "sessions_checked": len(days),
            "source_corpus_id": saved["corpus_id"]}


def main():
    a, manifest = s20.load_market("NQ")
    out = ROOT / "setups/S-20"
    nq = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert nq["corpus_id"] == manifest["corpus_id"]
    nq_check = check(a, pd.read_csv(out / "daily.csv"), nq, 1.0)
    es_a, es_manifest = s20.load_market("ES")
    es = json.loads((out / "result_ES.json").read_text(encoding="utf-8"))
    assert es["corpus_id"] == es_manifest["corpus_id"]
    es_check = check(es_a, pd.read_csv(out / "daily_ES.csv"), es, 0.4)
    report = {"status": "pass", "NQ": nq_check, "ES": es_check,
              "scope": "known fills, earliest exit predicate, accounting; not predictive validity or missing outcomes"}
    (ROOT / "setups/S-20/audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
