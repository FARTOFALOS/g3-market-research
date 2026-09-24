"""Frozen CW1 cell ("overnight survived the morning", 30 min) on a window not used in discovery — see FREEZE_CW1.md.

    python -B base/105/cw1_frozen_check.py --self-test          # reproduces CW1 numbers on 2025-07..12 (already seen)
    python -B base/105/cw1_frozen_check.py --window 2026        # ONLY with the trader's word: opens NQ 2026-01-02..07-10

The statistic is fixed: for the class "survived" (same definitions as cw1_cross_window.build), raw
r30 = side * (close 12:30 - open of the 12:00 bar), points: n, mean, median, share > 0, and the same for the two
other tested classes as contrast. No cells, no permutations, no other horizons.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import cw1_cross_window as CW  # noqa: E402


def stats(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) == 0: return dict(n=0)
    se = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan")
    return dict(n=int(len(x)), mean=float(x.mean()), median=float(np.median(x)), share_pos=float((x > 0).mean()),
                se=float(se), t=float(x.mean() / se) if se and se > 0 else None)


def verdict(s):
    if s["n"] == 0: return "no days"
    if s["mean"] > 0 and s["median"] > 0: return "same sign as 2020-25"
    if s["mean"] < 0 and s["median"] < 0: return "opposite sign"
    return "undecided"


def main():
    a = argparse.ArgumentParser(); g = a.add_mutually_exclusive_group(required=True)
    g.add_argument("--self-test", action="store_true"); g.add_argument("--window", choices=["2026"])
    x = a.parse_args()
    if x.self_test:
        P, _ = CW.build("NQ", d_lo=20250701, d_hi=20251231)
        ref = pd.read_csv(HERE / "cw1_out/days.csv")
        ref = ref[(ref.date >= 20250701) & (ref.date <= 20251231)].set_index("date")
        mine = P[P.cls.isin(CW.TESTED)].set_index("date")
        same = mine.index.equals(ref.index) and (mine.cls == ref.cls).all() and np.allclose(mine.r30, ref.r30, equal_nan=True)
        print("self-test on 2025-07..12 reproduces CW1 rows:", bool(same), "| survived days", int((mine.cls == "survived").sum()))
        if not same: raise SystemExit("self-test failed")
        return
    cal = CW.J.S17.history_calendar()
    cal = pd.concat([cal[cal.date < CW.J.S17.F_FROM], CW.J.S17.forward_calendar()], ignore_index=True)
    P, short = CW.build("NQ", cal=cal, root=ROOT / "data/forward/market", d_lo=20260101, d_hi=20260710)
    out = dict(freeze="base/105/FREEZE_CW1.md", window=[20260102, 20260710], short_sessions_skipped=short,
               population=len(P), classes=P.cls.value_counts().to_dict(),
               survived_r30=stats(P.loc[P.cls == "survived", "r30"]),
               aligned_r30=stats(P.loc[P.cls == "aligned", "r30"]), cancelled_r30=stats(P.loc[P.cls == "cancelled", "r30"]))
    out["verdict"] = verdict(out["survived_r30"])
    (HERE / "cw1_out/frozen_2026.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
