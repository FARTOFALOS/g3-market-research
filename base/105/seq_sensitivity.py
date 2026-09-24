"""RD1 resolution of the evaluation protocol before any reader is fitted.

Run from repository root: python -B base/105/seq_sensitivity.py
Uses only the saved paired A/B day table. "Noisy oracle" selectors:
  point 1: take morning iff D + sigma*z > 0; day = RA if taken else RB; compared with A and B.
  point 2: in the B world skip the PV2 branch iff RB + sigma*z < 0; day = RB if taken else 0; compared with B.
Detection = month-block t >= 2 on the paired daily difference. Day-level
approximation (no occupancy/budget knock-on), one family of selectors.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "seq_reader_out"
SIGMAS = [0, 5, 10, 20, 30, 50, 75, 100, 150, 250, 500]
DRAWS, SEED = 400, 20260924


def month_t(diff, months):
    n = len(diff)
    s = pd.Series(diff).groupby(months).sum().to_numpy()
    k = len(s)
    se = np.sqrt(k / (k - 1) * ((s - s.mean()) ** 2).sum()) / n
    return diff.mean() / se if se > 0 else 0.0, se


def curve(g, rng):
    ra, rb = g.RA.to_numpy(float), g.RB.to_numpy(float)
    d, months = ra - rb, (g.date // 100).to_numpy()
    _, se_a = month_t(rb - ra, months)
    out = dict(days=len(g), resolution_2se=dict(point1_vs_A=None, point2_vs_B=None), point1=[], point2=[])
    p1_se, p2_se = [], []
    for s in SIGMAS:
        rows1, rows2 = [], []
        for _ in range(DRAWS):
            z = rng.standard_normal(len(g))
            take = d + s * z > 0
            v = np.where(take, ra, rb)
            ta, sa = month_t(v - ra, months)
            tb, _ = month_t(v - rb, months)
            rows1.append(((v - ra).mean(), (v - rb).mean(), ta >= 2, tb >= 2, take.mean()))
            p1_se.append(sa)
            skip = rb + s * rng.standard_normal(len(g)) < 0
            w = np.where(skip, 0.0, rb)
            t2, s2 = month_t(w - rb, months)
            rows2.append(((w - rb).mean(), w.mean(), t2 >= 2, skip.mean()))
            p2_se.append(s2)
        r1, r2 = np.array(rows1, float), np.array(rows2, float)
        out["point1"].append(dict(sigma=s, lift_vs_A=r1[:, 0].mean(), lift_vs_B=r1[:, 1].mean(),
                                  p_detect_vs_A=r1[:, 2].mean(), p_detect_vs_B=r1[:, 3].mean(), take_share=r1[:, 4].mean()))
        out["point2"].append(dict(sigma=s, lift_vs_B=r2[:, 0].mean(), value=r2[:, 1].mean(),
                                  p_detect_vs_B=r2[:, 2].mean(), skip_share=r2[:, 3].mean()))
    out["resolution_2se"]["point1_vs_A"] = 2 * float(np.median(p1_se))
    out["resolution_2se"]["point2_vs_B"] = 2 * float(np.median(p2_se))
    return out


def main():
    days = pd.read_csv(HERE / "f0_ab_probe_out/f0_ab_days.csv")
    p = days[(days.p0_status == "yes") & (days.A_status == "resolved") & (days.B_status == "resolved")]
    rng = np.random.default_rng(SEED)
    res = dict(protocol=__doc__.strip().splitlines()[3:8], draws=DRAWS, seed=SEED,
               t2020_2025=curve(p[p.date >= 20200101], rng),
               t2013_2019=curve(p[(p.date >= 20130101) & (p.date < 20200101)], rng))
    OUT.mkdir(exist_ok=True)
    (OUT / "sensitivity.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    for terr in ("t2020_2025", "t2013_2019"):
        r = res[terr]
        print(terr, "days", r["days"], "2se:", {k: round(v, 2) for k, v in r["resolution_2se"].items()})
        print("  point1  sigma  lift_vs_A  P(det vs A)  lift_vs_B  P(det vs B)  take")
        for x in r["point1"]:
            print(f"  {x['sigma']:>12} {x['lift_vs_A']:10.2f} {x['p_detect_vs_A']:12.2f} {x['lift_vs_B']:10.2f} {x['p_detect_vs_B']:12.2f} {x['take_share']:5.2f}")
        print("  point2  sigma  lift_vs_B  P(det vs B)  skip")
        for x in r["point2"]:
            print(f"  {x['sigma']:>12} {x['lift_vs_B']:10.2f} {x['p_detect_vs_B']:12.2f} {x['skip_share']:5.2f}")


if __name__ == "__main__":
    main()
