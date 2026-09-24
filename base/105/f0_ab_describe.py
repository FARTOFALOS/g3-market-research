"""Descriptive tail/downstream anatomy of the saved paired replay, no reader.

Run from repository root: python -B base/105/f0_ab_describe.py
Only the existing CSV is read; no trade rule, sample eligibility or target changes.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def describe(g):
    d = g.D.to_numpy(float)
    morning = g.morning_A.to_numpy(float)
    downstream = g.downstream_A.to_numpy(float) - g.downstream_B.to_numpy(float)
    assert np.allclose(d, g.RA - g.RB)
    assert np.allclose(d, morning + downstream)
    positive = d[d > 0]
    negative = d[d < 0]
    at_stop = d == -19.75
    h = np.maximum(g.RA, g.RB).mean() - max(g.RA.mean(), g.RB.mean())
    neg_mass = np.maximum(-d, 0).mean()
    pos_mass = np.maximum(d, 0).mean()
    assert np.isclose(h, min(pos_mass, neg_mass))
    assert np.isclose(h, neg_mass + min(d.mean(), 0))
    return dict(
        pairs=len(g), mean_A=float(g.RA.mean()), mean_B=float(g.RB.mean()),
        mean_D=float(d.mean()), H=float(h),
        positive_days=len(positive), negative_days=len(negative),
        positive_share=float((d > 0).mean()), negative_share=float((d < 0).mean()),
        mean_positive_D=float(positive.mean()) if len(positive) else None,
        mean_negative_D=float(negative.mean()) if len(negative) else None,
        days_D_exactly_minus_19_75=int(at_stop.sum()),
        share_D_exactly_minus_19_75=float(at_stop.mean()),
        mean_positive_mass=pos_mass, mean_negative_mass=neg_mass,
        mean_negative_mass_at_minus_19_75=float(19.75 * at_stop.mean()),
        share_negative_mass_at_minus_19_75=float(19.75 * at_stop.sum() / -negative.sum()) if len(negative) else None,
        mean_correction_when_A_is_worse_constant=float(min(d.mean(), 0)),
        corr_D_morning=float(np.corrcoef(d, morning)[0, 1]),
        days_D_exactly_morning=int((d == morning).sum()),
        downstream_difference_days=int((downstream != 0).sum()),
        downstream_difference_share=float((downstream != 0).mean()),
        realized_sign_difference_days=int((np.sign(d) != np.sign(morning)).sum()),
        strict_realized_sign_flip_days=int((d * morning < 0).sum()),
        H_morning_only=float(np.maximum(morning, 0).mean() - max(morning.mean(), 0)),
        mean_morning=float(morning.mean()), mean_downstream_difference=float(downstream.mean()),
        downstream_reduction_of_mean_morning=float(-downstream.mean() / morning.mean()) if morning.mean() else None,
    )


if __name__ == "__main__":
    source = HERE / "f0_ab_probe_out/f0_ab_days.csv"
    days = pd.read_csv(source)
    paired = days[(days.p0_status == "yes") & (days.A_status == "resolved") & (days.B_status == "resolved")]
    modern = paired[(paired.date >= 20200101) & (paired.date <= 20251231)]
    result = dict(
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        scope="P_AB only; descriptive outcome anatomy, no prefix selection or model fitting",
        modern=describe(modern),
        by_year={str(y): describe(g) for y, g in modern.groupby(modern.date // 10000)},
        caveats=["Stable H alone does not establish predictable state variation.",
                 "Near-equivalence of realized outcomes does not imply equivalence of conditional action values.",
                 "The primary target remains full D, including downstream consequences."],
    )
    target = source.with_name("tail_anatomy.json")
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
