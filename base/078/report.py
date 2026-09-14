"""Render work/078a/support.json into the tables the card and the architects need.

python -B base/078/report.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
J = json.loads((ROOT / "work" / "078a" / "support.json").read_text(encoding="utf-8"))
G = json.loads((ROOT / "work" / "078a" / "genealogy.json").read_text(encoding="utf-8"))
BASE = ["G0", "G1", "G2"]


def f(x, n=3):
    return "—" if x is None else f"{x:.{n}f}"


print("=" * 78)
print("POPULATIONS")
print("=" * 78)
a, b = J["population_A"], J["population_B"]
print(f"A: {a['passport_rows']} passport rows -> {a['distinct_t0_minutes']} distinct "
      f"T0 minutes -> {a['eligible']} eligible "
      f"({a['dropped_not_eligible']} dropped), "
      f"{a['session_days']} session days, {a['minutes_carrying_more_than_one_riz']} "
      f"minutes carry >1 RIZ")
print(f"B: {b['eligible_non_t0_minutes']} eligible non-T0 minutes, "
      f"{b['session_days']} session days, B/A = {b['ratio_B_over_A']:.0f}x")
c = J["control_C2b_matching"]
print(f"C2b: {c['A_with_a_matched_control']} of {a['eligible']} A have a matched "
      f"control ({c['share_matched']:.3f}); candidates per A median "
      f"{c['candidates_per_A']['median']:.0f}, min {c['candidates_per_A']['min']}")
print("\nRIZ context of comparison minutes (tf54 scope):")
for k, v in J["comparison_minute_riz_context"]["share"].items():
    print(f"   {k:<14} {v:.4f}")

print()
print("=" * 78)
print("PRIMARY DIAGNOSTIC  R = d_k(A->B) / d_k(A->A), cross-day, same k")
print("=" * 78)
print(f"{'G':<20} {'pool':<22} {'k':>2} {'p25':>7} {'median':>7} {'p75':>7} "
      f"{'R<1':>7}")
for name in BASE:
    r = J["representations"][name]
    for tag, lab in (("C1_availability_full_pool", "full pool (avail.)"),
                     ("C2a_uniform", "C2a size-matched"),
                     ("C2b_composition", "C2b comp.-matched")):
        if tag not in r:
            continue
        for k in (1, 10):
            d = r[tag]["R_ratio"][f"k{k}"]
            if tag == "C1_availability_full_pool":
                print(f"{name:<20} {lab:<22} {k:>2} {f(d['p25'])!s:>7} "
                      f"{f(d['p50'])!s:>7} {f(d['p75'])!s:>7} "
                      f"{f(d['share_below_1'])!s:>7}")
            else:
                print(f"{name:<20} {lab:<22} {k:>2} "
                      f"{f(d['p25']['median'])!s:>7} {f(d['p50']['median'])!s:>7} "
                      f"{f(d['p75']['median'])!s:>7} "
                      f"{f(d['share_below_1']['median'])!s:>7}")

print()
print("=" * 78)
print("STRUCTURAL OVERLAP AND RECOVERABILITY (balanced pools, 20 reps)")
print("0.5 purity = full overlap;  0.5 accuracy / AUC = T0 not recoverable")
print("=" * 78)
print(f"{'G':<20} {'pool':<20} {'kNN purity':>22} {'1NN acc':>16} {'AUC':>16}")
for name in BASE + [s for s in J["representations"] if s not in BASE]:
    r = J["representations"][name]
    for tag, lab in (("C2a_uniform", "C2a size"), ("C2b_composition", "C2b comp")):
        p = r[tag]["C4_structural_overlap_knn_purity"]
        ac = r[tag]["balanced_1nn_accuracy"]
        au = r[tag]["auc_recoverability"]
        print(f"{name:<20} {lab:<20} "
              f"{f(p['median'])} [{f(p['p5'],3)},{f(p['p95'],3)}]".rjust(22)
              + f"{f(ac['median'])}".rjust(16) + f"{f(au['median'])}".rjust(16))

print()
print("=" * 78)
print("SECONDARY: coverage at tau = quantiles of the A->A cross-day NN distance")
print("=" * 78)
for name in BASE:
    r = J["representations"][name]
    print(f"\n{name}   dims={r['dims']}  A used={r['A_used']}")
    hdr = f"  {'tau':<10}" + "".join(f"{q:>26}" for q in
                                     ["full pool >=1 / >=5", "C2a >=1 / >=5",
                                      "C2b >=1 / >=5"])
    print(hdr)
    for q in (10, 25, 50, 75, 90):
        key = f"tau_p{q}"
        cells = []
        for tag in ("C1_availability_full_pool", "C2a_uniform", "C2b_composition"):
            if tag in r:
                cv = r[tag]["coverage_at_A_to_A_quantiles"][key]
                cells.append(f"{cv['share_ge1']:.3f} / {cv['share_ge5']:.3f}")
            else:
                cells.append("—")
        tau = r["C2a_uniform"]["coverage_at_A_to_A_quantiles"][key]["tau"]
        print(f"  p{q:<3} {tau:7.4f}" + "".join(f"{c:>26}" for c in cells))

print()
print("=" * 78)
print("RESIDUAL IMBALANCE AND EXTRAPOLATION (vs size-matched control)")
print("=" * 78)
for name in BASE:
    r = J["representations"][name]
    cb = r["coordinate_balance"]
    print(f"{name:<6} extrapolation share {r['extrapolation_share_vs_size_matched']:.4f}"
          f"   max|SMD| {cb['max_abs_smd']:.3f}"
          f"   median|SMD| {cb['median_abs_smd']:.3f}"
          f"   coords with |SMD|>0.25: {cb['share_coords_abs_smd_over_0p25']:.3f}")

print()
print("=" * 78)
print("HOW MUCH OF THE T0 GENEALOGY EACH WINDOW CAN SEE (measured separately)")
print("=" * 78)
v = G["visible_to_a_generic_window"]
print(f"  full generating price history inside  5 min: "
      f"{v['share_full_generating_price_history_inside_5']:.3f}")
print(f"  full generating price history inside 30 min: "
      f"{v['share_full_generating_price_history_inside_30']:.3f}")
print(f"  observed part of the T0 native bar inside  5 min: "
      f"{v['share_t0_native_bar_observed_part_inside_5']:.3f}")
print(f"  observed part of the T0 native bar inside 30 min: "
      f"{v['share_t0_native_bar_observed_part_inside_30']:.3f}")
print(f"  median depth of the generating history: "
      f"{G['depth_summary']['median_depth_minutes']:.0f} minutes")
