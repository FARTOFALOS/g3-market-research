"""079A second pass — the §7 items the first pass reported only through medians,
plus the attrition profile.

Four things, no new class, no new metric, no future:
  1. orientation of the current interaction per class (a median cannot see it:
     the distribution is two-moded, up-body and down-body);
  2. how much of the zone the body actually passes, per class;
  3. what the 1046 -> 587 attrition removes, and whether it is selective;
  4. residual imbalance across the 124 M coordinates.

python -B base/079/contrast2.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "base" / "078"))
sys.path.insert(0, str(ROOT / "research"))
from represent import G2, Tape, coords, sigma_sd          # noqa: E402
from decompose import Bars                                # noqa: E402
from contrast import build_instances, geometry, smd, GEOM, SEED, TF  # noqa: E402
from calendar_utils import year_of                        # noqa: E402

OUT = ROOT / "work" / "079a"
RNG = np.random.default_rng(SEED + 1)


def main() -> None:
    bars, rows = build_instances()
    tape = Tape("NQ")
    elig = tape.eligible()
    mod, yr = tape.clock()

    p = np.array([r["p"] for r in rows])
    cls = np.array([r["cls"] for r in rows])
    obj = np.array([r["riz_id"] for r in rows])
    ok = (p >= 0) & (p < tape.n)
    ok &= np.where(ok, elig[np.where(ok, p, 0)], False)

    rep = {}

    # ---- 3. attrition ------------------------------------------------
    good = set(obj[ok & (cls == 0)]) & set(obj[ok & (cls == 1)])
    keep = ok & np.isin(obj, list(good))
    lost_h0 = ~ok & (cls == 0)
    lost_h1 = ~ok & (cls == 1)

    def prof(mask):
        q = p[mask]
        q = q[(q >= 0) & (q < tape.n)]
        if q.size == 0:
            return {"n": int(mask.sum()), "no_address": int(mask.sum())}
        return {"n": int(mask.sum()),
                "by_et_hour": {str(int(h)): int(c) for h, c in
                               zip(*np.unique(mod[q] // 60, return_counts=True))},
                "by_year": {str(int(v)): int(c) for v, c in
                            zip(*np.unique(yr[q], return_counts=True))}}

    kept_h0 = keep & (cls == 0)
    rep["attrition"] = {
        "why": ("a relation-instance needs the 078A eligibility of its own minute "
                "(151 contiguous finite minutes ending at p) AND its partner event "
                "must survive too, so the design stays paired"),
        "objects_total": 1046,
        "H0_eligible": int((ok & (cls == 0)).sum()),
        "H1_eligible": int((ok & (cls == 1)).sum()),
        "objects_paired": len(good),
        "share_objects_kept": float(len(good) / 1046),
        "lost_H0": prof(lost_h0), "lost_H1": prof(lost_h1),
        "kept_H0": prof(kept_h0),
    }
    hk = rep["attrition"]["kept_H0"].get("by_et_hour", {})
    hl = rep["attrition"]["lost_H0"].get("by_et_hour", {})
    rep["attrition"]["H0_drop_share_by_et_hour"] = {
        str(h): round(hl.get(str(h), 0) / max(1, hl.get(str(h), 0) + hk.get(str(h), 0)), 3)
        for h in range(24) if hl.get(str(h), 0) + hk.get(str(h), 0) > 0}

    # ---- coordinates on the paired population ------------------------
    rows_k = [r for r, f in zip(rows, keep) if f]
    pk, ck, ok_obj = p[keep], cls[keep], obj[keep]
    sig = tape.sigma_at(pk, G2.ruler, G2.shift)
    R = geometry(bars, rows_k, sig)
    s_sd = sigma_sd(tape, G2, np.flatnonzero(elig)[
        RNG.choice(int(elig.sum()), 400_000, replace=False)])
    M = coords(tape, pk, G2, s_sd)
    fin = np.isfinite(M).all(1) & np.isfinite(R).all(1)
    surv = set(ok_obj[fin & (ck == 0)]) & set(ok_obj[fin & (ck == 1)])
    fin &= np.isin(ok_obj, list(surv))
    M, R, pk, ck, sig = M[fin], R[fin], pk[fin], ck[fin], sig[fin]
    rows_k = [r for r, f in zip(rows_k, fin) if f]

    # ---- 1. orientation ----------------------------------------------
    down = R[:, 0] < R[:, 1]                     # close below bar open
    rep["orientation_of_current_interaction"] = {
        "definition": "body points down when close[p] < open of the developing native bar",
        "H0_share_down": float(down[ck == 0].mean()),
        "H1_share_down": float(down[ck == 1].mean()),
        "difference": float(down[ck == 0].mean() - down[ck == 1].mean()),
        "note": ("a median of close_in_zone cannot see this: the distribution is "
                 "two-moded and no side-flip is applied, by the 078A rule"),
    }
    for lab, m in (("down-body", down), ("up-body", ~down)):
        sub = {}
        for i, n in enumerate(GEOM):
            a, b = R[m & (ck == 0), i], R[m & (ck == 1), i]
            if a.size and b.size:
                sub[n] = {"H0_median": float(np.median(a)),
                          "H1_median": float(np.median(b)), "smd": smd(a, b)}
        rep["orientation_of_current_interaction"][f"within_{lab}"] = sub

    # ---- 2. how much of the zone the body passes ----------------------
    over_far = np.maximum(R[:, 0], R[:, 1]) - 1.0        # beyond the top, in widths
    over_near = -np.minimum(R[:, 0], R[:, 1])            # beyond the bottom
    rep["span_depth"] = {
        "beyond_top_in_zone_widths": {
            "H0_median": float(np.median(over_far[ck == 0])),
            "H1_median": float(np.median(over_far[ck == 1])),
            "smd": smd(over_far[ck == 0], over_far[ck == 1])},
        "beyond_bottom_in_zone_widths": {
            "H0_median": float(np.median(over_near[ck == 0])),
            "H1_median": float(np.median(over_near[ck == 1])),
            "smd": smd(over_near[ck == 0], over_near[ck == 1])},
    }

    # ---- 4. residual imbalance across the M block ---------------------
    names = ["clock_cos", "clock_sin", "epoch_year", "log_sigma"] + [
        f"shape[min{j - 29}, {'OHLC'[i]}]" for j in range(30) for i in range(4)]
    sm = np.array([smd(M[ck == 0, i], M[ck == 1, i]) for i in range(M.shape[1])])
    top = np.argsort(-np.abs(sm))[:6]
    rep["M_block_residual_imbalance"] = {
        "max_abs_smd": float(np.max(np.abs(sm))),
        "median_abs_smd": float(np.median(np.abs(sm))),
        "share_coords_over_0p25": float(np.mean(np.abs(sm) > 0.25)),
        "worst": [{"coordinate": names[i], "smd": float(sm[i])} for i in top],
    }

    (OUT / "contrast2.json").write_text(
        json.dumps(rep, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep.items() if k != "attrition"},
                     indent=2, ensure_ascii=False, default=float))
    a = rep["attrition"]
    print("\nattrition: 1046 objects -> H0 eligible %d, H1 eligible %d, paired %d (%.3f)"
          % (a["H0_eligible"], a["H1_eligible"], a["objects_paired"],
             a["share_objects_kept"]))
    print("H0 drop share by ET hour:", a["H0_drop_share_by_et_hour"])
    print(f"\nwritten: {OUT / 'contrast2.json'}")


if __name__ == "__main__":
    main()
