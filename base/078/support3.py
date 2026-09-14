"""078A, third pass — section 7 of the task, and which coordinate is imbalanced.

Same frozen operator, two sub-pools of B instead of one:
  NO-RIZ        comparison minutes with no tf54 RIZ alive;
  RIZ-PRESENT   comparison minutes with at least one alive.

Answers: does the overlap depend on whether T0 is compared with ordinary tape or
with RIZ-organised stretches? No new representation, lookback or metric.

python -B base/078/support3.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from represent import BASE_SPECS, Tape, coords, sigma_sd
from support import balanced, knn_against, knn_self, K, REPS, SEED, INSTRUMENT, TF

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "work" / "078a"
RNG = np.random.default_rng(SEED + 2)
COORD_NAMES = ["clock_cos", "clock_sin", "epoch_year", "log_sigma"]


def main() -> None:
    tape = Tape(INSTRUMENT)
    elig = tape.eligible()
    mod, yr = tape.clock()
    sid = tape.session_id

    cell = ROOT / "data" / "field" / INSTRUMENT / f"cells/tf_{TF:04d}"
    pp = pq.read_table(cell / "passports.parquet")
    t0 = np.asarray(pp["t0_spine_pos"]).astype(np.int64)
    a_min = np.unique(t0)
    is_t0 = np.zeros(tape.n, dtype=bool)
    is_t0[a_min] = True
    A = a_min[elig[a_min]]
    B = np.flatnonzero(elig & ~is_t0)

    start = np.asarray(pp["t0_spine_pos"]).astype(np.int64)
    dele = pp["c1_deletion_spine_pos"].fill_null(-1).to_numpy(
        zero_copy_only=False).astype(np.int64)
    last = np.asarray(pp["last_observed_spine_pos"]).astype(np.int64)
    end = np.where(dele >= 0, dele, last + 1).astype(np.int64)
    delta = np.zeros(tape.n + 1, dtype=np.int32)
    np.add.at(delta, start, 1)
    np.add.at(delta, np.minimum(end, tape.n), -1)
    alive = np.cumsum(delta)[:tape.n]
    first_t0 = int(start.min())
    defined = np.arange(tape.n) >= first_t0

    pools = {
        "NO-RIZ": B[(alive[B] == 0) & defined[B]],
        "RIZ-PRESENT": B[(alive[B] > 0) & defined[B]],
    }
    rep: dict[str, object] = {
        "pool_sizes": {k: int(v.size) for k, v in pools.items()},
        "pool_session_days": {k: int(np.unique(sid[v]).size) for k, v in pools.items()},
        "A_size": int(A.size),
    }
    # how the two sub-pools differ in composition, before any distance is taken
    rep["pool_composition"] = {
        k: {"share_rth_0930_1600": float(np.mean((mod[v] >= 570) & (mod[v] < 960))),
            "median_et_hour": float(np.median(mod[v] // 60)),
            "by_epoch": {e: float(np.mean((yr[v] >= a) & (yr[v] <= b)))
                         for e, (a, b) in (("2006-2010", (2006, 2010)),
                                           ("2011-2015", (2011, 2015)),
                                           ("2016-2020", (2016, 2020)),
                                           ("2021-2026", (2021, 2026)))}}
        for k, v in pools.items()}
    rep["A_composition"] = {
        "share_rth_0930_1600": float(np.mean((mod[A] >= 570) & (mod[A] < 960))),
        "median_et_hour": float(np.median(mod[A] // 60))}

    sd_pool = B[RNG.choice(B.size, 400_000, replace=False)]
    out = {}
    for spec in BASE_SPECS:
        s_sd = sigma_sd(tape, spec, sd_pool)
        XA = coords(tape, A, spec, s_sd)
        fin = np.isfinite(XA).all(1)
        XA, Ae, sidA = XA[fin], A[fin], sid[A[fin]]
        dAA, _ = knn_self(XA, sidA, Ae, K)
        den = dAA[:, 0] > 0

        res = {}
        for label, pool in pools.items():
          for mode in ("C2a-uniform", "C2b-composition"):
            # C2b is the frozen composition-matched control of the freeze,
            # applied inside the sub-pool. The NO-RIZ pool is 93% RTH against
            # 68% for A, so without it the split measures clock composition.
            if mode == "C2b-composition":
                key = yr.astype(np.int64) * 10000 + mod.astype(np.int64)
                o = np.argsort(key[pool], kind="stable")
                Ps, ks = pool[o], key[pool[o]]
                lo = np.searchsorted(ks, key[Ae], side="left")
                hi = np.searchsorted(ks, key[Ae], side="right")
                cnd = [Ps[a:b][sid[Ps[a:b]] != sid[Ae[i]]]
                       for i, (a, b) in enumerate(zip(lo, hi))]
                matched = np.array([len(c) > 0 for c in cnd])
            rng = np.random.default_rng(SEED + 3)
            R1, pur, acc, auc = [], [], [], []
            for _ in range(REPS):
                if mode == "C2b-composition":
                    if not matched.any():
                        break
                    sel = np.array([c[rng.integers(len(c))] for c in cnd if len(c)],
                                   dtype=np.int64)
                else:
                    sel = rng.choice(pool, size=Ae.size, replace=False)
                XB = coords(tape, sel, spec, s_sd)
                g = np.isfinite(XB).all(1)
                XB, posB = XB[g], sel[g]
                d, _i = knn_against(XA, sidA, XB, sid[posB], posB, K)
                R1.append((d[:, 0] / dAA[:, 0])[den])
                p, a_, u = balanced(XA, sidA, XB, sid[posB], K)
                pur.append(p); acc.append(a_); auc.append(u)
            if not R1:
                res[f"{label} | {mode}"] = {"note": "no matched control exists"}
                continue
            r = np.concatenate(R1)
            r = r[np.isfinite(r)]
            res[f"{label} | {mode}"] = {
                "matched_share": (float(matched.mean())
                                  if mode == "C2b-composition" else 1.0),
                "R_k1": {"p25": float(np.percentile(r, 25)),
                         "median": float(np.median(r)),
                         "p75": float(np.percentile(r, 75)),
                         "share_below_1": float(np.mean(r < 1))},
                "knn_purity": float(np.median(pur)),
                "balanced_1nn_accuracy": float(np.median(acc)),
                "auc": float(np.median([x for x in auc if x is not None])),
            }

        # which coordinate carries the residual imbalance
        selb = RNG.choice(B, size=Ae.size, replace=False)
        XB = coords(tape, selb, spec, s_sd)
        XB = XB[np.isfinite(XB).all(1)]
        smd = (XA.mean(0) - XB.mean(0)) / np.sqrt((XA.var(0) + XB.var(0)) / 2 + 1e-12)
        top = np.argsort(-np.abs(smd))[:4]
        res["worst_imbalanced_coordinates"] = [
            {"coordinate": COORD_NAMES[i] if i < 4 else
             f"shape[min {i - 4 >> 2}, {'OHLC'[(i - 4) % 4]}]",
             "smd": float(smd[i])} for i in top]
        out[spec.name] = res
        print(f"  done {spec.name}", flush=True)

    rep["per_representation"] = out
    (OUT / "support3.json").write_text(
        json.dumps(rep, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(f"\nwritten: {OUT / 'support3.json'}")


if __name__ == "__main__":
    main()
