"""078A — sensitivity R2, the exclusion window the freeze declared.

The main run puts no exclusion window between A and B, as section 6 of the task
requires. R2 asks what happens when comparison minutes within +-W of ANY tf54 T0
are removed: W in {30, 1440} minutes. Same frozen operator, smaller B.

python -B base/078/support4.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from represent import BASE_SPECS, Tape, coords, sigma_sd
from support import balanced, knn_against, knn_self, K, REPS, SEED, INSTRUMENT, TF

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "work" / "078a"
RNG = np.random.default_rng(SEED + 4)


def main() -> None:
    tape = Tape(INSTRUMENT)
    elig = tape.eligible()
    mod, yr = tape.clock()
    sid = tape.session_id

    cell = ROOT / "data" / "field" / INSTRUMENT / f"cells/tf_{TF:04d}"
    pp = pq.read_table(cell / "passports.parquet")
    a_min = np.unique(np.asarray(pp["t0_spine_pos"]).astype(np.int64))
    is_t0 = np.zeros(tape.n, dtype=bool)
    is_t0[a_min] = True
    A = a_min[elig[a_min]]

    near = {}
    for W in (30, 1440):
        flag = np.zeros(tape.n, dtype=bool)
        for p in a_min:
            flag[max(0, p - W):min(tape.n, p + W + 1)] = True
        near[W] = flag

    pools = {"W=0 (frozen primary)": np.flatnonzero(elig & ~is_t0)}
    for W in (30, 1440):
        pools[f"W={W}"] = np.flatnonzero(elig & ~near[W])

    rep = {"pool_sizes": {k: int(v.size) for k, v in pools.items()},
           "A_size": int(A.size)}

    sd_pool = pools["W=0 (frozen primary)"][
        RNG.choice(pools["W=0 (frozen primary)"].size, 400_000, replace=False)]
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
            rng = np.random.default_rng(SEED + 5)
            R1, pur, acc, auc = [], [], [], []
            for _ in range(REPS):
                sel = rng.choice(pool, size=Ae.size, replace=False)
                XB = coords(tape, sel, spec, s_sd)
                g = np.isfinite(XB).all(1)
                XB, posB = XB[g], sel[g]
                d, _i = knn_against(XA, sidA, XB, sid[posB], posB, K)
                R1.append((d[:, 0] / dAA[:, 0])[den])
                p, a_, u = balanced(XA, sidA, XB, sid[posB], K)
                pur.append(p); acc.append(a_); auc.append(u)
            r = np.concatenate(R1)
            r = r[np.isfinite(r)]
            res[label] = {
                "R_k1_median": float(np.median(r)),
                "R_share_below_1": float(np.mean(r < 1)),
                "knn_purity": float(np.median(pur)),
                "balanced_1nn_accuracy": float(np.median(acc)),
                "auc": float(np.median([x for x in auc if x is not None])),
            }
        out[spec.name] = res
        print(f"  done {spec.name}", flush=True)

    rep["per_representation"] = out
    (OUT / "support4.json").write_text(
        json.dumps(rep, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(f"\nwritten: {OUT / 'support4.json'}")


if __name__ == "__main__":
    main()
