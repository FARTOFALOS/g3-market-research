"""078A — the support audit itself.

Executes base/078/FREEZE_078A.md and nothing else. Y is never defined, computed
or approached; no minute later than p enters any coordinate (checked by
selftest.py). Volume is never loaded. The field is read-only.

python -B base/078/support.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from zlib import crc32

from represent import (BASE_SPECS, SENSITIVITIES, PAD, Spec, Tape, coords,
                       sigma_sd)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research"))
from calendar_utils import date_key  # noqa: E402

INSTRUMENT, TF = "NQ", 54
K = 10                      # frozen
REPS = 20                   # frozen
SEED = 20780914
OUT = ROOT / "work" / "078a"
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(SEED)
INF = np.float32(np.inf)


# ------------------------------------------------------------------ helpers
def qs(a, ps=(5, 25, 50, 75, 95)):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {f"p{p}": None for p in ps}
    return {f"p{p}": float(np.percentile(a, p)) for p in ps}


def knn_against(XA, sidA, XB, sidB, posB, k=K):
    """k nearest CROSS-DAY neighbours of every row of XA inside XB.

    Returns (distances (nA,k), source positions (nA,k)). Same-session-day
    neighbours are excluded before ranking, never after.
    """
    nA = XA.shape[0]
    best = np.full((nA, k), INF, dtype=np.float32)
    bidx = np.full((nA, k), -1, dtype=np.int64)
    aa = (XA.astype(np.float32) ** 2).sum(1)
    step = max(1, int(2.0e7 // max(nA, 1)))
    for s in range(0, XB.shape[0], step):
        e = min(s + step, XB.shape[0])
        xb = XB[s:e]
        D = aa[:, None] + (xb ** 2).sum(1)[None, :] - 2.0 * (XA @ xb.T)
        np.maximum(D, 0.0, out=D)
        D[sidA[:, None] == sidB[s:e][None, :]] = INF
        m = min(k, D.shape[1])
        part = np.argpartition(D, m - 1, axis=1)[:, :m]
        cd = np.take_along_axis(D, part, 1)
        ci = posB[s:e][part]
        allD = np.concatenate([best, cd], 1)
        allI = np.concatenate([bidx, ci], 1)
        o = np.argsort(allD, axis=1, kind="stable")[:, :k]
        best = np.take_along_axis(allD, o, 1)
        bidx = np.take_along_axis(allI, o, 1)
    return np.sqrt(best), bidx


def knn_self(XA, sidA, posA, k=K):
    """k nearest CROSS-DAY neighbours of A inside A itself (self excluded)."""
    aa = (XA ** 2).sum(1)
    D = aa[:, None] + aa[None, :] - 2.0 * (XA @ XA.T)
    np.maximum(D, 0.0, out=D)
    D[sidA[:, None] == sidA[None, :]] = INF       # also removes the self pair
    k = min(k, D.shape[1] - 1)
    part = np.argpartition(D, k - 1, axis=1)[:, :k]
    cd = np.take_along_axis(D, part, 1)
    o = np.argsort(cd, axis=1, kind="stable")
    return np.sqrt(np.take_along_axis(cd, o, 1)), posA[np.take_along_axis(part, o, 1)]


def ratio_stats(cross, within, k):
    """R_i(k) = d_k(A->B) / d_k(A->A); the frozen primary diagnostic."""
    a, b = cross[:, k - 1], within[:, k - 1]
    ok = np.isfinite(a) & np.isfinite(b) & (b > 0)
    r = a[ok] / b[ok]
    out = qs(r)
    out["n"] = int(ok.sum())
    out["share_below_1"] = float(np.mean(r < 1.0)) if r.size else None
    out["mean_log10"] = float(np.mean(np.log10(r))) if r.size else None
    return out


# ------------------------------------------------------------------- main
def main() -> None:
    rep: dict[str, object] = {"instrument": INSTRUMENT, "tf_minutes": TF,
                              "seed": SEED, "k": K, "reps": REPS, "pad": PAD}
    tape = Tape(INSTRUMENT)
    elig = tape.eligible()
    mod, yr = tape.clock()
    sid = tape.session_id

    # ---- population A ---------------------------------------------------
    cell = ROOT / "data" / "field" / INSTRUMENT / f"cells/tf_{TF:04d}"
    pp = pq.read_table(cell / "passports.parquet")
    t0 = np.asarray(pp["t0_spine_pos"]).astype(np.int64)
    a_min, inv, cnt = np.unique(t0, return_inverse=True, return_counts=True)
    is_t0 = np.zeros(tape.n, dtype=bool)
    is_t0[a_min] = True
    A = a_min[elig[a_min]]
    rep["population_A"] = {
        "passport_rows": int(len(t0)),
        "distinct_t0_minutes": int(a_min.size),
        "eligible": int(A.size),
        "dropped_not_eligible": int(a_min.size - A.size),
        "minutes_carrying_more_than_one_riz": int((cnt > 1).sum()),
        "session_days": int(np.unique(sid[A]).size),
        "distinct_dates": int(np.unique(date_key(tape.ts[A])).size),
        "years": sorted({int(v) for v in yr[A]}),
    }

    # ---- population B ---------------------------------------------------
    Bmask = elig & ~is_t0
    B = np.flatnonzero(Bmask)
    rep["population_B"] = {
        "eligible_non_t0_minutes": int(B.size),
        "ratio_B_over_A": float(B.size / A.size),
        "session_days": int(np.unique(sid[B]).size),
    }

    # ---- section 7: other tf54 RIZ alive at a comparison minute ---------
    start = np.asarray(pp["t0_spine_pos"]).astype(np.int64)
    dele = pp["c1_deletion_spine_pos"].fill_null(-1).to_numpy(
        zero_copy_only=False).astype(np.int64)
    last = np.asarray(pp["last_observed_spine_pos"]).astype(np.int64)
    # an object exists from T0 until canonical deletion; where deletion is null
    # (10 censored rows) the last observed minute is the last minute it is known
    # to exist, and nothing after it is claimed.
    end = np.where(dele >= 0, dele, last + 1).astype(np.int64)
    delta = np.zeros(tape.n + 1, dtype=np.int32)
    np.add.at(delta, start, 1)
    np.add.at(delta, np.minimum(end, tape.n), -1)
    alive = np.cumsum(delta)[:tape.n]
    first_t0 = int(start.min())
    label = np.where(np.arange(tape.n) < first_t0, 2,      # UNDEFINED
                     np.where(alive > 0, 1, 0))            # RIZ-PRESENT / NO-RIZ
    names = {0: "NO-RIZ", 1: "RIZ-PRESENT", 2: "UNDEFINED"}
    rep["comparison_minute_riz_context"] = {
        "scope": "tf54 only; a minute may carry RIZ of other timeframes",
        "counts": {names[v]: int(c) for v, c in
                   zip(*np.unique(label[B], return_counts=True))},
        "share": {names[v]: float(c / B.size) for v, c in
                  zip(*np.unique(label[B], return_counts=True))},
    }

    # ---- controls C2a / C2b --------------------------------------------
    # C2b: same ET minute-of-day, same ET calendar year, different session day.
    key_all = yr.astype(np.int64) * 10000 + mod.astype(np.int64)
    order = np.argsort(key_all[B], kind="stable")
    Bs = B[order]
    ks = key_all[Bs]
    bounds = np.searchsorted(ks, key_all[A], side="left"), \
        np.searchsorted(ks, key_all[A], side="right")
    matched_ok = np.zeros(A.size, dtype=bool)
    for i in range(A.size):
        lo, hi = bounds[0][i], bounds[1][i]
        matched_ok[i] = np.any(sid[Bs[lo:hi]] != sid[A[i]])
    rep["control_C2b_matching"] = {
        "definition": "same ET minute-of-day, same ET calendar year, different session day",
        "A_with_a_matched_control": int(matched_ok.sum()),
        "A_without_any_matched_control": int((~matched_ok).sum()),
        "share_matched": float(matched_ok.mean()),
    }

    def draw_c2a(rng):
        return rng.choice(B, size=A.size, replace=False)

    c2b_cand = []
    for i in range(A.size):
        lo, hi = bounds[0][i], bounds[1][i]
        cand = Bs[lo:hi]
        c2b_cand.append(cand[sid[cand] != sid[A[i]]])
    rep["control_C2b_matching"]["candidates_per_A"] = {
        "median": float(np.median([len(c) for c in c2b_cand])),
        "min": int(min(len(c) for c in c2b_cand)),
        "max": int(max(len(c) for c in c2b_cand))}

    def draw_c2b(rng):
        return np.array([c[rng.integers(len(c))] if len(c) else -1
                         for c in c2b_cand], dtype=np.int64)

    # ---- the audit per representation ----------------------------------
    results: dict[str, dict] = {}
    all_specs = list(BASE_SPECS) + list(SENSITIVITIES)
    sd_pool = B[RNG.choice(B.size, 400_000, replace=False)]

    for spec in all_specs:
        base = spec in BASE_SPECS
        s_sd = sigma_sd(tape, spec, sd_pool)
        XA = coords(tape, A, spec, s_sd)
        finA = np.isfinite(XA).all(1)
        XA = XA[finA]
        Ae, sidA = A[finA], sid[A[finA]]
        r: dict[str, object] = {"dims": int(XA.shape[1]),
                                "sigma_log_sd": s_sd,
                                "A_used": int(Ae.size),
                                "A_dropped_bad_variant_ruler": int((~finA).sum())}

        # -- C3 denominator: A -> A, cross-day
        dAA, _ = knn_self(XA, sidA, Ae, K)
        r["within_class_A_to_A"] = {f"k{k}": qs(dAA[:, k - 1]) for k in (1, K)}

        # -- C1 availability: full pool (base specs only, frozen budget)
        if base:
            dAB, iAB = full_pool(tape, XA, sidA, B, spec, s_sd, sid)
            r["C1_availability_full_pool"] = {
                "pool_size": int(B.size),
                **{f"k{k}": qs(dAB[:, k - 1]) for k in (1, K)},
                "R_ratio": {f"k{k}": ratio_stats(dAB, dAA, k) for k in (1, K)},
                "nearest_neighbour_riz_context":
                    {names[v]: int(c) for v, c in
                     zip(*np.unique(label[iAB[:, 0]], return_counts=True))},
                "coverage_at_A_to_A_quantiles": coverage(dAB, dAA),
            }

        # -- C2a / C2b + C4 structural overlap
        for tag, draw in (("C2a_uniform", draw_c2a), ("C2b_composition", draw_c2b)):
            rng = np.random.default_rng(SEED + crc32((spec.name + tag).encode()) % 100_000)
            Rk = {1: [], K: []}
            purity, acc, auc, cov = [], [], [], []
            for _ in range(REPS):
                bsel = draw(rng)
                keep = bsel >= 0
                XB = coords(tape, bsel[keep], spec, s_sd)
                good = np.isfinite(XB).all(1)
                XB, posB = XB[good], bsel[keep][good]
                sidB = sid[posB]
                d, _ = knn_against(XA, sidA, XB, sidB, posB, K)
                for k in (1, K):
                    Rk[k].append(ratio_stats(d, dAA, k))
                p, a_, u = balanced(XA, sidA, XB, sidB, K)
                purity.append(p); acc.append(a_); auc.append(u)
                cov.append(coverage(d, dAA))
            r[tag] = {
                "control_size": int(A.size),
                "R_ratio": {f"k{k}": across(Rk[k]) for k in (1, K)},
                "C4_structural_overlap_knn_purity": across_scalar(purity),
                "balanced_1nn_accuracy": across_scalar(acc),
                "auc_recoverability": across_scalar(auc),
                "coverage_at_A_to_A_quantiles": across_cov(cov),
            }

        # -- extrapolation + coordinate balance (base specs)
        if base:
            bsel = draw_c2a(np.random.default_rng(SEED + 7))
            XB = coords(tape, bsel, spec, s_sd)
            XB = XB[np.isfinite(XB).all(1)]
            lo, hi = XB.min(0), XB.max(0)
            outside = (XA < lo) | (XA > hi)
            smd = (XA.mean(0) - XB.mean(0)) / np.sqrt(
                (XA.var(0) + XB.var(0)) / 2 + 1e-12)
            r["extrapolation_share_vs_size_matched"] = float(np.mean(outside.any(1)))
            r["coordinate_balance"] = {
                "max_abs_smd": float(np.max(np.abs(smd))),
                "median_abs_smd": float(np.median(np.abs(smd))),
                "share_coords_abs_smd_over_0p25": float(np.mean(np.abs(smd) > 0.25)),
            }

        results[spec.name] = r
        print(f"  done {spec.name}", flush=True)

    rep["representations"] = results
    path = OUT / "support.json"
    path.write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=float),
                    encoding="utf-8")
    print(f"\nwritten: {path}")


def full_pool(tape, XA, sidA, B, spec, s_sd, sid, chunk=20_000):
    nA = XA.shape[0]
    best = np.full((nA, K), INF, dtype=np.float32)
    bidx = np.full((nA, K), -1, dtype=np.int64)
    aa = (XA ** 2).sum(1)
    for s in range(0, B.size, chunk):
        pos = B[s:s + chunk]
        XB = coords(tape, pos, spec, s_sd)
        good = np.isfinite(XB).all(1)
        XB, pos = XB[good], pos[good]
        if not len(pos):
            continue
        D = aa[:, None] + (XB ** 2).sum(1)[None, :] - 2.0 * (XA @ XB.T)
        np.maximum(D, 0.0, out=D)
        D[sidA[:, None] == sid[pos][None, :]] = INF
        m = min(K, D.shape[1])
        part = np.argpartition(D, m - 1, axis=1)[:, :m]
        allD = np.concatenate([best, np.take_along_axis(D, part, 1)], 1)
        allI = np.concatenate([bidx, pos[part]], 1)
        o = np.argsort(allD, axis=1, kind="stable")[:, :K]
        best = np.take_along_axis(allD, o, 1)
        bidx = np.take_along_axis(allI, o, 1)
        if (s // chunk) % 40 == 0:
            print(f"    {spec.name} full pool {s / B.size:5.1%}", flush=True)
    return np.sqrt(best), bidx


def balanced(XA, sidA, XB, sidB, k):
    """C4: fraction of the k nearest CROSS-DAY neighbours from the other class."""
    X = np.vstack([XA, XB])
    s = np.concatenate([sidA, sidB])
    y = np.concatenate([np.zeros(len(XA), bool), np.ones(len(XB), bool)])
    aa = (X ** 2).sum(1)
    D = aa[:, None] + aa[None, :] - 2.0 * (X @ X.T)
    np.maximum(D, 0.0, out=D)
    D[s[:, None] == s[None, :]] = INF
    kk = min(k, D.shape[1] - 1)
    part = np.argpartition(D, kk - 1, axis=1)[:, :kk]
    other = (y[part] != y[:, None]).mean(1)
    nn = part[np.arange(len(X)), np.argmin(np.take_along_axis(D, part, 1), axis=1)]
    acc = float((y[nn] == y).mean())
    # same-session-day pairs are already INF, and a point shares a day with
    # itself, so the self pair is excluded here too.
    dA = D[:, :len(XA)].min(1)
    dB = D[:, len(XA):].min(1)
    score = dA - dB                      # large -> looks like B
    auc = roc_auc(score, y)
    return float(other.mean()), acc, auc


def roc_auc(score, y):
    ok = np.isfinite(score)
    score, y = score[ok], y[ok]
    if y.all() or not y.any():
        return None
    order = np.argsort(score, kind="stable")
    ranks = np.empty(len(score), float)
    ranks[order] = np.arange(1, len(score) + 1)
    n1, n0 = int(y.sum()), int((~y).sum())
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def coverage(cross, within):
    """Secondary: share of A with >=1 and >=5 cross-day controls inside tau."""
    out = {}
    for q in (10, 25, 50, 75, 90):
        tau = float(np.percentile(within[:, 0][np.isfinite(within[:, 0])], q))
        inside = (cross <= tau).sum(1)
        out[f"tau_p{q}"] = {"tau": tau,
                            "share_ge1": float(np.mean(inside >= 1)),
                            "share_ge5": float(np.mean(inside >= 5))}
    return out


def across(list_of_dicts):
    keys = [k for k in list_of_dicts[0] if isinstance(list_of_dicts[0][k], (int, float))]
    return {k: {"median": float(np.median([d[k] for d in list_of_dicts])),
                "p5": float(np.percentile([d[k] for d in list_of_dicts], 5)),
                "p95": float(np.percentile([d[k] for d in list_of_dicts], 95))}
            for k in keys if all(d[k] is not None for d in list_of_dicts)}


def across_scalar(vals):
    v = [x for x in vals if x is not None]
    return {"median": float(np.median(v)), "p5": float(np.percentile(v, 5)),
            "p95": float(np.percentile(v, 95))}


def across_cov(covs):
    out = {}
    for key in covs[0]:
        for field in ("share_ge1", "share_ge5"):
            out.setdefault(key, {})[field] = float(
                np.median([c[key][field] for c in covs]))
        out[key]["tau"] = covs[0][key]["tau"]
    return out


if __name__ == "__main__":
    main()
