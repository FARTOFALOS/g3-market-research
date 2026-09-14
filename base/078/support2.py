"""078A, second pass — the deliverables the first pass left uncounted, plus one
validity check of the control instrument itself.

Four things, no new representation, no new lookback, no new metric:

  1. effective session-days behind the support at the frozen tau grid;
  2. dependence of the primary ratio R on session time and on epoch;
  3. C2b validity: how often a point's OWN matched partner is its nearest
     control. C2b matches exactly on minute-of-day and year, which are two of
     G0's four coordinates, so the paired distance loses those blocks by
     construction. This is a property of the control, and it has to be named
     with a number rather than left inside the result;
  4. whether the 277 T0 minutes dropped by eligibility are dropped selectively.

python -B base/078/support2.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from represent import BASE_SPECS, Tape, coords, sigma_sd
from support import knn_against, knn_self, K, REPS, SEED, INSTRUMENT, TF

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research"))
OUT = ROOT / "work" / "078a"
RNG = np.random.default_rng(SEED + 1)


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
    keep = elig[a_min]
    A, dropped = a_min[keep], a_min[~keep]
    B = np.flatnonzero(elig & ~is_t0)

    rep: dict[str, object] = {}

    # ---- 4. is the eligibility drop selective? --------------------------
    def profile(pos):
        h = mod[pos] // 60
        return {"n": int(pos.size),
                "by_year": {str(int(v)): int(c) for v, c in
                            zip(*np.unique(yr[pos], return_counts=True))},
                "by_et_hour": {str(int(v)): int(c) for v, c in
                               zip(*np.unique(h, return_counts=True))}}

    kept_p, drop_p = profile(A), profile(dropped)
    hours_kept = np.array([kept_p["by_et_hour"].get(str(h), 0) for h in range(24)])
    hours_drop = np.array([drop_p["by_et_hour"].get(str(h), 0) for h in range(24)])
    rep["eligibility_drop"] = {
        "why": f"a representation needs {151} contiguous finite minutes ending at p",
        "kept": kept_p, "dropped": drop_p,
        "dropped_share": float(dropped.size / a_min.size),
        "et_hours_where_drop_share_exceeds_half": [
            int(h) for h in range(24)
            if (hours_kept[h] + hours_drop[h]) > 0
            and hours_drop[h] / (hours_kept[h] + hours_drop[h]) > 0.5],
        "note": ("dropped minutes are not negative cases and not evidence; they "
                 "are unknown for every representation equally"),
    }

    # ---- controls, rebuilt exactly as in support.py ---------------------
    key_all = yr.astype(np.int64) * 10000 + mod.astype(np.int64)
    order = np.argsort(key_all[B], kind="stable")
    Bs, ks = B[order], key_all[B[order]]
    lo = np.searchsorted(ks, key_all[A], side="left")
    hi = np.searchsorted(ks, key_all[A], side="right")
    cand = [Bs[a:b][sid[Bs[a:b]] != sid[A[i]]] for i, (a, b) in enumerate(zip(lo, hi))]

    sd_pool = B[RNG.choice(B.size, 400_000, replace=False)]
    out_specs = {}

    for spec in BASE_SPECS:
        s_sd = sigma_sd(tape, spec, sd_pool)
        XA = coords(tape, A, spec, s_sd)
        fin = np.isfinite(XA).all(1)
        XA, Ae, sidA = XA[fin], A[fin], sid[A[fin]]
        dAA, _ = knn_self(XA, sidA, Ae, K)
        # A ratio needs a positive denominator. In G0 a few T0 sit at distance
        # exactly zero from another T0 on a different day: four coordinates are
        # not enough to separate them. Counted, not quietly divided.
        pos_den = dAA[:, 0] > 0
        tau = {q: float(np.percentile(dAA[:, 0][np.isfinite(dAA[:, 0])], q))
               for q in (10, 25, 50, 75, 90)}

        cand_f = [cand[i] for i in np.flatnonzero(fin)]

        band = np.where(mod[Ae] < 9 * 60 + 30, "pre-0930",
                        np.where(mod[Ae] < 12 * 60, "0930-1200",
                                 np.where(mod[Ae] < 16 * 60, "1200-1600", "post-1600")))
        ep = np.select([yr[Ae] <= 2010, yr[Ae] <= 2015, yr[Ae] <= 2020],
                       ["2006-2010", "2011-2015", "2016-2020"], "2021-2026")

        res: dict[str, object] = {
            "tau_grid": tau,
            "A_with_a_zero_distance_twin_on_another_day": int((~pos_den).sum()),
            "A_used": int(Ae.size)}
        for tag in ("C2a", "C2b"):
            rng = np.random.default_rng(SEED + (0 if tag == "C2a" else 1))
            eff_days, partner_first, Rk1, Rk10 = [], [], [], []
            byband, byep = [], []
            for _ in range(REPS):
                if tag == "C2a":
                    sel = rng.choice(B, size=Ae.size, replace=False)
                else:
                    sel = np.array([c[rng.integers(len(c))] if len(c) else -1
                                    for c in cand_f], dtype=np.int64)
                ok = sel >= 0
                XB = coords(tape, sel[ok], spec, s_sd)
                g = np.isfinite(XB).all(1)
                XB, posB = XB[g], sel[ok][g]
                d, idx = knn_against(XA, sidA, XB, sid[posB], posB, K)
                Rk1.append((d[:, 0] / dAA[:, 0])[pos_den])
                Rk10.append((d[:, K - 1] / dAA[:, K - 1])[dAA[:, K - 1] > 0])
                # 1. effective session-days at each tau
                eff_days.append({q: int(np.unique(
                    sid[Ae[(d <= tau[q]).any(1)]]).size) for q in tau})
                # 3. is a point's own matched partner its nearest control?
                if tag == "C2b":
                    # `sel` is one control per A row, so a direct comparison
                    # answers whether that very control came out nearest.
                    partner_first.append(float(np.mean(idx[:, 0] == sel)))
                # 2. dependence on session time and epoch
                r1 = np.where(pos_den, d[:, 0] / np.where(pos_den, dAA[:, 0], 1.0),
                              np.nan)
                byband.append({b: float(np.nanmedian(r1[band == b]))
                               for b in np.unique(band)
                               if np.isfinite(r1[band == b]).any()})
                byep.append({e: float(np.nanmedian(r1[ep == e]))
                             for e in np.unique(ep)
                             if np.isfinite(r1[ep == e]).any()})
            R1 = np.concatenate(Rk1); R10 = np.concatenate(Rk10)
            res[tag] = {
                "R_k1_median": float(np.median(R1)),
                "R_k10_median": float(np.median(R10)),
                "effective_session_days": {
                    f"tau_p{q}": {"median": float(np.median([e[q] for e in eff_days])),
                                  "of_total_A_days": int(np.unique(sid[Ae]).size)}
                    for q in tau},
                "R_k1_median_by_session_time": {
                    b: float(np.median([x[b] for x in byband if b in x]))
                    for b in np.unique(band)},
                "R_k1_median_by_epoch": {
                    e: float(np.median([x[e] for x in byep if e in x]))
                    for e in np.unique(ep)},
            }
            if tag == "C2b":
                res[tag]["own_partner_is_nearest_control"] = {
                    "median": float(np.median(partner_first)),
                    "p5": float(np.percentile(partner_first, 5)),
                    "p95": float(np.percentile(partner_first, 95)),
                    "meaning": ("C2b matches exactly on minute-of-day and year, "
                                "so for the paired control the clock and epoch "
                                "blocks contribute exactly zero. A high value "
                                "means C2b is biased TOWARDS overlap for the "
                                "blocks it matches on, and the C2a column is the "
                                "one to read for those."),
                }
        out_specs[spec.name] = res
        print(f"  done {spec.name}", flush=True)

    rep["per_representation"] = out_specs
    (OUT / "support2.json").write_text(
        json.dumps(rep, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(f"\nwritten: {OUT / 'support2.json'}")


if __name__ == "__main__":
    main()
