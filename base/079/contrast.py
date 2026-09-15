"""079A — build H0/H1 relation-instances and measure their common support.

Executes base/079/FREEZE_079A.md. The future is never opened: F is not chosen,
not defined and not computed. Volume is never loaded. The field is read-only.

python -B base/079/contrast.py
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
from decompose import Bars, first_span_minute             # noqa: E402

INSTRUMENT, TF = "NQ", 54
K, REPS, SEED = 10, 20, 20790914
OUT = ROOT / "work" / "079a"
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(SEED)
INF = np.float32(np.inf)
GEOM = ("close_in_zone", "open_in_zone", "log_w_sigma", "bar_phase")


# ------------------------------------------------------------- instances
def build_instances():
    bars = Bars(INSTRUMENT, TF)
    cell = ROOT / "data" / "field" / INSTRUMENT / f"cells/tf_{TF:04d}"
    pp = pq.read_table(cell / "passports.parquet")
    ev = pq.read_table(cell / "events.parquet")

    riz = np.asarray(pp["riz_id"])
    t0 = np.asarray(pp["t0_spine_pos"]).astype(np.int64)
    zt = np.asarray(pp["zone_top"]).astype(float)
    zb = np.asarray(pp["zone_bottom"]).astype(float)
    prec = np.asarray(pp["precursor_formed_spine_pos"]).astype(np.int64)
    idx = {r: i for i, r in enumerate(riz)}

    m = np.asarray(ev["event_kind"]) == "accepted_span"
    e_riz = np.asarray(ev["riz_id"])[m]
    e_pos = np.asarray(ev["market_spine_pos"]).astype(np.int64)[m]
    order = np.lexsort((e_pos, e_riz))
    e_riz, e_pos = e_riz[order], e_pos[order]
    first = np.r_[True, e_riz[1:] != e_riz[:-1]]
    first_span_report = {r: int(p) for r, p in zip(e_riz[first], e_pos[first])}

    rows = []
    for r, cls, anchor in (
            [(r, 0, first_span_report[r]) for r in riz]
            + [(r, 1, int(t0[idx[r]])) for r in riz]):
        i = idx[r]
        pos, dur, nmin = first_span_minute(bars, anchor, zt[i], zb[i])
        rows.append({"riz_id": str(r), "cls": cls, "anchor_report": int(anchor),
                     "p": int(pos), "dur": int(dur), "bar_minutes": int(nmin),
                     "zone_top": float(zt[i]), "zone_bottom": float(zb[i]),
                     "precursor_pos": int(prec[i]), "t0_pos": int(t0[i])})
    return bars, rows


def geometry(bars: Bars, rows, sig):
    """The frozen four-coordinate current-geometry record R(M,A)."""
    out = np.full((len(rows), 4), np.nan)
    for j, row in enumerate(rows):
        p = row["p"]
        if p < 0 or not np.isfinite(sig[j]) or sig[j] <= 0:
            continue
        w = row["zone_top"] - row["zone_bottom"]
        if w <= 0:
            continue
        mins = bars.bar_minutes(p)
        if mins.size == 0:
            continue
        bar_open = float(bars.open[mins[0]])
        c = float(bars.close[p])
        out[j] = ((c - row["zone_bottom"]) / w,
                  (bar_open - row["zone_bottom"]) / w,
                  np.log(w / sig[j]),
                  row["dur"] / TF)
    return out


# ---------------------------------------------------------------- knn
def knn(XQ, okQ, XR, dayQ, dayR, objQ, objR, k=K, cross_object=True):
    """k nearest admissible neighbours: different session-day, and by default a
    different riz_id, so two events of one object are never two witnesses."""
    nQ = XQ.shape[0]
    D = ((XQ ** 2).sum(1)[:, None] + (XR ** 2).sum(1)[None, :]
         - 2.0 * (XQ @ XR.T))
    np.maximum(D, 0.0, out=D)
    bad = dayQ[:, None] == dayR[None, :]
    if cross_object:
        bad |= objQ[:, None] == objR[None, :]
    D[bad] = INF
    k = min(k, D.shape[1])
    part = np.argpartition(D, k - 1, axis=1)[:, :k]
    d = np.take_along_axis(D, part, 1)
    o = np.argsort(d, axis=1, kind="stable")
    return np.sqrt(np.take_along_axis(d, o, 1)), np.take_along_axis(part, o, 1)


def ratio(cross, within, k):
    a, b = cross[:, k - 1], within[:, k - 1]
    ok = np.isfinite(a) & np.isfinite(b) & (b > 0)
    r = a[ok] / b[ok]
    return {"n": int(ok.sum()), "p25": float(np.percentile(r, 25)),
            "median": float(np.median(r)), "p75": float(np.percentile(r, 75)),
            "share_below_1": float(np.mean(r < 1))}


def purity(X, day, obj, y, k=K):
    D = ((X ** 2).sum(1)[:, None] + (X ** 2).sum(1)[None, :] - 2.0 * (X @ X.T))
    np.maximum(D, 0.0, out=D)
    D[(day[:, None] == day[None, :]) | (obj[:, None] == obj[None, :])] = INF
    k = min(k, D.shape[1] - 1)
    part = np.argpartition(D, k - 1, axis=1)[:, :k]
    other = float((y[part] != y[:, None]).mean())
    nn = part[np.arange(len(X)), np.argmin(np.take_along_axis(D, part, 1), 1)]
    return other, float((y[nn] == y).mean())


def smd(a, b):
    return float((np.mean(a) - np.mean(b))
                 / np.sqrt((np.var(a) + np.var(b)) / 2 + 1e-12))


# ---------------------------------------------------------------- main
def main() -> None:
    bars, rows = build_instances()
    tape = Tape(INSTRUMENT)
    elig = tape.eligible()

    p = np.array([r["p"] for r in rows])
    cls = np.array([r["cls"] for r in rows])
    ok = (p >= 0) & (p < tape.n)
    ok &= np.where(ok, elig[np.where(ok, p, 0)], False)

    rep = {"instrument": INSTRUMENT, "tf_minutes": TF, "k": K, "reps": REPS,
           "cohort": "EVENTUAL-T0 COHORT: the 1046 tf54 objects of the frozen field",
           "instances_built": {"H0": int((cls == 0).sum()), "H1": int((cls == 1).sum())},
           "instances_eligible": {"H0": int((ok & (cls == 0)).sum()),
                                  "H1": int((ok & (cls == 1)).sum())}}

    # keep only objects that survive in BOTH classes, so the design stays paired
    obj_all = np.array([r["riz_id"] for r in rows])
    good_obj = set(obj_all[ok & (cls == 0)]) & set(obj_all[ok & (cls == 1)])
    keep = ok & np.isin(obj_all, list(good_obj))
    rows_k = [r for r, f in zip(rows, keep) if f]
    p, cls = p[keep], cls[keep]
    obj = obj_all[keep]
    day = np.asarray(tape.session_id)[p]
    rep["paired_objects"] = int(len(good_obj))
    rep["instances_used"] = {"H0": int((cls == 0).sum()), "H1": int((cls == 1).sum())}

    # separation of the two events of one object
    gap = {}
    for r in good_obj:
        a = p[(obj == r) & (cls == 0)][0]
        b = p[(obj == r) & (cls == 1)][0]
        gap[r] = int(b - a)
    g = np.array(list(gap.values()))
    rep["H0_to_H1_minutes"] = {"min": int(g.min()), "p25": float(np.percentile(g, 25)),
                               "median": float(np.median(g)),
                               "p75": float(np.percentile(g, 75)),
                               "max": int(g.max()),
                               "all_positive": bool((g > 0).all()),
                               "same_session_day": int((day[cls == 0] == day[cls == 1]).sum())}

    # multiplicity
    rep["multiplicity"] = {
        "distinct_relation_instances": int(len(p)),
        "distinct_physical_minutes": int(np.unique(p).size),
        "distinct_zones": int(np.unique(obj).size),
        "distinct_session_days": int(np.unique(day).size),
        "minutes_carrying_more_than_one_instance": int(
            np.sum(np.bincount(np.unique(p, return_inverse=True)[1]) > 1)),
        "objects_giving_more_than_one_instance_per_class": 0,
    }

    # ---- coordinates -------------------------------------------------
    s_sd = sigma_sd(tape, G2, np.flatnonzero(elig)[
        RNG.choice(int(elig.sum()), 400_000, replace=False)])
    M = coords(tape, p, G2, s_sd)
    sig = tape.sigma_at(p, G2.ruler, G2.shift)
    R = geometry(bars, rows_k, sig)
    fin = np.isfinite(M).all(1) & np.isfinite(R).all(1)
    # keep the pairing after the finiteness cut
    surv = set(obj[fin & (cls == 0)]) & set(obj[fin & (cls == 1)])
    fin &= np.isin(obj, list(surv))
    M, R, p, cls, obj, day = M[fin], R[fin], p[fin], cls[fin], obj[fin], day[fin]
    rows_k = [r for r, f in zip(rows_k, fin) if f]
    rep["instances_after_coordinates"] = {"H0": int((cls == 0).sum()),
                                          "H1": int((cls == 1).sum()),
                                          "objects": int(np.unique(obj).size)}

    geom_sd = R.std(0)
    rep["geom_sd_frozen"] = {n: float(v) for n, v in zip(GEOM, geom_sd)}
    Rw = (R / geom_sd) * np.sqrt(1.0 / len(GEOM))

    # ---- section 10: what price actually does in p, per class ---------
    rep["current_event_balance"] = {
        n: {"H0_median": float(np.median(R[cls == 0, i])),
            "H1_median": float(np.median(R[cls == 1, i])),
            "smd": smd(R[cls == 0, i], R[cls == 1, i])}
        for i, n in enumerate(GEOM)}
    body = np.abs(R[:, 0] - R[:, 1])          # body height in zone widths
    rep["current_event_balance"]["body_height_in_zone_widths"] = {
        "H0_median": float(np.median(body[cls == 0])),
        "H1_median": float(np.median(body[cls == 1])),
        "smd": smd(body[cls == 0], body[cls == 1])}
    rep["current_event_balance"]["local_sigma_points"] = {
        "H0_median": float(np.median(sig[fin][cls == 0])),
        "H1_median": float(np.median(sig[fin][cls == 1])),
        "smd": smd(np.log(sig[fin][cls == 0]), np.log(sig[fin][cls == 1]))}

    # ---- section 9: biography differences, DESCRIPTION ONLY -----------
    age = np.array([r["p"] - r["precursor_pos"] for r in rows_k], float)
    rep["biography_difference_description"] = {
        "note": "not predictors, not matched on; the subject of the question",
        "age_minutes": {"H0_median": float(np.median(age[cls == 0])),
                        "H1_median": float(np.median(age[cls == 1])),
                        "smd": smd(np.log(age[cls == 0] + 1), np.log(age[cls == 1] + 1))},
        "prior_accepted_spans": {"H0": 0, "H1": 1},
        "minutes_since_first_interaction": {
            "H0_median": 0.0, "H1_median": float(np.median(g))},
    }

    # ---- support under the three frozen metric variants ---------------
    variants = {
        "P M+R": np.hstack([M, Rw]),
        "Q R only": Rw,
        "S M only": M,
    }
    res = {}
    for name, X in variants.items():
        X = np.ascontiguousarray(X, dtype=np.float32)
        out = {"dims": int(X.shape[1])}
        for a, b, tag in ((0, 1, "H0->H1"), (1, 0, "H1->H0")):
            qa, qb = cls == a, cls == b
            dx, _ = knn(X[qa], qa, X[qb], day[qa], day[qb], obj[qa], obj[qb], K)
            dw, _ = knn(X[qa], qa, X[qa], day[qa], day[qa], obj[qa], obj[qa], K)
            out[tag] = {f"R_k{k}": ratio(dx, dw, k) for k in (1, K)}
        o, acc = purity(X, day, obj, cls, K)
        out["cross_object_knn_purity"] = o
        out["cross_object_1nn_accuracy"] = acc
        # paired same-object support, reported separately and never mixed in
        pair = []
        for r in np.unique(obj):
            ia = np.flatnonzero((obj == r) & (cls == 0))
            ib = np.flatnonzero((obj == r) & (cls == 1))
            if ia.size and ib.size:
                pair.append(float(np.sqrt(((X[ia[0]] - X[ib[0]]) ** 2).sum())))
        pair = np.array(pair)
        qa = cls == 0
        dw, _ = knn(X[qa], qa, X[qa], day[qa], day[qa], obj[qa], obj[qa], K)
        ref = dw[:, 0]
        out["paired_same_object"] = {
            "n": int(pair.size),
            "distance_median": float(np.median(pair)),
            "vs_within_class_nn_median": float(np.median(pair) / np.median(ref)),
            "note": "separate from cross-object support, never added to it",
        }
        # coverage at tau from the within-class cross-day cross-object NN
        tau = {q: float(np.percentile(ref, q)) for q in (10, 25, 50, 75, 90)}
        qa, qb = cls == 0, cls == 1
        dx, _ = knn(X[qa], qa, X[qb], day[qa], day[qb], obj[qa], obj[qb], K)
        out["coverage"] = {
            f"tau_p{q}": {
                "tau": tau[q],
                "share_ge1": float(np.mean((dx <= tau[q]).any(1))),
                "share_ge5": float(np.mean((dx <= tau[q]).sum(1) >= 5)),
                "effective_session_days": int(np.unique(
                    day[qa][(dx <= tau[q]).any(1)]).size),
                "effective_objects": int(np.unique(
                    obj[qa][(dx <= tau[q]).any(1)]).size),
            } for q in tau}
        lo, hi = X[cls == 1].min(0), X[cls == 1].max(0)
        out["extrapolation_share_H0_outside_H1"] = float(
            np.mean(((X[cls == 0] < lo) | (X[cls == 0] > hi)).any(1)))
        res[name] = out
        print(f"  done {name}", flush=True)

    rep["support"] = res
    rep["examples"] = pick_examples(rows_k, p, cls, obj, day, variants["P M+R"], R, tape)
    (OUT / "contrast.json").write_text(
        json.dumps(rep, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(f"\nwritten: {OUT / 'contrast.json'}")


def pick_examples(rows, p, cls, obj, day, X, R, tape, n=4):
    """Nearest cross-class, cross-day, cross-object pairs. Chosen without any F."""
    X = np.ascontiguousarray(X, dtype=np.float32)
    qa, qb = cls == 0, cls == 1
    d, idx = knn(X[qa], qa, X[qb], day[qa], day[qb], obj[qa], obj[qb], 1)
    order = np.argsort(d[:, 0])[:n]
    ia = np.flatnonzero(qa)
    ib = np.flatnonzero(qb)
    out = []
    for j in order:
        a, b = ia[j], ib[idx[j, 0]]
        out.append({
            "distance": float(d[j, 0]),
            "H0": describe(rows[a], p[a], R[a], tape),
            "H1": describe(rows[b], p[b], R[b], tape),
            "selection": "nearest cross-class neighbour under the frozen metric; no F exists",
        })
    return out


def describe(row, pos, r, tape):
    return {
        "riz_id": row["riz_id"],
        "utc": str(np.datetime64(int(tape.ts[pos]), "ns")),
        "zone": [row["zone_top"], row["zone_bottom"]],
        "close_in_zone": float(r[0]), "open_in_zone": float(r[1]),
        "log_w_sigma": float(r[2]), "bar_phase": float(r[3]),
        "minutes_into_native_bar": row["dur"],
        "age_minutes_since_precursor": int(pos - row["precursor_pos"]),
    }


if __name__ == "__main__":
    main()
