"""Map of missing payoff: where B1 (S-07 v1r1, then S-18 v0) earns, and where NQ moves without it.

No setup is born here. For NQ 2020-01-02 .. 2025-12-31, per 30-minute bucket of the New York day:
  - market amplitude: mean bar-range of the bucket (H - L) and mean |close_end - open_start|, points;
  - B1 exposure: share of sessions with an open S-07 / S-18 position in the bucket;
  - B1 money by bucket: minute mark-to-market of the frozen engines, cost booked at each exit;
  - after 10:00, amplitude split by state: 'free' = B1 holds nothing in the bucket AND the unblocked
    S-18 is flat (day inside its norm); otherwise 'busy'. (Audit 2026-09-24: the blocked S-18 state
    alone does not mean 'inside the norm' before 12:00, because S-18 waits for S-07.)
Engines are taken unchanged: setups/S-17/forward.py::engine (S-07 v1r1) and the S-18 loop of
setups/library/joint.py::s18_blocked with positions recorded. Daily totals must equal
joint_daily_NQ.csv (columns s07, s18_after_s07) to the cent, otherwise the script stops.

    python -B setups/library/payoff_map.py
Output: setups/library/payoff_map.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import joint as J  # noqa: E402

S17, S18 = J.S17, J.S18
LO, HI = pd.Timestamp("2020-01-02"), pd.Timestamp("2025-12-31")
PT, COST = 20.0, 15.0
BUCKETS = [(m, m + 30) for m in range(120, 960, 30)]          # 02:00 .. 16:00 ET, bar-open minutes


def s18_positions(cal, root, block_k):
    """joint.s18_blocked line by line, returning per-session (usd, [(k_entry_bar, k_exit_bar, pos, exit_px_is_open)])."""
    look, mult, step, first = 14, 1.0, 30, 30
    cal, K, O, C = S18.sessions("NQ", cal, root)
    S = len(cal)
    op = O[:, 1]
    cl = np.array([C[s, min(K[s], S18.KMAX)] for s in range(S)])
    move = np.abs(C / op[:, None] - 1.0)
    out = {}
    for s in range(look, S):
        prev = cl[s - 1]
        if not np.isfinite(op[s]) or not np.isfinite(prev) or not np.isfinite(cl[s]):
            continue
        hist = move[s - look:s]
        cnt = np.isfinite(hist).sum(axis=0)
        with np.errstate(invalid="ignore"):
            norm = np.where(cnt >= int(np.ceil(0.7 * look)), np.nanmean(hist, axis=0), np.nan)
        up = max(op[s], prev) * (1 + mult * norm)
        dn = min(op[s], prev) * (1 - mult * norm)
        marks = [k for k in range(first, S18.KMAX, step) if k + 1 <= K[s] - 1]
        pos, entry, ek = 0, np.nan, 0
        usd, bad, legs = 0.0, False, []
        for k in marks:
            if k < block_k[s]:
                continue
            p = C[s, k]
            if not np.isfinite(p) or not np.isfinite(up[k]):
                bad = True; break
            want = 1 if p > up[k] else (-1 if p < dn[k] else 0)
            if want != pos:
                px = O[s, k + 1]
                if not np.isfinite(px):
                    bad = True; break
                if pos != 0:
                    usd += pos * (px - entry) * PT - COST
                    legs.append((ek, k + 1, pos, entry, px))
                pos, entry, ek = want, px, k + 1
        if bad:
            continue
        if pos != 0:
            kend = min(K[s], S18.KMAX)
            usd += pos * (cl[s] - entry) * PT - COST
            legs.append((ek, kend + 1, pos, entry, cl[s]))
        out[s] = (usd, legs)
    return cal, K, O, C, out


def bucket_of_k(k):
    """Session bar k (close at 09:30 + k min) opened at minute 570 + k - 1."""
    return (570 + k - 1 - 120) // 30


def main():
    cal = S17.history_calendar()
    root = ROOT / "data/market/NQ"
    e = J.s07_s17(root, cal)
    block = J.block_from(e, "s07_exit_min", cal)
    cal2, K, O, C, s18 = s18_positions(cal, root.parent, block)
    _, _, _, _, s18n = s18_positions(cal, root.parent, np.zeros(len(cal)))   # unblocked: the norm state
    ref = pd.read_csv(HERE / "joint_daily_NQ.csv", parse_dates=["date"]).set_index("date")
    dates = pd.to_datetime(cal2.date)
    nb = len(BUCKETS)
    rows, bad07, bad18, approx07 = [], [], [], 0
    em = e.set_index("date")
    for s, d in enumerate(dates):
        if not (LO <= d <= HI) or d not in ref.index:
            continue
        r = dict(date=d)
        m07 = np.zeros(nb); x07 = np.zeros(nb, bool)
        st = em.status.get(d, "")
        if isinstance(st, str) and st.startswith("no_entry"):
            v07 = 0.0
        else:
            v07 = em.s07_usd.get(d, np.nan)
            if np.isfinite(v07):
                side, e0 = em.side[d], em.entry[d]
                kx = int(3 + em.s07_exit_min[d])
                path = C[s, 4:kx + 1]
                m07[bucket_of_k(kx)] -= COST
                if np.all(np.isfinite(path)):
                    prev = e0
                    for k in range(4, kx + 1):
                        m07[bucket_of_k(k)] += side * (C[s, k] - prev) * PT; prev = C[s, k]
                    for k in range(4, kx + 1): x07[bucket_of_k(k)] = True
                    m07[bucket_of_k(kx)] += v07 + COST - side * (prev - e0) * PT   # exact exit price
                else:
                    m07[bucket_of_k(kx)] += v07 + COST; x07[bucket_of_k(kx)] = True; approx07 += 1
        if not np.isclose(m07.sum(), v07 if np.isfinite(v07) else 0.0, atol=1e-6) or \
           (np.isfinite(ref.s07[d]) != np.isfinite(v07)) or (np.isfinite(v07) and abs(ref.s07[d] - v07) > 0.01):
            bad07.append(str(d.date()))
        m18 = np.zeros(nb); x18 = np.zeros(nb, bool); state18 = np.zeros(nb, np.int8)
        v18 = np.nan
        if s in s18:
            v18, legs = s18[s]
            for ke, kx, pos, ep, xp in legs:
                prev = ep
                for k in range(ke, kx):
                    if not np.isfinite(C[s, k]): continue
                    m18[bucket_of_k(k)] += pos * (C[s, k] - prev) * PT; prev = C[s, k]; x18[bucket_of_k(k)] = True
                kb = min(kx - 1, S18.KMAX)
                m18[bucket_of_k(kb)] += pos * (xp - prev) * PT - COST
                for k in range(ke, kx): state18[bucket_of_k(min(k, S18.KMAX))] = pos
        ok18 = np.isfinite(ref.s18_after_s07[d])
        if ok18 != np.isfinite(v18) or (ok18 and (abs(ref.s18_after_s07[d] - v18) > 0.01 or abs(m18.sum() - v18) > 0.01)):
            bad18.append(str(d.date()))
        r.update({f"m07_{b}": m07[b] for b in range(nb)}); r.update({f"x07_{b}": x07[b] for b in range(nb)})
        r.update({f"m18_{b}": m18[b] for b in range(nb)}); r.update({f"x18_{b}": x18[b] for b in range(nb)})
        r.update({f"st_{b}": state18[b] for b in range(nb)})
        stn = np.zeros(nb, np.int8)
        if s in s18n:
            for ke, kx, pos, ep, xp in s18n[s][1]:
                for k in range(ke, kx): stn[bucket_of_k(min(k, S18.KMAX))] = pos
        r.update({f"free_{b}": bool((not x07[b]) and state18[b] == 0 and stn[b] == 0) for b in range(nb)})
        r["known"] = bool(np.isfinite(v07) and np.isfinite(v18) and s in s18n)
        rows.append(r)
    if bad07 or bad18:
        raise SystemExit(f"daily totals do not reconcile: S-07 {bad07[:5]} ({len(bad07)}), S-18 {bad18[:5]} ({len(bad18)})")
    P = pd.DataFrame(rows)
    P = P[P.known].reset_index(drop=True)

    # market amplitude per bucket from the minute tape (same bytes as the library's tape)
    sys.path.insert(0, str(ROOT / "base/105"))
    from tape import load_minutes
    df = load_minutes("2020-01-01", "2026-01-01", "NQ")
    df = df[(df["mod"] >= 120) & (df["mod"] < 960)].copy()
    df["b"] = (df["mod"] - 120) // 30
    g = df.groupby(["date", "b"])
    A = pd.DataFrame(dict(n=g.size(), hi=g.h.max(), lo=g.l.min(), o=g.o.first(), c=g.c.last())).reset_index()
    A = A[A.n >= 25]
    A["range"] = A.hi - A.lo; A["net"] = (A.c - A.o).abs()
    A["date"] = pd.to_datetime(A.date.astype(str))
    A = A[A.date.isin(set(P.date))]

    out = dict(territory=[str(LO.date()), str(HI.date())], sessions=len(P), s07_bucket_approx_days=approx07,
               reconciled="daily S-07 and S-18-after-S-07 totals equal joint_daily_NQ.csv on every mapped session",
               buckets=[])
    Pi = P.set_index("date")
    for b, (m0, m1) in enumerate(BUCKETS):
        a = A[A.b == b]
        st = np.where(Pi[f"free_{b}"].reindex(a.date).to_numpy().astype(bool), 0, 1)
        row = dict(bucket=f"{m0 // 60:02d}:{m0 % 60:02d}-{m1 // 60:02d}:{m1 % 60:02d}",
                   range_pts=round(float(a.range.mean()), 2), net_pts=round(float(a.net.mean()), 2),
                   s07_exposed=round(float(Pi[f"x07_{b}"].mean()), 3), s18_exposed=round(float(Pi[f"x18_{b}"].mean()), 3),
                   s07_usd_per_session=round(float(Pi[f"m07_{b}"].mean()), 2), s18_usd_per_session=round(float(Pi[f"m18_{b}"].mean()), 2))
        if m0 >= 600:
            fl = st == 0
            row.update(share_free=round(float(np.mean(fl)), 3),
                       range_when_free=round(float(a.range[fl].mean()), 2) if fl.any() else None,
                       net_when_free=round(float(a.net[fl].mean()), 2) if fl.any() else None,
                       range_when_busy=round(float(a.range[~fl].mean()), 2) if (~fl).any() else None,
                       net_when_busy=round(float(a.net[~fl].mean()), 2) if (~fl).any() else None)
        out["buckets"].append(row)
    tot07 = sum(r["s07_usd_per_session"] for r in out["buckets"]); tot18 = sum(r["s18_usd_per_session"] for r in out["buckets"])
    out["b1_usd_per_session"] = dict(s07=round(tot07, 2), s18_after_s07=round(tot18, 2), total=round(tot07 + tot18, 2))
    (HERE / "payoff_map.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"sessions {len(P)}; B1 $/session S-07 {tot07:.1f} + S-18 {tot18:.1f} = {tot07 + tot18:.1f}; S-07 approx days {approx07}")
    print(f"{'bucket':12s} {'range':>6s} {'|net|':>6s} {'x07':>5s} {'x18':>5s} {'$07':>7s} {'$18':>7s} | {'free':>5s} {'rng_fr':>6s} {'net_fr':>6s} {'rng_bz':>6s} {'net_bz':>6s}")
    for r in out["buckets"]:
        tail = "" if "share_free" not in r else \
            f"| {r['share_free']:5.2f} {r['range_when_free'] or 0:6.1f} {r['net_when_free'] or 0:6.1f} {r['range_when_busy'] or 0:6.1f} {r['net_when_busy'] or 0:6.1f}"
        print(f"{r['bucket']:12s} {r['range_pts']:6.1f} {r['net_pts']:6.1f} {r['s07_exposed']:5.2f} {r['s18_exposed']:5.2f} "
              f"{r['s07_usd_per_session']:7.1f} {r['s18_usd_per_session']:7.1f} {tail}")


if __name__ == "__main__":
    main()
