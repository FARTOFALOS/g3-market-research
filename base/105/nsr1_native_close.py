"""NSR1 — the close of the RIZ's native bar as the observation minute; fork b vs M (declared in base/105/LOG.md).

Population: TF 120..240, T0 bar open in 09:30-16:00 ET, 2006-01-01 .. 2025-12-31. Unit: cluster (t0_spine_pos, exit
side); representative = member with the earliest eligible native close q (q+1 present one minute later and opening
before 16:00 ET), ties by smaller TF then riz_id. Twin diagnostic before outcomes (same day, same side, exit boundaries
within 1.0 pt, q within 120 min of a kept one; greedy by q; > 10 % absorbed -> the absorbing episode is the unit).
At q: b exit boundary, M outward extreme of the native bar through q, a first tick beyond M, entry e = open(q+1),
d = |e-b|, s = |a-e|, p0 = s/(d+s). Touch automaton from q+1; censor at 120 min, 16:00 ET or a missing minute.
G = (d+s)(Y-p0) in points = payoff of the bracket "to b" (censored: marked to market). Money: brackets "to b" and
"beyond M" separately, gap-through stops filled at the minute open, targets at their price, cost 0.75 pt.

Run from repository root: python -B base/105/nsr1_native_close.py [NQ|ES]
"""
import json, math, sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from g3riz.query import Field  # noqa: E402
from g3riz.entry_check import native_grid  # noqa: E402

MIN = 60_000_000_000
TICK, COST, HORIZON, TWIN_PT, TWIN_MIN, TWIN_SHARE, LIMIT = 0.25, 0.75, 120, 1.0, 120, 0.10, 20.0
EPOCHS = (("2006-12", 20060101, 20121231), ("2013-19", 20130101, 20191231), ("2020-25", 20200101, 20251231))
COLS = ["riz_id", "tf_minutes", "zone_top", "zone_bottom", "t0_ts_ns", "t0_spine_pos", "t0_kind", "t0_exit_side",
        "native_blue_confirmation_spine_pos", "precursor_formed_ts_ns"]


def et_parts(ns):
    t = pd.to_datetime(np.asarray(ns), utc=True).tz_convert("America/New_York")
    return (t.year * 10000 + t.month * 100 + t.day).to_numpy(), (t.hour * 60 + t.minute).to_numpy()


def members(F):
    m = F.market; ts = np.asarray(m.close_ts_utc_ns)
    out = []
    for tf in range(120, 241):
        P = F.passports(tf=tf, t0_start_ns=pd.Timestamp("2006-01-01", tz="UTC").value,
                        t0_stop_ns=pd.Timestamp("2026-01-01", tz="UTC").value).select(COLS).to_pandas()
        if not len(P): continue
        st, sp = native_grid(m, tf)
        j0 = np.searchsorted(sp, P.t0_spine_pos.to_numpy(), side="right")
        P["bar_start"] = st[j0]; P["q"] = sp[j0] - 1
        out.append(P)
    P = pd.concat(out, ignore_index=True)
    _, P["t0_mod"] = et_parts(P.t0_ts_ns.to_numpy() - MIN)
    P = P[(P.t0_mod >= 570) & (P.t0_mod < 960)].copy()
    q = P.q.to_numpy()
    ok = q + 1 < len(ts)
    nxt = np.where(ok, q + 1, q)
    P["q_date"], P["entry_mod"] = et_parts(ts[q])            # ts[q] = close of q = open of q+1
    P["eligible"] = ok & (ts[nxt] - ts[q] == MIN) & (P.entry_mod >= 570) & (P.entry_mod < 960)
    P["confirmed"] = P.native_blue_confirmation_spine_pos.notna() & (P.native_blue_confirmation_spine_pos == P.q)
    assert (P.native_blue_confirmation_spine_pos.isna() | P.confirmed).all(), "confirmation off the native close"
    return P


def representatives(P):
    E = P[P.eligible].sort_values(["t0_spine_pos", "t0_exit_side", "q", "tf_minutes", "riz_id"])
    R = E.drop_duplicates(["t0_spine_pos", "t0_exit_side"]).copy()
    R["b"] = np.where(R.t0_exit_side == "north", R.zone_top, R.zone_bottom)
    return R


def twins(R, ts):
    R = R.sort_values(["q", "tf_minutes", "riz_id"]).copy()
    kept = []                                                  # (date, side, b, ts_q)
    absorbed = np.zeros(len(R), bool)
    for i, r in enumerate(R.itertuples()):
        tq = ts[r.q]
        for (dt, sd, b, tk) in reversed(kept):
            if tq - tk > TWIN_MIN * MIN: break
            if dt == r.q_date and sd == r.t0_exit_side and abs(b - r.b) <= TWIN_PT + 1e-9:
                absorbed[i] = True; break
        if not absorbed[i]: kept.append((r.q_date, r.t0_exit_side, r.b, tq))
    R["twin_absorbed"] = absorbed
    return R


def fork(R, F):
    m = F.market
    o, h, l, c = (np.asarray(getattr(m, k)) for k in ("open", "high", "low", "close"))
    ts = np.asarray(m.close_ts_utc_ns)
    _, mod_open = et_parts(ts - MIN)
    rows = []
    for r in R.itertuples():
        north = r.t0_exit_side == "north"; sg = 1.0 if north else -1.0
        q, a0, t0 = int(r.q), int(r.bar_start), int(r.t0_spine_pos)
        b = r.b
        M = h[a0:q + 1].max() if north else l[a0:q + 1].min()
        a = M + sg * TICK
        e = o[q + 1]
        d = sg * (e - b); s = sg * (a - e)
        pre = dict(minutes_t0_to_q=int(round((ts[q] - ts[t0]) / MIN)), run_t0_q=sg * (c[q] - c[t0]), M_minus_b=sg * (M - b),
                   close_q_from_b=sg * (c[q] - b),
                   touched_b_before_q=bool(((l[t0 + 1:q + 1] <= b) if north else (h[t0 + 1:q + 1] >= b)).any()) if q > t0 else False)
        base = dict(riz_id=r.riz_id, tf=int(r.tf_minutes), date=int(r.q_date), entry_mod=int(r.entry_mod), side=r.t0_exit_side,
                    t0_kind=r.t0_kind, confirmed=bool(r.confirmed), twin_absorbed=bool(r.twin_absorbed),
                    zone_age_d=(r.t0_ts_ns - r.precursor_formed_ts_ns) / 8.64e13, e=e, b=b, M=M, d=d, s=s, **pre)
        if d <= 0 or s <= 0:
            rows.append(dict(base, status="b_at_entry" if d <= 0 else "a_at_entry")); continue
        p0 = s / (d + s)
        status, k_end, last = "censored_horizon", None, q
        for k in range(q + 1, q + 1 + HORIZON):
            if k >= len(ts) or (k > q + 1 and ts[k] - ts[k - 1] != MIN):
                status = "censored_gap"; break
            if mod_open[k] >= 960 or mod_open[k] < 570:
                status = "censored_window"; break
            hb = (l[k] <= b) if north else (h[k] >= b)
            ha = (h[k] >= a) if north else (l[k] <= a)
            last = k
            if hb or ha:
                status = "same_minute" if (hb and ha) else ("b_first" if hb else "a_first"); k_end = k; break
        # bracket "to b": short (north) from e, target b, stop a; bracket "beyond M": long (north), target a, stop b
        if status == "b_first":
            Y = 1.0; G = d; ret = d
            stop_fill = min(o[k_end], b) if north else max(o[k_end], b)
            cont = sg * (stop_fill - e)
            G_lo = G_hi = G; ret_lo = ret_hi = ret; cont_lo = cont_hi = cont
        elif status == "a_first":
            Y = 0.0; G = -s; cont = s
            stop_fill = max(o[k_end], a) if north else min(o[k_end], a)
            ret = -sg * (stop_fill - e)
            G_lo = G_hi = G; ret_lo = ret_hi = ret; cont_lo = cont_hi = cont
        elif status == "same_minute":
            Y = np.nan; G = np.nan; ret = np.nan; cont = np.nan
            sa = max(o[k_end], a) if north else min(o[k_end], a)
            sb = min(o[k_end], b) if north else max(o[k_end], b)
            G_lo, G_hi = -s, d
            ret_lo, ret_hi = -sg * (sa - e), d
            cont_lo, cont_hi = sg * (sb - e), s
        else:
            Y = np.nan; mtm = sg * (c[last] - e)
            G = -mtm; ret = -mtm; cont = mtm
            G_lo = G_hi = G; ret_lo = ret_hi = ret; cont_lo = cont_hi = cont
        rows.append(dict(base, status=status, p0=p0, Y=Y, X=(Y - p0) if np.isfinite(Y) else np.nan, G=G, G_lo=G_lo, G_hi=G_hi,
                         ret=ret, ret_lo=ret_lo, ret_hi=ret_hi, cont=cont, cont_lo=cont_lo, cont_hi=cont_hi,
                         minutes_held=(k_end if k_end is not None else last) - q))
    return pd.DataFrame(rows)


def clustered(v, days):
    v = np.asarray(v, float); ok = np.isfinite(v); v, days = v[ok], np.asarray(days)[ok]
    n = len(v)
    if n < 2: return dict(n=n, mean=float(v.mean()) if n else None, t=None, days=int(len(np.unique(days))))
    mu = v.mean()
    s = pd.Series(v - mu).groupby(days).sum().to_numpy()
    se = math.sqrt((s * s).sum()) / n
    return dict(n=int(n), mean=round(float(mu), 4), t=round(float(mu / se), 2) if se > 0 else None, days=int(len(np.unique(days))))


def contrast(A, B, col):
    a, b = A[col].to_numpy(float), B[col].to_numpy(float)
    ma, mb = np.isfinite(a), np.isfinite(b)
    if ma.sum() < 2 or mb.sum() < 2: return None
    mua, mub = a[ma].mean(), b[mb].mean()
    psi = pd.concat([pd.Series((a[ma] - mua) / ma.sum(), index=A.date.to_numpy()[ma]),
                     pd.Series(-(b[mb] - mub) / mb.sum(), index=B.date.to_numpy()[mb])])
    se = math.sqrt((psi.groupby(level=0).sum() ** 2).sum())
    return dict(diff=round(float(mua - mub), 4), t=round(float((mua - mub) / se), 2) if se > 0 else None)


def block(D):
    res = dict(episodes=int(len(D)), status={k: int(v) for k, v in D.status.value_counts().items()})
    F_ = D[D.status.isin(["b_first", "a_first", "same_minute", "censored_horizon", "censored_window", "censored_gap"])]
    days = F_.date.to_numpy()
    res["fork_episodes"] = int(len(F_))
    if not len(F_): return res
    res["prefix"] = {k: round(float(D[k].median()), 2) for k in ("minutes_t0_to_q", "run_t0_q", "M_minus_b", "close_q_from_b", "zone_age_d")}
    res["prefix"]["share_touched_b_before_q"] = round(float(D.touched_b_before_q.mean()), 3)
    res["remaining"] = dict(d_median=round(float(F_.d.median()), 2), s_median=round(float(F_.s.median()), 2),
                            p0_mean=round(float(F_.p0.mean()), 4), minutes_held_median=float(F_.minutes_held.median()))
    rs = F_[F_.status.isin(["b_first", "a_first"])]
    res["Y_mean_resolved"] = round(float(rs.Y.mean()), 4) if len(rs) else None
    res["p0_mean_resolved"] = round(float(rs.p0.mean()), 4) if len(rs) else None
    res["X_resolved"] = clustered(rs.X, rs.date)
    res["G_pts"] = clustered(F_.G, days)                      # same-minute excluded
    res["G_pts_lo"] = clustered(F_.G_lo, days); res["G_pts_hi"] = clustered(F_.G_hi, days)
    for side in ("ret", "cont"):
        net = F_[side] - COST; lo = F_[f"{side}_lo"] - COST; hi = F_[f"{side}_hi"] - COST
        res[f"{side}_net"] = dict(excl_same=clustered(net, days), adverse=clustered(lo, days), favorable=clustered(hi, days))
        stop = F_.s if side == "ret" else F_.d
        feas = (stop + COST <= LIMIT).to_numpy()
        res[f"{side}_net_contract"] = dict(share=round(float(feas.mean()), 3), adverse=clustered(lo[feas], days[feas]),
                                           favorable=clustered(hi[feas], days[feas]))
    return res


def main():
    inst = (sys.argv[1] if len(sys.argv) > 1 else "NQ").upper()
    OUT = HERE / "nsr1_out" / inst; OUT.mkdir(parents=True, exist_ok=True)
    F = Field(ROOT, inst); ts = np.asarray(F.market.close_ts_utc_ns)
    P = members(F)
    R = twins(representatives(P), ts)
    share_tw = float(R.twin_absorbed.mean())
    unit = "absorbing episode" if share_tw > TWIN_SHARE else "(t0_spine_pos, side)"
    res = dict(declaration="base/105/LOG.md NSR1", instrument=inst, rows=int(len(P)),
               clusters=int(P.groupby(["t0_spine_pos", "t0_exit_side"]).ngroups),
               clusters_without_eligible_close=int(P.groupby(["t0_spine_pos", "t0_exit_side"]).eligible.any().eq(False).sum()),
               representatives=int(len(R)), twin_share=round(share_tw, 4), unit=unit)
    print({k: v for k, v in res.items()}, flush=True)       # unit fixed before any outcome is read
    U = R[~R.twin_absorbed] if unit == "absorbing episode" else R
    D = fork(U, F)
    D.to_csv(OUT / "episodes.csv", index=False)
    D["epoch"] = D.date.map(lambda x: next(n for n, lo, hi in EPOCHS if lo <= x <= hi))
    res["by_state"] = {}
    for st_name, S in (("confirmed", D[D.confirmed]), ("flicker", D[~D.confirmed])):
        res["by_state"][st_name] = {"all": block(S)}
        for ep, _, _ in EPOCHS: res["by_state"][st_name][ep] = block(S[S.epoch == ep])
    res["confirmed_by_t0_kind"] = {k: block(g) for k, g in D[D.confirmed].groupby("t0_kind")}
    res["contrast_confirmed_minus_flicker_G"] = {}
    for ep, _, _ in EPOCHS + (("all", 0, 0),):
        A = D[D.confirmed] if ep == "all" else D[D.confirmed & (D.epoch == ep)]
        B = D[~D.confirmed] if ep == "all" else D[~D.confirmed & (D.epoch == ep)]
        res["contrast_confirmed_minus_flicker_G"][ep] = contrast(A, B, "G")
    # decision rule on confirmed episodes, both bounds of the unknown order
    C = res["by_state"]["confirmed"]
    verdict = {}
    for bound in ("G_pts_lo", "G_pts_hi"):
        g = {ep: C[ep][bound] for ep, _, _ in EPOCHS}
        sgn = np.sign(g["2020-25"]["mean"] or 0)
        verdict[bound] = bool(sgn != 0 and all(np.sign(g[ep]["mean"] or 0) == sgn for ep in g)
                              and all(abs(g[ep]["t"] or 0) >= 3 for ep in ("2013-19", "2020-25")))
    res["asymmetry"] = bool(all(verdict.values())); res["asymmetry_by_bound"] = verdict
    if res["asymmetry"]:
        side = "ret" if C["2020-25"]["G_pts_lo"]["mean"] > 0 else "cont"
        res["money_side"] = side
        res["money"] = bool(all((C[ep][f"{side}_net"]["adverse"]["mean"] or -1) > 0 and (C[ep][f"{side}_net"]["adverse"]["t"] or 0) >= 2
                                for ep in ("2013-19", "2020-25")))
    (OUT / "summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for st_name in ("confirmed", "flicker"):
        for ep in ("all",) + tuple(e for e, _, _ in EPOCHS):
            b = res["by_state"][st_name][ep]
            if not b.get("fork_episodes"): continue
            print(f"{st_name:9s} {ep:7s} ep {b['episodes']:5d} fork {b['fork_episodes']:5d} status {b['status']}")
            print(f"   prefix {b['prefix']} remaining {b['remaining']}")
            print(f"   Y {b['Y_mean_resolved']} p0 {b['p0_mean_resolved']} X {b['X_resolved']}")
            print(f"   G {b['G_pts']} lo {b['G_pts_lo']['mean']} (t {b['G_pts_lo']['t']}) hi {b['G_pts_hi']['mean']} (t {b['G_pts_hi']['t']})")
            for side in ("ret", "cont"):
                x = b[f"{side}_net"]; y = b[f"{side}_net_contract"]
                print(f"   {side:4s} net adv {x['adverse']['mean']} (t {x['adverse']['t']}) fav {x['favorable']['mean']} (t {x['favorable']['t']})"
                      f" | contract share {y['share']} adv {y['adverse']['mean']} (t {y['adverse']['t']})")
    print("contrast G confirmed - flicker:", res["contrast_confirmed_minus_flicker_G"])
    for k, b in res["confirmed_by_t0_kind"].items():
        print("confirmed", k, "fork", b.get("fork_episodes"), "G", b.get("G_pts"), "X", b.get("X_resolved"))
    print("asymmetry:", res["asymmetry"], res["asymmetry_by_bound"], "money:", res.get("money"), res.get("money_side"))


if __name__ == "__main__":
    main()
