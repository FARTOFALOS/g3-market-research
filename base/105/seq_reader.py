"""RD1 — two-point sequential reader of F0 (declared in base/105/LOG.md before counting).

Point 1, 09:34 (bar 09:33 closed): take the morning S07 attempt or defer it.
Point 2, PV2 recognition (bar rec closed): take the pullback or skip it; asked only
when the position is free. "No trade today" is the trajectory (defer, skip).

The engine is the phase replay of f0_ab_probe.py with one change: authorization
of each opportunity is a policy call. Constant policies must reproduce A and B of
the probe on every P0 day before anything is fitted (hard assertion).
Model at both points: OLS with intercept on standardized features, target money,
take iff prediction > 0. Nested yearly walk-forward (pi2 first, then pi1).

Run from repository root: python -B base/105/seq_reader.py
"""
from __future__ import annotations

import json, math, os
from pathlib import Path

import numpy as np
import pandas as pd

import f0_ab_probe as P
from f0_ab_probe import (Market, SuffixMarket, cal_close, parent_membership, market_stream, materialize_entry,
                         band_state, event_signature, S07_BOUNDARY, LOCK, MIN_LIM, LIMIT, DAY_LIMIT, COST, SLIP, TICK)
from trade import Tape

HERE = Path(__file__).resolve().parent
OUT = HERE / "seq_reader_out"
Y0, Y1 = 20060101, 20251231
PI2_FIRST, PI1_FIRST = 2009, 2013
NORM_N, NORM_MIN = 20, 10
F1 = ["pre_range_rel", "side_strength", "overnight_move_rel", "prev_session_range_rel"]
F2 = ["day_pnl", "minutes_since_open", "risk_points", "day_range_rel"]
FALSIFIER_N = 256


# ---------------------------------------------------------------- engine
def replay_seq(M, T, D, ops, unknown_mod, unknown_reason, decide):
    """f0_ab_probe.replay with authorization delegated to decide(op, snap, M, D)."""
    pnl = 0.; morning = 0.; downstream = 0.; active = None; log = []
    close = cal_close(T, D)
    last_boundary = close if close is not None else 960
    by_boundary = {}
    for op in ops: by_boundary.setdefault(op["boundary"], []).append(op)

    def state():
        lim = min(LIMIT, DAY_LIMIT + pnl)
        if LOCK and pnl > 0: lim = min(lim, pnl)
        return dict(pnl=pnl, busy=active is not None,
                    position=None if active is None else active["op"]["oid"],
                    remaining_budget=DAY_LIMIT + pnl, lock_active=bool(LOCK and pnl > 0), limit=lim)

    def unknown(reason, boundary, phase, op=None):
        log.append(dict(date=D, event="unknown", boundary=boundary, phase=phase,
                        oid=None if op is None else op["oid"], reason=reason, **state()))
        return dict(status="unknown", reason=reason, net=np.nan, morning=morning,
                    downstream=downstream, unknown_boundary=boundary), log

    for boundary in range(S07_BOUNDARY, last_boundary + 1):
        if active is not None:
            op, e, st, last = active["op"], active["entry"], active["stop"], active["last"]
            m = boundary - 1; b = M.prefix(D, m).bar(m)
            if b is None: return unknown(f"missing_open_position_bar:{m}", boundary, "observe_close", op)
            side = op["side"]; px = None; why = None; xtype = None
            if side > 0 and b["l"] <= st: px = min(b["o"], st) - SLIP; why = "stop"; xtype = 1
            elif side < 0 and b["h"] >= st: px = max(b["o"], st) + SLIP; why = "stop"; xtype = 1
            elif np.isfinite(e["target"]) and side > 0 and b["h"] >= e["target"] + TICK:
                px = max(b["o"], e["target"]); why = "target"; xtype = 2
            elif np.isfinite(e["target"]) and side < 0 and b["l"] <= e["target"] - TICK:
                px = min(b["o"], e["target"]); why = "target"; xtype = 2
            elif m == last: px = b["c"]; why = "time"; xtype = 3
            if px is not None:
                before = pnl; net = side * (px - e["entry"]) - COST; pnl += net
                if op["branch"] == "S07": morning += net
                else: downstream += net
                active = None
                log.append(dict(date=D, event="settlement", phase="observe_close", boundary=boundary,
                                oid=op["oid"], branch=op["branch"], rec=op["rec"], side=side,
                                entry_mod=e["entry_mod"], entry=e["entry"], stop=st,
                                exit_mod=m, exit=px, xtype=xtype, net=net, reason=why,
                                day_before=before, day_after=pnl, **state()))
        if unknown_mod is not None and boundary == unknown_mod + 1:
            return unknown(f"detector:{unknown_reason}", boundary, "recognition")
        for op in by_boundary.get(boundary, []):
            snap = state()
            auth = bool(decide(op, snap, M, D))
            z = dict(date=D, event="opportunity", phase="decision", oid=op["oid"], branch=op["branch"],
                     rec=op["rec"], boundary=boundary, side=op["side"], authorized=auth,
                     day_before=pnl, busy_before=snap["busy"], **snap)
            z["result"] = "declined" if not auth else ("policy_skip" if active is not None else "authorized")
            z["reason"] = "policy_decline" if not auth else ("busy" if active is not None else "ok")
            log.append(z)
            if not auth or active is not None: continue
            e = materialize_entry(M, T, op)
            if e["status"] == "unknown": return unknown(e["reason"], boundary, "execution", op)
            ex = dict(date=D, event="execution", phase="execution", boundary=boundary, oid=op["oid"], branch=op["branch"], **state())
            if e["status"] == "no_trade":
                ex.update(result="no_trade", reason=e["reason"]); log.append(ex); continue
            if e["plan_loss"] > LIMIT:
                ex.update(result="policy_skip", reason="original_plan_loss_over_LIMIT"); log.append(ex); continue
            lim = state()["limit"]
            if lim < MIN_LIM:
                ex.update(result="policy_skip", reason="remaining_limit_below_min_lim"); log.append(ex); continue
            if close is None: return unknown("calendar_close_unknown", boundary, "execution", op)
            last = min(op["rec"] + op["hold"], close - 1)
            if op["iend"] >= 0: last = min(last, op["iend"])
            if boundary > last:
                ex.update(result="no_trade", reason="no_horizon_after_entry"); log.append(ex); continue
            st = e["stop"] if e["plan_loss"] <= lim else e["entry"] - op["side"] * (lim - COST - SLIP)
            active = dict(op=op, entry=e, stop=st, last=last)
            ex.update(result="opened", reason="ok", entry=e["entry"], stop=st, planned_last=last)
            log.append(ex)
    assert active is None, "position survived prescribed session end"
    return dict(status="resolved", reason="ok", net=pnl, morning=morning, downstream=downstream), log


def const(morning, pv2):
    return lambda op, snap, M, D: morning if op["branch"] == "S07" else pv2


# ---------------------------------------------------------------- state features
class Norms:
    """Per-date session quantities; a date's norm uses only the 20 dates before it."""
    def __init__(self, T):
        self.dates = sorted(T.days)
        self.pos = {d: k for k, d in enumerate(self.dates)}
        pre, sess, close = [], [], []
        for d in self.dates:
            ix = T.days[d]; md = T.mod[ix]; cm = int(T.close_mod[ix[0]])
            w = ix[(md >= 544) & (md <= 573)]
            pre.append(T.h[w].max() - T.l[w].min() if len(w) == 30 else np.nan)
            s = ix[(md >= 570) & (md < cm)]
            sess.append(T.h[s].max() - T.l[s].min() if len(s) > 0 and len(s) >= 0.8 * (cm - 570) else np.nan)
            last = ix[md == cm - 1]
            close.append(float(T.c[last[0]]) if len(last) else np.nan)
        self.pre, self.sess, self.close = np.array(pre), np.array(sess), np.array(close)

    def norm(self, arr, d):
        k = self.pos.get(d)
        if k is None: return np.nan
        v = arr[max(0, k - NORM_N):k]; v = v[np.isfinite(v)]
        return float(np.median(v)) if len(v) >= NORM_MIN else np.nan

    def prev(self, d):
        k = self.pos.get(d)
        return None if not k else self.dates[k - 1]


def features1(M, N, D):
    Pf = M.prefix(D, 573)
    bs = [Pf.bar(m) for m in range(544, 574)]
    if any(b is None for b in bs): return None
    hi = max(b["h"] for b in bs); lo = min(b["l"] for b in bs); c = bs[-1]["c"]
    mid = (hi + lo) / 2; side = 1 if c > mid else -1; r = hi - lo
    pv = N.prev(D)
    npre, nsess = N.norm(N.pre, D), N.norm(N.sess, D)
    if pv is None or not np.isfinite(npre) or not np.isfinite(nsess) or r <= 0: return None
    kp = N.pos[pv]
    nprev = N.norm(N.sess, pv)
    f = [r / npre, abs(c - mid) / (r / 2), side * (c - N.close[kp]) / nsess, N.sess[kp] / nprev]
    return f if all(np.isfinite(f)) else None


def features2(M, N, D, op, snap):
    rec = op["rec"]; Pf = M.prefix(D, rec)
    bs = [Pf.bar(m) for m in range(570, rec + 1)]
    nsess = N.norm(N.sess, D)
    if any(b is None for b in bs) or not np.isfinite(nsess): return None
    c = bs[-1]["c"]
    f = [snap["pnl"], rec - 570, op["side"] * (c - op["stop"]),
         (max(b["h"] for b in bs) - min(b["l"] for b in bs)) / nsess]
    return f if all(np.isfinite(f)) else None


# ---------------------------------------------------------------- model
class OLS:
    def __init__(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y, float)
        self.mu = X.mean(0); sd = X.std(0); self.sd = np.where(sd > 0, sd, 1.0)
        Z = np.c_[np.ones(len(X)), (X - self.mu) / self.sd]
        self.beta = np.linalg.lstsq(Z, y, rcond=None)[0]; self.n = len(y)

    def predict(self, x):
        return float(self.beta[0] + ((np.asarray(x, float) - self.mu) / self.sd) @ self.beta[1:])

    def to_dict(self):
        return dict(n=self.n, intercept=float(self.beta[0]), beta=[float(b) for b in self.beta[1:]],
                    mu=[float(v) for v in self.mu], sd=[float(v) for v in self.sd])


class Counter:
    def __init__(self): self.asked = 0; self.taken = 0; self.fallback = 0


def pi2_policy(model, N, morning, counter=None, pv2_default=True):
    """morning: bool or callable for the S07 decision; PV2 decided by model when the position is free."""
    def decide(op, snap, M, D):
        if op["branch"] == "S07":
            return morning(op, snap, M, D) if callable(morning) else morning
        if snap["busy"] or model is None: return pv2_default
        f = features2(M, N, D, op, snap)
        take = True if f is None else model.predict(f) > 0
        if counter is not None:
            counter.asked += 1; counter.taken += int(take); counter.fallback += int(f is None)
        return take
    return decide


def pi1_morning(model, N, counter=None):
    def decide(op, snap, M, D):
        f = features1(M, N, D)
        take = True if f is None else model.predict(f) > 0
        if counter is not None:
            counter.asked += 1; counter.taken += int(take); counter.fallback += int(f is None)
        return take
    return decide


# ---------------------------------------------------------------- falsifier
def prefix_falsifier(T, N, streams, rng):
    s07, pv2 = [], []
    for D, (ops, _, _) in streams.items():
        for op in ops: (s07 if op["branch"] == "S07" else pv2).append((D, op))
    fails, n = [], 0
    snap = dict(pnl=-7.25)
    for pool, after in ((s07, lambda op: 573), (pv2, lambda op: op["rec"])):
        pick = [pool[i] for i in rng.choice(len(pool), min(FALSIFIER_N, len(pool)), replace=False)]
        for D, op in pick:
            a = Market(T); b = SuffixMarket(T, D, after(op)); n += 1
            fa = features1(a, N, D) if op["branch"] == "S07" else features2(a, N, D, op, snap)
            fb = features1(b, N, D) if op["branch"] == "S07" else features2(b, N, D, op, snap)
            if fa != fb: fails.append((D, op["oid"]))
    return dict(tests=n, failures=fails, passed=not fails)


# ---------------------------------------------------------------- evaluation
def month_stats(x, dates):
    x = np.asarray(x, float); n = len(x)
    s = pd.Series(x).groupby(np.asarray(dates) // 100).sum().to_numpy(); k = len(s)
    se = math.sqrt(k / (k - 1) * ((s - s.mean()) ** 2).sum()) / n
    return dict(mean=float(x.mean()), se_month=float(se), t=float(x.mean() / se) if se > 0 else 0.0)


def contract(r, trades):
    r = np.asarray(r, float); traded = np.asarray(trades) > 0
    k = math.ceil(0.05 * traded.sum()) if traded.sum() else 0; tot = r.sum()
    top = np.sort(r[traded])[::-1][:k].sum() if k else 0.0
    return dict(mean=float(r.mean()), share_days_ge_0=float(np.mean(r >= 0)), days_with_trade=int(traded.sum()),
                top5pct_share=float(top / tot) if tot > 0 else None, worst_day=float(r.min()))


def main():
    os.chdir(HERE); OUT.mkdir(exist_ok=True)
    print("Loading tape...", flush=True)
    T = Tape(end="2026-01-01"); M = Market(T); info, _, _ = band_state(T); N = Norms(T)
    cal = T.cal.reset_index(); cal = cal[(cal.date >= Y0) & (cal.date <= Y1)]
    streams = {}
    for c in cal.itertuples():
        D = int(c.date); member, _, _ = parent_membership(M, T, D)
        if member == "yes": streams[D] = market_stream(M, T, D, info)
    print(f"P0 days {len(streams)}; regression of constant policies against the probe...", flush=True)
    bad = []
    for D, (ops, um, ur) in streams.items():
        for auth in (True, False):
            r0, e0 = P.replay(M, T, D, ops, um, ur, auth)
            r1, e1 = replay_seq(M, T, D, ops, um, ur, const(auth, True))
            same = r0["status"] == r1["status"] and (r0["status"] != "resolved" or abs(r0["net"] - r1["net"]) <= 1e-9) \
                and event_signature(e0) == event_signature(e1)
            if not same: bad.append((D, auth))
    if bad: raise AssertionError(f"constant policies do not reproduce the probe: {bad[:5]}")
    print("  reproduced A and B on every P0 day", flush=True)
    fals = prefix_falsifier(T, N, streams, np.random.default_rng(20260924))
    if not fals["passed"]: raise AssertionError(f"future access in features: {fals['failures'][:5]}")
    print(f"  prefix falsifier passed ({fals['tests']} decisions)", flush=True)

    year = lambda D: D // 10000
    print("Point 2 rows (both morning worlds)...", flush=True)
    rows2 = []
    for D, (ops, um, ur) in streams.items():
        for op in [o for o in ops if o["branch"] == "PV2_V7m"]:
            for world in (True, False):
                def dec(take):
                    return lambda o, snap, M_, D_: world if o["branch"] == "S07" else (take if o["oid"] == op["oid"] else True)
                rt, et = replay_seq(M, T, D, ops, um, ur, dec(True))
                rs, _ = replay_seq(M, T, D, ops, um, ur, dec(False))
                ev = [e for e in et if e["event"] == "opportunity" and e["oid"] == op["oid"]]
                if not ev or ev[0]["busy_before"]: continue
                snap = dict(pnl=ev[0]["pnl"])
                f = features2(M, N, D, op, snap)
                ok = rt["status"] == rs["status"] == "resolved"
                rows2.append(dict(date=D, year=year(D), world="take" if world else "defer", oid=op["oid"],
                                  label=rt["net"] - rs["net"] if ok else np.nan, resolved=ok,
                                  **{k: (f[i] if f else np.nan) for i, k in enumerate(F2)}))
    r2 = pd.DataFrame(rows2)
    usable2 = r2[r2.resolved & r2[F2].notna().all(1)]
    pi2 = {}
    for y in range(PI2_FIRST, 2026):
        tr = usable2[usable2.year < y]
        pi2[y] = OLS(tr[F2].to_numpy(), tr.label.to_numpy())

    print("Point 1 labels with out-of-past pi2...", flush=True)
    rows1 = []
    for D, (ops, um, ur) in streams.items():
        y = year(D)
        if y < PI2_FIRST: continue
        rt, _ = replay_seq(M, T, D, ops, um, ur, pi2_policy(pi2[y], N, True))
        rd, _ = replay_seq(M, T, D, ops, um, ur, pi2_policy(pi2[y], N, False))
        f = features1(M, N, D)
        ok = rt["status"] == rd["status"] == "resolved"
        rows1.append(dict(date=D, year=y, label=rt["net"] - rd["net"] if ok else np.nan, resolved=ok,
                          **{k: (f[i] if f else np.nan) for i, k in enumerate(F1)}))
    r1 = pd.DataFrame(rows1)
    usable1 = r1[r1.resolved & r1[F1].notna().all(1)]
    pi1 = {}
    for Y in range(PI1_FIRST, 2026):
        tr = usable1[usable1.year < Y]
        pi1[Y] = OLS(tr[F1].to_numpy(), tr.label.to_numpy())

    print("Sequential test replays 2013-2025...", flush=True)
    c1, c2 = Counter(), Counter()
    out = []
    for D, (ops, um, ur) in streams.items():
        Y = year(D)
        if Y < PI1_FIRST: continue
        pols = dict(pi=pi2_policy(pi2[Y], N, pi1_morning(pi1[Y], N, c1), c2),
                    A=const(True, True), B=const(False, True), M=const(True, False), Z=const(False, False),
                    pi1_take=pi2_policy(None, N, pi1_morning(pi1[Y], N)),
                    take_pi2=pi2_policy(pi2[Y], N, True), defer_pi2=pi2_policy(pi2[Y], N, False))
        row = dict(date=D, year=Y)
        for k, pol in pols.items():
            r, ev = replay_seq(M, T, D, ops, um, ur, pol)
            row[f"R_{k}"] = r["net"] if r["status"] == "resolved" else np.nan
            row[f"n_{k}"] = sum(e["event"] == "settlement" for e in ev)
            row[f"s_{k}"] = r["status"]
        out.append(row)
    days = pd.DataFrame(out)
    keys = ["pi", "A", "B", "M", "Z", "pi1_take", "take_pi2", "defer_pi2"]
    paired = days[days[[f"R_{k}" for k in keys]].notna().all(1)]

    summary = dict(declaration="base/105/LOG.md RD1", features=dict(point1=F1, point2=F2),
                   regression="constant policies reproduced probe A and B on all P0 days",
                   prefix_falsifier=dict(tests=fals["tests"], passed=fals["passed"]),
                   rows=dict(point2_total=len(r2), point2_usable=len(usable2), point1_total=len(r1), point1_usable=len(usable1),
                             test_days=len(days), paired_days=len(paired)),
                   decisions=dict(point1=vars(c1), point2=vars(c2)),
                   models=dict(pi2={y: m.to_dict() for y, m in pi2.items()}, pi1={Y: m.to_dict() for Y, m in pi1.items()}),
                   territories={})
    for name, lo, hi in (("2013-2019", 20130101, 20191231), ("2020-2025", 20200101, 20251231)):
        g = paired[(paired.date >= lo) & (paired.date <= hi)]
        terr = dict(days=len(g), policies={k: contract(g[f"R_{k}"], g[f"n_{k}"]) for k in keys}, pi_minus={})
        for k in ("A", "B", "Z", "M"):
            terr["pi_minus"][k] = month_stats(g.R_pi - g[f"R_{k}"], g.date)
        terr["components"] = dict(pi1_take_minus_A=month_stats(g.R_pi1_take - g.R_A, g.date),
                                  take_pi2_minus_A=month_stats(g.R_take_pi2 - g.R_A, g.date),
                                  defer_pi2_minus_B=month_stats(g.R_defer_pi2 - g.R_B, g.date))
        terr["by_year_pi_minus_A"] = {int(y): float((x.R_pi - x.R_A).mean()) for y, x in g.groupby("year")}
        terr["by_year_pi_minus_B"] = {int(y): float((x.R_pi - x.R_B).mean()) for y, x in g.groupby("year")}
        summary["territories"][name] = terr
    t20, t13 = summary["territories"]["2020-2025"]["pi_minus"], summary["territories"]["2013-2019"]["pi_minus"]
    if all(t20[k]["t"] >= 2 for k in ("A", "B", "Z")) and all(t13[k]["mean"] > 0 for k in ("A", "B", "Z")):
        verdict = "decision-edge (discovery)"
    elif any(t20[k]["t"] <= -2 for k in ("A", "B")):
        verdict = "worse than a constant policy"
    else:
        verdict = "not distinguished at the protocol's resolution"
    summary["verdict"] = verdict
    r2.to_csv(OUT / "point2_rows.csv", index=False); r1.to_csv(OUT / "point1_rows.csv", index=False)
    days.to_csv(OUT / "test_days.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(json.dumps(dict(verdict=verdict, rows=summary["rows"], decisions=summary["decisions"],
                          territories={k: dict(pi_minus=v["pi_minus"], components=v["components"],
                                               policies={p: v["policies"][p] for p in ("pi", "A", "B", "Z")})
                                       for k, v in summary["territories"].items()}),
                     ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    main()
