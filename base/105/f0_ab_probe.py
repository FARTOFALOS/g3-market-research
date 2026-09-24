"""F0 A/B counterfactual probe — first engineering stage, no reader/ML.

Frozen object (source commit 0cc0dee33af9ebff35c3e1c4778b9a77f93a8524):
obs1.py F0 = S07 + PV2_V7m_120, lock=True, min_lim=3.  Its effective
morning decision boundary is 09:34 ET: the bar opened 09:33 closes, the side is
known, then the historical model fills at the next bar open.  That boundary-fill
is preserved as a probe convention; real-world executability is not established.

The script rebuilds market opportunities from the tape (not from executed-trade
CSVs), replays A=authorize morning S07 and B=skip only that authorization,
propagates archive unknowns, reconciles A with the legacy calculation, runs
prefix-causality falsifiers, and reports D=R_A-R_B and descriptive H.

Run: python base/105/f0_ab_probe.py
"""
from __future__ import annotations

import argparse, hashlib, json, math, os, subprocess
from pathlib import Path
import numpy as np
import pandas as pd

from trade import Tape, COST, SLIP, TICK, LIMIT, DAY_LIMIT, epoch, simulate
from v0_s18 import bands
from composite import s07_signals, run as legacy_run

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE_F0_COMMIT = "0cc0dee33af9ebff35c3e1c4778b9a77f93a8524"
Y0, Y1 = 20060101, 20251231
LOCK, MIN_LIM, S07_CAP = True, 3.0, 18.5
S07_REC, S07_BOUNDARY, S07_END = 573, 574, 693
BOOT_N, BOOT_SEED = 2000, 1050934
ALLOWED_CORRECTIONS = [
    "remove future-dependent parent-population filtering",
    "retain calendar rows whose status reflects missing future tape",
    "separate opportunity from execution/outcome",
    "keep deterministic entry rejection as an event",
    "propagate archive uncertainty instead of numeric surrogate P&L",
    "interpret mod as bar-open minute and label S07 decision boundary 09:34",
    "book settlements only when their bar has closed; preserve policy decisions",
]


def jdefault(x):
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, Path): return str(x)
    raise TypeError(type(x).__name__)


def git_head():
    try: return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception: return None


def sha256(path):
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
        return h.hexdigest()
    except OSError: return None


class Market:
    """Phase boundary: detectors read closed bars; execution reads only next open."""
    def __init__(self, T): self.T = T
    def bar(self, D, m):
        i = self.T.at(D, m)
        if i < 0: return None
        return dict(i=i, o=float(self.T.o[i]), h=float(self.T.h[i]),
                    l=float(self.T.l[i]), c=float(self.T.c[i]))
    def open(self, D, m):
        i = self.T.at(D, m)
        return None if i < 0 else float(self.T.o[i])
    def prefix(self, D, closed_through):
        return Prefix(self, D, closed_through)


class Prefix:
    """Closed-bar view that makes future OHLC inaccessible by construction."""
    def __init__(self, market, D, closed_through):
        self.market, self.D, self.closed_through = market, D, closed_through
    def bar(self, m):
        if m > self.closed_through:
            raise AssertionError(f"future OHLC access: requested {m}, closed through {self.closed_through}")
        return self.market.bar(self.D, m)


class SuffixMarket(Market):
    """Changes only the continuation; used as a prefix-causality falsifier."""
    def __init__(self, T, D, after): super().__init__(T); self.D, self.after = D, after
    def z(self, m): return 137.0 + (m % 17) * 3.25
    def open(self, D, m):
        x = super().open(D, m)
        return x if x is None or D != self.D or m <= self.after else x + self.z(m)
    def bar(self, D, m):
        b = super().bar(D, m)
        if b is None or D != self.D or m <= self.after: return b
        z = self.z(m); b = b.copy()
        b.update(o=b["o"]+z, h=b["h"]+z+11, l=b["l"]+z-7, c=b["c"]+z+3)
        return b


def cal_close(T, D):
    if D not in T.cal.index or pd.isna(T.cal.loc[D, "close_mod"]): return None
    return int(T.cal.loc[D, "close_mod"])


def parent_membership(M, T, D):
    """Tape-derived calendar completeness is never an eligibility filter."""
    row = T.cal.loc[D]
    # REG's prescribed close is independent of its observed tape end/status.
    close = 960 if row.get("type") == "REG" else cal_close(T, D)
    if close is not None and close <= S07_BOUNDARY:
        return "no", "calendar_closed_before_decision", None
    op, reason = op_s07(M, D)
    if op is None:
        if row.get("status") == "closed":
            return "no", "calendar_closed", None
        return "unknown", reason, None
    if close is None:
        return "unknown", "calendar_decision_eligibility_unknown", None
    return "yes", "ok", op


def op_s07(M, D):
    """Opportunity exists from 30 closed bars 09:04..09:33 (close known 09:34)."""
    bs = []; P = M.prefix(D, S07_REC)
    for m in range(544, 574):
        b = P.bar(m)
        if b is None: return None, f"missing_decision_prefix_bar:{m}"
        bs.append(b)
    mid = (max(b["h"] for b in bs) + min(b["l"] for b in bs)) / 2
    side = 1 if bs[-1]["c"] > mid else -1
    return dict(oid=f"{D}:S07:573", date=D, branch="S07", rec=S07_REC,
                boundary=S07_BOUNDARY, side=side, stop_kind="offset", stop=S07_CAP,
                target=np.nan, iend=S07_END, hold=120, source=S07_REC, pivot=-1,
                max_read=S07_REC), "ok"


def band_state(T):
    info, UB, LB = bands(T)
    return {int(D):(K, up, dn) for D, K, up, dn in info}, UB, LB


def v7_anchor(M, T, D, up, dn, side):
    close = cal_close(T, D)
    for k in range(30, 361):
        m = 570 + k - 1
        if close is not None and m >= close: return None, None, "session_ended"
        bound = up[k] if side > 0 else dn[k]
        if not np.isfinite(bound): continue
        b = M.prefix(D, m).bar(m)
        if b is None: return None, m, f"missing_v7m_bar:{m}"
        if side * (b["c"] - bound) > 0: return (b["i"], m), None, "ok"
    return None, None, "no_anchor"


def pv2_after_anchor(M, T, D, ai, am, up, dn, side):
    close = cal_close(T, D); a = M.prefix(D, am).bar(am)
    if a is None: return None, am, f"missing_anchor_bar:{am}"
    extreme = a["h"] if side > 0 else a["l"]
    u0 = float(T.u[ai])
    if not np.isfinite(u0): return None, None, "anchor_scale_unavailable"
    for pm in range(am+1, am+31):
        if close is not None and pm+2 >= close: return None, None, "session_ended_before_confirmation"
        # A broken guard closes the recognizer on pm; later missing bars
        # cannot undo that already observable terminal fact.
        q = M.prefix(D, pm).bar(pm)
        if q is None: return None, pm, f"missing_pv2_recognizer_bar:{pm}"
        k = pm - 570 + 1; guard = up[k] if side > 0 else dn[k]
        if (side > 0 and q["l"] <= guard) or (side < 0 and q["h"] >= guard):
            return None, None, "guard_broken"
        P = M.prefix(D, pm+2)
        bars = {m:P.bar(m) for m in (pm-1, pm, pm+1, pm+2)}
        miss = [m for m,b in bars.items() if b is None]
        if miss: return None, min(miss), f"missing_pv2_recognizer_bar:{min(miss)}"
        q0,q,q1,q2 = bars[pm-1],bars[pm],bars[pm+1],bars[pm+2]
        if side > 0:
            ok = q["l"] < q0["l"] and q["l"] <= q1["l"] and q["l"] <= q2["l"] and extreme-q["l"] >= u0
        else:
            ok = q["h"] > q0["h"] and q["h"] >= q1["h"] and q["h"] >= q2["h"] and q["h"]-extreme >= u0
        if ok:
            rec = pm+2; st = (q["l"] if side > 0 else q["h"]) - side*TICK
            return dict(oid=f"{D}:PV2:{side:+d}:{rec}", date=D, branch="PV2_V7m",
                        rec=rec, boundary=rec+1, side=side, stop_kind="absolute", stop=float(st),
                        target=np.nan, iend=-1, hold=120, source=am, pivot=pm, max_read=rec), None, "ok"
        extreme = max(extreme,q["h"]) if side > 0 else min(extreme,q["l"])
    return None, None, "no_pivot"


def market_stream(M, T, D, info):
    ops=[]; s07, reason = op_s07(M,D)
    if s07 is None: return ops, S07_REC, reason
    ops.append(s07)
    if D not in info: return ops, None, "ok:no_v7m_band_state"
    _,up,dn = info[D]; unknown=[]
    for side in (1,-1):
        anchor, um, rs = v7_anchor(M,T,D,up,dn,side)
        if um is not None: unknown.append((um,rs)); continue
        if anchor is None: continue
        op, um, rs = pv2_after_anchor(M,T,D,anchor[0],anchor[1],up,dn,side)
        if um is not None: unknown.append((um,rs))
        elif op is not None: ops.append(op)
    ops.sort(key=lambda o:(o["rec"],o["branch"],o["side"]))
    if unknown:
        unknown.sort(); return ops, unknown[0][0], unknown[0][1]
    return ops, None, "ok"


def materialize_entry(M,T,op):
    m=op["boundary"]; close=cal_close(T,op["date"])
    if close is not None and m >= close: return dict(status="no_trade",reason="calendar_closed_before_entry")
    e=M.open(op["date"],m)
    if e is None: return dict(status="unknown",reason=f"missing_entry_open:{m}")
    st=e-op["side"]*op["stop"] if op["stop_kind"]=="offset" else op["stop"]
    risk=op["side"]*(e-st)
    if risk <= 0: return dict(status="no_trade",reason="idea_dead_at_entry")
    if np.isfinite(op["target"]) and op["side"]*(op["target"]-e) <= 0:
        return dict(status="no_trade",reason="target_already_passed")
    return dict(status="ready",reason="ok",entry_mod=m,entry=e,stop=st,target=op["target"],plan_loss=risk+COST+SLIP)


def position(M,T,op,e,stop_override=None):
    st=e["stop"] if stop_override is None else stop_override; side=op["side"]
    close=cal_close(T,op["date"])
    if close is None: return dict(status="unknown",reason="calendar_close_unknown")
    last=min(op["rec"]+op["hold"],close-1)
    if op["iend"] >= 0: last=min(last,op["iend"])
    if e["entry_mod"] > last: return dict(status="no_trade",reason="no_horizon_after_entry")
    for m in range(e["entry_mod"],last+1):
        b=M.bar(op["date"],m)
        if b is None: return dict(status="unknown",reason=f"missing_open_position_bar:{m}")
        if side>0:
            if b["l"] <= st:
                px=min(b["o"],st)-SLIP; return dict(status="trade",reason="stop",exit_mod=m,exit=px,xtype=1,net=px-e["entry"]-COST,stop=st)
            if np.isfinite(e["target"]) and b["h"] >= e["target"]+TICK:
                px=max(b["o"],e["target"]); return dict(status="trade",reason="target",exit_mod=m,exit=px,xtype=2,net=px-e["entry"]-COST,stop=st)
        else:
            if b["h"] >= st:
                px=max(b["o"],st)+SLIP; return dict(status="trade",reason="stop",exit_mod=m,exit=px,xtype=1,net=e["entry"]-px-COST,stop=st)
            if np.isfinite(e["target"]) and b["l"] <= e["target"]-TICK:
                px=min(b["o"],e["target"]); return dict(status="trade",reason="target",exit_mod=m,exit=px,xtype=2,net=e["entry"]-px-COST,stop=st)
        if m==last:
            return dict(status="trade",reason="time",exit_mod=m,exit=b["c"],xtype=3,
                        net=side*(b["c"]-e["entry"])-COST,stop=st)
    raise AssertionError("position fell through")


def replay(M,T,D,ops,unknown_mod,unknown_reason,authorize_morning,through=None):
    """Boundary phases: observe prior closed bar, decide, then read next open.

    No future exit time or final P&L enters the live policy state. `through`
    stops just after the decision phase of a boundary, for causal falsifiers.
    """
    pnl=0.; morning=0.; downstream=0.; active=None; log=[]
    close=cal_close(T,D)
    last_boundary=close if close is not None else 960
    by_boundary={}
    for op in ops: by_boundary.setdefault(op["boundary"],[]).append(op)

    def state():
        lim=min(LIMIT,DAY_LIMIT+pnl)
        if LOCK and pnl>0: lim=min(lim,pnl)
        return dict(pnl=pnl,busy=active is not None,
                    position=None if active is None else active["op"]["oid"],
                    remaining_budget=DAY_LIMIT+pnl,lock_active=bool(LOCK and pnl>0),limit=lim)

    def unknown(reason,boundary,phase,op=None):
        log.append(dict(date=D,event="unknown",boundary=boundary,phase=phase,
                        oid=None if op is None else op["oid"],reason=reason,**state()))
        return dict(status="unknown",reason=reason,net=np.nan,morning=morning,
                    downstream=downstream,unknown_boundary=boundary),log

    for boundary in range(S07_BOUNDARY,last_boundary+1):
        if active is not None:
            op,e,st,last=active["op"],active["entry"],active["stop"],active["last"]
            m=boundary-1; b=M.prefix(D,m).bar(m)
            if b is None: return unknown(f"missing_open_position_bar:{m}",boundary,"observe_close",op)
            side=op["side"]; px=None; why=None; xtype=None
            if side>0 and b["l"]<=st: px=min(b["o"],st)-SLIP; why="stop"; xtype=1
            elif side<0 and b["h"]>=st: px=max(b["o"],st)+SLIP; why="stop"; xtype=1
            elif np.isfinite(e["target"]) and side>0 and b["h"]>=e["target"]+TICK:
                px=max(b["o"],e["target"]); why="target"; xtype=2
            elif np.isfinite(e["target"]) and side<0 and b["l"]<=e["target"]-TICK:
                px=min(b["o"],e["target"]); why="target"; xtype=2
            elif m==last: px=b["c"]; why="time"; xtype=3
            if px is not None:
                before=pnl; net=side*(px-e["entry"])-COST; pnl+=net
                if op["branch"]=="S07": morning+=net
                else: downstream+=net
                active=None
                log.append(dict(date=D,event="settlement",phase="observe_close",boundary=boundary,
                                oid=op["oid"],branch=op["branch"],rec=op["rec"],side=side,
                                entry_mod=e["entry_mod"],entry=e["entry"],stop=st,
                                exit_mod=m,exit=px,xtype=xtype,net=net,reason=why,
                                day_before=before,day_after=pnl,**state()))
        if unknown_mod is not None and boundary==unknown_mod+1:
            return unknown(f"detector:{unknown_reason}",boundary,"recognition")
        for op in by_boundary.get(boundary,[]):
            auth=not(op["branch"]=="S07" and not authorize_morning)
            snap=state()
            z=dict(date=D,event="opportunity",phase="decision",oid=op["oid"],branch=op["branch"],
                   rec=op["rec"],boundary=boundary,side=op["side"],authorized=auth,
                   day_before=pnl,busy_before=snap["busy"],**snap)
            z["result"]="authorization_skip" if not auth else ("policy_skip" if active is not None else "authorized")
            z["reason"]="B_intervention" if not auth else ("busy" if active is not None else "ok")
            log.append(z)
            if through is not None and boundary>=through: continue
            if not auth or active is not None: continue
            e=materialize_entry(M,T,op)
            if e["status"]=="unknown": return unknown(e["reason"],boundary,"execution",op)
            ex=dict(date=D,event="execution",phase="execution",boundary=boundary,oid=op["oid"],branch=op["branch"],**state())
            if e["status"]=="no_trade":
                ex.update(result="no_trade",reason=e["reason"]); log.append(ex); continue
            if e["plan_loss"]>LIMIT:
                ex.update(result="policy_skip",reason="original_plan_loss_over_LIMIT"); log.append(ex); continue
            lim=state()["limit"]
            if lim<MIN_LIM:
                ex.update(result="policy_skip",reason="remaining_limit_below_min_lim"); log.append(ex); continue
            if close is None: return unknown("calendar_close_unknown",boundary,"execution",op)
            last=min(op["rec"]+op["hold"],close-1)
            if op["iend"]>=0: last=min(last,op["iend"])
            if boundary>last:
                ex.update(result="no_trade",reason="no_horizon_after_entry"); log.append(ex); continue
            st=e["stop"] if e["plan_loss"]<=lim else e["entry"]-op["side"]*(lim-COST-SLIP)
            active=dict(op=op,entry=e,stop=st,last=last)
            ex.update(result="opened",reason="ok",entry=e["entry"],stop=st,planned_last=last)
            log.append(ex)
        if through is not None and boundary>=through:
            return dict(status="prefix",reason="decision_phase",**state()),log
    assert active is None, "position survived prescribed session end"
    return dict(status="resolved",reason="ok",net=pnl,morning=morning,downstream=downstream),log


# Legacy reconstruction: intentionally reproduces the old future-filter/surrogate semantics for reconciliation only.
def legacy_v7(T,info):
    rows=[]
    for D,(K,up,dn) in info.items():
        for side in (1,-1):
            for k in range(30,361):
                i=int(K[k])
                if i<0 or not np.isfinite(up[k]): continue
                b=up[k] if side>0 else dn[k]
                if side*(T.c[i]-b)>0: rows.append((D,i,side)); break
    return rows


def legacy_pivot(T,UB,LB,i0,side,D):
    last=T.days[D][-1]; ext=T.h[i0] if side>0 else T.l[i0]
    for p in range(i0+1,min(i0+31,last-2)):
        if T.mod[p+2]-T.mod[i0] != p+2-i0: return None
        guard=UB[p] if side>0 else LB[p]
        if side>0:
            if T.l[p] <= guard: return None
            ok=T.l[p]<T.l[p-1] and T.l[p]<=T.l[p+1] and T.l[p]<=T.l[p+2] and ext-T.l[p]>=T.u[i0]
        else:
            if T.h[p] >= guard: return None
            ok=T.h[p]>T.h[p-1] and T.h[p]>=T.h[p+1] and T.h[p]>=T.h[p+2] and T.h[p]-ext>=T.u[i0]
        if ok:return p
        ext=max(ext,T.h[p]) if side>0 else min(ext,T.l[p])
    return None


def legacy_f0(T,info,UB,LB):
    pv=[]
    for D,i0,side in legacy_v7(T,info):
        p=legacy_pivot(T,UB,LB,i0,side,D)
        if p is not None: pv.append(dict(date=D,irec=p+2,side=side,stop=(T.l[p] if side>0 else T.h[p])-side*TICK,target=np.nan,iend=-1))
    ps=pd.DataFrame(pv,columns=["date","irec","side","stop","target","iend"])
    pv=simulate(T,ps,120) if len(ps) else pd.DataFrame()
    if len(pv): pv["branch"]="PV2_V7m"
    ss=s07_signals(T); s=simulate(T,ss) if len(ss) else pd.DataFrame()
    if len(s): s["branch"]="S07"
    sdates=set(map(int,ss.date.tolist())) if len(ss) else set()
    cols=["date","irec","side","stop","target","iend","branch","entry","exit","xtype","jexit","risk","plan_loss","net","fits"]
    parts=[]
    if len(s):parts.append(s[cols])
    if len(pv):parts.append(pv[cols])
    if not parts:return pd.Series(dtype=float),pd.DataFrame(),sdates
    tk=legacy_run(T,pd.concat(parts,ignore_index=True),lock=True,min_lim=MIN_LIM,y0=Y0,y1=Y1)
    return tk.groupby("date").net.sum(),tk,sdates


def fp(ops,upto):
    return [(o["branch"],o["rec"],o["boundary"],o["side"],o["stop_kind"],round(float(o["stop"]),10),o["iend"],o["source"],o["pivot"])
            for o in ops if o["rec"]<=upto]


def causality_tests(T,info,streams,max_pv=256):
    sf=[];pf=[];phase=[];cand=[]; sn=0; state_fail=[]; state_n=0
    for D,(ops,_,_) in streams.items():
        s=[o for o in ops if o["branch"]=="S07"]
        if s:
            alt,rs=op_s07(SuffixMarket(T,D,S07_REC),D); sn+=1
            if alt is None or fp(s,S07_REC)!=fp([alt],S07_REC): sf.append((D,rs))
        for o in ops:
            if o["max_read"]>o["rec"]:phase.append((D,o["oid"],o["max_read"],o["rec"]))
            if o["branch"]=="PV2_V7m":cand.append((D,o))
    if len(cand)>max_pv:
        ix=np.linspace(0,len(cand)-1,max_pv).round().astype(int); cand=[cand[i] for i in ix]
    for D,o in cand:
        alt,_,_=market_stream(SuffixMarket(T,D,o["rec"]),T,D,info)
        if fp(streams[D][0],o["rec"])!=fp(alt,o["rec"]):pf.append((D,o["oid"]))
        for auth in (True,False):
            s=streams[D]
            before,_=replay(Market(T),T,D,*s,auth,through=o["boundary"])
            after,_=replay(SuffixMarket(T,D,o["rec"]),T,D,*s,auth,through=o["boundary"])
            state_n+=1
            if before["status"]!=after["status"] or any(before.get(k)!=after.get(k) for k in ("pnl","position","busy","remaining_budget","limit")):
                state_fail.append((D,o["oid"],auth))
    return dict(s07_tests=sn,s07_failures=sf,pv2_tests=len(cand),pv2_failures=pf,phase_access_violations=phase,
                policy_state_tests=state_n,policy_state_failures=state_fail,
                passed=not(sf or pf or phase or state_fail),note="Substitution tests are falsifiers, not a universal proof.")


def bootstrap(ra,rb,nboot,dates=None):
    n=len(ra)
    if n<2:return dict(n=n,H=np.nan,H_lo=np.nan,H_hi=np.nan,mean_D=np.nan,D_lo=np.nan,D_hi=np.nan)
    H=float(np.mean(np.maximum(ra,rb))-max(np.mean(ra),np.mean(rb))); D=ra-rb
    rng=np.random.default_rng(BOOT_SEED); hb=[];db=[]
    # Calendar-month blocks preserve within-month dependence; A/B stay paired.
    if dates is None: blocks=[np.array([i]) for i in range(n)]
    else:
        months=np.asarray(dates,dtype=np.int64)//100
        blocks=[np.flatnonzero(months==m) for m in np.unique(months)]
    for _ in range(max(1,nboot)):
        i=np.concatenate([blocks[k] for k in rng.integers(0,len(blocks),len(blocks))]); a=ra[i];b=rb[i]
        hb.append(np.mean(np.maximum(a,b))-max(np.mean(a),np.mean(b))); db.append(np.mean(a-b))
    k=max(1,math.ceil(.05*n)); pos=np.maximum(D,0); neg=np.maximum(-D,0)
    return dict(n=n,H=H,H_lo=float(np.quantile(hb,.025)),H_hi=float(np.quantile(hb,.975)),
                bootstrap_unit="calendar_month" if dates is not None else "paired_day",bootstrap_blocks=len(blocks),bootstrap_replicates=nboot,
                mean_A=float(np.mean(ra)),mean_B=float(np.mean(rb)),mean_oracle=float(np.mean(np.maximum(ra,rb))),
                mean_D=float(np.mean(D)),D_lo=float(np.quantile(db,.025)),D_hi=float(np.quantile(db,.975)),
                median_D=float(np.median(D)),D_quantiles={str(q):float(np.quantile(D,q)) for q in [0,.01,.05,.25,.5,.75,.95,.99,1]},
                positive_mass_top5pct_paired_days=float(np.sort(pos)[-k:].sum()/pos.sum()) if pos.sum() else None,
                negative_mass_top5pct_paired_days=float(np.sort(neg)[-k:].sum()/neg.sum()) if neg.sum() else None,
                H_identity_from_D=float(min(np.mean(np.maximum(D,0)),np.mean(np.maximum(-D,0)))),
                positive_D_mass=float(np.maximum(D,0).sum()),negative_D_mass=float(np.maximum(-D,0).sum()),
                share_D_gt0=float(np.mean(D>0)),share_D_lt0=float(np.mean(D<0)),share_D_eq0=float(np.mean(D==0)))


def reference_replay(T,streams,authorize_morning):
    """Independent unchanged legacy execution/policy applied to corrected ops.

    Used only as a verifier. Its gap surrogates are NEVER reported as outcomes.
    """
    rows=[]
    for D,(ops,_,_) in streams.items():
        for op in ops:
            if op["branch"]=="S07" and not authorize_morning: continue
            i=T.at(D,op["rec"]); j=T.at(D,op["boundary"])
            if i<0 or j<0: continue
            st=float(T.o[j])-op["side"]*op["stop"] if op["stop_kind"]=="offset" else op["stop"]
            end=T.at(D,op["iend"]) if op["iend"]>=0 else -1
            rows.append(dict(date=D,irec=i,side=op["side"],stop=st,target=op["target"],
                             iend=end,branch=op["branch"],oid=op["oid"]))
    if not rows: return pd.DataFrame()
    return legacy_run(T,simulate(T,pd.DataFrame(rows)),lock=LOCK,min_lim=MIN_LIM,y0=Y0,y1=Y1)


def trade_signature(T,frame):
    return [(r.branch,int(T.mod[int(r.irec)]),int(r.side),float(r.entry),
             int(T.mod[int(r.jexit)]),float(r.exit),float(r.net)) for r in frame.itertuples()]


def event_signature(events):
    return [(e["branch"],e["rec"],e["side"],e["entry"],e["exit_mod"],e["exit"],e["net"])
            for e in events if e["event"]=="settlement"]


def signatures_equal(a,b):
    return len(a)==len(b) and all(x[:3]==y[:3] and np.allclose(x[3:],y[3:],rtol=0,atol=1e-9)
                                 for x,y in zip(a,b))


def unknown_witness(M,T,D,reason,stream=None):
    if reason=="calendar_close_unknown": return cal_close(T,D) is None
    if reason=="calendar_decision_eligibility_unknown":
        return cal_close(T,D) is None and op_s07(M,D)[0] is not None
    if reason.startswith("detector:"):
        if stream is None or stream[1] is None or reason!="detector:"+stream[2]: return False
        reason=reason.removeprefix("detector:")
    prefixes=("missing_decision_prefix_bar:","missing_entry_open:","missing_open_position_bar:",
              "missing_v7m_bar:","missing_anchor_bar:","missing_pv2_recognizer_bar:")
    if not reason.startswith(prefixes): return False
    minute=int(reason.rsplit(":",1)[1])
    if reason.startswith("missing_decision_prefix_bar:") and not 544<=minute<=573: return False
    return M.bar(D,minute) is None


def verify_branch(T,M,D,status,net,events,reference,stream):
    actual=event_signature(events)
    if status=="resolved":
        expected=trade_signature(T,reference)
        return signatures_equal(actual,expected) and abs(sum(x[-1] for x in actual)-net)<=1e-9
    if status!="unknown": return False
    missing=[e for e in events if e["event"]=="unknown"]
    if len(missing)!=1: return False
    e=missing[0]
    if not unknown_witness(M,T,D,e["reason"],stream): return False
    # Every already settled, unaffected trade must match the independent engine.
    known=reference[(reference.xtype!=4) & (reference.jexit.map(lambda i:int(T.mod[int(i)])+1)<=e["boundary"])] if len(reference) else reference
    return signatures_equal(actual,trade_signature(T,known))


def reconcile(days,old,sdates,context=None):
    rows=[]
    for r in days.itertuples():
        D=int(r.date); ov=float(old.loc[D]) if D in old.index else np.nan; nv=float(r.RA) if r.A_status=="resolved" else np.nan
        cat,ok="unverified_difference",False; va=vb=False; detail="missing_verification_context"
        if context is not None:
            T,M=context["T"],context["M"]
            status,why,_=parent_membership(M,T,D)
            if r.p0_status in ("no","unknown"):
                ok=status==r.p0_status and why==r.p0_reason
                cat="calendar_not_eligible" if status=="no" else "parent_membership_unknown"
                detail=why
            else:
                ea=context["events"].get((D,"A"),[]); eb=context["events"].get((D,"B"),[])
                empty=pd.DataFrame()
                ra=context["ref_a"].get(D,empty); rb=context["ref_b"].get(D,empty)
                raw=context["old_trades"].get(D,empty); stream=context["streams"].get(D)
                va=verify_branch(T,M,D,r.A_status,r.RA,ea,ra,stream)
                vb=verify_branch(T,M,D,r.B_status,r.RB,eb,rb,stream)
                if r.A_status=="unknown": va=va and any(e.get("reason")==r.A_reason for e in ea if e["event"]=="unknown")
                if r.B_status=="unknown": vb=vb and any(e.get("reason")==r.B_reason for e in eb if e["event"]=="unknown")
                raw_match=signatures_equal(event_signature(ea),trade_signature(T,raw))
                if not (status=="yes" and va and vb):
                    cat="execution_or_unknown_evidence_failed"; detail=f"A_verified={va}; B_verified={vb}"
                elif r.A_status=="unknown":
                    cat="old_numeric_outcome_to_unknown" if np.isfinite(ov) else "new_A_unknown"
                    ok=True; detail=r.A_reason
                elif raw_match and (abs((ov if np.isfinite(ov) else 0.)-nv)<=1e-9):
                    cat="exact_reproduction"; ok=True; detail="full_trade_trace_and_independent_execution_match"
                elif D not in sdates and T.at(D,S07_END)<0 and r.B_status=="resolved" and signatures_equal(event_signature(eb),trade_signature(T,raw)):
                    cat="returned_future_filtered_morning"; ok=True
                    detail="missing_legacy_future_exit_bar; corrected_A_verified; B_exactly_reproduces_legacy"
                else:
                    cat="unexplained_or_unpermitted_semantic_difference"; detail="independent_execution_matches_but_raw_legacy_difference_not_authorized"
        rows.append(dict(date=D,old_A=ov,new_A=nv,category=cat,allowed=ok,
                         A_verified=va,B_verified=vb,evidence=detail,A_status=r.A_status,A_reason=r.A_reason))
    return pd.DataFrame(rows)


def passport():
    files=["f0_ab_probe.py","test_f0_ab_probe.py","tape.py","trade.py","v0_batch1.py","v0_s18.py","pivot2.py","composite.py","obs1.py","calendar_nq.csv"]
    return dict(object="F0 probe: S07 + PV2_V7m_120, lock=True, min_lim=3",f0_source_commit=SOURCE_F0_COMMIT,
                runtime_git_head=git_head(),decision_boundary="09:34 ET after close of bar opened 09:33",
                execution_convention="next minute open; boundary-fill convention; executability not established",
                s07=dict(recognition_bar_mod=S07_REC,decision_boundary_mod=S07_BOUNDARY,stop_distance=S07_CAP,time_exit_bar_mod=S07_END),
                policy=dict(cost=COST,stop_slippage=SLIP,trade_limit=LIMIT,day_limit=DAY_LIMIT,lock=LOCK,min_lim=MIN_LIM,one_position=True),
                territory=[Y0,Y1],data_origin=dict(tape="data/forward/market/NQ/*.npy",source_gate="base/091/source_gate_forward_2026-09-21.json",calendar="base/105/calendar_nq.csv"),
                allowed_measurement_corrections=ALLOWED_CORRECTIONS,file_sha256={f:sha256(HERE/f) for f in files},
                calendar_note="Frozen inherited calendar conventions; full row census retained, completeness/status unknown does not exclude a valid regular-day prefix. Historical schedule provenance limitations remain.",
                non_goals=["G_S","ML","feature selection","policy optimization","real-world execution validation"])


def run_probe(outdir,max_pv_tests=256,nboot=BOOT_N):
    outdir=outdir.resolve(); os.chdir(HERE); outdir.mkdir(parents=True,exist_ok=True)
    print("Loading tape and frozen band state...",flush=True)
    T=Tape(end="2026-01-01"); M=Market(T); info,UB,LB=band_state(T)
    frozen_passport=passport()
    data=ROOT/"data/forward/market/NQ"
    manifest=json.loads((data/"manifest.json").read_text(encoding="utf-8"))
    identities={name:sha256(data/name) for name in ("close_ts_utc_ns.npy","open.npy","high.npy","low.npy","close.npy")}
    if any(v!=manifest["array_sha256"][k] for k,v in identities.items()): raise AssertionError("market array identity mismatch")
    frozen_passport["data_identity"]=dict(corpus_id=manifest["corpus_id"],array_sha256=identities,source_gate_sha256=sha256(ROOT/"base/091/source_gate_forward_2026-09-21.json"))
    (outdir/"passport.json").write_text(json.dumps(frozen_passport,ensure_ascii=False,indent=2),encoding="utf-8")
    cal=T.cal.reset_index(); cal=cal[(cal.date>=Y0)&(cal.date<=Y1)]
    rows=[];opsout=[];events=[];streams={}
    event_map={}
    print(f"Replaying {len(cal)} calendar dates...",flush=True)
    for count,c in enumerate(cal.itertuples(),1):
        D=int(c.date); member,reason,s=parent_membership(M,T,D)
        if member!="yes":
            st="not_in_P0" if member=="no" else "unknown"
            rows.append(dict(date=D,p0_status=member,p0_reason=reason,side=np.nan,A_status=st,A_reason=reason,RA=np.nan,B_status=st,B_reason=reason,RB=np.nan,morning_A=np.nan,downstream_A=np.nan,downstream_B=np.nan));continue
        ops,um,ur=market_stream(M,T,D,info); streams[D]=(ops,um,ur);opsout+=ops
        A,ea=replay(M,T,D,ops,um,ur,True);B,eb=replay(M,T,D,ops,um,ur,False)
        event_map[(D,"A")]=ea; event_map[(D,"B")]=eb
        for e in ea:e["policy"]="A";events.append(e)
        for e in eb:e["policy"]="B";events.append(e)
        rows.append(dict(date=D,p0_status="yes",p0_reason="ok",side=s["side"],A_status=A["status"],A_reason=A["reason"],RA=A["net"],B_status=B["status"],B_reason=B["reason"],RB=B["net"],morning_A=A["morning"],downstream_A=A["downstream"],downstream_B=B["downstream"]))
        if count%1000==0: print(f"  {count}/{len(cal)} dates",flush=True)
    days=pd.DataFrame(rows);days["epoch"]=days.date.map(epoch)
    pair=(days.p0_status=="yes")&(days.A_status=="resolved")&(days.B_status=="resolved")
    days["D"]=np.where(pair,days.RA-days.RB,np.nan)
    days["D_decomp"]=np.where(pair,days.morning_A+(days.downstream_A-days.downstream_B),np.nan)
    if ((days.loc[pair,"D"]-days.loc[pair,"D_decomp"]).abs()>1e-9).any():raise AssertionError("D decomposition failed")
    print("Independent legacy execution and reconciliation...",flush=True)
    old,oldtk,sdates=legacy_f0(T,info,UB,LB)
    refa=reference_replay(T,streams,True); refb=reference_replay(T,streams,False)
    groups=lambda df:dict(tuple(df.groupby("date"))) if len(df) else {}
    context=dict(T=T,M=M,events=event_map,streams=streams,ref_a=groups(refa),ref_b=groups(refb),old_trades=groups(oldtk))
    rec=reconcile(days,old,sdates,context);rpass=bool(rec.allowed.all())
    print("Continuation and policy-state falsifiers...",flush=True)
    tests=causality_tests(T,info,streams,max_pv_tests); p=days[pair]
    passed=bool(tests["passed"] and rpass)
    head=bootstrap(p.RA.to_numpy(float),p.RB.to_numpy(float),nboot,p.date) if passed else None
    p0=days[days.p0_status=="yes"]
    epochs={}
    for ep,g in days.groupby("epoch"):
        gp=g[g.p0_status=="yes"]; pp=g[(g.p0_status=="yes")&(g.A_status=="resolved")&(g.B_status=="resolved")]
        epochs[ep]=dict(P0=len(gp),P_AB=len(pp),membership_unknown=int((g.p0_status=="unknown").sum()),
                       A_unknown_reasons=gp.loc[gp.A_status=="unknown","A_reason"].value_counts().to_dict(),
                       B_unknown_reasons=gp.loc[gp.B_status=="unknown","B_reason"].value_counts().to_dict(),
                       headroom=bootstrap(pp.RA.to_numpy(float),pp.RB.to_numpy(float),nboot,pp.date) if passed else None,
                       mean_morning_A=float(pp.morning_A.mean()),mean_downstream_A=float(pp.downstream_A.mean()),mean_downstream_B=float(pp.downstream_B.mean()))
    summary=dict(passport=frozen_passport,counts=dict(calendar_dates=len(days),calendar_decision_dates=int((days.p0_status!="no").sum()),P0_known=int((days.p0_status=="yes").sum()),P0_membership_unknown=int((days.p0_status=="unknown").sum()),P_A=int(((days.p0_status=="yes")&(days.A_status=="resolved")).sum()),P_B=int(((days.p0_status=="yes")&(days.B_status=="resolved")).sum()),P_AB=int(pair.sum())),
                 resolution=dict(A=float((p0.A_status=="resolved").mean()) if len(p0) else np.nan,B=float((p0.B_status=="resolved").mean()) if len(p0) else np.nan,AB=float(pair.sum()/len(p0)) if len(p0) else np.nan,A_unknown_reasons=p0.loc[p0.A_status=="unknown","A_reason"].value_counts().to_dict(),B_unknown_reasons=p0.loc[p0.B_status=="unknown","B_reason"].value_counts().to_dict()),
                 by_epoch=epochs,headroom_resolved=head,headroom_scope="conditional on P_AB; all-P0 headroom is not established without justified bounds for unresolved days",epsilon="not set; no practical-smallness verdict is allowed",continuation_substitution=tests,reconciliation=dict(passed=rpass,categories=rec.category.value_counts().to_dict(),forbidden_differences=int((~rec.allowed).sum())),stage1_passed=passed)
    days.to_csv(outdir/"f0_ab_days.csv",index=False);pd.DataFrame(opsout).to_csv(outdir/"f0_ab_opportunities.csv",index=False);pd.DataFrame(events).to_csv(outdir/"f0_ab_events.csv",index=False);rec.to_csv(outdir/"f0_ab_reconciliation.csv",index=False)
    if len(oldtk):oldtk.to_csv(outdir/"legacy_f0_trades_rebuilt.csv",index=False)
    with open(outdir/"f0_ab_summary.json","w",encoding="utf-8") as f:json.dump(summary,f,ensure_ascii=False,indent=2,allow_nan=True,default=jdefault)
    if not tests["passed"]:raise AssertionError("prefix-causality tests failed; see f0_ab_summary.json")
    if not rpass:raise AssertionError("non-permitted reconciliation differences; see f0_ab_reconciliation.csv")
    return summary


def main():
    a=argparse.ArgumentParser();a.add_argument("--out",default=str(HERE/"f0_ab_probe_out"));a.add_argument("--pv-tests",type=int,default=256);a.add_argument("--bootstrap",type=int,default=BOOT_N);x=a.parse_args()
    s=run_probe(Path(x.out),x.pv_tests,x.bootstrap)
    print(json.dumps(dict(stage1_passed=s["stage1_passed"],counts=s["counts"],resolution={k:v for k,v in s["resolution"].items() if not k.endswith("reasons")},headroom_resolved=s["headroom_resolved"],reconciliation=s["reconciliation"],continuation_substitution={k:s["continuation_substitution"][k] for k in ("passed","s07_tests","pv2_tests")}),ensure_ascii=False,indent=2,allow_nan=True,default=jdefault))


if __name__=="__main__":main()
