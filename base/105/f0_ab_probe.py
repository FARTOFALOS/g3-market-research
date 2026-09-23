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
    "separate opportunity from execution/outcome",
    "keep deterministic entry rejection as an event",
    "propagate archive uncertainty instead of numeric surrogate P&L",
    "interpret mod as bar-open minute and label S07 decision boundary 09:34",
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


def op_s07(M, D):
    """Opportunity exists from 30 closed bars 09:04..09:33 (close known 09:34)."""
    bs = []
    for m in range(544, 574):
        b = M.bar(D, m)
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
        b = M.bar(D, m)
        if b is None: return None, m, f"missing_v7m_bar:{m}"
        if side * (b["c"] - bound) > 0: return (b["i"], m), None, "ok"
    return None, None, "no_anchor"


def pv2_after_anchor(M, T, D, ai, am, up, dn, side):
    close = cal_close(T, D); a = M.bar(D, am)
    if a is None: return None, am, f"missing_anchor_bar:{am}"
    extreme = a["h"] if side > 0 else a["l"]
    u0 = float(T.u[ai])
    if not np.isfinite(u0): return None, None, "anchor_scale_unavailable"
    for pm in range(am+1, am+31):
        if close is not None and pm+2 >= close: return None, None, "session_ended_before_confirmation"
        bars = {m:M.bar(D,m) for m in (pm-1, pm, pm+1, pm+2)}
        miss = [m for m,b in bars.items() if b is None]
        if miss: return None, min(miss), f"missing_pv2_recognizer_bar:{min(miss)}"
        q0,q,q1,q2 = bars[pm-1],bars[pm],bars[pm+1],bars[pm+2]
        k = pm - 570 + 1; guard = up[k] if side > 0 else dn[k]
        if side > 0:
            if np.isfinite(guard) and q["l"] <= guard: return None, None, "guard_broken"
            ok = q["l"] < q0["l"] and q["l"] <= q1["l"] and q["l"] <= q2["l"] and extreme-q["l"] >= u0
        else:
            if np.isfinite(guard) and q["h"] >= guard: return None, None, "guard_broken"
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


def replay(M,T,D,ops,unknown_mod,unknown_reason,authorize_morning):
    pnl=0.; busy=-1; morning=0.; downstream=0.; log=[]
    timeline=[(o["rec"],"op",o) for o in ops]
    if unknown_mod is not None: timeline.append((unknown_mod,"unknown",unknown_reason))
    timeline.sort(key=lambda x:(x[0],0 if x[1]=="op" else 1))
    for mod,kind,x in timeline:
        if kind=="unknown":
            log.append(dict(date=D,event="detector_unknown",mod=mod,reason=x))
            return dict(status="unknown",reason=f"detector:{x}",net=np.nan,morning=np.nan,downstream=np.nan),log
        op=x; auth=not(op["branch"]=="S07" and not authorize_morning)
        z=dict(date=D,event="opportunity",oid=op["oid"],branch=op["branch"],rec=op["rec"],boundary=op["boundary"],
               side=op["side"],authorized=auth,day_before=pnl,busy_before=busy)
        if not auth: z.update(result="authorization_skip",reason="B_intervention",day_after=pnl,busy_after=busy); log.append(z); continue
        if op["boundary"] <= busy: z.update(result="policy_skip",reason="busy",day_after=pnl,busy_after=busy); log.append(z); continue
        e=materialize_entry(M,T,op)
        if e["status"]=="unknown":
            z.update(result="unknown",reason=e["reason"]); log.append(z)
            return dict(status="unknown",reason=e["reason"],net=np.nan,morning=np.nan,downstream=np.nan),log
        if e["status"]=="no_trade": z.update(result="no_trade",reason=e["reason"],day_after=pnl,busy_after=busy); log.append(z); continue
        if e["plan_loss"] > LIMIT:
            z.update(result="policy_skip",reason="original_plan_loss_over_LIMIT",plan_loss=e["plan_loss"],day_after=pnl,busy_after=busy); log.append(z); continue
        lim=min(LIMIT,DAY_LIMIT+pnl)
        if LOCK and pnl>0: lim=min(lim,pnl)
        if lim<MIN_LIM: z.update(result="policy_skip",reason="remaining_limit_below_min_lim",limit=lim,day_after=pnl,busy_after=busy); log.append(z); continue
        stop=None
        if e["plan_loss"]>lim: stop=e["entry"]-op["side"]*(lim-COST-SLIP)
        q=position(M,T,op,e,stop)
        if q["status"]=="unknown":
            z.update(result="unknown",reason=q["reason"],entry=e["entry"]); log.append(z)
            return dict(status="unknown",reason=q["reason"],net=np.nan,morning=np.nan,downstream=np.nan),log
        if q["status"]=="no_trade": z.update(result="no_trade",reason=q["reason"],day_after=pnl,busy_after=busy); log.append(z); continue
        pnl+=q["net"]; busy=q["exit_mod"]
        if op["branch"]=="S07": morning+=q["net"]
        else: downstream+=q["net"]
        z.update(result="trade",reason=q["reason"],entry_mod=e["entry_mod"],entry=e["entry"],stop=q["stop"],
                 exit_mod=q["exit_mod"],exit=q["exit"],xtype=q["xtype"],net=q["net"],day_after=pnl,busy_after=busy)
        log.append(z)
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
    sf=[];pf=[];phase=[];cand=[]; sn=0
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
    return dict(s07_tests=sn,s07_failures=sf,pv2_tests=len(cand),pv2_failures=pf,phase_access_violations=phase,
                passed=not(sf or pf or phase),note="Substitution tests are falsifiers; phase-limited interfaces carry the main burden.")


def bootstrap(ra,rb,nboot):
    n=len(ra)
    if n<2:return dict(n=n,H=np.nan,H_lo=np.nan,H_hi=np.nan,mean_D=np.nan,D_lo=np.nan,D_hi=np.nan)
    H=float(np.mean(np.maximum(ra,rb))-max(np.mean(ra),np.mean(rb))); D=ra-rb
    rng=np.random.default_rng(BOOT_SEED); hb=[];db=[]
    for _ in range(max(1,nboot)):
        i=rng.integers(0,n,n); a=ra[i];b=rb[i]
        hb.append(np.mean(np.maximum(a,b))-max(np.mean(a),np.mean(b))); db.append(np.mean(a-b))
    return dict(n=n,H=H,H_lo=float(np.quantile(hb,.025)),H_hi=float(np.quantile(hb,.975)),
                mean_D=float(np.mean(D)),D_lo=float(np.quantile(db,.025)),D_hi=float(np.quantile(db,.975)),
                H_identity_from_D=float(min(np.mean(np.maximum(D,0)),np.mean(np.maximum(-D,0)))),
                positive_D_mass=float(np.maximum(D,0).sum()),negative_D_mass=float(np.maximum(-D,0).sum()),
                share_D_gt0=float(np.mean(D>0)),share_D_lt0=float(np.mean(D<0)),share_D_eq0=float(np.mean(D==0)))


def reconcile(days,old,sdates):
    rows=[]
    for r in days.itertuples():
        D=int(r.date); ov=float(old.loc[D]) if D in old.index else np.nan; nv=float(r.RA) if r.A_status=="resolved" else np.nan
        if r.p0_status=="unknown":cat,ok="parent_membership_unknown",True
        elif r.A_status=="unknown":cat,ok=("old_numeric_surrogate_to_unknown" if np.isfinite(ov) else "new_A_unknown"),True
        elif np.isfinite(ov) and abs(ov-nv)<=1e-9:cat,ok="exact_reproduction",True
        elif D not in sdates:cat,ok="future_filter_measurement_correction",True
        elif np.isfinite(ov):cat,ok="unexplained_numeric_difference",False
        else:cat,ok="old_missing_new_resolved",False
        rows.append(dict(date=D,old_A=ov,new_A=nv,category=cat,allowed=ok,A_status=r.A_status,A_reason=r.A_reason))
    return pd.DataFrame(rows)


def passport():
    files=["tape.py","trade.py","v0_s18.py","pivot2.py","composite.py","obs1.py","calendar_nq.csv"]
    return dict(object="F0 probe: S07 + PV2_V7m_120, lock=True, min_lim=3",f0_source_commit=SOURCE_F0_COMMIT,
                runtime_git_head=git_head(),decision_boundary="09:34 ET after close of bar opened 09:33",
                execution_convention="next minute open; boundary-fill convention; executability not established",
                s07=dict(recognition_bar_mod=S07_REC,decision_boundary_mod=S07_BOUNDARY,stop_distance=S07_CAP,time_exit_bar_mod=S07_END),
                policy=dict(cost=COST,stop_slippage=SLIP,trade_limit=LIMIT,day_limit=DAY_LIMIT,lock=LOCK,min_lim=MIN_LIM,one_position=True),
                territory=[Y0,Y1],data_origin=dict(tape="data/forward/market/NQ/*.npy",source_gate="base/091/source_gate_forward_2026-09-21.json",calendar="base/105/calendar_nq.csv"),
                allowed_measurement_corrections=ALLOWED_CORRECTIONS,file_sha256={f:sha256(HERE/f) for f in files},
                non_goals=["G_S","ML","feature selection","policy optimization","real-world execution validation"])


def run_probe(outdir,max_pv_tests=256,nboot=BOOT_N):
    os.chdir(HERE); outdir.mkdir(parents=True,exist_ok=True)
    T=Tape(); M=Market(T); info,UB,LB=band_state(T)
    cal=T.cal.reset_index(); cal=cal[(cal.date>=Y0)&(cal.date<=Y1)]
    cal=cal[cal.status.isin(["regular","short"])|((cal.status=="special")&cal["last"].notna())]
    rows=[];opsout=[];events=[];streams={}
    for c in cal.itertuples():
        D=int(c.date); close=None if pd.isna(c.close_mod) else int(c.close_mod)
        if close is not None and close<=S07_BOUNDARY:
            rows.append(dict(date=D,p0_status="no",p0_reason="calendar_closed_before_decision",side=np.nan,A_status="not_in_P0",A_reason="",RA=np.nan,B_status="not_in_P0",B_reason="",RB=np.nan,morning_A=np.nan,downstream_A=np.nan,downstream_B=np.nan));continue
        s,reason=op_s07(M,D)
        if s is None:
            rows.append(dict(date=D,p0_status="unknown",p0_reason=reason,side=np.nan,A_status="unknown",A_reason=reason,RA=np.nan,B_status="unknown",B_reason=reason,RB=np.nan,morning_A=np.nan,downstream_A=np.nan,downstream_B=np.nan));continue
        ops,um,ur=market_stream(M,T,D,info); streams[D]=(ops,um,ur);opsout+=ops
        A,ea=replay(M,T,D,ops,um,ur,True);B,eb=replay(M,T,D,ops,um,ur,False)
        for e in ea:e["policy"]="A";events.append(e)
        for e in eb:e["policy"]="B";events.append(e)
        rows.append(dict(date=D,p0_status="yes",p0_reason="ok",side=s["side"],A_status=A["status"],A_reason=A["reason"],RA=A["net"],B_status=B["status"],B_reason=B["reason"],RB=B["net"],morning_A=A["morning"],downstream_A=A["downstream"],downstream_B=B["downstream"]))
    days=pd.DataFrame(rows);days["epoch"]=days.date.map(epoch)
    pair=(days.p0_status=="yes")&(days.A_status=="resolved")&(days.B_status=="resolved")
    days["D"]=np.where(pair,days.RA-days.RB,np.nan)
    days["D_decomp"]=np.where(pair,days.morning_A+(days.downstream_A-days.downstream_B),np.nan)
    if ((days.loc[pair,"D"]-days.loc[pair,"D_decomp"]).abs()>1e-9).any():raise AssertionError("D decomposition failed")
    old,oldtk,sdates=legacy_f0(T,info,UB,LB);rec=reconcile(days,old,sdates);rpass=bool(rec.allowed.all())
    tests=causality_tests(T,info,streams,max_pv_tests); p=days[pair]
    head=bootstrap(p.RA.to_numpy(float),p.RB.to_numpy(float),nboot)
    p0=days[days.p0_status=="yes"]
    summary=dict(passport=passport(),counts=dict(calendar_decision_dates=int((days.p0_status!="no").sum()),P0_known=int((days.p0_status=="yes").sum()),P0_membership_unknown=int((days.p0_status=="unknown").sum()),P_A=int(((days.p0_status=="yes")&(days.A_status=="resolved")).sum()),P_B=int(((days.p0_status=="yes")&(days.B_status=="resolved")).sum()),P_AB=int(pair.sum())),
                 resolution=dict(A=float((p0.A_status=="resolved").mean()) if len(p0) else np.nan,B=float((p0.B_status=="resolved").mean()) if len(p0) else np.nan,AB=float(pair.sum()/len(p0)) if len(p0) else np.nan,A_unknown_reasons=p0.loc[p0.A_status=="unknown","A_reason"].value_counts().to_dict(),B_unknown_reasons=p0.loc[p0.B_status=="unknown","B_reason"].value_counts().to_dict()),
                 headroom_resolved=head,headroom_scope="conditional on P_AB; all-P0 headroom is not established without justified bounds for unresolved days",epsilon="not set; no practical-smallness verdict is allowed",continuation_substitution=tests,reconciliation=dict(passed=rpass,categories=rec.category.value_counts().to_dict(),forbidden_differences=int((~rec.allowed).sum())),stage1_passed=bool(tests["passed"] and rpass))
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
