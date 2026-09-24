"""Acceptance regressions for measurement invariants; no market fitting."""
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import f0_ab_probe as p

D=20250102


def tape(missing=(), final_close=100.):
    mods=np.array([m for m in range(544,960) if m not in missing])
    o=np.full(len(mods),100.); h=o+1; l=o-1; c=o.copy()
    pattern={599:(104.,106.,104.,105.),600:(106.,110.,106.,108.),
             601:(106.,109.,104.,105.),602:(105.,109.,105.,107.),603:(107.,109.,106.,108.)}
    for minute,vals in pattern.items():
        for arr,value in zip((o,h,l,c),vals): arr[mods==minute]=value
    c[mods==693]=final_close; h[mods==693]=max(101.,final_close+1)
    idx={int(m):i for i,m in enumerate(mods)}
    return SimpleNamespace(o=o,h=h,l=l,c=c,u=np.full(len(mods),2.),mod=mods,
        date=np.full(len(mods),D,dtype=np.int64),days={D:np.arange(len(mods))},
        at=lambda day,m:idx.get(m,-1) if day==D else -1,
        cal=pd.DataFrame([dict(date=D,type="REG",status="regular",close_mod=960,last=960)]).set_index("date"))


def stream(t):
    info={D:(None,np.full(391,102.),np.full(391,98.))}
    return info,p.market_stream(p.Market(t),t,D,info)


def test_parent_retains_prefix_when_future_exit_bar_is_missing():
    t=tape(missing=(693,)); t.cal.loc[D,"status"]="unknown"; t.cal.loc[D,"last"]=700
    assert p.parent_membership(p.Market(t),t,D)[:2]==("yes","ok")


def test_missing_prefix_is_unknown_not_no_opportunity():
    t=tape(missing=(560,))
    assert p.parent_membership(p.Market(t),t,D)[:2]==("unknown","missing_decision_prefix_bar:560")


def test_calendar_close_before_boundary_is_not_eligible():
    t=tape(); t.cal.loc[D,["type","status","close_mod"]]=["GOODFRI","short",555]
    assert p.parent_membership(p.Market(t),t,D)[0]=="no"


def test_prefix_denies_future_bar():
    with pytest.raises(AssertionError,match="future OHLC"):
        p.Market(tape()).prefix(D,573).bar(574)


def test_decision_does_not_read_next_open():
    class NoOpen(p.Market):
        def open(self,*args): raise AssertionError("open not yet available")
    t=tape(); m=NoOpen(t); op,_=p.op_s07(m,D)
    state,logs=p.replay(m,t,D,[op],None,"ok",True,through=574)
    assert state["status"]=="prefix"
    assert logs[-1]["phase"]=="decision"


def test_policy_trace_is_invariant_to_later_settlement():
    results=[]
    for t in (tape(),tape(final_close=105.)):
        _,s=stream(t)
        state,logs=p.replay(p.Market(t),t,D,*s,True,through=604)
        results.append((state,logs))
    assert results[0]==results[1]
    assert results[0][0]["pnl"]==0
    assert results[0][1][-1]["reason"]=="busy"


def test_only_authorization_differs_at_intervention():
    t=tape(); _,s=stream(t)
    _,a=p.replay(p.Market(t),t,D,*s,True,through=574)
    _,b=p.replay(p.Market(t),t,D,*s,False,through=574)
    assert a[0]["authorized"] and not b[0]["authorized"]
    assert {k:v for k,v in a[0].items() if k not in ("authorized","result","reason")}=={k:v for k,v in b[0].items() if k not in ("authorized","result","reason")}


@pytest.mark.parametrize("missing,reason",[(574,"missing_entry_open:574"),(580,"missing_open_position_bar:580")])
def test_missing_execution_or_position_is_unknown(missing,reason):
    t=tape(missing=(missing,)); m=p.Market(t); op,_=p.op_s07(m,D)
    a,_=p.replay(m,t,D,[op],None,"ok",True)
    b,_=p.replay(m,t,D,[op],None,"ok",False)
    assert a["status"]=="unknown" and a["reason"]==reason and np.isnan(a["net"])
    assert b["status"]=="resolved" and b["net"]==0


def test_settlement_precedes_next_boundary_decision():
    t=tape(); t.h[t.mod==600]=120.
    m=p.Market(t); op,_=p.op_s07(m,D)
    nextop=dict(op,oid="later",branch="PV2_V7m",rec=600,boundary=601,side=-1,
                stop_kind="absolute",stop=130.,iend=-1)
    _,log=p.replay(m,t,D,[op,nextop],None,"ok",True,through=601)
    assert log[-2]["event"]=="settlement"
    assert log[-1]["event"]=="opportunity" and not log[-1]["busy"]
    assert log[-1]["pnl"]==-19.75


def test_broken_guard_is_known_before_later_missing_confirmation():
    t=tape(missing=(601,)); t.l[t.mod==600]=101.
    up=np.full(391,102.); dn=np.full(391,98.)
    op,um,why=p.pv2_after_anchor(p.Market(t),t,D,t.at(D,599),599,up,dn,1)
    assert op is None and um is None and why=="guard_broken"


def test_independent_reference_matches_both_resolved_policies():
    t=tape(); _,s=stream(t)
    for auth in (True,False):
        result,events=p.replay(p.Market(t),t,D,*s,auth)
        ref=p.reference_replay(t,{D:s},auth)
        assert p.verify_branch(t,p.Market(t),D,result["status"],result["net"],events,ref,s)
        assert not p.verify_branch(t,p.Market(t),D,"resolved",result["net"]+1,events,ref,s)


@pytest.mark.parametrize("status,net,dates",[("unknown",np.nan,{D}),("resolved",999999.,set())])
def test_reconciliation_without_evidence_fails_closed(status,net,dates):
    days=pd.DataFrame([dict(date=D,p0_status="yes",A_status=status,A_reason="unexplained",RA=net)])
    rec=p.reconcile(days,pd.Series({D:123.}),dates)
    assert not rec.allowed.any()


def test_unknown_requires_a_real_missing_observation():
    t=tape(); m=p.Market(t)
    assert not p.unknown_witness(m,t,D,"missing_entry_open:574")
    assert not p.unknown_witness(m,t,D,"arbitrary_unknown")
    assert p.unknown_witness(p.Market(tape(missing=(574,))),t,D,"missing_entry_open:574")


def test_causality_suite_includes_policy_states():
    t=tape(); info,s=stream(t)
    result=p.causality_tests(t,info,{D:s})
    assert result["passed"] and result["policy_state_tests"]==2


def test_headroom_identity_and_paired_month_bootstrap():
    a=np.array([10.,-10.,8.,-4.]); b=np.zeros(4)
    r=p.bootstrap(a,b,50,np.array([20250102,20250103,20250203,20250204]))
    assert r["H"]==r["H_identity_from_D"]==3.5
    assert r["bootstrap_unit"]=="calendar_month" and r["bootstrap_blocks"]==2
