"""Executes FROZEN_TRANSITION.md. No barriers in the outcome."""
import numpy as np, pyarrow.parquet as pq
from pathlib import Path
import open_y as M
ROOT=Path('C:/Users/Admin/Claude/g3-market-research'); NS=60_000_000_000
SEARCH, HOR = 240, 30

def build(inst, tf):
    A,se=M.market(inst); ts=A['close_ts_utc_ns']
    rows=pq.read_table(ROOT/f'data/field/{inst}/cells/tf_{tf:04d}/passports.parquet',
        columns=['zone_bottom','zone_top','t0_spine_pos','t0_exit_side']).to_pylist()
    rows.sort(key=lambda r:r['t0_spine_pos']); out=[]
    for r in rows:
        t0=int(r['t0_spine_pos']); p=t0+2
        si=se['lookup'].get(int(A['session_id'][t0]))
        if si is None: continue
        an=int(se['session_open_utc_ns'][si]); k=((int(ts[t0])-an)//NS-1)//tf; bg=an+k*tf*NS
        st=int(np.searchsorted(ts[:p+1],bg,side='right'))
        tw=np.asarray(ts[st:p+1])
        if len(tw)<3 or int(tw[0])!=bg+NS or np.any(np.diff(tw)!=NS): continue
        w=r['zone_top']-r['zone_bottom']
        if w<=0: continue
        north=r['t0_exit_side']=='north'; ex=r['zone_top'] if north else r['zone_bottom']
        hi=st+SEARCH
        if hi+HOR+1>=len(A['close']): continue
        seg=np.asarray(ts[p+1:hi])
        if len(seg)==0 or np.any(np.diff(seg)!=NS): continue
        sh=np.asarray(A['high'][p+1:hi],float); sl=np.asarray(A['low'][p+1:hi],float)
        olow=(sl-ex) if north else (ex-sh)
        h=np.where(olow<=0)[0]
        if len(h)==0: continue
        q=p+1+int(h[0])
        tp=np.asarray(ts[st:q+1])
        if len(tp)<4 or np.any(np.diff(tp)!=NS): continue
        cp=np.asarray(A['close'][st:q+1],float)
        hp=np.asarray(A['high'][st:q+1],float); lp=np.asarray(A['low'][st:q+1],float)
        if not (np.isfinite(cp).all() and np.isfinite(hp).all() and np.isfinite(lp).all()): continue
        d=np.diff(cp); sig=float(np.std(d))
        if len(d)<3 or sig<=0: continue
        # sigma of the 60 minutes before the interval, for the regime-ratio candidate
        a=st-60; sig_pre=np.nan
        if a>=0:
            tq=np.asarray(ts[a:st])
            if len(tq)==60 and not np.any(np.diff(tq)!=NS):
                cq=np.asarray(A['close'][a:st],float)
                if np.isfinite(cq).all(): sig_pre=float(np.std(np.diff(cq)))
        oc_pre=(cp-ex) if north else (ex-cp)
        oh=(hp-ex) if north else (ex-lp); ol=(lp-ex) if north else (ex-hp)
        e=(ol<=0)&(oh>=0); f=(ol<=-w)&(oh>=-w)
        loc=np.select([oc_pre<-w,oc_pre==-w,oc_pre<0,oc_pre==0],[-2,-1,0,1],default=2)
        reads=list(zip(e.astype(int),f.astype(int),loc.astype(int)))
        runs=sum(1 for i,v in enumerate(reads) if i==0 or v!=reads[i-1])
        n=float(len(cp))
        ft=np.asarray(ts[q:q+HOR+1])
        if len(ft)<HOR+1 or np.any(np.diff(ft)!=NS): continue
        cf=np.asarray(A['close'][q:q+HOR+1],float)
        if not np.isfinite(cf).all(): continue
        ocf=(cf-ex) if north else (ex-cf)
        rr=np.diff(ocf)/sig
        ac1=float(np.corrcoef(rr[:-1],rr[1:])[0,1]) if np.std(rr[:-1])>0 and np.std(rr[1:])>0 else np.nan
        out.append(dict(close_s=float(oc_pre[-1])/sig, depth_s=float(-ol[-1])/sig,
                        sigma=sig, sig_ratio=(sig/sig_pre if np.isfinite(sig_pre) and sig_pre>0 else np.nan),
                        runs_rate=runs/n, exit_share=float(e.sum())/n, far_share=float(f.sum())/n,
                        inside_share=float((loc==0).sum())/n, plen=n,
                        drift=float(ocf[-1]-ocf[0])/sig, ac1=ac1, volratio=float(np.std(rr)),
                        day=int(A['session_id'][q]),
                        date=str(np.datetime64(int(ts[q]),'ns'))[:10]))
    return out
