"""Actual truncated prefixes and conservative bracket execution for S-06 v2."""
import numpy as np
import pandas as pd
from reject import execute,decisions,ROOT,HERE,OUT,M,market,recognition,dump,sha

def executions():
    ts=np.arange(8,dtype=np.int64)*M;op=np.arange(99.,107.);hi=op+.5;lo=op-.5;cl=op+.25
    def f(o=op,h=hi,l=lo,c=cl,t=ts,target=True,optimistic=False):
        return execute(t,o,h,l,c,np.array([0]),np.array([99.]),np.array([103.]),np.array([1]),
            np.array([98.]),np.array([7*M]),3,.25,target,optimistic)[0]
    a=f(target=False);assert a[0]==2 and a[3]==100. and a[4]==102.25
    hh=hi.copy();hh[2]=103.25;a=f(h=hh);assert a[0]==3 and a[4]==103.
    hh[2]=103.;assert f(h=hh)[0]==2  # touch alone is not a limit fill
    hh[1]=103.25;ll=lo.copy();ll[1]=97.5
    a=f(h=hh,l=ll);b=f(h=hh,l=ll,optimistic=True)
    assert a[5]==-2.25 and b[5]==3. and a[12]==b[12]==1
    oo=op.copy();oo[2]=97.;a=f(o=oo);assert a[4]==97. and a[13]==1
    oo[1]=104.;assert f(o=oo)[0]==-1
    broken=ts.copy();broken[1:]+=M;assert f(t=broken)[0]==-2
    broken=ts.copy();broken[2:]+=M;assert f(t=broken)[0]==-3
    hh=hi.copy();hh[2]=103.25
    a=execute(ts,200-op,200-lo,200-hh,200-cl,np.array([0]),np.array([97.]),np.array([101.]),np.array([-1]),
        np.array([102.]),np.array([7*M]),3,.25,True)[0]
    assert a[5]==3. and a[4]==97.
    return ['clock_exit','target_trade_through','touch_not_fill','both_barriers_bounds',
            'stop_gap','entry_outside_cancel','missing_entry','missing_path','mirror_short']

def prefixes():
    records=[]
    for ins in ['ES','NQ','YM']:
        p=pd.read_parquet(OUT/f'{ins}_objects.parquet').sort_values('riz_id').iloc[::1800].copy();m=market(ins)
        for k in [3,5]:
            full,_=recognition(ins,p,m,k);signals=decisions(full,m)
            for i in np.linspace(0,len(signals)-1,2,dtype=int):
                q=int(signals.iloc[i].q)
                for cutoff in [q-1,q,q+1]:
                    rows,_=recognition(ins,p,m,k,cutoff=cutoff)
                    mc={key:value[:cutoff+1] for key,value in m.items()}
                    truncated=decisions(rows,mc)
                    cols=['riz_id','p','q','direction','probe_extreme','riz_count']
                    a=signals.loc[signals.q<=cutoff,cols].reset_index(drop=True)
                    pd.testing.assert_frame_equal(a,truncated[cols].reset_index(drop=True))
                    records.append(dict(instrument=ins,k=k,cutoff=cutoff,decisions=len(a),equal=True))
        print('REJECTION PREFIXES',ins,flush=True)
    return records

if __name__=='__main__':
    cases=executions();p=prefixes()
    dump(HERE/'checks_v2_before_outcomes.json',dict(prefixes=p,execution_cases=cases,code_sha256=sha(HERE/'reject.py')))
    print('PASS',len(p),'prefix cutoffs;',len(cases),'execution cases',flush=True)
