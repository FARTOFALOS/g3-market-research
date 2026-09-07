"""Prefix equivalence and execution cases that distinguish timing and OCO errors."""
import numpy as np
import pandas as pd
from run import ROOT,HERE,OUT,M,market,recognition,decisions,execute,dump,sha

def execution_checks():
    ts=np.arange(9,dtype=np.int64)*M
    op=np.arange(100.,109.);op[1]=100.
    hi=op+1.;lo=op-.5;cl=op+.25
    hi[1]=101.5;lo[1]=99.5;cl[1]=101.25
    def f(o=op,h=hi,l=lo,c=cl,t=ts,q=np.array([0]),d=np.array([0]),stop_entry=True):
        n=len(q)
        return execute(t,o,h,l,c,q,np.full(n,99.),np.full(n,101.),d,np.full(n,8*M),3,.25,stop_entry)
    a=f()[0];assert a[0]==2 and a[3]==101.25 and a[4]==103.25 and a[5]==2 and a[2]==3
    a=f(o=200-op,h=200-lo,l=200-hi,c=200-cl)[0]
    assert a[5]==2 and a[9]==-1
    hh=hi.copy();ll=lo.copy();hh[1]=102.;ll[1]=98.
    a=f(h=hh,l=ll)[0];assert a[0]==1 and a[5]==-2.5 and a[9]==0 and a[10]==1
    oo=op.copy();oo[2]=98.;a=f(o=oo)[0];assert a[0]==1 and a[4]==98. and a[5]==-3.25
    hh[1]=101.;ll[1]=99.;assert f(h=hh,l=ll)[0,0]==0
    oo=op.copy();oo[1]=102.;hh=hi.copy();hh[1]=103.;ll=lo.copy();ll[1]=101.
    a=f(o=oo,h=hh,l=ll)[0];assert a[3]==102. and a[5]==1.25
    assert f(d=np.array([1]),stop_entry=False)[0,0]==-1
    broken=ts.copy();broken[1:]+=M;assert f(t=broken)[0,0]==-2
    broken=ts.copy();broken[2:]+=M;assert f(t=broken)[0,0]==-3
    a=f(q=np.array([0,1,3]),d=np.array([0,0,0]));assert a[1,0]==-4 and a[2,0]>0
    ll=lo.copy();ll[4]=90.;assert f(l=ll)[0,5]==2.
    return ['timed_trigger_fill','mirror_short','both_OCO_orders_same_bar',
        'gap_stop_worse_open','untriggered_expiry','gap_entry_actual_open',
        'close_entry_return_cancel','unknown_entry','unknown_path',
        'nonoverlap_and_release','future_low_ignored']

def prefixes():
    records=[]
    for ins in ['ES','NQ','YM']:
        allp=pd.read_parquet(OUT/f'{ins}_objects.parquet')
        p=allp.sort_values('riz_id').iloc[::max(1,len(allp)//250)].copy();m=market(ins)
        for k in [3,5]:
            full,_=recognition(ins,p,m,k)
            moments=np.sort(full.p.unique())
            for index in np.linspace(0,len(moments)-1,4,dtype=int):
                q=int(moments[index])
                for cutoff in [q-1,q,q+1]:
                    truncated,_=recognition(ins,p,m,k,cutoff=cutoff)
                    cols=['riz_id','p','box_low','box_high','age_native_bars']
                    a=full.loc[full.p<=cutoff,cols].sort_values(['p','riz_id']).reset_index(drop=True)
                    b=truncated[cols].sort_values(['p','riz_id']).reset_index(drop=True)
                    pd.testing.assert_frame_equal(a,b)
                    mc={key:value[:cutoff+1] for key,value in m.items()}
                    for entry in ['stop','close']:
                        aa=decisions(full,m,entry);aa=aa.loc[aa.q<=cutoff]
                        bb=decisions(truncated,mc,entry)
                        cols2=['riz_id','p','q','direction','riz_count']
                        pd.testing.assert_frame_equal(aa[cols2].reset_index(drop=True),bb[cols2].reset_index(drop=True))
                    records.append(dict(instrument=ins,k=k,cutoff=cutoff,pauses=len(a),equal=True))
        print('PREFIXES',ins,flush=True)
    return records

if __name__=='__main__':
    exec_cases=execution_checks();prefix=prefixes()
    result={'prefixes':prefix,'execution_cases':exec_cases,'code_sha256':sha(HERE/'run.py')}
    dump(HERE/'checks_before_outcomes.json',result)
    print('PASS',len(prefix),'prefix cutoffs; both entry modes;',len(exec_cases),'execution cases',flush=True)
