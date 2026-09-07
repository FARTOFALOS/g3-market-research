"""Targeted checks of actual prefix recognition and order execution, not general infrastructure."""
import json
import numpy as np
import pandas as pd
from run import ROOT,HERE,OUT,M,market,dump,execute,sha
from reversal import signal_rows,decide,replay


def prefix_checks():
    cal=pd.read_parquet(HERE/'calendar.parquet');records=[]
    for ins in ['ES','NQ','YM']:
        p=pd.read_parquet(OUT/f'{ins}_objects.parquet')
        sample=p.sort_values('riz_id').iloc[::max(1,len(p)//480)].copy()
        m=market(ins)
        for k in [2,3,4]:
            full=signal_rows(sample,m,cal,k)
            found=full.loc[full.recognition_status==1].sort_values('q')
            for idx in np.linspace(0,len(found)-1,5,dtype=int):
                q=int(found.iloc[idx].q)
                for cutoff in [q-1,q,q+1]:
                    truncated=signal_rows(sample,m,cal,k,cutoff=cutoff)
                    a=full.loc[(full.recognition_status==1)&(full.q<=cutoff),['riz_id','q','direction','low_so_far','high_so_far']].sort_values('riz_id').reset_index(drop=True)
                    b=truncated.loc[truncated.recognition_status==1,a.columns].sort_values('riz_id').reset_index(drop=True)
                    pd.testing.assert_frame_equal(a,b)
                    records.append(dict(instrument=ins,k=k,cutoff=cutoff,signals=len(a),equal=True))
        # Initial facts actually refer to observed candle positions, not another time axis.
        t=p.t0_spine_pos.to_numpy(np.int64)
        assert np.array_equal(m['close_ts_utc_ns'][t],p.t0_ts_ns.to_numpy())
        side=np.where(p.t0_exit_side=='north',1,-1)
        edge=np.where(side==1,p.zone_top,p.zone_bottom)
        assert np.all((m['close'][t]-edge)*side>0)
    return records


def execution_checks():
    ts=np.arange(6,dtype=np.int64)*M
    op=np.array([100.,100.,101.,102.,103.,104.]);hi=op+1;lo=op-.5;cl=op+.25
    def one(t=ts,o=op,h=hi,l=lo,c=cl,side=1,stop=98.,horizon=3,use=True):
        return execute(t,o,h,l,c,np.array([0]),np.array([5*M]),np.array([side]),np.array([stop]),horizon,use)[0]
    r=one();assert r[0]==2 and r[2]==3 and r[5]==2.25 and r[6]==3
    # Open gaps through stop fill at worse open; long and mirror short.
    gap=op.copy();gap[2]=97.;r=one(o=gap);assert r[0]==1 and r[4]==97. and r[9]==1
    r=one(o=200-gap,h=200-lo,l=200-hi,c=200-cl,side=-1,stop=102.)
    assert r[0]==1 and r[4]==103. and r[5]==-3.
    ll=lo.copy();ll[2]=97.;r=one(l=ll);assert r[0]==1 and r[4]==98.
    broken=ts.copy();broken[2:]+=M;r=one(t=broken);assert r[0]==-3
    r=one(stop=101.);assert r[0]==-1
    # Future low beyond the declared exit cannot retroactively stop the trade.
    ll=lo.copy();ll[4]=90.;assert one(l=ll)[0]==2
    return {'time_exit':True,'long_and_short_gap_fill':True,'intrabar_stop':True,
            'missing_minute_unknown':True,'entry_behind_stop_cancelled':True,'future_stop_ignored':True}


def main():
    r={'prefixes':prefix_checks(),'execution':execution_checks()}
    dump(HERE/'checks_before_outcomes.json',r)
    print('PREFIX_CHECKS',len(r['prefixes']),'EXECUTION',r['execution'],flush=True)


if __name__=='__main__':main()
