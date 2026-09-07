"""Continuation action on the same observed per-RIZ return; adaptive successor to S-04."""
from pathlib import Path
import argparse
import itertools
import sys
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE.parent/'S-04'))
from run import M,INSTRUMENTS,POINT,TICK,COST,market,dump,sha,execute,stats
from reversal import decide
INPUT=ROOT/'data/research/S-04'
OUT=ROOT/'data/research/S-05'


def family():
    v=[]
    for ins,k,wait,horizon,stop in itertools.product(INSTRUMENTS,[2,3,4],[10,30,60],[15,30,60,120],['far','time']):
        v.append(dict(id=f'{ins}_k{k}_w{wait}_h{horizon}_{stop}',instrument=ins,k=k,wait=wait,horizon=horizon,stop=stop))
    dump(HERE/'family_v1.json',v);return v


def replay(sub,m,v,delay=0):
    ins=v['instrument'];d=-sub.direction.to_numpy(np.int64)
    stop=np.where(d==1,sub.zone_bottom-TICK[ins],sub.zone_top+TICK[ins])
    arr=execute(*[np.asarray(m[k]) for k in ['close_ts_utc_ns','open','high','low','close']],
                sub.q.to_numpy(np.int64),sub.calendar_close_ns.to_numpy(np.int64),d,stop,
                v['horizon'],v['stop']=='far',delay)
    r=sub.copy();r['direction']=d
    for j,c in enumerate(['status','entry_pos','exit_pos','entry','exit','gross_points','minutes',
                            'bar_heat_bound','bar_mfe_bound','stop_gap']):r[c]=arr[:,j]
    b=sub.q.to_numpy(np.int64)+1+delay;bounded=np.minimum(b,len(m['open'])-1)
    # Entry only while the counter-move still lies back across its own exit edge.
    cancel=(m['open'][bounded]-sub.edge.to_numpy())*d>=0
    r.loc[cancel & (r.status>=-1),'status']=-1
    r.loc[r.conflict,'status']=-4
    r['stop_price']=stop if v['stop']=='far' else np.nan
    r['gross_dollars']=r.gross_points*POINT[ins];r['net_dollars']=r.gross_dollars-COST[ins]
    r.loc[r.status<=0,['gross_points','gross_dollars','net_dollars','minutes']]=np.nan
    r['variant']=v['id'];r['instrument']=ins
    return r


def backtest():
    OUT.mkdir(parents=True,exist_ok=True)
    cal=pd.read_parquet(HERE.parent/'S-04/calendar.parquet')
    variants=family();days={'date':cal.date};counts={'date':cal.date};summary=[];ledgers=[]
    for ins in INSTRUMENTS:
        m=market(ins);rows={k:pd.read_parquet(INPUT/f'{ins}_signals_k{k}.parquet') for k in [2,3,4]}
        cov=pd.read_parquet(INPUT/f'{ins}_coverage.parquet').known.to_numpy()
        for v in [v for v in variants if v['instrument']==ins]:
            r=replay(decide(rows[v['k']],v['wait']),m,v)
            r['day_known']=cov[r.day.to_numpy()]
            d=np.zeros(len(cal));n=np.zeros(len(cal));a=r.day.to_numpy()
            good=(r.status>0).to_numpy();unknown=r.status.isin([-2,-3]).to_numpy()
            d[a[good]]=r.net_dollars.to_numpy()[good];n[a[good]]=1
            d[a[unknown]]=np.nan;n[a[unknown]]=np.nan;d[~cov]=np.nan;n[~cov]=np.nan
            days[v['id']]=d;counts[v['id']]=n
            valid=r.loc[r.day_known&(r.status>0)]
            summary.append(dict(**v,signals=len(r),no_entry=int((r.status==-1).sum()),
                conflicts=int((r.status==-4).sum()),unknown_outcomes=int(unknown.sum()),
                excluded_days=int((~cov).sum()),**stats(d,valid)))
            ledgers.append(r)
        print('S-05 backtest',ins,flush=True)
    pd.DataFrame(days).to_parquet(OUT/'daily_v1.parquet',index=False)
    pd.DataFrame(counts).to_parquet(OUT/'counts_v1.parquet',index=False)
    pd.concat(ledgers,ignore_index=True).to_parquet(OUT/'all_decisions_v1.parquet',index=False)
    r=pd.DataFrame(summary);r.to_csv(HERE/'summary_v1.csv',index=False)
    print(r.sort_values('total',ascending=False).head(15).to_string(index=False),flush=True)
    dump(HERE/'run_hashes_v1.json',{str(p.relative_to(ROOT)):sha(p) for p in
       [HERE/'follow.py',HERE/'SEARCH.md',HERE/'family_v1.json',HERE.parent/'S-04/run.py',HERE.parent/'S-04/reversal.py']})


def selection():
    for scope,folders in [('combined',[INPUT,OUT]),('S05_only',[OUT])]:
        xs=[];cs=[]
        for folder in folders:
            prefix=folder.name+'/'
            xs.append(pd.read_parquet(folder/'daily_v1.parquet').set_index('date').add_prefix(prefix))
            cs.append(pd.read_parquet(folder/'counts_v1.parquet').set_index('date').add_prefix(prefix))
        daily=pd.concat(xs,axis=1);count=pd.concat(cs,axis=1)
        years=pd.to_datetime(daily.index).year.to_numpy();common=daily.notna().all(axis=1).to_numpy()
        records=[];curve=np.full(len(daily),np.nan)
        for year in range(2010,2027):
            train=(years>=year-3)&(years<year)&common;test=(years==year)&common
            expected=((years>=year-3)&(years<year)).sum()
            means=daily.loc[train].mean().where(count.loc[train].sum()>=30,-np.inf)
            winner=means.idxmax() if means.max()>0 and train.sum()>=.8*expected else None
            curve[test]=daily.loc[test,winner] if winner else 0.
            records.append(dict(year=year,train_start=f'{year-3}-01-01',train_stop=f'{year-1}-12-31',
                train_days=int(train.sum()),expected_train_days=int(expected),variant=winner,
                train_mean=float(means[winner]) if winner else None,test_days=int(test.sum()),
                test_trades=int(count.loc[test,winner].sum()) if winner else 0,
                test_net=float(np.nansum(curve[test]))))
        pd.DataFrame(records).to_csv(HERE/f'selection_{scope}_v1.csv',index=False)
        pd.DataFrame({'date':daily.index,'known_all_variants':common,'selection_net':curve}).to_parquet(OUT/f'selection_daily_{scope}_v1.parquet',index=False)
        eq=np.r_[0,np.nancumsum(curve)]
        print(scope,'NET',np.nansum(curve),'DD',np.max(np.maximum.accumulate(eq)-eq),flush=True)
        print(pd.DataFrame(records).to_string(index=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['backtest','selection'])
    args=p.parse_args();globals()[args.phase]()
