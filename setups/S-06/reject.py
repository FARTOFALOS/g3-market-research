"""S-06 v2: one-bar rejected departure from an observed pause at a living RIZ."""
import argparse,itertools,json
import numpy as np
import pandas as pd
from numba import njit
from run import ROOT,HERE,OUT,M,POINT,TICK,COST,market,dump,sha,recognition

def decisions(rows,m):
    q=rows.p.to_numpy(np.int64)+1;safe=np.minimum(q,len(m['close'])-1)
    o,h,l,c=[m[key][safe] for key in ['open','high','low','close']]
    L=rows.box_low.to_numpy();U=rows.box_high.to_numpy()
    known=(q<len(m['close']))&(m['close_ts_utc_ns'][safe]==m['close_ts_utc_ns'][rows.p]+M)
    inside=(o>=L)&(o<=U)&(c>=L)&(c<=U)
    d=np.where((l<L)&(h<=U),1,np.where((h>U)&(l>=L),-1,0))
    alive=rows.c1_deletion_spine_pos.isna()|(rows.c1_deletion_spine_pos>q)
    valid=known&inside&(d!=0)&alive.to_numpy()&(m['close_ts_utc_ns'][safe]<rows.calendar_close_ns)
    s=rows.loc[valid].copy();s['q']=q[valid];s['direction']=d[valid]
    s['probe_extreme']=np.where(d[valid]==1,l[valid],h[valid])
    counts=s.groupby('q').size()
    s=s.sort_values(['q','tf_minutes','riz_id']).drop_duplicates('q').copy()
    s['riz_count']=s.q.map(counts).astype(np.int64)
    return s.reset_index(drop=True)

@njit(cache=False)
def execute(ts,op,hi,lo,cl,q,L,U,direction,extreme,session_end,horizon,tick,use_target,optimistic=False,delay=0):
    out=np.full((len(q),14),np.nan);busy=-1
    for i in range(len(q)):
        if q[i]<busy:out[i,0]=-4;continue
        b=q[i]+1+delay
        if b>=len(ts) or ts[b]!=ts[q[i]]+(delay+1)*M:out[i,0]=-2;continue
        if ts[b]>session_end[i] or op[b]<L[i] or op[b]>U[i]:out[i,0]=-1;continue
        price=op[b];d=direction[i];stop=extreme[i]-d*tick;target=U[i] if d==1 else L[i]
        if (price-stop)*d<=0 or (use_target and (target-price)*d<=0):out[i,0]=-1;continue
        end=min(ts[q[i]]+(horizon+delay)*M,session_end[i]);heat=0.;mfe=0.
        for j in range(b,len(ts)):
            if ts[j]!=ts[q[i]]+(j-q[i])*M or ts[j]>end:out[i,0]=-3;busy=np.searchsorted(ts,end);break
            status=0;exit_price=0.;tie=0.;gap=0.
            if j>b and (op[j]-stop)*d<=0:status=1;exit_price=op[j];gap=1.
            elif j>b and use_target and (op[j]-target)*d>=tick:status=3;exit_price=target;gap=1.
            heat=max(heat,(price-(lo[j] if d==1 else hi[j]))*d)
            mfe=max(mfe,((hi[j] if d==1 else lo[j])-price)*d)
            if status==0:
                hitstop=(lo[j]<=stop if d==1 else hi[j]>=stop)
                hittarget=use_target and (hi[j]>=target+tick if d==1 else lo[j]<=target-tick)
                if hitstop and hittarget:
                    tie=1.;status=3 if optimistic else 1;exit_price=target if optimistic else stop
                elif hitstop:status=1;exit_price=stop
                elif hittarget:status=3;exit_price=target
                elif ts[j]==end:status=2;exit_price=cl[j]
            if status:
                out[i]=np.array([float(status),b,j,price,exit_price,(exit_price-price)*d,
                    (ts[j]-ts[q[i]]-delay*M)/M,heat,mfe,(price-stop)*d,stop,target,tie,gap]);busy=j;break
        if np.isnan(out[i,0]):out[i,0]=-3;busy=len(ts)
    return out

def replay(s,m,v,optimistic=False,delay=0):
    a=execute(*[np.asarray(m[c]) for c in ['close_ts_utc_ns','open','high','low','close']],
        s.q.to_numpy(np.int64),s.box_low.to_numpy(),s.box_high.to_numpy(),s.direction.to_numpy(np.int64),
        s.probe_extreme.to_numpy(),s.calendar_close_ns.to_numpy(np.int64),v['horizon'],TICK[v['instrument']],v['target'],optimistic,delay)
    r=s.copy()
    for j,key in enumerate(['status','entry_pos','exit_pos','entry_price','exit_price','gross_points',
        'minutes','heat_bound','mfe_bound','initial_risk_points','stop_price','target_price','both_barriers','gap_fill']):r[key]=a[:,j]
    r['gross_dollars']=r.gross_points*POINT[v['instrument']];r['net_dollars']=r.gross_dollars-COST[v['instrument']]
    r.loc[r.status<=0,['gross_dollars','net_dollars']]=np.nan
    r['variant']=v['id'];r['instrument']=v['instrument']
    cov=pd.read_parquet(ROOT/f"data/research/S-04/{v['instrument']}_coverage.parquet")
    r['day_known']=cov.known.to_numpy()[r.day.to_numpy(np.int64)]
    return r

def family():
    return [dict(id=f"{ins}_k{k}_reject_{'target' if target else 'time'}_h{h}",instrument=ins,k=k,target=target,horizon=h)
        for ins,k,target,h in itertools.product(['ES','NQ','YM'],[3,5],[True,False],[5,15,30])]

def backtest():
    cal=pd.read_parquet(ROOT/'setups/S-04/calendar.parquet');daily={'date':cal.date};counts={'date':cal.date};summaries=[];ledgers=[]
    variants=family();dump(HERE/'family_v2.json',variants)
    dump(HERE/'run_hashes_v2.json',{p.name:sha(p) for p in [HERE/'run.py',HERE/'reject.py',HERE/'SEARCH.md',HERE/'family_v2.json']})
    for ins in ['ES','NQ','YM']:
        m=market(ins);cache={}
        for k in [3,5]:
            rows=pd.read_parquet(OUT/f'{ins}_pauses_k{k}.parquet');cache[k]=decisions(rows,m)
            cache[k].to_parquet(OUT/f'{ins}_rejected_k{k}.parquet',index=False)
        cov=pd.read_parquet(ROOT/f'data/research/S-04/{ins}_coverage.parquet').known.to_numpy()
        for v in [v for v in variants if v['instrument']==ins]:
            r=replay(cache[v['k']],m,v);valid=r.loc[r.day_known&(r.status>0)]
            d=np.zeros(len(cal));n=np.zeros(len(cal));pnl=valid.groupby('day').net_dollars.sum();num=valid.groupby('day').size()
            d[pnl.index]=pnl;n[num.index]=num;unknown=r.loc[r.status.isin([-2,-3]),'day'].unique()
            d[unknown]=np.nan;n[unknown]=np.nan;d[~cov]=np.nan;n[~cov]=np.nan
            daily[v['id']]=d;counts[v['id']]=n;eq=np.r_[0,np.nancumsum(d)]
            recent=(cal.date>='2020-01-01').to_numpy();known=recent&np.isfinite(d)
            optimistic=replay(cache[v['k']],m,v,optimistic=True)
            optimistic=optimistic.loc[optimistic.day_known&(optimistic.status>0)]
            summaries.append(dict(**v,decisions=len(r),trades=len(valid),trade_days=int((n>0).sum()),
                known_days=int(np.isfinite(d).sum()),frequency_2020=float((n[known]>0).mean()),
                total=float(np.nansum(d)),gross=float(valid.gross_dollars.sum()),mean_trade=float(valid.net_dollars.mean()),
                max_dd=float((np.maximum.accumulate(eq)-eq).max()),win_rate=float((valid.net_dollars>0).mean()),
                median_hold=float(valid.minutes.median()),ties=int(valid.both_barriers.sum()),
                optimistic_total=float(optimistic.net_dollars.sum()),unknown=int(r.status.isin([-2,-3]).sum()),
                net_before_2020=float(np.nansum(d[~recent])),net_2020=float(np.nansum(d[recent]))))
            ledgers.append(r)
        print('REJECT',ins,flush=True)
    pd.DataFrame(daily).to_parquet(OUT/'daily_v2.parquet',index=False);pd.DataFrame(counts).to_parquet(OUT/'counts_v2.parquet',index=False)
    pd.concat(ledgers,ignore_index=True).to_parquet(OUT/'all_decisions_v2.parquet',index=False)
    summary=pd.DataFrame(summaries);summary.to_csv(HERE/'summary_v2.csv',index=False)
    print(summary.sort_values('total',ascending=False).head(12).to_string(index=False),flush=True)

def selection():
    for scope,families in [('v2',[('S-06',2)]),('S06',[('S-06',1),('S-06',2)]),('combined',[('S-04',1),('S-05',1),('S-06',1),('S-06',2)])]:
        daily=pd.concat([pd.read_parquet(ROOT/f'data/research/{f}/daily_v{v}.parquet').set_index('date').add_prefix(f+f'/v{v}/') for f,v in families],axis=1)
        counts=pd.concat([pd.read_parquet(ROOT/f'data/research/{f}/counts_v{v}.parquet').set_index('date').add_prefix(f+f'/v{v}/') for f,v in families],axis=1)
        years=pd.to_datetime(daily.index).year.to_numpy();known=daily.notna().all(axis=1).to_numpy();curve=np.full(len(daily),np.nan);records=[]
        for year in range(2010,2027):
            train=(years>=year-3)&(years<year)&known;test=(years==year)&known;expected=((years>=year-3)&(years<year)).sum()
            means=daily.loc[train].mean().where(counts.loc[train].sum()>=30,-np.inf)
            winner=means.idxmax() if means.max()>0 and train.sum()>=.8*expected else None
            curve[test]=daily.loc[test,winner] if winner else 0.
            records.append(dict(year=year,train_days=int(train.sum()),expected_train_days=int(expected),variant=winner,
                train_mean=float(means[winner]) if winner else None,test_days=int(test.sum()),
                test_trades=int(counts.loc[test,winner].sum()) if winner else 0,test_net=float(np.nansum(curve[test]))))
        pd.DataFrame(records).to_csv(HERE/f'selection_after_v2_{scope}.csv',index=False)
        pd.DataFrame({'date':daily.index,'known_all':known,'selection_net':curve}).to_parquet(OUT/f'selection_after_v2_{scope}_daily.parquet',index=False)
        eq=np.r_[0,np.nancumsum(curve)]
        print(scope,{'net':float(np.nansum(curve)),'max_dd':float((np.maximum.accumulate(eq)-eq).max())},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['backtest','selection']);a=p.parse_args();globals()[a.phase]()
