"""Candle reversal after an observed RIZ interaction; all research variants retained."""
import argparse
import itertools
import json
import warnings
import numpy as np
import pandas as pd
from numba import njit
from run import ROOT, HERE, OUT, M, INSTRUMENTS, POINT, TICK, COST, market, dump, sha, execute, stats


@njit(cache=False)
def recognize(ts, high, low, close, t0, edge, t0side, end, consecutive):
    result = np.full((len(t0), 5), np.nan)
    for i in range(len(t0)):
        t = t0[i]
        if t >= len(ts):
            continue
        a = low[t]; b = high[t]
        d = -t0side[i]
        result[i, 0] = 0
        for q in range(t+1, min(t+61, len(ts))):
            if ts[q] > end[i]:
                break
            if ts[q] != ts[t]+(q-t)*M:
                result[i,0] = -1; break
            a = min(a, low[q]); b = max(b, high[q])
            if q-t < consecutive:
                continue
            if (close[q-consecutive]-edge[i])*d >= 0 or (close[q]-edge[i])*d <= 0:
                continue
            valid = True
            for j in range(q-consecutive+1, q+1):
                if (close[j]-close[j-1])*d <= 0:
                    valid = False; break
            if valid:
                result[i] = np.array([1., q, q-t, a, b]); break
        if result[i,0] == 0 and len(ts) <= t+60 and ts[-1] < end[i]:
            result[i,0] = -1
    return result


def signal_rows(p, m, cal, k, cutoff=None):
    if cutoff is not None:
        m = {key: value[:cutoff+1] for key,value in m.items()}
        p = p.loc[p.t0_spine_pos<=cutoff].copy()
    t = p.t0_ts_ns.to_numpy()
    day = np.searchsorted(cal.open_ns.to_numpy(), t, side='right')-1
    safe = np.maximum(day,0)
    valid = (day>=0)&(t>=cal.open_ns.to_numpy()[safe])&(t<cal.close_ns.to_numpy()[safe]-30*M)
    p = p.loc[valid].copy(); day = day[valid]
    side = np.where(p.t0_exit_side=='north',1,-1)
    edge = np.where(side==1,p.zone_top,p.zone_bottom)
    end = cal.close_ns.to_numpy()[day]-30*M
    result = recognize(*[np.asarray(m[c]) for c in ['close_ts_utc_ns','high','low','close']],
                       p.t0_spine_pos.to_numpy(np.int64), edge, side, end, k)
    p['day'] = day; p['date'] = cal.date.to_numpy()[day]
    p['calendar_close_ns'] = cal.close_ns.to_numpy()[day]
    p['direction'] = -side; p['edge'] = edge; p['k'] = k
    for j,key in enumerate(['recognition_status','q','wait_minutes','low_so_far','high_so_far']):
        p[key] = result[:,j]
    return p


def signals():
    cal = pd.read_parquet(HERE/'calendar.parquet')
    totals=[]
    for ins in INSTRUMENTS:
        m=market(ins);p=pd.read_parquet(OUT/f'{ins}_objects.parquet')
        for k in [2,3,4]:
            r=signal_rows(p,m,cal,k)
            r.to_parquet(OUT/f'{ins}_signals_k{k}.parquet',index=False)
            found=r.loc[r.recognition_status==1]
            totals.append(dict(instrument=ins,k=k,eligible_objects=len(r),
                recognized_objects=len(found),moments=found.q.nunique(),days=found.date.nunique(),
                unknown=int((r.recognition_status<0).sum()),median_wait=found.wait_minutes.median()))
        print('signals',ins,totals[-3:],flush=True)
    pd.DataFrame(totals).to_csv(HERE/'recognition_v1.csv',index=False)


def family():
    result=[]
    for ins,k,wait,horizon,stop in itertools.product(INSTRUMENTS,[2,3,4],[10,30,60],[15,30,60,120],['swing','time']):
        result.append(dict(id=f'{ins}_k{k}_w{wait}_h{horizon}_{stop}',instrument=ins,k=k,
                           wait=wait,horizon=horizon,stop=stop))
    dump(HERE/'family_v1.json',result)
    return result


def decide(rows, wait):
    s=rows.loc[(rows.recognition_status==1)&(rows.wait_minutes<=wait)].copy()
    s['q']=s.q.astype(np.int64)
    counts=s.groupby('q').agg(riz_count=('riz_id','size'),sides=('direction','nunique'))
    s=s.sort_values(['q','tf_minutes','riz_id']).drop_duplicates('q').copy()
    s['riz_count']=s.q.map(counts.riz_count)
    s['conflict']=s.q.map(counts.sides)>1
    return s.drop_duplicates('day').copy()


def replay(sub,m,v,delay=0):
    ins=v['instrument']
    stop=np.where(sub.direction==1,sub.low_so_far-TICK[ins],sub.high_so_far+TICK[ins])
    arr=execute(*[np.asarray(m[k]) for k in ['close_ts_utc_ns','open','high','low','close']],
                sub.q.to_numpy(np.int64),sub.calendar_close_ns.to_numpy(np.int64),
                sub.direction.to_numpy(np.int64),stop,v['horizon'],v['stop']=='swing',delay)
    r=sub.copy()
    for j,c in enumerate(['status','entry_pos','exit_pos','entry','exit','gross_points','minutes',
                            'bar_heat_bound','bar_mfe_bound','stop_gap']):
        r[c]=arr[:,j]
    b=sub.q.to_numpy(np.int64)+1+delay
    bounded=np.minimum(b,len(m['open'])-1)
    cancel=(((m['open'][bounded]-sub.edge)*sub.direction<=0)|
            ((m['open'][bounded]-stop)*sub.direction<=0)).to_numpy()
    r.loc[cancel & (r.status>=-1),'status']=-1
    r.loc[r.conflict,'status']=-4
    r['stop_price']=stop
    r['gross_dollars']=r.gross_points*POINT[ins]
    r['net_dollars']=r.gross_dollars-COST[ins]
    r.loc[r.status<=0,['gross_points','gross_dollars','net_dollars','minutes']]=np.nan
    r['variant']=v['id'];r['instrument']=ins
    return r


def backtest():
    variants=family();cal=pd.read_parquet(HERE/'calendar.parquet')
    days={'date':cal.date};counts={'date':cal.date};summary=[];ledgers=[]
    for ins in INSTRUMENTS:
        m=market(ins)
        rows={k:pd.read_parquet(OUT/f'{ins}_signals_k{k}.parquet') for k in [2,3,4]}
        cov=pd.read_parquet(OUT/f'{ins}_coverage.parquet').known.to_numpy()
        for v in [v for v in variants if v['instrument']==ins]:
            r=replay(decide(rows[v['k']],v['wait']),m,v)
            r['day_known']=cov[r.day.to_numpy()]
            d=np.zeros(len(cal));n=np.zeros(len(cal));a=r.day.to_numpy()
            good=(r.status>0).to_numpy();unknown=r.status.isin([-2,-3]).to_numpy()
            d[a[good]]=r.net_dollars.to_numpy()[good];n[a[good]]=1
            d[a[unknown]]=np.nan;n[a[unknown]]=np.nan
            d[~cov]=np.nan;n[~cov]=np.nan
            days[v['id']]=d;counts[v['id']]=n
            valid=r.loc[r.day_known & (r.status>0)]
            summary.append(dict(**v,signals=len(r),no_entry=int((r.status==-1).sum()),
                conflicts=int((r.status==-4).sum()),unknown_outcomes=int(unknown.sum()),
                excluded_days=int((~cov).sum()),**stats(d,valid)))
            ledgers.append(r)
        print('backtest complete',ins,flush=True)
    pd.DataFrame(days).to_parquet(OUT/'daily_v1.parquet',index=False)
    pd.DataFrame(counts).to_parquet(OUT/'counts_v1.parquet',index=False)
    pd.concat(ledgers,ignore_index=True).to_parquet(OUT/'all_decisions_v1.parquet',index=False)
    result=pd.DataFrame(summary);result.to_csv(HERE/'summary_v1.csv',index=False)
    print(result.sort_values('total',ascending=False).head(15).to_string(index=False),flush=True)
    dump(HERE/'run_hashes_v1.json',{p.name:sha(p) for p in [HERE/'run.py',HERE/'reversal.py',HERE/'SEARCH.md',HERE/'family_v1.json']})


def selection():
    daily=pd.read_parquet(OUT/'daily_v1.parquet').set_index('date')
    count=pd.read_parquet(OUT/'counts_v1.parquet').set_index('date')
    years=pd.to_datetime(daily.index).year.to_numpy()
    common=daily.notna().all(axis=1).to_numpy()
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
    pd.DataFrame(records).to_csv(HERE/'selection_v1.csv',index=False)
    pd.DataFrame({'date':daily.index,'known_all_variants':common,'selection_net':curve}).to_parquet(OUT/'selection_daily_v1.parquet',index=False)
    print(pd.DataFrame(records).to_string(index=False),flush=True)
    eq=np.r_[0,np.nancumsum(curve)];drawdown=float(np.max(np.maximum.accumulate(eq)-eq))
    print('TOTAL',np.nansum(curve),'MAX_DRAWDOWN',drawdown,'COMMON_DAYS',common.sum(),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['signals','backtest','selection'])
    args=p.parse_args();globals()[args.phase]()
