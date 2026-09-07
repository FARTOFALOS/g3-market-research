"""Bounded candle-pause study at living RIZ; frozen data, no Volume."""
from pathlib import Path
import argparse,itertools,json,sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from numba import njit
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'data/research/S-06'
sys.path.insert(0,str(ROOT/'setups/S-04'))
# Load unchanged common OHLC/time primitives without importing this file as run.
import importlib.util
_spec=importlib.util.spec_from_file_location('s04_common',ROOT/'setups/S-04/run.py')
common=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(common)
M,POINT,TICK,COST,market,dump,sha=(getattr(common,k) for k in ['M','POINT','TICK','COST','market','dump','sha'])

@njit(cache=False)
def pause_boxes(ts,hi,lo,k):
    lower=np.full(len(ts),np.nan);upper=lower.copy()
    for p in range(k-1,len(ts)):
        if ts[p]-ts[p-k+1]!=(k-1)*M:continue
        a=lo[p-k+1];b=hi[p-k+1];overlo=a;overhi=b
        for j in range(p-k+2,p+1):
            a=min(a,lo[j]);b=max(b,hi[j]);overlo=max(overlo,lo[j]);overhi=min(overhi,hi[j])
        if overlo<overhi:lower[p]=a;upper[p]=b
    return lower,upper

@njit(cache=False)
def attach(ts,t0,edge,deleted,end,lower,upper,k):
    found=[]
    for i in range(len(t0)):
        t=t0[i]
        for p in range(t+1,len(ts)):
            if ts[p]>=end[i] or p>=deleted[i]:break
            if ts[p]!=ts[t]+(p-t)*M:break
            if p<t+k:continue
            if lower[p]<=edge[i]<=upper[p]:found.append((i,p))
    return found

def recognition(ins,objects=None,m=None,k=3,cutoff=None):
    cal=pd.read_parquet(ROOT/'setups/S-04/calendar.parquet')
    p=pd.read_parquet(OUT/f'{ins}_objects.parquet') if objects is None else objects.copy()
    m=market(ins) if m is None else m
    if cutoff is not None:
        m={key:value[:cutoff+1] for key,value in m.items()}
        p=p.loc[p.t0_spine_pos<=cutoff].copy()
    day=np.searchsorted(cal.open_ns.to_numpy(),p.t0_ts_ns,side='right')-1
    safe=np.maximum(day,0)
    good=(day>=0)&(p.t0_ts_ns.to_numpy()>=cal.open_ns.to_numpy()[safe])&(p.t0_ts_ns.to_numpy()<cal.close_ns.to_numpy()[safe])
    p=p.loc[good].reset_index(drop=True);day=day[good]
    p['day']=day;p['date']=cal.date.to_numpy()[day];p['calendar_close_ns']=cal.close_ns.to_numpy()[day]
    p['edge']=np.where(p.t0_exit_side=='north',p.zone_top,p.zone_bottom)
    deletion=p.c1_deletion_spine_pos.fillna(len(m['close'])).to_numpy(np.int64)
    low,high=pause_boxes(m['close_ts_utc_ns'],m['high'],m['low'],k)
    found=attach(m['close_ts_utc_ns'],p.t0_spine_pos.to_numpy(np.int64),p.edge.to_numpy(),deletion,
                 p.calendar_close_ns.to_numpy(np.int64),low,high,k)
    a=np.asarray(found,dtype=np.int64).reshape(-1,2);r=p.iloc[a[:,0]].copy();r['p']=a[:,1]
    r['box_low']=low[a[:,1]];r['box_high']=high[a[:,1]];r['k']=k
    r['age_minutes']=a[:,1]-r.t0_spine_pos.to_numpy();r['age_native_bars']=r.age_minutes/r.tf_minutes
    return r.reset_index(drop=True),len(p)

def prepare():
    OUT.mkdir(parents=True,exist_ok=True);receipts={};summary=[]
    cols=common.COLS+['c1_deletion_spine_pos']
    for ins in ['ES','NQ','YM']:
        files=[ROOT/'data/field'/ins/'cells'/f'tf_{tf:04d}'/'passports.parquet' for tf in range(1,1441)]
        # Exact inputs are bound by the earlier unchanged corpus receipt.
        p=pq.read_table(files,columns=cols).to_pandas();p.to_parquet(OUT/f'{ins}_objects.parquet',index=False)
        receipts[ins]={'objects':len(p),'copy_sha256':sha(OUT/f'{ins}_objects.parquet')}
        m=market(ins)
        for k in [3,5]:
            r,eligible=recognition(ins,p,m,k)
            r.to_parquet(OUT/f'{ins}_pauses_k{k}.parquet',index=False)
            summary.append(dict(instrument=ins,k=k,eligible_objects=eligible,object_pauses=len(r),
                recognized_objects=r.riz_id.nunique(),minutes=r.p.nunique(),days=r.date.nunique()))
            print('PAUSES',summary[-1],flush=True)
    dump(HERE/'inputs.json',{'previous_input_receipt':'data/research/S-04/input_hashes.json',
        'previous_receipt_sha256':sha(ROOT/'data/research/S-04/input_hashes.json'),'new_copies':receipts})
    pd.DataFrame(summary).to_csv(HERE/'recognition.csv',index=False)

def decisions(rows,m,entry):
    s=rows.copy();s['q']=s.p
    if entry=='close':
        q=s.p.to_numpy(np.int64)+1;safe=np.minimum(q,len(m['close'])-1);cl=m['close'][safe]
        d=np.where(cl>s.box_high,1,np.where(cl<s.box_low,-1,0))
        known=(q<len(m['close']))&(m['close_ts_utc_ns'][safe]==m['close_ts_utc_ns'][s.p]+M)
        alive=s.c1_deletion_spine_pos.isna()|(s.c1_deletion_spine_pos>q)
        valid=known&alive.to_numpy()&(d!=0)&(m['close_ts_utc_ns'][safe]<s.calendar_close_ns)
        s=s.loc[valid].copy();s['q']=q[valid];s['direction']=d[valid]
    else:s['direction']=0
    counts=s.groupby('q').size()
    s=s.sort_values(['q','tf_minutes','riz_id']).drop_duplicates('q').copy()
    s['riz_count']=s.q.map(counts).astype(np.int64)
    return s.reset_index(drop=True)

@njit(cache=False)
def execute(ts,op,hi,lo,cl,q,lower,upper,direction,session_end,horizon,tick,stop_entry,delay=0):
    # status: 0 untriggered, 1 stopped, 2 timed, -1 cancelled, -2 entry missing,
    # -3 later path missing, -4 overlaps an existing position. Ambiguous side is 0.
    out=np.full((len(q),13),np.nan);busy=-1
    for i in range(len(q)):
        if q[i]<busy:out[i,0]=-4;continue
        b=q[i]+1+delay
        if b>=len(ts) or ts[b]!=ts[q[i]]+(1+delay)*M:
            out[i,0]=-2;continue
        if ts[b]>session_end[i]:out[i,0]=-1;continue
        L=lower[i]-tick;U=upper[i]+tick;d=direction[i];ambiguous=False
        price=op[b]
        if stop_entry:
            if op[b]>=U:d=1
            elif op[b]<=L:d=-1
            elif hi[b]>=U and lo[b]<=L:
                # Either first side reaches the opposite stop; the money is identical.
                out[i]=np.array([1.,b,b,U,L,L-U,1.,U-L,max(hi[b]-U,L-lo[b]),0.,1.,U-L,L]);busy=b;continue
            elif hi[b]>=U:d=1;price=U
            elif lo[b]<=L:d=-1;price=L
            else:out[i,0]=0;continue
        elif (d==1 and price<=upper[i]) or (d==-1 and price>=lower[i]):
            out[i,0]=-1;continue
        stop=L if d==1 else U;initial=(price-stop)*d
        end=min(ts[q[i]]+(horizon+delay)*M,session_end[i]);heat=0.;mfe=0.
        # Holding includes the entry minute. For trigger entries full-bar heat/MFE
        # may include a pre-entry extreme and are deliberately upper bounds.
        for j in range(b,len(ts)):
            if ts[j]!=ts[q[i]]+(j-q[i])*M or ts[j]>end:
                out[i,0]=-3;busy=np.searchsorted(ts,end);break
            if j>b and (op[j]-stop)*d<=0:
                out[i]=np.array([1.,b,j,price,op[j],(op[j]-price)*d,
                    (ts[j]-M-ts[q[i]]-delay*M)/M,max(heat,(price-op[j])*d),mfe,d,0.,initial,stop]);busy=j;break
            heat=max(heat,(price-(lo[j] if d==1 else hi[j]))*d)
            mfe=max(mfe,((hi[j] if d==1 else lo[j])-price)*d)
            if (lo[j]<=stop if d==1 else hi[j]>=stop):
                out[i]=np.array([1.,b,j,price,stop,(stop-price)*d,
                    (ts[j]-ts[q[i]]-delay*M)/M,heat,mfe,d,0.,initial,stop]);busy=j;break
            if ts[j]==end:
                out[i]=np.array([2.,b,j,price,cl[j],(cl[j]-price)*d,
                    (ts[j]-ts[q[i]]-delay*M)/M,heat,mfe,d,0.,initial,stop]);busy=j;break
        if np.isnan(out[i,0]):out[i,0]=-3;busy=len(ts)
    return out

def replay(s,m,v,delay=0):
    a=execute(*[np.asarray(m[c]) for c in ['close_ts_utc_ns','open','high','low','close']],
        s.q.to_numpy(np.int64),s.box_low.to_numpy(),s.box_high.to_numpy(),s.direction.to_numpy(np.int64),
        s.calendar_close_ns.to_numpy(np.int64),v['horizon'],TICK[v['instrument']],v['entry']=='stop',delay)
    r=s.copy()
    for j,key in enumerate(['status','entry_pos','exit_pos','entry_price','exit_price','gross_points',
        'minutes','heat_bound','mfe_bound','actual_direction','ambiguous_first_side','initial_risk_points','stop_price']):r[key]=a[:,j]
    r['gross_dollars']=r.gross_points*POINT[v['instrument']]
    r['net_dollars']=r.gross_dollars-COST[v['instrument']]
    r.loc[r.status<=0,['gross_dollars','net_dollars']]=np.nan
    r['variant']=v['id'];r['instrument']=v['instrument']
    cov=pd.read_parquet(ROOT/f"data/research/S-04/{v['instrument']}_coverage.parquet")
    r['day_known']=cov.known.to_numpy()[r.day.to_numpy(np.int64)]
    return r

def family():
    return [dict(id=f'{ins}_k{k}_{entry}_h{h}',instrument=ins,k=k,entry=entry,horizon=h)
        for ins,k,entry,h in itertools.product(['ES','NQ','YM'],[3,5],['stop','close'],[5,15,30])]

def backtest():
    cal=pd.read_parquet(ROOT/'setups/S-04/calendar.parquet');daily={'date':cal.date};counts={'date':cal.date};summaries=[];ledgers=[]
    variants=family();dump(HERE/'family_v1.json',variants)
    dump(HERE/'run_hashes_v1.json',{p.name:sha(p) for p in [HERE/'run.py',HERE/'SEARCH.md',HERE/'family_v1.json']})
    for ins in ['ES','NQ','YM']:
        m=market(ins);cache={}
        for k in [3,5]:
            rows=pd.read_parquet(OUT/f'{ins}_pauses_k{k}.parquet')
            for entry in ['stop','close']:cache[k,entry]=decisions(rows,m,entry)
        cov=pd.read_parquet(ROOT/f'data/research/S-04/{ins}_coverage.parquet').known.to_numpy()
        for v in [v for v in variants if v['instrument']==ins]:
            r=replay(cache[v['k'],v['entry']],m,v);valid=r.loc[r.day_known&(r.status>0)]
            d=np.zeros(len(cal));n=np.zeros(len(cal))
            pnl=valid.groupby('day').net_dollars.sum();num=valid.groupby('day').size()
            d[pnl.index]=pnl;n[num.index]=num
            unknown=r.loc[r.status.isin([-2,-3]),'day'].unique();d[unknown]=np.nan;n[unknown]=np.nan
            d[~cov]=np.nan;n[~cov]=np.nan
            daily[v['id']]=d;counts[v['id']]=n;eq=np.r_[0,np.nancumsum(d)]
            recent=(cal.date>='2020-01-01').to_numpy();recent_known=recent&np.isfinite(d)
            summaries.append(dict(**v,decisions=len(r),trades=len(valid),
                trade_days=int((n>0).sum()),known_days=int(np.isfinite(d).sum()),
                frequency_2020=float((n[recent_known]>0).mean()),total=float(np.nansum(d)),
                mean_trade=float(valid.net_dollars.mean()),max_dd=float((np.maximum.accumulate(eq)-eq).max()),
                win_rate=float((valid.net_dollars>0).mean()),median_hold=float(valid.minutes.median()),
                ambiguous_sides=int(valid.ambiguous_first_side.sum()),
                unknown=int(r.status.isin([-2,-3]).sum()),
                net_before_2020=float(np.nansum(d[~recent])),net_2020=float(np.nansum(d[recent]))))
            ledgers.append(r)
        print('BACKTEST',ins,flush=True)
    pd.DataFrame(daily).to_parquet(OUT/'daily_v1.parquet',index=False)
    pd.DataFrame(counts).to_parquet(OUT/'counts_v1.parquet',index=False)
    pd.concat(ledgers,ignore_index=True).to_parquet(OUT/'all_decisions_v1.parquet',index=False)
    summary=pd.DataFrame(summaries);summary.to_csv(HERE/'summary_v1.csv',index=False)
    print(summary.sort_values('total',ascending=False).head(12).to_string(index=False),flush=True)

def selection():
    for scope,folders in [('new',['S-06']),('combined',['S-04','S-05','S-06'])]:
        daily=pd.concat([pd.read_parquet(ROOT/f'data/research/{f}/daily_v1.parquet').set_index('date').add_prefix(f+'/') for f in folders],axis=1)
        counts=pd.concat([pd.read_parquet(ROOT/f'data/research/{f}/counts_v1.parquet').set_index('date').add_prefix(f+'/') for f in folders],axis=1)
        years=pd.to_datetime(daily.index).year.to_numpy();known=daily.notna().all(axis=1).to_numpy();curve=np.full(len(daily),np.nan);records=[]
        for year in range(2010,2027):
            train=(years>=year-3)&(years<year)&known;test=(years==year)&known
            expected=((years>=year-3)&(years<year)).sum()
            means=daily.loc[train].mean().where(counts.loc[train].sum()>=30,-np.inf)
            winner=means.idxmax() if means.max()>0 and train.sum()>=.8*expected else None
            curve[test]=daily.loc[test,winner] if winner else 0.
            records.append(dict(year=year,train_days=int(train.sum()),expected_train_days=int(expected),
                variant=winner,train_mean=float(means[winner]) if winner else None,
                test_days=int(test.sum()),test_trades=int(counts.loc[test,winner].sum()) if winner else 0,
                test_net=float(np.nansum(curve[test]))))
        pd.DataFrame(records).to_csv(HERE/f'selection_{scope}.csv',index=False)
        pd.DataFrame({'date':daily.index,'known_all':known,'selection_net':curve}).to_parquet(OUT/f'selection_{scope}_daily.parquet',index=False)
        eq=np.r_[0,np.nancumsum(curve)]
        print(scope,{'net':float(np.nansum(curve)),'max_dd':float((np.maximum.accumulate(eq)-eq).max())},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','backtest','selection']);a=p.parse_args();globals()[a.phase]()
