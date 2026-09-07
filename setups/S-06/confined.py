"""S-06 v3: closes must remain in the shared candle-range intersection."""
import json
import numpy as np
import pandas as pd
import run as breakout
import reject
from run import ROOT,HERE,OUT,M,POINT,COST,market,dump,sha

def gate(rows,m,k):
    p=np.sort(rows.p.unique()).astype(np.int64)
    idx=p[:,None]-np.arange(k-1,-1,-1)
    low=m['low'][idx].max(axis=1);high=m['high'][idx].min(axis=1)
    valid=(low<high)&(m['close'][idx].min(axis=1)>=low)&(m['close'][idx].max(axis=1)<=high)
    s=rows.loc[rows.p.isin(p[valid])].copy()
    s['common_low']=s.p.map(pd.Series(low[valid],index=p[valid]))
    s['common_high']=s.p.map(pd.Series(high[valid],index=p[valid]))
    return s.reset_index(drop=True)

def family():
    result=[]
    for kind,module in [('breakout',breakout),('reject',reject)]:
        for v in module.family():result.append(dict(**{**v,'id':v['id']+'_confined'},kind=kind))
    return result

def prepare_and_check():
    checks=[];counts=[]
    for ins in ['ES','NQ','YM']:
        m=market(ins)
        for k in [3,5]:
            rows=pd.read_parquet(OUT/f'{ins}_pauses_k{k}.parquet');full=gate(rows,m,k)
            full.to_parquet(OUT/f'{ins}_confined_k{k}.parquet',index=False)
            counts.append(dict(instrument=ins,k=k,object_pauses=len(full),objects=full.riz_id.nunique(),minutes=full.p.nunique(),days=full.date.nunique()))
            moments=np.sort(full.p.unique())
            for j in np.linspace(0,len(moments)-1,3,dtype=int):
                for cutoff in [int(moments[j]),int(moments[j])+1]:
                    mc={key:value[:cutoff+1] for key,value in m.items()}
                    t=gate(rows.loc[rows.p<=cutoff],mc,k)
                    columns=['riz_id','p','common_low','common_high']
                    pd.testing.assert_frame_equal(full.loc[full.p<=cutoff,columns].reset_index(drop=True),t[columns].reset_index(drop=True))
                    for entry in ['stop','close','reject']:
                        a=reject.decisions(full,m) if entry=='reject' else breakout.decisions(full,m,entry)
                        b=reject.decisions(t,mc) if entry=='reject' else breakout.decisions(t,mc,entry)
                        cols=['riz_id','q','direction','riz_count']
                        pd.testing.assert_frame_equal(a.loc[a.q<=cutoff,cols].reset_index(drop=True),b[cols].reset_index(drop=True))
                    checks.append(dict(instrument=ins,k=k,cutoff=cutoff,equal=True))
        print('CONFINED',ins,counts[-2:],flush=True)
    pd.DataFrame(counts).to_csv(HERE/'recognition_v3.csv',index=False)
    dump(HERE/'checks_v3_before_outcomes.json',{'physical_gate_and_decision_checks':checks,'code_sha256':sha(HERE/'confined.py')})

def backtest():
    variants=family();dump(HERE/'family_v3.json',variants)
    dump(HERE/'run_hashes_v3.json',{p.name:sha(p) for p in [HERE/'run.py',HERE/'reject.py',HERE/'confined.py',HERE/'SEARCH.md',HERE/'family_v3.json']})
    cal=pd.read_parquet(ROOT/'setups/S-04/calendar.parquet');daily={'date':cal.date};counts={'date':cal.date};summaries=[];ledgers=[]
    for ins in ['ES','NQ','YM']:
        m=market(ins);cache={}
        for k in [3,5]:
            rows=pd.read_parquet(OUT/f'{ins}_confined_k{k}.parquet')
            cache[k,'reject']=reject.decisions(rows,m)
            for entry in ['stop','close']:cache[k,entry]=breakout.decisions(rows,m,entry)
        cov=pd.read_parquet(ROOT/f'data/research/S-04/{ins}_coverage.parquet').known.to_numpy()
        for v in [v for v in variants if v['instrument']==ins]:
            module=reject if v['kind']=='reject' else breakout
            s=cache[v['k'],'reject' if v['kind']=='reject' else v['entry']]
            r=module.replay(s,m,v);valid=r.loc[r.day_known&(r.status>0)]
            d=np.zeros(len(cal));n=np.zeros(len(cal));pnl=valid.groupby('day').net_dollars.sum();num=valid.groupby('day').size()
            d[pnl.index]=pnl;n[num.index]=num;unknown=r.loc[r.status.isin([-2,-3]),'day'].unique()
            d[unknown]=np.nan;n[unknown]=np.nan;d[~cov]=np.nan;n[~cov]=np.nan
            daily[v['id']]=d;counts[v['id']]=n;eq=np.r_[0,np.nancumsum(d)]
            recent=(cal.date>='2020-01-01').to_numpy();known=recent&np.isfinite(d)
            optim=module.replay(s,m,v,optimistic=True) if v['kind']=='reject' else r
            optim=optim.loc[optim.day_known&(optim.status>0)]
            summaries.append(dict(**v,decisions=len(r),trades=len(valid),trade_days=int((n>0).sum()),known_days=int(np.isfinite(d).sum()),
                frequency_2020=float((n[known]>0).mean()),total=float(np.nansum(d)),gross=float(valid.gross_dollars.sum()),
                mean_trade=float(valid.net_dollars.mean()),max_dd=float((np.maximum.accumulate(eq)-eq).max()),
                win_rate=float((valid.net_dollars>0).mean()),median_hold=float(valid.minutes.median()),
                optimistic_total=float(optim.net_dollars.sum()),net_before_2020=float(np.nansum(d[~recent])),net_2020=float(np.nansum(d[recent]))))
            ledgers.append(r)
        print('CONFINED RESULTS',ins,flush=True)
    pd.DataFrame(daily).to_parquet(OUT/'daily_v3.parquet',index=False);pd.DataFrame(counts).to_parquet(OUT/'counts_v3.parquet',index=False)
    pd.concat(ledgers,ignore_index=True).to_parquet(OUT/'all_decisions_v3.parquet',index=False)
    summary=pd.DataFrame(summaries);summary.to_csv(HERE/'summary_v3.csv',index=False)
    print(summary.sort_values('total',ascending=False).head(12).to_string(index=False),flush=True)

def selection():
    for scope,families in [('v3',[('S-06',3)]),('S06',[('S-06',1),('S-06',2),('S-06',3)]),('combined',[('S-04',1),('S-05',1),('S-06',1),('S-06',2),('S-06',3)])]:
        daily=pd.concat([pd.read_parquet(ROOT/f'data/research/{f}/daily_v{v}.parquet').set_index('date').add_prefix(f+f'/v{v}/') for f,v in families],axis=1)
        counts=pd.concat([pd.read_parquet(ROOT/f'data/research/{f}/counts_v{v}.parquet').set_index('date').add_prefix(f+f'/v{v}/') for f,v in families],axis=1)
        years=pd.to_datetime(daily.index).year.to_numpy();known=daily.notna().all(axis=1).to_numpy();curve=np.full(len(daily),np.nan);records=[]
        for year in range(2010,2027):
            train=(years>=year-3)&(years<year)&known;test=(years==year)&known;expected=((years>=year-3)&(years<year)).sum()
            means=daily.loc[train].mean().where(counts.loc[train].sum()>=30,-np.inf)
            winner=means.idxmax() if means.max()>0 and train.sum()>=.8*expected else None
            curve[test]=daily.loc[test,winner] if winner else 0.
            records.append(dict(year=year,train_days=int(train.sum()),variant=winner,test_trades=int(counts.loc[test,winner].sum()) if winner else 0,test_net=float(np.nansum(curve[test]))))
        pd.DataFrame(records).to_csv(HERE/f'selection_after_v3_{scope}.csv',index=False)
        pd.DataFrame({'date':daily.index,'known_all':known,'selection_net':curve}).to_parquet(OUT/f'selection_after_v3_{scope}_daily.parquet',index=False)
        eq=np.r_[0,np.nancumsum(curve)]
        print(scope,{'net':float(np.nansum(curve)),'max_dd':float((np.maximum.accumulate(eq)-eq).max())},flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare_and_check','backtest','selection']);a=p.parse_args();globals()[a.phase]()
