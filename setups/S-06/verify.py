"""Fresh recognition replay and independent raw-OHLC reconciliation of two daily examples."""
import json
import numpy as np
import pandas as pd
import run as breakout
import reject
from run import ROOT,HERE,OUT,M,POINT,COST,TICK,market,dump,recognition

def direct(z,m,version,target,horizon):
    q=int(z.q);b=q+1;op,hi,lo,cl,ts=[m[c] for c in ['open','high','low','close','close_ts_utc_ns']]
    p=int(z.p);k=int(z.k);a=p-k+1
    L=float(np.min(lo[a:p+1]));U=float(np.max(hi[a:p+1]))
    assert L==z.box_low and U==z.box_high
    assert np.max(lo[a:p+1])<np.min(hi[a:p+1])
    assert L<=z.edge<=U and a>z.t0_spine_pos
    assert pd.isna(z.c1_deletion_spine_pos) or z.c1_deletion_spine_pos>q
    tick=TICK[z.instrument];end=min(int(ts[q])+horizon*M,int(z.calendar_close_ns))
    if version==1:
        lower=L-tick;upper=U+tick
        if op[b]>=upper:d=1;price=float(op[b])
        elif op[b]<=lower:d=-1;price=float(op[b])
        elif hi[b]>=upper and lo[b]<=lower:
            assert z.ambiguous_first_side==1 and z.actual_direction==0 and z.exit_pos==b
            assert z.gross_points==lower-upper
            return
        elif hi[b]>=upper:d=1;price=upper
        else:
            assert lo[b]<=lower;d=-1;price=lower
        stop=lower if d==1 else upper;goal=None
    else:
        assert L<=op[q]<=U and L<=cl[q]<=U
        if lo[q]<L and hi[q]<=U:d=1;extreme=lo[q]
        else:
            assert hi[q]>U and lo[q]>=L;d=-1;extreme=hi[q]
        assert d==z.direction and L<=op[b]<=U
        price=float(op[b]);stop=float(extreme-d*tick);goal=U if d==1 else L
    exit_price=None
    for j in range(b,len(ts)):
        assert ts[j]==ts[q]+(j-q)*M and ts[j]<=end
        if j>b and (op[j]-stop)*d<=0:exit_price=float(op[j]);break
        if version==2 and target and j>b and (op[j]-goal)*d>=tick:exit_price=goal;break
        hitstop=(lo[j]<=stop) if d==1 else (hi[j]>=stop)
        hittarget=version==2 and target and ((hi[j]>=goal+tick) if d==1 else (lo[j]<=goal-tick))
        if hitstop:exit_price=stop;break
        if hittarget:exit_price=goal;break
        if ts[j]==end:exit_price=float(cl[j]);break
    assert exit_price is not None and z.entry_pos==b and z.exit_pos==j
    assert price==z.entry_price and exit_price==z.exit_price
    assert abs((exit_price-price)*d-z.gross_points)<1e-9
    assert abs(z.gross_points*POINT[z.instrument]-COST[z.instrument]-z.net_dollars)<1e-9

def main():
    m=market('NQ');pauses,_=recognition('NQ',k=3)
    reports=[]
    for version,key in [(1,'NQ_k3_stop_h5'),(2,'NQ_k3_reject_target_h5')]:
        module=breakout if version==1 else reject
        config=next(x for x in module.family() if x['id']==key)
        decisions=module.decisions(pauses,m,'stop') if version==1 else module.decisions(pauses,m)
        fresh=module.replay(decisions,m,config).sort_values('q').reset_index(drop=True)
        saved=pd.read_parquet(OUT/f'all_decisions_v{version}.parquet',filters=[('variant','==',key)]).sort_values('q').reset_index(drop=True)
        pd.testing.assert_frame_equal(fresh[saved.columns],saved,check_dtype=False)
        trades=fresh.loc[fresh.day_known&(fresh.status>0)]
        for z in trades.itertuples():direct(z,m,version,version==2,5)
        fresh.to_parquet(OUT/f'illustrative_v{version}_replay.parquet',index=False)
        years=[]
        daily=pd.read_parquet(OUT/f'daily_v{version}.parquet').set_index('date')[key]
        count=pd.read_parquet(OUT/f'counts_v{version}.parquet').set_index('date')[key]
        for name,start,stop in [('2006-2012','2006-01-01','2012-12-31'),('2013-2019','2013-01-01','2019-12-31'),('2020-2026','2020-01-01','2026-12-31')]:
            d=daily.loc[(daily.index>=start)&(daily.index<=stop)]
            z=trades.loc[(trades.date>=start)&(trades.date<=stop)]
            years.append(dict(period=name,trades=len(z),gross=float(z.gross_dollars.sum()),
                net=float(z.net_dollars.sum()),mean_net=float(z.net_dollars.mean()),
                known_days=int(d.notna().sum()),trade_days=z.date.nunique()))
        risks=trades.initial_risk_points*POINT['NQ']
        recent=trades.loc[trades.date>='2020-01-01']
        info=dict(variant=key,full_replay_equal=True,independent_trade_checks=len(trades),
            total=float(trades.net_dollars.sum()),median_risk=float(risks.median()),
            risk_q90=float(risks.quantile(.9)),worst_trade=float(trades.net_dollars.min()),
            average_trades_per_active_day_2020=float(len(recent)/recent.date.nunique()),
            trade_days_2020=recent.date.nunique(),known_days_2020=int(daily.loc[daily.index>='2020-01-01'].notna().sum()),
            exposure_age_native_quantiles=trades.age_native_bars.quantile([.1,.5,.9]).tolist(),epochs=years)
        if version==2:
            optim=module.replay(decisions,m,config,optimistic=True)
            delay=module.replay(decisions,m,config,delay=1)
            delay=delay.loc[delay.day_known&(delay.status>0)]
            info.update(optimistic_total=float(optim.loc[optim.day_known&(optim.status>0),'net_dollars'].sum()),
                one_minute_delay=dict(trades=len(delay),net=float(delay.net_dollars.sum())),
                without_best_ten_days=float(daily.sum()-daily.nlargest(10).sum()))
        reports.append(info)
        print('INDEPENDENT',key,len(trades),flush=True)
    # Reconcile all summaries to their retained daily money, including negative rules.
    for version in [1,2]:
        summary=pd.read_csv(HERE/f'summary_v{version}.csv')
        daily=pd.read_parquet(OUT/f'daily_v{version}.parquet').set_index('date')
        count=pd.read_parquet(OUT/f'counts_v{version}.parquet').set_index('date')
        for z in summary.itertuples():
            assert daily[z.id].sum()==z.total and count[z.id].sum()==z.trades
    dump(HERE/'verification.json',dict(illustrative_rules=reports,all_72_daily_results_reconciled=True))
    print(json.dumps(reports,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
