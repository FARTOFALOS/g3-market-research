"""Frozen version replay, independent price reconciliation and economic sensitivities."""
import json
import platform
import numpy as np
import pandas as pd
from follow import HERE,OUT,INPUT,ROOT,POINT,COST,M,market,replay,decide,dump,sha
from reversal import signal_rows


def curve_stats(x):
    x=np.asarray(x,dtype=float);x=x[np.isfinite(x)];e=np.r_[0,np.cumsum(x)]
    return dict(total=float(x.sum()),days=len(x),mean_day=float(x.mean()),
                drawdown=float(np.max(np.maximum.accumulate(e)-e)))


def main():
    v=json.loads((HERE/'selected_v2.json').read_text());ins=v['instrument']
    cal=pd.read_parquet(HERE.parent/'S-04/calendar.parquet');m=market(ins)
    # Recompute all recognition from the fresh frozen-field copy, not signal cache.
    p=pd.read_parquet(INPUT/f'{ins}_objects.parquet')
    rows=signal_rows(p,m,cal,v['k']);sub=decide(rows,v['wait'])
    cov=pd.read_parquet(INPUT/f'{ins}_coverage.parquet');mask=cov.known.to_numpy()
    r=replay(sub,m,v);r['day_known']=mask[r.day.to_numpy()]
    stored=pd.read_parquet(OUT/'all_decisions_v1.parquet')
    stored=stored.loc[stored.variant==v['id']].sort_values('day').reset_index(drop=True)
    r=r.sort_values('day').reset_index(drop=True)
    pd.testing.assert_frame_equal(r[stored.columns],stored,check_dtype=False)
    valid=r.loc[r.day_known&(r.status>0)].copy()
    ts=m['close_ts_utc_ns'];op=m['open'];cl=m['close']
    # Independent direct clock lookup of every timed exit and fill.
    for z in valid.itertuples():
        q=int(z.q);b=q+1
        target=min(int(ts[q])+v['horizon']*M,int(z.calendar_close_ns))
        end=int(np.searchsorted(ts,target))
        assert ts[end]==target
        expected=(float(cl[end])-float(op[b]))*int(z.direction)*POINT[ins]-COST[ins]
        assert abs(expected-z.net_dollars)<1e-9
        assert end==z.exit_pos and b==z.entry_pos
        # Candle decisions independently checked on raw quotes for every selected object.
        k=v['k'];t=int(z.t0_spine_pos);direction=-int(z.direction)
        cc=np.asarray(cl[q-k:q+1])
        assert np.all(np.diff(cc)*direction>0)
        assert (cc[0]-z.edge)*direction<0 and (cc[-1]-z.edge)*direction>0
        assert 0<q-t<=v['wait']
    r.to_parquet(OUT/'selected_v2_replay.parquet',index=False)
    daily=pd.read_parquet(OUT/'daily_v1.parquet').set_index('date')[v['id']]
    years=[]
    for year in range(2006,2027):
        z=valid.loc[valid.date.str.startswith(str(year))]
        d=daily.loc[daily.index.str.startswith(str(year))]
        years.append(dict(year=year,known_days=int(d.notna().sum()),trades=len(z),
                          net=float(z.net_dollars.sum()),frequency=len(z)/max(1,d.notna().sum())))
    pd.DataFrame(years).to_csv(HERE/'yearly_v2.csv',index=False)
    delayed=replay(sub,m,v,delay=1);delayed['day_known']=mask[delayed.day.to_numpy()]
    delayed.to_parquet(OUT/'selected_v2_delay1.parquet',index=False)
    latency=delayed.loc[delayed.day_known&(delayed.status>0)]
    sensitivities=[dict(check='base',trades=len(valid),net=float(valid.net_dollars.sum()))]
    for cost in [25.,45.]:
        sensitivities.append(dict(check=f'round_cost_{cost}',trades=len(valid),net=float((valid.gross_dollars-cost).sum())))
    sensitivities.append(dict(check='one_minute_entry_delay',trades=len(latency),net=float(latency.net_dollars.sum())))
    sensitivities.append(dict(check='delay_and_cost_25',trades=len(latency),net=float((latency.gross_dollars-25.).sum())))
    pd.DataFrame(sensitivities).to_csv(HERE/'sensitivities_v2.csv',index=False)
    # Moving-block bootstrap for the selected rule: ordinary uncertainty, NOT selection correction.
    x=daily.to_numpy();rng=np.random.default_rng(20260907);block=20;B=3000
    means=[]
    for _ in range(B):
        starts=rng.integers(0,len(x),size=(len(x)+block-1)//block)
        idx=((starts[:,None]+np.arange(block))%len(x)).ravel()[:len(x)]
        means.append(np.nanmean(x[idx]))
    ci=np.quantile(means,[.025,.975])
    net=valid.net_dollars.sort_values(ascending=False).to_numpy()
    # Candle-based intratrade heat is exact high/low exposure for the time-only rule.
    heat=valid.bar_heat_bound*POINT[ins];mfe=valid.bar_mfe_bound*POINT[ins]
    # Lifetime is descriptive only: no future deletion is a trading input.
    lifetimes=[]
    for tf,z in valid.groupby('tf_minutes'):
        path=ROOT/'data/field'/ins/'cells'/f'tf_{int(tf):04d}'/'passports.parquet'
        p=pd.read_parquet(path,columns=['riz_id','c1_deletion_spine_pos'])
        p=z[['riz_id','q']].merge(p,on='riz_id',how='left',validate='many_to_one')
        p['deleted_by_signal']=p.c1_deletion_spine_pos.notna()&(p.c1_deletion_spine_pos<=p.q)
        lifetimes.append(p)
    life=pd.concat(lifetimes,ignore_index=True).sort_values('q')
    life[['riz_id','q','deleted_by_signal','c1_deletion_spine_pos']].to_parquet(
        OUT/'selected_v2_lifecycle.parquet',index=False)
    recognized=rows.loc[rows.recognition_status==1]
    recent_minutes=recognized.loc[recognized.date>='2020-01-01'].groupby('date').q.nunique()
    info=dict(version=v,replay_equal=True,direct_price_checks=len(valid),
        calendar_days=len(cal),known_days=int(daily.notna().sum()),unknown_days=int(daily.isna().sum()),
        recognized_objects=int((rows.recognition_status==1).sum()),
        unique_signal_minutes=int(rows.loc[rows.recognition_status==1,'q'].nunique()),
        eligible_objects=len(rows),
        signal_minutes_per_event_day_2020_quantiles=np.quantile(recent_minutes,[.1,.5,.9]).tolist(),
        deleted_by_signal=int(life.deleted_by_signal.sum()),
        decision_days=len(r),entry_cancelled=int((r.status==-1).sum()),
        unknown_outcomes_in_all_decisions=int(r.status.isin([-2,-3]).sum()),
        unknown_outcomes_on_complete_days=int((r.day_known&r.status.isin([-2,-3])).sum()),
        trades=len(valid),**curve_stats(daily),mean_trade=float(valid.net_dollars.mean()),
        median_trade=float(valid.net_dollars.median()),win_rate=float((valid.net_dollars>0).mean()),
        worst_trade=float(valid.net_dollars.min()),best_trade=float(valid.net_dollars.max()),
        mean_wait=float(valid.wait_minutes.mean()),median_wait=float(valid.wait_minutes.median()),
        wait_q10_q90=np.quantile(valid.wait_minutes,[.1,.9]).tolist(),
        heat_dollars_q50_q90_q99_max=np.quantile(heat,[.5,.9,.99,1]).tolist(),
        mfe_dollars_q50_q90_q99_max=np.quantile(mfe,[.5,.9,.99,1]).tolist(),
        selected_rule_daily_mean_ci95=ci.tolist(),bootstrap=dict(block=block,repetitions=B,
        limitation='Approximate moving-day-block interval under weak stability; unadjusted for adaptive rule choice.'),
        without_best_day=float(net[1:].sum()),without_best_10_days=float(net[10:].sum()),
        best_one_percent_share=float(net[:max(1,int(np.ceil(len(net)*.01)))].sum()/net.sum()),
        without_best_one_percent=float(net[max(1,int(np.ceil(len(net)*.01))):].sum()),
        sensitivities=sensitivities,python=platform.python_version())
    dump(HERE/'verification_v2.json',info)
    print(json.dumps(info,ensure_ascii=False,indent=2),flush=True)
    print(pd.DataFrame(years).to_string(index=False),flush=True)


if __name__=='__main__':main()
