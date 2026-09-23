"""Conditional diagnostics, temporal uncertainty and library comparison.
Never label resolved-only results as the complete policy result.
"""
import csv
import json
import numpy as np
import pandas as pd
from census import ROOT,load,MINUTE

def csv_period(path):
    with path.open(encoding='utf-8-sig',newline='') as f:
        rows=[r for r in csv.DictReader(f) if '2021-01-01'<=r['date'][:10]<'2026-01-01']
    return pd.DataFrame(rows)

def intervals(num,den,block):
    rng=np.random.default_rng(10319)
    n=len(num); results=[]
    for _ in range(10000):
        starts=rng.integers(0,n-block+1,size=int(np.ceil(n/block)))
        ix=(starts[:,None]+np.arange(block)).ravel()[:n]
        ds=den[ix].sum()
        results.append(num[ix].sum()/ds if ds else np.nan)
    return [float(x) for x in np.nanquantile(results,[.025,.975])]

def main():
    a,_,_=load(); out=ROOT/'work/103/replay'; report={}
    cal=pd.read_csv(ROOT/'base/103/calendar.csv'); dates=cal.date.tolist()
    for v in ('v1','v2'):
        t=pd.read_parquet(out/f'{v}_trades.parquet')
        day=pd.read_parquet(out/f'{v}_daily.parquet')
        known=day.status=='known'; k=t[t.status=='known']; y=day.net.fillna(0).to_numpy(); mask=known.to_numpy(float)
        kd=day[known]; positive=k[k.net>0]; negative=k[k.net<0]
        # Do not join separate known runs into a made-up equity curve.
        maxdd=0.; peak=0.; eq=0.
        for r in day.itertuples():
            if r.status!='known':
                peak=eq=0.; continue
            eq+=r.net; peak=max(peak,eq); maxdd=max(maxdd,peak-eq)
        costs={str(c):float((kd.net+(1-c)*kd.entries).mean()) for c in (0,1,2)}
        summary={'scope':'conditional on completely identified sessions; NOT full policy economics',
            'known_sessions':int(known.sum()),'unknown_sessions':int((~known).sum()),
            'known_zero_sessions':int(((day.entries==0)&known).sum()),
            'net_per_known_session':float(kd.net.mean()),
            'ci95_conditional_by_block':{str(b):intervals(y,mask,b) for b in (5,20,60)},
            'net_per_known_session_cost_sensitivity':costs,
            'known_sessions_break_even_roundtrip_cost':float((kd.net+kd.entries).sum()/kd.entries.sum()),
            'known_trade_net_mean':float(k.net.mean()),'known_trade_win_rate':float((k.net>0).mean()),
            'known_trade_avg_win':float(positive.net.mean()),'known_trade_avg_loss':float(negative.net.mean()),
            'known_trade_min_net':float(k.net.min()),'known_trade_max_net':float(k.net.max()),
            'known_trade_mae_quantiles':k.mae.quantile([.5,.9,.99,1]).to_dict(),
            'known_trade_held_quantiles':k.held_minutes.quantile([.5,.9,.99,1]).to_dict(),
            'known_trade_held_total_minutes':float(k.held_minutes.sum()),
            'known_session_worst':float(kd.net.min()),'max_drawdown_within_contiguous_known_runs':float(maxdd),
            'known_session_sum':float(kd.net.sum()),
            'known_session_sum_without_best_5pct':float(kd.net.sort_values().iloc[:len(kd)-max(1,int(np.ceil(.05*len(kd))))].sum()),
            'unknown_full_policy_drawdown':True,
            'known_trade_tf_quantiles':k.tf.quantile([0,.5,.9,1]).to_dict()}
        summary['by_year_conditional']=kd.assign(year=kd.date.str[:4]).groupby('year').agg(
            sessions=('net','size'),net=('net','mean'),entries=('entries','sum')).reset_index().to_dict('records')
        report[v]=summary
    pairs=pd.read_parquet(out/'v1_pairs.parquet'); p=pairs[pairs.continue_status=='known'].copy()
    daily=p.groupby('date').delta_exit_minus_continue.agg(['sum','size']).reindex(dates,fill_value=0)
    report['pairs']={'scope':'events in certified v1 position prefixes, with known counterfactual continuation; not all potential events',
        'observed_events':len(pairs),'known_pairs':len(p),'unknown_counterfactual':len(pairs)-len(p),
        'mean_exit_minus_continue':float(p.delta_exit_minus_continue.mean()),
        'ci95_conditional_by_block':{str(b):intervals(daily['sum'].to_numpy(),daily['size'].to_numpy(),b) for b in (5,20,60)},
        'recovered_winners':int(p.recovered_winner.sum()),
        'recovered_winners_share':float(p.recovered_winner.mean()),
        'exit_better_fraction':float((p.delta_exit_minus_continue>0).mean()),
        'delta_quantiles':p.delta_exit_minus_continue.quantile([0,.01,.1,.5,.9,.99,1]).to_dict()}
    # Existing library results: read only 2021-2025. No rerun, no data after 2025.
    s07=csv_period(ROOT/'setups/S-17/engine_history_NQ.csv')
    s18=csv_period(ROOT/'setups/S-18/band_daily_NQ.csv')
    s18t=csv_period(ROOT/'setups/S-18/band_trades_NQ.csv')
    s07=s07.set_index('date'); s18=s18.set_index('date')
    s07active=set(s07[~s07.status.str.startswith('no_entry')].index)
    s18active=set(s18[pd.to_numeric(s18.n_tr,errors='coerce')>0].index)
    lib={}
    calmap=cal.set_index('date')
    for v in ('v1','v2'):
        t=pd.read_parquet(out/f'{v}_trades.parquet'); day=pd.read_parquet(out/f'{v}_daily.parquet')
        known_days=set(day[day.status=='known'].date); taken=set(t.date)
        shared07=[]; shared18=[]
        total_minutes=0; overlap07=0; overlap18=0; same07=0; same18=0
        for r in t[t.status=='known'].itertuples():
            start=int(a['ts'][int(r.entry_pos)]-MINUTE)
            end=int(a['ts'][int(r.exit_pos)]-(0 if r.reason=='session_close' else MINUTE))
            total_minutes+=(end-start)/MINUTE
            op=pd.Timestamp(calmap.loc[r.date,'open']).value
            if r.date in s07active:
                z=s07.loc[r.date]
                if z.s07_exact=='True' and z.s07_exit_min:
                    b=op+3*MINUTE; e=b+int(float(z.s07_exit_min))*MINUTE
                    dur=max(0,min(end,e)-max(start,b))/MINUTE
                    overlap07+=dur; same07+=dur*(int(float(z.side))==r.direction)
            for z in s18t[s18t.date==r.date].itertuples():
                b=op+(int(float(z.k_in))-1)*MINUTE
                close=int(calmap.loc[r.date,'close_ns']); K=(close-op)//MINUTE
                e=close if int(float(z.k_out))==K else op+(int(float(z.k_out))-1)*MINUTE
                dur=max(0,min(end,e)-max(start,b))/MINUTE
                overlap18+=dur; same18+=dur*(int(float(z.side))==r.direction)
        for r in day[day.status=='known'].itertuples():
            if r.date in s07.index:
                z=s07.loc[r.date]
                if z.s07_exact=='True' and z.s07_usd:
                    shared07.append((r.net,float(z.s07_usd)))
            if r.date in s18.index and s18.loc[r.date,'status']=='ok':
                shared18.append((r.net,float(s18.loc[r.date,'usd'])))
        corr=lambda x:float(np.corrcoef(np.array(x).T)[0,1]) if len(x)>2 else None
        lib[v]={'s19_certified_active_sessions':len(taken),'s07_active_sessions':len(s07active),
                's18_active_sessions':len(s18active),'shared_active_s07':len(taken&s07active),
                'shared_active_s18':len(taken&s18active),
                'known_held_minutes':total_minutes,'overlap_s07_minutes':overlap07,'same_direction_s07_minutes':same07,
                'overlap_s18_minutes':overlap18,'same_direction_s18_minutes':same18,
                's07_common_exact_days_for_correlation':len(shared07),'s07_daily_correlation':corr(shared07),
                's18_common_exact_days_for_correlation':len(shared18),'s18_daily_correlation':corr(shared18),
                'qualification':'descriptive selected known support; not independence or executable portfolio'}
    report['library_comparison']=lib
    # A common fully observed calendar is a diagnostic, not an ex-ante signal.
    d1=pd.read_parquet(out/'v1_daily.parquet')
    d2=pd.read_parquet(out/'v2_daily.parquet')
    joined=d1.merge(d2,on='date',suffixes=('_v1','_v2'),validate='one_to_one').sort_values('date')
    common=((joined.status_v1=='known')&(joined.status_v2=='known')).to_numpy(float)
    v2_only=((joined.status_v1!='known')&(joined.status_v2=='known')).to_numpy(float)
    def clean_slice(mask,version):
        net=joined[f'net_{version}'].fillna(0).to_numpy()
        entries=joined[f'entries_{version}'].to_numpy()
        n=int(mask.sum())
        return {'sessions':n,'trades':int((entries*mask).sum()),
                'net_sum':float((net*mask).sum()),'net_per_session':float((net*mask).sum()/n),
                'zero_cost_mean':float(((net+entries)*mask).sum()/n),
                'ci95_block20':intervals(net*mask,mask,20)}
    extra_dates=set(joined.loc[v2_only.astype(bool),'date'])
    extra_trades=pd.read_parquet(out/'v2_trades.parquet')
    extra_trades=extra_trades[(extra_trades.date.isin(extra_dates))&(extra_trades.status=='known')]
    report['clean_base_decision']={
        'common_fully_known':{'v1':clean_slice(common,'v1'),'v2':clean_slice(common,'v2')},
        'v2_known_while_v1_unknown':{
            'v2':clean_slice(v2_only,'v2'),
            'exit_reason':extra_trades.groupby('reason').agg(
                trades=('net','size'),net_sum=('net','sum')).reset_index().to_dict('records')},
        'qualification':('Common-day membership is determined by both policy paths and data completeness; '
                         'it is not an ex-ante trading selector or a clean holdout. Full-population signs remain unknown.')}
    (ROOT/'base/103/diagnostics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
