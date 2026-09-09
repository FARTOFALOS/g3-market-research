"""After-outcome diagnostics, not a new selection rule for v1."""
import json
import numpy as np
import pandas as pd
import run as s

def scalar_oracle(m,r):
    # Deliberately independent event search: first hit indices via arrays,
    # not the production event loop. Used only on continuous entered paths.
    p=int(r.p);b=p+1;e=int(r.exit_pos);d=int(r.side)
    op,hi,lo,cl=[m[k][b:e+1] for k in ['open','high','low','close']]
    assert np.all(np.diff(m['close_ts_utc_ns'][b:e+1])==s.M)
    events=[]
    for k in np.flatnonzero((op-r.stop)*d<=0):events.append((int(k),0,float(op[k])))
    for k in np.flatnonzero((op-r.target)*d>=.25):events.append((int(k),1,float(r.target)))
    sh=lo<=r.stop if d>0 else hi>=r.stop
    th=hi>=r.target+.25 if d>0 else lo<=r.target-.25
    for k in np.flatnonzero(sh):events.append((int(k),2,float(r.stop)))
    for k in np.flatnonzero(th):events.append((int(k),3,float(r.target)))
    events.append((len(op)-1,4,float(cl[-1])))
    event=min(events)
    if r.execution=='time':
        assert event[1]==4 and m['close_ts_utc_ns'][e]==min(int(m['close_ts_utc_ns'][p])+60*s.M,int(r.session_end))
    else:
        assert event[1]!=4
    return (event[2]-float(op[0]))*d,b+event[0]

def main():
    m,cal=s.load();all_t=pd.read_parquet(s.OUT/'trades.parquet')
    t=all_t[(all_t.variant=='v1')&(all_t.bound=='lower')&all_t.net_usd.notna()].copy()
    for _,r in t.iterrows():
        gp,ep=scalar_oracle(m,r);assert gp==r.gross_points and ep==r.exit_pos
    yearly=t.groupby(t.date.str[:4]).net_usd.agg(['size','sum','mean'])
    yearly.to_csv(s.HERE/'yearly.csv')
    rec=pd.read_parquet(s.OUT/'recognition.parquet').set_index('date')
    # Common calendar includes true zero decisions; unavailable prefixes stay NaN.
    daily=pd.DataFrame(index=rec.index)
    for variant in ['v1','h30','h120','delay1']:
        z=all_t[(all_t.variant==variant)&(all_t.bound=='lower')]
        daily[variant]=0.
        daily.loc[rec.status.str.startswith('unknown'),variant]=np.nan
        known=z[z.net_usd.notna()];daily.loc[known.date,variant]=known.net_usd.to_numpy()
        unknown=z[z.execution.str.startswith('unknown')];daily.loc[unknown.date,variant]=np.nan
    daily.to_csv(s.HERE/'daily.csv')
    # Whole calendar months as blocks: dependence within month retained. This
    # describes this observed sample, not independent discovery validation.
    monthly=daily.groupby(daily.index.str[:7]).sum(min_count=1).dropna()
    rng=np.random.default_rng(808);vals=monthly.v1.to_numpy()
    bs=np.array([rng.choice(vals,len(vals),replace=True).sum() for _ in range(10000)])
    adjusted=[]
    # Four fixed neighbour variants, simultaneously inspected: Bonferroni
    # percentile intervals only for this family; earlier research is not covered.
    for col in monthly:
        x=monthly[col].to_numpy();v=np.array([rng.choice(x,len(x),replace=True).sum() for _ in range(10000)])
        adjusted.append(dict(variant=col,net=float(x.sum()),family_interval_98_75=np.quantile(v,[.00625,.99375]).tolist()))
    recent=t[t.date>='2020'];net=t.net_usd;best=net.nlargest(5).sum()
    z={'independent_execution_agreements':len(t),'full_month_block_bootstrap_95_total_usd':np.quantile(bs,[.025,.975]).tolist(),
       'four_variant_family':adjusted,'known_decision_days':int(daily.v1.notna().sum()),'unknown_decision_days':int(daily.v1.isna().sum()),
       'active_day_fraction_known':len(t)/int(daily.v1.notna().sum()),'net_at_cost30':float((net-15).sum()),
       'net_without_best_five':float(net.sum()-best),'largest_five_profit':float(best),
       'recent_2020_count':len(recent),'recent_average_usd':float(recent.net_usd.mean()),
       'yearly':yearly.reset_index().to_dict(orient='records')}
    # Adverse move while position is open: include the whole last candle as
    # an upper bound, never pretend its post-exit extreme was necessarily held.
    mtm=[];equity=0.;peak=0.;drawdown_bound=0.
    for _,r in t.iterrows():
        b=int(r.entry_pos);e=int(r.exit_pos);d=int(r.side)
        for j in range(b,e+1):
            lowmark=equity+((m['low'][j]-r.entry) if d>0 else (r.entry-m['high'][j]))*20-15
            drawdown_bound=max(drawdown_bound,peak-lowmark)
            if j<e:
                mark=equity+(m['close'][j]-r.entry)*d*20-15
                peak=max(peak,mark)
        equity+=r.net_usd;peak=max(peak,equity)
    z['drawdown_from_minute_close_peaks_to_bar_adverse_bound_usd']=float(drawdown_bound)
    stopped=t[t.execution.str.startswith('stop')];later=0
    for _,r in stopped.iterrows():
        a=int(r.exit_pos)+1;e=int(r.p)+60
        if e>=len(m['close']):continue
        # Only strictly later candles; no guessed order on the stop candle.
        hit=m['high'][a:e+1]>=r.target+.25 if r.side>0 else m['low'][a:e+1]<=r.target-.25
        later+=int(np.any(hit))
    z['stopped_trades']=len(stopped)
    z['stops_followed_by_target_on_later_bar_before_60min']=later
    manifest=json.loads((s.ROOT/'data/market/NQ/manifest.json').read_text(encoding='utf-8'))
    hashes={}
    for k in s.KEYS:
        p=s.ROOT/'data/market/NQ'/f'{k}.npy';h=s.sha(p)
        assert h==manifest['array_sha256'][f'{k}.npy']
        hashes[str(p.relative_to(s.ROOT))]=h
    z['verified_input_hashes']=hashes
    z['corpus_id']=manifest['corpus_id']
    z['limitations']='Month resampling assumes representative exchangeable months; regime shift and unknown prior selection not corrected. Incomplete months contain observed days only.'
    (s.HERE/'validation.json').write_text(json.dumps(z,indent=2),encoding='utf-8')
    print(json.dumps(z,indent=2))

if __name__=='__main__':main()
