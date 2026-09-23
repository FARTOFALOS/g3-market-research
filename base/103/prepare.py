"""Finalize prefix census, calendar, simultaneous signals; NO trade outcomes.
Raw census target is observed-only when history has gaps. Never use that target
to declare a definite target-visited exclusion before resolving availability.
"""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
from census import load,ROOT,OUT,MINUTE

sys.path.append(str(ROOT/'work/103/deps'))
import exchange_calendars as xc

def main():
    a,_,_=load()
    files=[OUT/f'tf_{tf:04d}.parquet' for tf in range(1,1441)]
    assert all(f.exists() for f in files),'incomplete 1440-TF parent'
    p=pd.concat([pd.read_parquet(f) for f in files],ignore_index=True)
    assert not p.riz_id.duplicated().any()
    # SER-8788: the 16:15-16:30 ET scheduled halt ended 2021-06-28.
    # Earliest ancestry in this parent is July 2020. The 2020 settlement FAQ
    # confirms these trading hours; the 2021 notice specifies their end.
    ts=a['ts']; diff=np.diff(ts)
    ix=np.flatnonzero((diff==16*MINUTE)&(np.diff(a['session_id'])==0))+1
    et=pd.to_datetime(ts[ix],utc=True).tz_convert('America/New_York')
    scheduled=(et.year>=2020)&(et.strftime('%Y-%m-%d')<'2021-06-28')&(et.hour==16)&(et.minute==31)
    pause=np.zeros(len(ts),np.int64); pause[ix[scheduled]]=1; pause=np.cumsum(pause)
    p['raw_prefix_gap_count']=p.prefix_gap_count
    good=p.c3_start.notna()
    p.loc[good,'prefix_gap_count']=p.loc[good,'prefix_gap_count'].to_numpy()-\
        (pause[p.loc[good,'t0_spine_pos'].to_numpy(np.int64)]-pause[p.loc[good,'c3_start'].to_numpy(np.int64)])
    p['gap_before_first_span_count']=np.nan
    has_span=good&p.first_span_pos.notna()
    A=p.loc[has_span,'c3_start'].to_numpy(np.int64)
    B=p.loc[has_span,'first_span_pos'].to_numpy(np.int64)
    p.loc[has_span,'gap_before_first_span_count']=a['gaps'][B]-a['gaps'][A]-(pause[B]-pause[A])
    assert (p.loc[has_span,'gap_before_first_span_count']>=0).all()
    # E is fixed when C3..first-span has no gap. An observed later visit to E
    # then excludes the RIZ even if a different minute before T0 is missing.
    # With a pre-span gap, the true E may differ, so an apparent visit is unknown.
    fixed_E=p.gap_before_first_span_count==0
    certain_visit=(p.already_visited==1)&fixed_E
    p['structural_status']=np.select(
        [p.target.isna(),~p.origin_aligned,certain_visit,p.prefix_gap_count>0,p.already_visited==1],
        ['ancestry_unavailable','opposite_origin','target_already_visited','prefix_unknown','target_already_visited'],
        default='structural_signal')
    cal=xc.get_calendar('XNYS',start='2021-01-01',end='2025-12-31')
    sch=cal.schedule[['open','close']].copy()
    sch['date']=sch.index.strftime('%Y-%m-%d')
    sch['entry_start_ns']=[pd.Timestamp(d+' 03:00',tz='America/New_York').value for d in sch.date]
    sch['close_ns']=sch['close'].astype('int64')
    # pandas/exchange_calendars versions can use us instead of ns.
    sch['close_ns']=[pd.Timestamp(v).value for v in sch['close']]
    sch.to_csv(ROOT/'base/103/calendar.csv',index=False)
    dates=pd.to_datetime(p.t0_ts_ns,utc=True).dt.tz_convert('America/New_York').dt.strftime('%Y-%m-%d')
    p['session_date']=dates
    p=p.merge(sch[['date','entry_start_ns','close_ns']],left_on='session_date',right_on='date',how='left',validate='many_to_one')
    p['clock_status']=np.select([p.close_ns.isna(),p.t0_ts_ns<p.entry_start_ns,p.t0_ts_ns>=p.close_ns],
        ['non_XNYS_day','before_03','at_or_after_close'],default='entry_window')
    p['signal']=(p.structural_status=='structural_signal')&(p.clock_status=='entry_window')
    p.to_parquet(ROOT/'work/103/parent.parquet',index=False)
    sig=p[p.signal].copy()
    groups=sig.groupby('t0_spine_pos').agg(rows=('riz_id','size'),directions=('direction','nunique'),
        targets=('target','nunique'),far_levels=('far','nunique'))
    # Same q, direction, E, far means identical subsequent rule and bindings.
    unique=sig.drop_duplicates(['t0_spine_pos','direction','target','far'])
    stats={'parent':len(p),'cells':1440,'sessions':len(sch),
        'structural_status':p.structural_status.value_counts().to_dict(),
        'clock_status':p.clock_status.value_counts().to_dict(),
        'signals':len(sig),'signal_minutes':len(groups),'distinct_actions':len(unique),
        'minutes_with_multiple_actions':int((unique.groupby('t0_spine_pos').size()>1).sum()),
        'minutes_with_opposite_directions':int((groups.directions>1).sum()),
        'signal_remaining_quantiles':sig.remaining.quantile([0,.1,.5,.9,1]).to_dict(),
        'signal_tf_quantiles':sig.tf_minutes.quantile([0,.1,.5,.9,1]).to_dict(),
        'prefix_unknown_in_entry_window':int(((p.structural_status=='prefix_unknown')&(p.clock_status=='entry_window')).sum()),
        'calendar_package':xc.__version__,
        'scheduled_2020_2021_pauses_not_missing':int(scheduled.sum()),
        'pause_sources':['https://www.cmegroup.com/content/dam/cmegroup/notices/ser/2021/06/SER-8788.pdf',
        'https://www.cmegroup.com/education/articles-and-reports/faq-daily-settlement-price-determination-time-change.html']}
    (ROOT/'base/103/prefix_census.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
    print(json.dumps(stats,indent=2))

if __name__=='__main__': main()
