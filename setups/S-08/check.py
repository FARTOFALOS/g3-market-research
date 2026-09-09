"""Independent checks of timing, gaps, ambiguous ordering and prefix recognition."""
import json
import numpy as np
import pandas as pd
import run as s

def fixture():
    m={'close_ts_utc_ns':np.arange(6,dtype=np.int64)*s.M,
       'open':np.array([100.,100.,100.,100.,100.,100.]),
       'high':np.array([101.,101.,101.,101.,101.,101.]),
       'low':np.array([99.,99.,99.,99.,99.,99.]),
       'close':np.array([100.,100.,100.,100.,100.,100.])}
    r=dict(p=0,side=1,stop=95.,target=110.,prev_low=90.,prev_high=115.,session_end=5*s.M)
    return m,r

def checks():
    passed=[]
    m,r=fixture();z=s.execute(m,r,3);assert z['execution']=='time' and z['held_minutes']==3
    passed.append('wall_clock_hold')
    m,r=fixture();m['low'][1]=94.;z=s.execute(m,r,3);assert z['exit']==95 and z['exit_pos']==1
    passed.append('stop_in_entry_minute')
    m,r=fixture();m['open'][2]=92.;m['low'][2]=91.;z=s.execute(m,r,3);assert z['exit']==92
    passed.append('stop_gap_actual_open')
    m,r=fixture();m['high'][1]=110.;z=s.execute(m,r,3);assert z['execution']=='time'
    m['high'][1]=110.25;z=s.execute(m,r,3);assert z['exit']==110.
    passed.append('target_touch_vs_trade_through')
    m,r=fixture();m['low'][1]=94.;m['high'][1]=111.
    assert s.execute(m,r,3)['exit']==95. and s.execute(m,r,3,optimistic=True)['exit']==110.
    passed.append('both_orders_retained')
    m,r=fixture();m={k:np.delete(v,2) for k,v in m.items()};assert s.execute(m,r,3)['execution']=='unknown_path'
    passed.append('gap_unknown_not_zero')
    m,r=fixture();m['open'][1]=116.;assert s.execute(m,r,3)['execution']=='cancelled_open'
    passed.append('entry_outside_cancel')
    m,r=fixture();m['low'][1]=94.;orig=s.execute(m,r,3)
    mm={'close_ts_utc_ns':m['close_ts_utc_ns'],'open':200-m['open'],'high':200-m['low'],'low':200-m['high'],'close':200-m['close']}
    rr=dict(r,side=-1,stop=105.,target=90.,prev_low=85.,prev_high=110.)
    assert s.execute(mm,rr,3)['gross_points']==orig['gross_points']
    passed.append('long_short_mirror')
    m,cal=s.load();n=0;neg=0
    for i in range(1,len(cal)):
        row=s.recognize(m,cal.iloc[i-1],cal.iloc[i])
        if row['status']!='signal':continue
        p=row['p'];cut={k:v[:p+1] for k,v in m.items()}
        assert s.recognize(cut,cal.iloc[i-1],cal.iloc[i])==row
        early={k:v[:p] for k,v in m.items()}
        assert s.recognize(early,cal.iloc[i-1],cal.iloc[i])['status']!='signal'
        n+=1;neg+=1
    result={'synthetic':passed,'real_signal_prefix_checks':n,'one_minute_earlier_not_signal':neg}
    (s.HERE/'checks_before_outcomes.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':checks()
