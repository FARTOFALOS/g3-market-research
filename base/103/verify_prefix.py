"""Independent native-grid and prefix-truncation checks before economics."""
from types import SimpleNamespace
import json
import numpy as np
import pandas as pd
from census import load,grid,facts,ROOT,OUT
from g3riz.entry_check import native_grid

a,s,_=load()
# Adapt only the last incomplete archive session to the permitted endpoint.
keep=s['first_minute_pos']<len(a['ts'])
s={k:v[keep] for k,v in s.items()}
s['stop_minute_pos']=np.minimum(s['stop_minute_pos'],len(a['ts']))
market=SimpleNamespace(close_ts_utc_ns=a['ts'],sessions=s)
checks=[]
for tf in (1,5,54,240):
    st,en=grid(a['ts'],s['first_minute_pos'],s['stop_minute_pos'],s['session_open_utc_ns'],tf)
    rs,re=native_grid(market,tf)
    assert np.array_equal(st,rs) and np.array_equal(en,re)
    p=pd.read_parquet(OUT/f'tf_{tf:04d}.parquet')
    sample=p.sort_values('riz_id').iloc[::max(1,len(p)//12)].head(12)
    count=0
    for r in sample.itertuples():
        q=int(r.t0_spine_pos); b=int(r.first_span_pos)
        def run(cut):
            return facts(a['high'][:cut],a['low'][:cut],a['close'][:cut],a['ts'][:cut],a['gaps'][:cut],st,en,
                np.array([r.precursor_native_bar_index]),np.array([r.precursor_formed_spine_pos]),
                np.array([r.first_span_native],dtype=np.int64),np.array([b]),np.array([q]),np.array([r.direction]))
        full,clipped=run(len(a['ts'])),run(q+1)
        assert np.array_equal(full,clipped,equal_nan=True)
        if np.isfinite(full[0,0]):
            start=int(full[0,0]); sign=r.direction
            vals=a['high'][start:b+1] if sign>0 else a['low'][start:b+1]
            E=float(max(vals) if sign>0 else min(vals))
            future=a['high'][b+1:q+1] if sign>0 else a['low'][b+1:q+1]
            visited=bool(any(sign*(future-E)>=0))
            assert E==r.target and visited==bool(r.already_visited)
        count+=1
    checks.append({'tf':tf,'native_bars_compared':len(st),'full_vs_truncated_cases':count})
result={'checks':checks,'status':'pass','scope':'grid and prefix facts; does not certify selection, execution or economics'}
(ROOT/'base/103/prefix_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))
