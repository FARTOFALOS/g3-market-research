"""Bounded counterexamples for 060. Never read OHLC at/after cutoff."""
import json, pickle
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import run as s

def main():
    m,cal=s.load();ts=m['close_ts_utc_ns'];hi=m['high'];lo=m['low'];n=len(ts)
    source=s.ROOT/'work/060/visits.pkl'
    out=pickle.loads(source.read_bytes())
    q={'stored_clusters':len(out),'cutoff_pos':n,'stored_visit_starts_at_or_after_cutoff':0,
       'observation_budgets_cross_cutoff':0,'prefix_windows_with_gaps':0,
       'finite_windows_longer_than_2880_wall_minutes':0,'checked_starts_before_cutoff':0,
       'starts_not_containing_line':0,'examples':[]}
    keys=[]
    for row in out:
        p,nu,nr,side,u,line,nv,starts,gaps,fv,cen=row
        if p+2880>=n:q['observation_budgets_cross_cutoff']+=1
        if p>=30 and ts[p-1]-ts[p-30]!=29*s.M:q['prefix_windows_with_gaps']+=1
        e=min(p+2880,n-1)
        if p+2880<n and ts[e]-ts[p]>2880*s.M:q['finite_windows_longer_than_2880_wall_minutes']+=1
        for st in starts:
            a=p+st
            if a>=n:q['stored_visit_starts_at_or_after_cutoff']+=1;continue
            q['checked_starts_before_cutoff']+=1
            if not lo[a]<=line<=hi[a]:
                q['starts_not_containing_line']+=1
                if len(q['examples'])<3:q['examples'].append(dict(p=p,position=a,UTC=str(pd.Timestamp(int(ts[a]),tz='UTC')),line=line,low=float(lo[a]),high=float(hi[a]),side=side))
        if nv>=2 and u>0:
            for st in starts[:4]:
                a=p+st
                if a+12>=e:break
                hh=hi[a+1:e+1];ll=lo[a+1:e+1]
                away=np.flatnonzero(hh>=line+2*u) if side=='north' else np.flatnonzero(ll<=line-2*u)
                if len(away):keys.append((p,a+1+int(away[0])))
    c=Counter(keys)
    q['repeat_visit_away_decisions_d2']=len(keys)
    q['unique_cluster_away_decisions_d2']=len(c)
    q['duplicate_labels_of_same_departure_d2']=sum(v-1 for v in c.values())
    # A cold scene, chosen by row coordinate, not future outcome. Saved prefix
    # explicitly excludes the later events seen during manual inspection.
    cell=s.ROOT/'data/field/NQ/cells/tf_0054'
    p=pq.read_table(cell/'passports.parquet').to_pandas()
    rid='riz_01bc3138842bf7f259b33190ef30f6b9';p=p[p.riz_id==rid].iloc[0];pos=int(p.t0_spine_pos)
    ev=pq.read_table(cell/'events.parquet').to_pandas();ev=ev[(ev.riz_id==rid)&(ev.market_spine_pos<=pos)]
    ev.to_json(s.HERE/'riz_prefix_events.json',orient='records',indent=2)
    df=pd.DataFrame({k:m[k][pos-10:pos+1] for k in s.KEYS})
    df.to_csv(s.HERE/'riz_prefix_candles.csv',index=False)
    q['prefix_scene']={'riz_id':rid,'tf':54,'t0_pos':pos,'t0_UTC':str(pd.Timestamp(int(ts[pos]),tz='UTC')),
                      'bottom':float(p.zone_bottom),'top':float(p.zone_top),'accepted_span_count':int(p.t0_span_count)}
    # Hash evidence files; contents were not modified by this audit.
    paths=[source]+[s.ROOT/'work/060'/x for x in ['visits.py','magnet.py','second.py','sanity.py','control.py']]
    q['evidence_hashes']={str(p.relative_to(s.ROOT)):s.sha(p) for p in paths}
    (s.HERE/'audit_checks.json').write_text(json.dumps(q,indent=2),encoding='utf-8')
    print(json.dumps(q,indent=2))

if __name__=='__main__':main()
