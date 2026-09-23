"""S-19 v1 parent census: prefix-only, all NQ TF1..1440, T0 2021..2025.
No outcome computation. Resume writes one research shard per TF; source read-only.
python -B base/103/census.py [--stop-tf 1440]
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import hashlib
import json
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from numba import njit
from g3riz.identity import build_identity

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'work/103/census'
MINUTE = 60_000_000_000
START = pd.Timestamp('2021-01-01', tz='UTC').value
STOP = pd.Timestamp('2026-01-01', tz='UTC').value

@njit
def grid(ts, first, stop, anchor, tf):
    starts = np.empty(len(ts), np.int64)
    ends = np.empty(len(ts), np.int64)
    n = 0
    for si in range(len(first)):
        a = first[si]; b = min(stop[si], len(ts))
        if a >= b:
            break
        prev = -1000000
        for p in range(a, b):
            k = ((ts[p]-anchor[si])//MINUTE-1)//tf
            if k != prev:
                if p > a:
                    ends[n-1] = p
                starts[n] = p
                n += 1
                prev = k
        ends[n-1] = b
    return starts[:n].copy(), ends[:n].copy()

@njit
def facts(H,L,C,ts,gaps,starts,ends,birth_idx,birth_pos,s1_idx,s1_pos,q,d):
    n = len(q)
    result = np.full((n, 8), np.nan)
    for i in range(n):
        j = birth_idx[i]; b = s1_pos[i]; p = q[i]; sj=s1_idx[i]
        if j < 2 or sj < 0 or b < 0 or b >= p:
            continue
        assert ends[j]-1 == birth_pos[i]
        assert ends[sj]-1 == b
        a = starts[j]
        if a > b:
            continue
        E = H[a] if d[i] > 0 else L[a]
        ep = a
        for k in range(a+1,b+1):
            v = H[k] if d[i] > 0 else L[k]
            if d[i]*(v-E)>0:
                E=v; ep=k
        visited=False
        for k in range(b+1,p+1):
            v=H[k] if d[i]>0 else L[k]
            if d[i]*(v-E)>=0:
                visited=True
        result[i,0]=a; result[i,1]=E; result[i,2]=ep
        result[i,3]=visited; result[i,4]=d[i]*(E-C[p])
        result[i,5]=gaps[p]-gaps[a]
        result[i,6]=starts[j-2]
        result[i,7]=(ts[p]-ts[a])/MINUTE
    return result

def load():
    mk = ROOT/'data/market/NQ'
    manifest=json.loads((mk/'manifest.json').read_text())
    src=json.loads((ROOT/'SOURCE_DATA.json').read_text())
    assert manifest['corpus_id']==src['instruments']['NQ']['corpus_id']
    ts=np.load(mk/'close_ts_utc_ns.npy',mmap_mode='r')
    limit=int(np.searchsorted(ts,STOP)); ts=ts[:limit]
    a={k:np.load(mk/f'{k}.npy',mmap_mode='r')[:limit] for k in ('open','high','low','close','session_id')}
    a['ts']=ts
    with np.load(mk/'sessions.npz') as z:
        sessions=dict(z)
    gaps=np.zeros(len(ts),np.int64)
    gaps[1:]=np.cumsum((np.diff(ts)!=MINUTE)&(np.diff(a['session_id'])==0))
    a['gaps']=gaps
    return a,sessions,manifest

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--stop-tf',type=int,default=1440)
    args=ap.parse_args(); OUT.mkdir(parents=True,exist_ok=True)
    a,s,manifest=load()
    codehash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    cols=['riz_id','corpus_id','tf_minutes','zone_top','zone_bottom','bullish',
          'precursor_formed_spine_pos','precursor_native_bar_index','t0_spine_pos',
          't0_ts_ns','t0_exit_side','t0_native_bar_index']
    for tf in range(1,args.stop_tf+1):
        dest=OUT/f'tf_{tf:04d}.parquet'; stamp=OUT/f'tf_{tf:04d}.json'
        if dest.exists() and stamp.exists():
            assert json.loads(stamp.read_text())['code_sha256']==codehash, 'stale census code'
            continue
        cell=ROOT/f'data/field/NQ/cells/tf_{tf:04d}'
        meta=json.loads((cell/'manifest.json').read_text())
        assert meta['status']=='complete'
        assert meta['build_identity']==build_identity(SimpleNamespace(manifest=manifest),'NQ',tf)
        p=pq.read_table(cell/'passports.parquet',columns=cols,
                        filters=[('t0_ts_ns','>=',START),('t0_ts_ns','<',STOP)]).to_pandas()
        assert p.empty or set(p.corpus_id)=={manifest['corpus_id']}
        e=pq.read_table(cell/'events.parquet',columns=['riz_id','market_spine_pos','native_bar_index','event_seq'],
                        filters=[('event_kind','=','accepted_span'),('event_ts_ns','<',STOP)]).to_pandas()
        e=e[e.riz_id.isin(p.riz_id)].sort_values(['riz_id','event_seq']).drop_duplicates('riz_id')
        e=e.rename(columns={'market_spine_pos':'first_span_pos','native_bar_index':'first_span_native'})
        p=p.merge(e,how='left',on='riz_id',validate='one_to_one')
        starts,ends=grid(a['ts'],s['first_minute_pos'],s['stop_minute_pos'],s['session_open_utc_ns'],tf)
        q=p.t0_spine_pos.to_numpy(np.int64)
        d=np.where(p.t0_exit_side=='north',1,-1).astype(np.int64)
        if len(p):
            j=p.t0_native_bar_index.to_numpy(np.int64)
            assert ((starts[j]<=q)&(ends[j]>q)).all()
        f=facts(a['high'],a['low'],a['close'],a['ts'],a['gaps'],starts,ends,
            p.precursor_native_bar_index.to_numpy(np.int64),p.precursor_formed_spine_pos.to_numpy(np.int64),
            p.first_span_native.fillna(-1).to_numpy(np.int64),p.first_span_pos.fillna(-1).to_numpy(np.int64),q,d)
        for k,name in enumerate(['c3_start','target','target_pos','already_visited','remaining',
                                  'prefix_gap_count','c1_start','ancestry_minutes']):
            p[name]=f[:,k]
        p['direction']=d; p['t0_close']=a['close'][q]
        p['far']=np.where(d>0,p.zone_bottom,p.zone_top)
        p['origin_aligned']=d==np.where(p.bullish,1,-1)
        # Unknown evidence does not get silently removed from the parent.
        p['structural_status']=np.select([
            p.target.isna(),~p.origin_aligned,p.already_visited==1,p.prefix_gap_count>0],
            ['ancestry_unavailable','opposite_origin','target_already_visited','prefix_unknown'],
            default='structural_signal')
        p.to_parquet(dest,index=False)
        stamp.write_text(json.dumps({'tf':tf,'rows':len(p),'code_sha256':codehash,
                        'corpus_id':manifest['corpus_id'],'manifest_sha256':hashlib.sha256((cell/'manifest.json').read_bytes()).hexdigest()}),encoding='utf-8')
        if tf<=5 or tf%100==0:
            print(tf,len(p),p.structural_status.value_counts().to_dict(),flush=True)
    files=[OUT/f'tf_{tf:04d}.parquet' for tf in range(1,args.stop_tf+1)]
    p=pd.concat([pd.read_parquet(f) for f in files],ignore_index=True)
    summary={'cells':len(files),'parent':len(p),'unique_t0':int(p.t0_spine_pos.nunique()),
             'statuses':p.structural_status.value_counts().to_dict(),
             'code_sha256':codehash,'corpus_id':manifest['corpus_id']}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary),flush=True)

if __name__=='__main__': main()
