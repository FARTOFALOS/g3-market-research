"""Executes FROZEN_EXTERNAL.md. External state only; no new prefix-derived features."""
import numpy as np, pyarrow.parquet as pq
from pathlib import Path
import open_y as M, transition as T
ROOT=Path('C:/Users/Admin/Claude/g3-market-research'); NS=60_000_000_000
OTHER_TF=[5,15,30,90,120]
_z={}
def zones(inst):
    if inst in _z: return _z[inst]
    acc=[]
    for tf in OTHER_TF:
        pth=ROOT/f'data/field/{inst}/cells/tf_{tf:04d}/passports.parquet'
        if not pth.exists(): continue
        t=pq.read_table(pth,columns=['zone_bottom','zone_top','t0_spine_pos']).to_pydict()
        acc.append((np.asarray(t['t0_spine_pos']),np.asarray(t['zone_bottom'],float),
                    np.asarray(t['zone_top'],float)))
    out=[]
    for pos,zb,zt in acc:
        o=np.argsort(pos,kind='stable'); out.append((pos[o],zb[o],zt[o]))
    _z[inst]=out; return out

def field_count(inst,q,price):
    n=0
    for pos,zb,zt in zones(inst):
        k=int(np.searchsorted(pos,q,side='left'))
        if k==0: continue
        b,t=zb[:k],zt[:k]
        n+=int(np.count_nonzero((b<=price)&(price<=t)))
    return n

def enrich(inst,tf,rows_idx):
    """rows_idx: list of (q, price_at_q, ts_at_q). returns external coords."""
    A,se=M.market(inst); ts=A['close_ts_utc_ns']
    others=[x for x in ['NQ','ES','YM'] if x!=inst]
    OA={o:M.market(o)[0] for o in others}
    OT={o:np.asarray(OA[o]['close_ts_utc_ns']) for o in others}
    out=[]
    for q,price,tq in rows_idx:
        fc=field_count(inst,q,price)
        # cross-instrument: normalised 30-min displacement at the same UTC minute
        cross=[]
        for o in others:
            j=int(np.searchsorted(OT[o],tq,side='left'))
            if j<31 or j>=len(OT[o]) or int(OT[o][j])!=tq: cross.append(np.nan); continue
            seg=np.asarray(OA[o]['close'][j-30:j+1],float)
            tseg=OT[o][j-30:j+1]
            if len(seg)<31 or np.any(np.diff(tseg)!=NS) or not np.isfinite(seg).all():
                cross.append(np.nan); continue
            s=float(np.std(np.diff(seg)))
            cross.append((seg[-1]-seg[0])/s if s>0 else np.nan)
        # session context
        si=se['lookup'].get(int(A['session_id'][q]))
        smin=np.nan; srange=np.nan
        if si is not None:
            an=int(se['session_open_utc_ns'][si]); smin=(tq-an)/NS
            j0=int(np.searchsorted(ts,an,side='left'))
            if 0<=j0<=q:
                hh=np.asarray(A['high'][j0:q+1],float); ll=np.asarray(A['low'][j0:q+1],float)
                if np.isfinite(hh).all() and np.isfinite(ll).all() and hh.max()>ll.min():
                    srange=(price-ll.min())/(hh.max()-ll.min())
        out.append(dict(field_zones=fc,
                        cross_mean=np.nanmean(cross) if np.any(np.isfinite(cross)) else np.nan,
                        cross_absdiff=abs(cross[0]-cross[1]) if all(np.isfinite(c) for c in cross) else np.nan,
                        session_min=smin, session_pos=srange))
    return out
