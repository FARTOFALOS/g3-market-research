"""New coordinate-selected minute scenes; no profit-based selection."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'data/research/S-06'
sys.path.insert(0,str(ROOT/'setups/S-04'))
from run import market,M,dump
from scenes import candles
from g3riz.query import Field

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cal=pd.read_parquet(ROOT/'setups/S-04/calendar.parquet')
    for ins in ['ES','NQ','YM']:
        p=pd.read_parquet(ROOT/f'data/research/S-04/{ins}_objects.parquet')
        local=pd.to_datetime(p.t0_ts_ns,utc=True).dt.tz_convert('America/New_York')
        minutes=local.dt.hour*60+local.dt.minute
        p=p.loc[local.dt.year.between(2022,2025)&minutes.between(575,900)].copy()
        p['date']=local.loc[p.index].dt.strftime('%Y-%m-%d')
        p=p.loc[p.date.isin(cal.date)]
        p=p.sort_values(['t0_spine_pos','tf_minutes','riz_id']).drop_duplicates('t0_spine_pos')
        rng=np.random.default_rng(20260908);chosen=[]
        for lo,hi in [(1,15),(16,60),(61,240),(241,1440)]:
            sub=p.loc[p.tf_minutes.between(lo,hi)]
            chosen.append(sub.iloc[int(rng.integers(len(sub)))].to_dict())
        m=market(ins);ts=m['close_ts_utc_ns'];f=Field(ROOT,ins)
        fig,axes=plt.subplots(4,1,figsize=(14,15),layout='constrained')
        for ax,z in zip(axes,chosen):
            t=int(z['t0_spine_pos']);n=int(ts[t]);a=np.searchsorted(ts,n-15*M);b=np.searchsorted(ts,n+30*M,side='right')
            candles(ax,m,np.arange(a,b),n)
            for edge in [z['zone_top'],z['zone_bottom']]:ax.axhline(edge,color='#6687ad',lw=.8)
            ax.axhspan(z['zone_bottom'],z['zone_top'],color='#6687ad',alpha=.10)
            ax.axvline(0,color='#333',lw=.9)
            ev=f.events(tf=int(z['tf_minutes']),riz_ids=[z['riz_id']]).to_pandas()
            dd=ev.loc[ev.event_kind=='deleted','market_spine_pos']
            if len(dd):
                d=int(dd.iloc[0]);z['deletion_pos']=d
                offset=(int(ts[d])-n)/M
                if offset<=30:ax.axvspan(offset,30,color='#777',alpha=.1,label='после удаления RIZ')
            z['events']=ev.loc[ev.market_spine_pos<b,['event_kind','market_spine_pos','span_count']].to_dict('records')
            when=pd.Timestamp(n,tz='UTC').tz_convert('America/New_York')
            ax.set_title(f"{ins} {when:%Y-%m-%d %H:%M} ET | TF {z['tf_minutes']} | {z['t0_exit_side']} | "
                         f"{z['zone_bottom']:.2f}–{z['zone_top']:.2f}\n{z['riz_id']}",loc='left',fontsize=10)
            ax.set_xlabel('Минуты от T0; серое — после записанного удаления')
        fig.savefig(HERE/f'{ins}_observe.png',dpi=120);plt.close(fig)
        dump(OUT/f'{ins}_observed.json',chosen)
        print(ins,[(z['date'],z['tf_minutes']) for z in chosen],flush=True)

if __name__=='__main__':main()
