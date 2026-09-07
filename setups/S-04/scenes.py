"""Outcome-blind scene selection; full post-T0 cash-session candle display."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from run import HERE, OUT, M, market, dump


def candles(ax, m, pos, t0):
    x = (m['close_ts_utc_ns'][pos]-t0)/M
    for x0, j in zip(x, pos):
        o,h,l,c = [float(m[k][j]) for k in ['open','high','low','close']]
        color = '#16877f' if c>=o else '#c65057'
        ax.vlines(x0,l,h,color=color,linewidth=.55)
        ax.add_patch(Rectangle((x0-.32,min(o,c)),.64,max(abs(c-o),.025),
                              facecolor=color,edgecolor=color,linewidth=.35))
    ax.grid(axis='y',alpha=.15)


def select(ins):
    p = pd.read_parquet(OUT/f'{ins}_objects.parquet')
    cal = pd.read_parquet(HERE/'calendar.parquet')
    local = pd.to_datetime(p.t0_ts_ns,utc=True).dt.tz_convert('America/New_York')
    # Sampling coordinates only; neither outcomes nor prefix shapes are inspected.
    p = p.loc[(local.dt.hour>=10)&(local.dt.hour<12)&(local.dt.year>=2020)].copy()
    p['date'] = local.loc[p.index].dt.strftime('%Y-%m-%d')
    p = p.loc[p.date.isin(cal.date)]
    p = p.sort_values(['t0_spine_pos','tf_minutes','riz_id']).drop_duplicates('t0_spine_pos')
    rng=np.random.default_rng(20260907)
    rows=[]
    for lo,hi in [(1,30),(31,120),(121,1440)]:
        sub=p.loc[p.tf_minutes.between(lo,hi)]
        ix=rng.choice(len(sub),size=2,replace=False)
        rows.extend(sub.iloc[ix].to_dict('records'))
    for r in rows:
        r['view_end_ns']=int(cal.loc[cal.date==r['date'],'close_ns'].iloc[0])
    dump(OUT/f'{ins}_visual_selection.json',rows)
    return rows


def draw(ins):
    rows=select(ins)
    m=market(ins);ts=m['close_ts_utc_ns']
    for page in range(2):
        fig,axes=plt.subplots(3,1,figsize=(16,12),layout='constrained')
        for ax,r in zip(axes,rows[page*3:(page+1)*3]):
            t=int(r['t0_spine_pos']);n=int(r['t0_ts_ns'])
            a=int(np.searchsorted(ts,n-30*M));b=int(np.searchsorted(ts,r['view_end_ns'],side='right'))
            candles(ax,m,np.arange(a,min(b,len(ts))),n)
            ax.axhspan(r['zone_bottom'],r['zone_top'],color='#7699d0',alpha=.12)
            ax.axhline(r['zone_top'],color='#6484ac',lw=.8)
            ax.axhline(r['zone_bottom'],color='#6484ac',lw=.8)
            ax.axvline(0,color='#222',lw=1)
            local=pd.Timestamp(n,tz='UTC').tz_convert('America/New_York')
            ax.set_title(f"{ins} | {local:%Y-%m-%d %H:%M} ET | TF {r['tf_minutes']} | "
                         f"{r['zone_bottom']:g}–{r['zone_top']:g} | T0 {r['t0_exit_side']}\n{r['riz_id']}",loc='left',fontsize=10)
            ax.set_xlabel('Минуты относительно T0; справа вся доступная основная сессия')
        path=HERE/f'{ins}_read_{page+1}.png';fig.savefig(path,dpi=135);plt.close(fig)
        print(path,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('instrument');args=p.parse_args();draw(args.instrument)
