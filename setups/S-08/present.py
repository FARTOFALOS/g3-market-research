"""Coordinate-selected scenes; prefixes and outcomes drawn separately."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import run as s

def candle(ax,m,a,b):
    for p in range(a,b+1):
        o,h,l,c=[float(m[k][p]) for k in ['open','high','low','close']]
        col='#13857c' if c>=o else '#bf4852'
        ax.plot([p-a,p-a],[l,h],color=col,lw=.8)
        ax.add_patch(Rectangle((p-a-.32,min(o,c)),.64,max(abs(o-c),.12),color=col))
    ticks=np.unique(np.linspace(0,b-a,min(7,b-a+1),dtype=int))
    ax.set_xticks(ticks,[pd.Timestamp(int(m['close_ts_utc_ns'][a+x]),tz='UTC').tz_convert('America/New_York').strftime('%H:%M') for x in ticks])
    ax.grid(alpha=.14);ax.tick_params(labelsize=8)

def main():
    m,cal=s.load();t=pd.read_parquet(s.OUT/'trades.parquet')
    t=t[(t.variant=='v1')&(t.bound=='lower')]
    picks=t[t.date.isin(['2024-01-03','2024-01-11','2024-02-15'])]
    fig,axs=plt.subplots(3,2,figsize=(14,12));rows=[]
    for ix,(_,r) in enumerate(picks.iterrows()):
        p=int(r.p);start=int(r.session_start);end=min(p+60,len(m['close'])-1)
        for ax,a,b,label in [(axs[ix,0],start,p,'Только известное к решению'),(axs[ix,1],start,end,'Последующий час')]:
            candle(ax,m,a,b)
            for val,name,col in [(r.boundary,'Вчерашний край','#526f94'),(r.target,'Цель: вчерашнее закрытие','#13857c'),(r.stop,'Стоп: край попытки','#bf4852')]:
                ax.axhline(val,color=col,ls='--',lw=.85,label=f'{name} {val:.2f}')
            ax.axvline(p-a,color='#333',lw=1)
            ax.set_title(f'{r.date} · {label}',loc='left',fontsize=11)
            ax.legend(fontsize=7,loc='best')
        ax=axs[ix,1];ax.scatter([int(r.entry_pos)-start],[r.entry],marker='o',color='#111',s=25)
        ax.scatter([int(r.exit_pos)-start],[r.exit],marker='x',color='#111',s=40)
        ax.set_xlabel(f'{r.execution}: {r.net_usd:+,.0f} USD; до {r.held_minutes:.0f} мин. в позиции',fontsize=10)
        rows.append({k:r[k] for k in ['date','prev_date','prev_low','prev_high','target','signal_close','opening_price','boundary','entry','stop','exit','execution','net_usd','held_minutes']})
    fig.suptitle('S-08 · Возврат внутрь вчерашнего диапазона после внешнего открытия\nПримеры выбраны по датам до просмотра результатов; NQ, время Нью-Йорка',fontsize=14)
    fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(s.HERE/'scenes.png',dpi=125);plt.close(fig)
    (s.HERE/'scenes.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    print(s.HERE/'scenes.png')

if __name__=='__main__':main()
