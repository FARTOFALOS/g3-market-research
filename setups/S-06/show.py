"""Show actual executed S-06 v2 trades; default examples are outcome-selected illustrations."""
import argparse,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run import ROOT,HERE,OUT,M,market,dump
from scenes import candles

def main(date=None):
    r=pd.read_parquet(OUT/'illustrative_v2_replay.parquet')
    r=r.loc[r.day_known&(r.status>0)].sort_values('q')
    if date:
        chosen=r.loc[r.date==date].head(1)
        if chosen.empty:raise ValueError('No executed illustrative-v2 trade on this known day')
        name=f'scene_{date}';output=OUT/f'{name}.png'
    else:
        recent=r.loc[r.date>='2023-01-01'];rows=[]
        for positive in [True,False]:
            x=recent.loc[(recent.net_dollars>0) if positive else (recent.net_dollars<0)].sort_values(['net_dollars','q'])
            rows.append(x.iloc[len(x)//2])
        chosen=pd.DataFrame(rows);name='scenes';output=HERE/'scenes.png'
    m=market('NQ');ts=m['close_ts_utc_ns'];fig,axes=plt.subplots(len(chosen),2,figsize=(15,4.5*len(chosen)),layout='constrained',squeeze=False)
    records=[]
    for row,(_,z) in enumerate(chosen.iterrows()):
        q=int(z.q);p=int(z.p);end=int(z.exit_pos);a=q-8;n=int(ts[q])
        when=pd.Timestamp(n,tz='UTC').tz_convert('America/New_York')
        for col,pos in enumerate([np.arange(a,q+1),np.arange(p-2,end+1)]):
            ax=axes[row,col];candles(ax,m,pos,n)
            ax.axhline(z.edge,color='#4c7fa7',lw=1,label='выходная граница RIZ')
            ax.axhspan(z.box_low,z.box_high,color='#d7ad36',alpha=.13,label='диапазон паузы')
            ax.axvline(0,color='#444',ls=':',lw=1)
            if col==0:
                ax.axvspan(-3,-1,color='#d7ad36',alpha=.10)
                ax.set_title(f'{when:%Y-%m-%d %H:%M} ET | NQ TF {int(z.tf_minutes)} | до решения\n'
                             f'Пауза {z.box_low:.2f}–{z.box_high:.2f}; RIZ существует',fontsize=10,loc='left')
            else:
                ax.axhline(z.stop_price,color='#bb4c53',ls='--',lw=.8,label='стоп')
                ax.axhline(z.target_price,color='#148975',ls='--',lw=.8,label='цель')
                ax.scatter(0,z.entry_price,color='#222',s=25,zorder=5)
                ax.scatter((ts[end]-n)/M,z.exit_price,color='#8057ac',s=30,zorder=5)
                side='покупка' if z.direction==1 else 'продажа'
                ax.set_title(f'{side} {z.entry_price:.2f} → {z.exit_price:.2f} | ${z.net_dollars:+,.0f}\n'
                             'Продолжение после решения, один NQ, расходы $15',fontsize=10,loc='left')
            ax.set_xlabel('Минуты относительно закрытия пробной свечи');ax.legend(fontsize=7,loc='best')
        record={key:z[key] for key in ['riz_id','tf_minutes','date','p','q','t0_spine_pos','zone_bottom','zone_top','edge',
            'box_low','box_high','direction','probe_extreme','entry_price','exit_price','stop_price','target_price','net_dollars','status']}
        record['decision_et']=str(when);records.append(record)
        pd.DataFrame({'close_time_utc':pd.to_datetime(ts[a:end+1],utc=True),**{c:m[c][a:end+1] for c in ['open','high','low','close']}}).to_csv(OUT/f"{name}_{row}_candles.csv",index=False)
    fig.suptitle('S-06 v2: выход не удержался к закрытию минуты; последующий исход ещё неизвестен',fontsize=12)
    fig.savefig(output,dpi=140);plt.close(fig);dump(OUT/f'{name}.json',records)
    print(output,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--date');main(p.parse_args().date)
