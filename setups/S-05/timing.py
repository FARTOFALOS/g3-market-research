"""Minute-by-minute overlap of already selected entry paths; exploratory timing exposure."""
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from follow import HERE,OUT,INPUT,ROOT,M,POINT,COST,market


def main(version='v1'):
    v=json.loads((HERE/f'selected_{version}.json').read_text())
    allr=pd.read_parquet(OUT/'all_decisions_v1.parquet')
    r=allr.loc[(allr.variant==v['id'])&allr.day_known&(allr.status>0)].sort_values('day')
    m=market(v['instrument']);ts=m['close_ts_utc_ns']
    paths=np.full((len(r),150),np.nan)
    for i,z in enumerate(r.itertuples()):
        q=int(z.q);t=int(ts[q]);entry=float(z.entry)
        for j in range(-29,121):
            n=t+j*M
            if n>z.calendar_close_ns:
                n=z.calendar_close_ns
            p=int(np.searchsorted(ts,n))
            if p<len(ts) and ts[p]==n:
                paths[i,j+29]=(m['close'][p]-entry)*z.direction
    np.save(OUT/f'selected_{version}_aligned_points.npy',paths)
    stats=[]
    for j in range(1,121):
        x=paths[:,j+29]
        stats.append(dict(minutes=j,n=int(np.isfinite(x).sum()),mean_points=np.nanmean(x),
            median_points=np.nanmedian(x),q10=np.nanquantile(x,.1),q90=np.nanquantile(x,.9),
            net_total=float(np.nansum(x*POINT[v['instrument']]-COST[v['instrument']])),
            mean_net=float(np.nanmean(x*POINT[v['instrument']]-COST[v['instrument']]))))
    pd.DataFrame(stats).to_csv(HERE/f'timing_exploration_{version}.csv',index=False)
    print(pd.DataFrame(stats).iloc[[0,1,2,3,4,7,9,14,19,29,44,59,89,119]].to_string(index=False))
    fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    x=np.arange(-29,121);ax=axes[0]
    # All paths contribute; density displayed in points and minute time, no ATR or risk units.
    clipped=np.clip(paths,-120,120)
    hist=np.array([np.histogram(col[np.isfinite(col)],bins=np.linspace(-120,120,121),density=True)[0]
                   for col in clipped.T]).T
    ax.imshow(hist,origin='lower',aspect='auto',extent=[-29,120,-120,120],cmap='Blues',vmax=np.quantile(hist,.97))
    ax.plot(x,np.nanmedian(paths,axis=0),color='#ed9231',label='Медиана')
    ax.plot(x,np.nanmean(paths,axis=0),color='#c5485a',label='Среднее')
    ax.axvline(0,color='black',ls='--',lw=.8);ax.axhline(0,color='black',lw=.5)
    ax.set(xlabel='Минуты относительно решения',ylabel='Пункты по стороне сделки',title=f"Наложение {len(r):,} путей NQ; плотность обрезана ±120 пунктами")
    ax.legend(fontsize=9)
    ax=axes[1];s=pd.DataFrame(stats)
    ax.plot(s.minutes,s.mean_net,color='#16877f',lw=2);ax.axhline(0,color='black',lw=.6)
    ax.set(xlabel='Минуты после входа',ylabel='Средние доллары после $15 расходов',title='Где накапливается денежный результат')
    ax.grid(alpha=.2)
    fig.savefig(HERE/f'timing_{version}.png',dpi=150);plt.close(fig)
    # Frequency belongs to actual days. Signal frequency and feasible entries are separate.
    cov=pd.read_parquet(INPUT/'NQ_coverage.parquet');records=[]
    for name,start in [('2006–2026','2006-01-01'),('2020–2026','2020-01-01'),('2023–2026','2023-01-01')]:
        c=cov.loc[(cov.date>=start)&cov.known]
        z=r.loc[r.date>=start]
        records.append(dict(period=name,known_days=len(c),trading_days=len(z),share=len(z)/len(c)))
    print('FREQUENCY',records)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--version',default='v1',choices=['v1','v2'])
    main(p.parse_args().version)
