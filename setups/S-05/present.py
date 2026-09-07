"""Trader-facing candle scenes, money curves and risk from the fixed v2 ledger."""
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from follow import HERE,OUT,INPUT,ROOT,M,POINT,COST,market,dump
from scenes import candles
from g3riz.query import Field


def examples(r,m,date=None):
    if date:
        chosen=r.loc[r.date==date]
        if chosen.empty:raise ValueError('No executed v2 trade on this complete data day')
        output=OUT/f'scene_{date}.png'
    else:
        recent=r.loc[r.date>='2024-01-01']
        win=recent.loc[recent.net_dollars>0].sort_values('net_dollars')
        loss=recent.loc[recent.net_dollars<0].sort_values('net_dollars')
        chosen=pd.concat([win.iloc[[len(win)//2]],loss.iloc[[len(loss)//2]]])
        output=HERE/'scenes_v2.png'
    fig,axes=plt.subplots(len(chosen),2,figsize=(16,5*len(chosen)),layout='constrained',squeeze=False)
    details=[];f=Field(ROOT,'NQ');ts=m['close_ts_utc_ns']
    allsignals=pd.read_parquet(INPUT/'NQ_signals_k3.parquet')
    for row,(_,z) in enumerate(chosen.iterrows()):
        t=int(z.t0_spine_pos);q=int(z.q);end=int(z.exit_pos)
        ev=f.events(tf=int(z.tf_minutes),riz_ids=[z.riz_id]).to_pandas()
        known=ev.loc[ev.market_spine_pos<=q]
        deleted=bool((known.event_kind=='deleted').any())
        loc=pd.Timestamp(int(ts[q]),tz='UTC').tz_convert('America/New_York')
        side='покупка' if z.direction==1 else 'продажа'
        a=max(0,t-10)
        for col,positions in enumerate([np.arange(a,q+1),np.arange(max(t,q-3),end+1)]):
            ax=axes[row,col];candles(ax,m,positions,int(ts[t]))
            ax.axhspan(z.zone_bottom,z.zone_top,color='#6f97c6',alpha=.10)
            for edge in [z.zone_bottom,z.zone_top]:ax.axhline(edge,color='#6687ad',lw=.8)
            ax.axvline((ts[q]-ts[t])/M,color='#d1953c',ls='--',lw=1)
            if col==0:
                ax.axvline(0,color='#444',lw=.8)
                ax.axvspan((ts[q-3]-ts[t])/M,(ts[q]-ts[t])/M,color='#f3c35c',alpha=.2)
                ax.set_title(f"До решения: {loc:%Y-%m-%d %H:%M} ET | TF {int(z.tf_minutes)}\n"
                             f"RIZ {z.zone_bottom:.2f}–{z.zone_top:.2f}; T0 {int(z.wait_minutes)} мин назад",loc='left',fontsize=10)
            else:
                ax.scatter((ts[q]-ts[t])/M,z.entry,color='#222',s=25,zorder=4)
                ax.scatter((ts[end]-ts[t])/M,z.exit,color='#7b52a0',s=30,zorder=4)
                ax.axhline(z.entry,color='#555',lw=.5,ls=':')
                ax.set_title(f"Продолжение: {side} {z.entry:.2f} → {z.exit:.2f}\n"
                             f"{int(z.minutes)} минут; после расходов ${z.net_dollars:+,.0f}",loc='left',fontsize=10)
            ax.set_xlabel('Минуты от T0; свеча обозначена временем закрытия')
        pre=known[['event_kind','market_spine_pos','span_count','north_alive','south_alive']].to_dict('records')
        constituent=allsignals.loc[(allsignals.q==q)&(allsignals.recognition_status==1)]
        constituent.to_parquet(OUT/f'{z.date}_participating_riz.parquet',index=False)
        pd.DataFrame({'time_utc':pd.to_datetime(ts[a:end+1],utc=True),
            **{c:m[c][a:end+1] for c in ['open','high','low','close']}}).to_csv(OUT/f'{z.date}_candles.csv',index=False)
        details.append(dict(date=z.date,riz_id=z.riz_id,tf=int(z.tf_minutes),
            t0_et=str(pd.Timestamp(int(ts[t]),tz='UTC').tz_convert('America/New_York')),
            decision_et=str(loc),north=float(z.zone_top),south=float(z.zone_bottom),
            entry=float(z.entry),exit=float(z.exit),net=float(z.net_dollars),direction=int(z.direction),
            deleted_by_decision=deleted,known_events=pre,
            trigger_closes=[float(m['close'][j]) for j in range(q-3,q+1)]))
    fig.suptitle('S-05 v2 — одна и та же узнаваемая сцена допускает разные исходы. '
                 'Линии сохраняют координаты RIZ, а не обещают его дальнейшую жизнь.',fontsize=11)
    fig.savefig(output,dpi=145);plt.close(fig)
    dump(OUT/('illustrations_v2.json' if not date else f'scene_{date}.json'),details)
    print('SCENES',details,flush=True)


def main(date=None):
    r=pd.read_parquet(OUT/'selected_v2_replay.parquet')
    r=r.loc[r.day_known&(r.status>0)].sort_values('day')
    m=market('NQ');examples(r,m,date)
    if date:return
    daily=pd.read_parquet(OUT/'daily_v1.parquet').set_index('date')['NQ_k3_w60_h60_time']
    common=pd.read_parquet(OUT/'selection_daily_combined_v1.parquet').set_index('date')
    select=common.selection_net
    comparable=daily.reindex(common.index).where(select.notna())
    fig,axes=plt.subplots(2,1,figsize=(13,8),layout='constrained')
    ax=axes[0];x=pd.to_datetime(daily.index);eq=daily.cumsum()
    ax.plot(x,eq,color='#16877f',lw=1.7)
    ax.set(title='S-05 v2: один NQ, $15 расходов за круг, 4 395 сделок',ylabel='Накопленные доллары')
    ax.axhline(0,color='#444',lw=.6);ax.grid(alpha=.2)
    ax=axes[1];x=pd.to_datetime(common.index)
    ax.plot(x,comparable.cumsum(),label='Фиксированный v2, выбранный после просмотра',color='#16877f')
    ax.plot(x,select.cumsum(),label='Выбор по предыдущим 3 годам, семейство S-04/S-05',color='#b34c59')
    ax.set(title='Сравнимые общие дни 2010–2026: выбор правила существенно меняет результат',ylabel='Накопленные доллары')
    ax.axhline(0,color='#444',lw=.6);ax.grid(alpha=.2);ax.legend(fontsize=9)
    fig.savefig(HERE/'equity_v2.png',dpi=150);plt.close(fig)
    # Open-position drawdown bounds under admissible within-minute high/low orders.
    peak=[0.,0.];dd=[0.,0.];cash=0.
    for z in r.itertuples():
        for j in range(int(z.entry_pos),int(z.exit_pos)+1):
            prices=[float(m[c][j]) for c in ['open','high','low','close']]
            o,h,l,c=[cash+(p-z.entry)*z.direction*20.-15. for p in prices]
            adverse=min(h,l);favorable=max(h,l)
            for v,path in enumerate([[o,adverse,favorable,c],[o,favorable,adverse,c]]):
                for value in path:
                    peak[v]=max(peak[v],value);dd[v]=max(dd[v],peak[v]-value)
        cash+=z.net_dollars
    cov=pd.read_parquet(INPUT/'NQ_coverage.parquet');frequency=[]
    for label,start in [('all','2006-01-01'),('2020+','2020-01-01'),('2023+','2023-01-01')]:
        known=cov.loc[cov.known&(cov.date>=start)]
        trades=r.loc[r.date>=start]
        frequency.append(dict(period=label,known_days=len(known),trade_days=len(trades),
                              no_trade_days=len(known)-len(trades),fraction=len(trades)/len(known)))
    dump(HERE/'presentation_numbers.json',dict(intraday_drawdown_order_bounds=dd,
          fixed_rule_common_days_2010_net=float(comparable.sum()),selection_net=float(select.sum()),frequency=frequency))
    print('INTRADAY_DD',dd,'FREQUENCY',frequency,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--date');main(p.parse_args().date)
