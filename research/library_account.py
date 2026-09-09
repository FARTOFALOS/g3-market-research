"""Общий исторический счёт библиотеки: решения, конфликты, ежедневность, деньги.

Правило объявлено до счёта: узнавание на закрытии минуты 12 фильма RIZ; если
узнаётся несколько семей, исполняется с наименьшим номером (порядок задан
поиском, не результатом сделок); одна позиция одновременно; вход `open` минуты
13, выход `open` минуты 13+H. Сторона каждой семьи заморожена на разведке NQ.
Расход показан как СЦЕНАРИЙ: валовое, безубыточный расход и чувствительность.
"""
import sys, json
sys.path.insert(0,'research')
import numpy as np
from film_corpus import build
from repetition_map import language, atom_name, Packed
from calendar_utils import date_key, year_of, minute_of_day
from candidate_check import POINT

CUR,H=12,37
MIN=60_000_000_000
COSTS=[0,2,5,10,15,20]
r1=json.load(open('work/070/action_NQ_c12.json',encoding='utf-8'))
tr=json.load(open('work/070/transfer_frozen.json',encoding='utf-8'))
side={r['id']:(1 if (r.get('difference') or 0)>0 else -1) for r in tr['instruments']['NQ']}
pts,recs=language(0,CUR); names=[atom_name(x) for x in recs]; idx={nm:i for i,nm in enumerate(names)}
out={}
for ins in ('NQ',):
    c=build(ins); n=c['ohlc'].shape[0]; O=c['ohlc'][:,:,0]; pt=POINT[ins]
    P=Packed(c,recs,0,CUR)
    masks=[]
    for f in r1['families']:
        m=np.ones(n,bool)
        for rel in f['relations']:
            i=idx[rel]; m&=P.bool_block(i,i+1)[:,0]
        masks.append((f['id'],m))
    t0=c['t0']; order=np.argsort(t0)
    entry=O[:,CUR+1]; exit_=O[:,CUR+1+H]
    fired=np.zeros(n,dtype=np.int16)-1
    for k,(fid,m) in enumerate(masks):
        fired[(fired<0)&m]=k
    ok=(fired>=0)&np.isfinite(entry)&np.isfinite(exit_)
    busy=-1; trades=[]
    for i in order:
        if not ok[i]: continue
        t_in=int(t0[i])+(CUR+1)*MIN
        if t_in<=busy: continue
        k=int(fired[i]); s=side[masks[k][0]]
        trades.append((t_in,s*(exit_[i]-entry[i])*pt,masks[k][0]))
        busy=int(t0[i])+(CUR+1+H)*MIN
    ts=np.array([x[0] for x in trades]); g=np.array([x[1] for x in trades])
    d=date_key(ts); yr=year_of(ts); mod=minute_of_day(ts)
    alld=np.unique(date_key(t0))
    u,cn=np.unique(d,return_counts=True)
    pos=np.searchsorted(alld,u); gaps=np.diff(pos)-1
    sem=g.std(ddof=1)/np.sqrt(len(g))
    r={'instrument':ins,'trades':int(len(g)),
       'candidates_recognised':int(ok.sum()),
       'blocked_by_open_position':int(ok.sum()-len(g)),
       'days_with_trade':int(len(u)),'trading_days_in_corpus':int(len(alld)),
       'share_days_with_trade':float(len(u)/len(alld)),
       'median_trades_per_active_day':float(np.median(cn)),
       'max_gap_days_without_trade':int(gaps.max()) if len(gaps) else None,
       'p95_gap_days':float(np.percentile(gaps,95)) if len(gaps) else None,
       'gross_mean_usd':float(g.mean()),'gross_sem':float(sem),
       't_gross':float(g.mean()/sem),
       'breakeven_round_trip_cost_usd':float(g.mean()),
       'share_positive':float((g>0).mean()),
       'by_cost':{},
       'by_year_gross':{int(y):float(g[yr==y].mean()) for y in np.unique(yr)},
       'rth_share_of_trades':float(((mod>=570)&(mod<960)).mean())}
    for cst in COSTS:
        eq=np.cumsum(g-cst)
        r['by_cost'][str(cst)]={'mean_net':float(g.mean()-cst),
                                'total_net':float((g-cst).sum()),
                                'drawdown':float(np.max(np.maximum.accumulate(eq)-eq))}
    out[ins]=r
    print(json.dumps(r,ensure_ascii=False,indent=1))
open('work/070/library_account.json','w',encoding='utf-8').write(json.dumps(out,ensure_ascii=False,indent=1))
