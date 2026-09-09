"""Три слоя оставшегося пути, одинаковый набор метрик для ВСЕХ 18 семей.

  1. сырой путь: доходности на горизонтах, MFE и MAE после узнавания;
  2. добавка: разница с контролем, сопоставленным по положению закрытия
     внутри диапазона префикса (20 корзин), ошибка кластерная по дням;
  3. исполнимое действие: вход `open` минуты 13, барьеры +/- b = min(вверх,вниз),
     выход по первому касанию либо по open минуты 50, расход $15,
     обе допустимые модели внутриминутной неоднозначности.

Сторона каждой семьи взята из знака разницы вероятности на разведке NQ и
на слоях 1 и 3 не переподбиралась.
"""
import sys, json
sys.path.insert(0,'research')
import numpy as np
from film_corpus import build
from repetition_map import language, atom_name, Packed
from motif_to_action import barriers, first_touch
from calendar_utils import date_key
from candidate_check import POINT, COST

CUR, BINS = 12, 20
HOR=[1,3,5,10,20,37]
r1=json.load(open('work/070/action_NQ_c12.json',encoding='utf-8'))
tr=json.load(open('work/070/transfer_frozen.json',encoding='utf-8'))
side_by_id={r['id']:(1 if (r.get('difference') or 0)>0 else -1) for r in tr['instruments']['NQ']}
pts,recs=language(0,CUR); names=[atom_name(x) for x in recs]; idx={nm:i for i,nm in enumerate(names)}

ins='NQ'; c=build(ins); n=c['ohlc'].shape[0]
train=c['t0']<np.datetime64('2020-01-01','ns').astype('int64')
day=date_key(c['t0'])
O,H,L=c['ohlc'][:,:,0],c['ohlc'][:,:,1],c['ohlc'][:,:,2]
hi,lo,Cc=barriers(c,CUR); du,dd=hi-Cc,Cc-lo
ratio=np.divide(du,du+dd,out=np.full(n,np.nan),where=(du+dd)>0)
b=np.minimum(du,dd); fu,fd=first_touch(c,CUR,Cc+b,Cc-b)
bins=np.clip(np.digitize(ratio,np.linspace(0,1,BINS+1))-1,0,BINS-1)
entry=O[:,CUR+1]; pt,cost=POINT[ins],COST[ins]
base=np.isfinite(entry)&np.isfinite(ratio)&np.isfinite(b)&(b>0)&train
P=Packed(c,recs,0,CUR)

def clustered(res, d):
    o=np.argsort(d); rs,ds=res[o],d[o]
    s=np.add.reduceat(rs,np.r_[0,np.flatnonzero(np.diff(ds))+1])
    return np.sqrt((s**2).sum())/len(res), len(s)

rows=[]
for f in r1['families']:
    m=np.ones(n,bool)
    for rel in f['relations']:
        i=idx[rel]; m&=P.bool_block(i,i+1)[:,0]
    m&=base
    if m.sum()<200: continue
    s=side_by_id[f['id']]
    ix=np.flatnonzero(m)
    e=entry[ix]
    rec={'id':f['id'],'relations':f['relations'],'n':int(len(ix)),
         'side':'long' if s>0 else 'short',
         'median_b_usd':float(np.nanmedian(b[ix])*pt)}
    # слой 1 + слой 2
    for h in HOR:
        v=s*(O[:,CUR+1+h]-entry)*pt
        ok=np.isfinite(v)
        p=np.full(BINS,np.nan)
        for k in range(BINS):
            ctl=base&ok&(bins==k)&~m
            if ctl.sum()>=30: p[k]=v[ctl].mean()
        sel=ix[np.isfinite(v[ix])&np.isfinite(p[bins[ix]])]
        if len(sel)<200: continue
        rec[f'raw_h{h}_usd']=float(v[sel].mean())
        res=v[sel]-p[bins[sel]]
        se,nd=clustered(res,day[sel])
        rec[f'incr_h{h}_usd']=float(res.mean())
        rec[f'incr_h{h}_t']=float(res.mean()/se) if se else None
    seg=slice(CUR+1,51)
    mfe=(np.nanmax(H[ix,seg],axis=1)-e)*pt if s>0 else (e-np.nanmin(L[ix,seg],axis=1))*pt
    mae=(e-np.nanmin(L[ix,seg],axis=1))*pt if s>0 else (np.nanmax(H[ix,seg],axis=1)-e)*pt
    rec['median_mfe_usd']=float(np.nanmedian(mfe)); rec['median_mae_usd']=float(np.nanmedian(mae))
    # слой 3
    u,d_=fu[ix],fd[ix]
    hu=(u>=0)&((d_<0)|(u<d_)); hd=(d_>=0)&((u<0)|(d_<u))
    tie=(u>=0)&(d_>=0)&(u==d_); edge=(u<0)&(d_<0)
    for nm,tv in (('fav',1),('unfav',-1)):
        g=np.zeros(len(ix))
        g[hu&~tie]=b[ix][hu&~tie]; g[hd&~tie]=-b[ix][hd&~tie]
        g[tie]=tv*b[ix][tie]; g[edge]=(O[ix,50][edge]-e[edge])
        v=s*g*pt; v=v[np.isfinite(v)]
        rec[f'action_gross_{nm}_usd']=float(v.mean())
        rec[f'action_net15_{nm}_usd']=float(v.mean()-cost)
    rec['tie_share']=float(tie.mean()); rec['edge_share']=float(edge.mean())
    rows.append(rec)

print('id   n     b$   MFE$ MAE$ | сырой h5/h37 | добавка h5 (t) h37 (t) | действие валовое неблаг..благ, нетто15')
for r in rows:
    print(f"{r['id']:>4} {r['n']:>6} {r['median_b_usd']:>5.0f} {r['median_mfe_usd']:>4.0f} {r['median_mae_usd']:>4.0f} | "
          f"{r['raw_h5_usd']:+6.2f} {r['raw_h37_usd']:+7.2f} | "
          f"{r['incr_h5_usd']:+6.2f}({r['incr_h5_t']:+5.2f}) {r['incr_h37_usd']:+7.2f}({r['incr_h37_t']:+5.2f}) | "
          f"{r['action_gross_unfav_usd']:+6.2f}..{r['action_gross_fav_usd']:+6.2f} "
          f"нетто {r['action_net15_unfav_usd']:+7.2f}")
mx=max(abs(r['incr_h37_t']) for r in rows)
print('максимум |t| добавки на h37:',round(mx,2),'; семей',len(rows))
open('work/070/three_layers.json','w',encoding='utf-8').write(json.dumps(rows,ensure_ascii=False,indent=1))
