"""Перенос ЗАМОРОЖЕННОГО NQ-представления на ES и YM.

Заморожено до этого расчёта и на ES/YM не переотбиралось: курсор 12, состав
18 семей (те же пары отношений), симметричные барьеры b = min(вверх, вниз),
контроль по положению закрытия в диапазоне префикса (20 корзин), кластерная по
дням ошибка. Это проверка переноса и чувствительности, НЕ независимое
подтверждение: ES и YM того же архива уже открывались в прежних циклах.
"""
import sys, json
sys.path.insert(0,'research')
import numpy as np
from film_corpus import build
from repetition_map import language, atom_name, Packed
from motif_to_action import barriers, first_touch
from calendar_utils import date_key

CUR, BINS = 12, 20
r1=json.load(open('work/070/action_NQ_c12.json',encoding='utf-8'))
frozen=[(f['id'],f['relations']) for f in r1['families']]
pts,recs=language(0,CUR); names=[atom_name(x) for x in recs]
idx={nm:i for i,nm in enumerate(names)}
out={'frozen_from':'NQ, work/070/action_NQ_c12.json','cursor':CUR,
     'status':'перенос и чувствительность; независимым подтверждением не является',
     'instruments':{}}
for ins in ('NQ','ES','YM'):
    c=build(ins); n=c['ohlc'].shape[0]
    train=c['t0']<np.datetime64('2020-01-01','ns').astype('int64')
    day=date_key(c['t0'])
    hi,lo,Cc=barriers(c,CUR); du,dd=hi-Cc,Cc-lo
    ratio=np.divide(du,du+dd,out=np.full(n,np.nan),where=(du+dd)>0)
    b=np.minimum(du,dd); fu,fd=first_touch(c,CUR,Cc+b,Cc-b)
    up=(fu>=0)&((fd<0)|(fu<fd)); dn=(fd>=0)&((fu<0)|(fd<fu))
    decided=(up^dn)&np.isfinite(ratio)&train
    y=up.astype(float)
    bins=np.clip(np.digitize(ratio,np.linspace(0,1,BINS+1))-1,0,BINS-1)
    P=Packed(c,recs,0,CUR)
    rows=[]
    for fid,rels in frozen:
        m=np.ones(n,bool)
        for rel in rels:
            i=idx[rel]; m&=P.bool_block(i,i+1)[:,0]
        m&=decided
        if m.sum()<200: rows.append({'id':fid,'films':int(m.sum()),'note':'мало случаев'}); continue
        p=np.full(BINS,np.nan)
        for k in range(BINS):
            ctl=decided&(bins==k)&~m
            if ctl.sum()>=30: p[k]=y[ctl].mean()
        ok=m&np.isfinite(p[bins])
        res=y[ok]-p[bins[ok]]; N=len(res)
        d=day[ok]; o=np.argsort(d); rs,ds=res[o],d[o]
        s=np.add.reduceat(rs,np.r_[0,np.flatnonzero(np.diff(ds))+1])
        se=np.sqrt((s**2).sum())/N
        rows.append({'id':fid,'films':int(N),'days':int(len(s)),
                     'difference':float(res.mean()),
                     't_clustered':float(res.mean()/se) if se else None})
    out['instruments'][ins]=rows
    print(ins, ' '.join(f"{r['id']}:{r.get('difference',float('nan')):+.3f}/{r.get('t_clustered') or 0:+.1f}" for r in rows))
open('work/070/transfer_frozen.json','w',encoding='utf-8').write(json.dumps(out,ensure_ascii=False,indent=1))
# согласие знаков NQ vs ES/YM
nq={r['id']:r.get('difference') for r in out['instruments']['NQ']}
for ins in ('ES','YM'):
    o={r['id']:r.get('difference') for r in out['instruments'][ins]}
    pairs=[(nq[k],o[k]) for k in nq if nq[k] is not None and o.get(k) is not None]
    agree=sum(1 for a,b in pairs if a*b>0)
    print(f'{ins}: совпадение знака с NQ {agree}/{len(pairs)}; '
          f'корреляция {np.corrcoef([a for a,_ in pairs],[b for _,b in pairs])[0,1]:.3f}')
