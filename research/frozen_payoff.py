"""Перенос ЗАМОРОЖЕННЫХ семей И ЗАМОРОЖЕННЫХ сторон на другие участки.

Заморожено на разведке NQ (T0 < 2020) и дальше не переподбиралось: состав
18 семей, курсор 12, сторона каждой семьи, барьеры, сопоставление по положению.
Проверяется одно: держится ли добавка h37, или она была платой за выбор стороны
на том же участке. Плюс диагностика зависимости инструментов по минутам T0.
"""
import sys, json
sys.path.insert(0,'research')
import numpy as np
from film_corpus import build
from repetition_map import language, atom_name, Packed
from motif_to_action import barriers
from calendar_utils import date_key
from candidate_check import POINT, COST

CUR,BINS,H=12,20,37
r1=json.load(open('work/070/action_NQ_c12.json',encoding='utf-8'))
tr=json.load(open('work/070/transfer_frozen.json',encoding='utf-8'))
side={r['id']:(1 if (r.get('difference') or 0)>0 else -1) for r in tr['instruments']['NQ']}
pts,recs=language(0,CUR); names=[atom_name(x) for x in recs]; idx={nm:i for i,nm in enumerate(names)}
res={'frozen':'семьи, курсор и стороны с разведки NQ (T0<2020)','horizon':H,'parts':{}}
T0={}
for ins in ('NQ','ES','YM'):
    c=build(ins); n=c['ohlc'].shape[0]; T0[ins]=set(int(t) for t in c['t0'])
    train=c['t0']<np.datetime64('2020-01-01','ns').astype('int64')
    day=date_key(c['t0']); O=c['ohlc'][:,:,0]
    hi,lo,Cc=barriers(c,CUR); du,dd=hi-Cc,Cc-lo
    ratio=np.divide(du,du+dd,out=np.full(n,np.nan),where=(du+dd)>0)
    bins=np.clip(np.digitize(ratio,np.linspace(0,1,BINS+1))-1,0,BINS-1)
    entry=O[:,CUR+1]; pt=POINT[ins]
    P=Packed(c,recs,0,CUR)
    for tag,part in (('разведка' if ins=='NQ' else 'вся история', train if ins=='NQ' else np.ones(n,bool)),
                     ('второй участок' if ins=='NQ' else None, ~train if ins=='NQ' else None)):
        if tag is None: continue
        base=np.isfinite(entry)&np.isfinite(ratio)&part
        rows=[]
        for f in r1['families']:
            m=np.ones(n,bool)
            for rel in f['relations']:
                i=idx[rel]; m&=P.bool_block(i,i+1)[:,0]
            m&=base
            if m.sum()<200: continue
            s=side[f['id']]
            v=s*(O[:,CUR+1+H]-entry)*pt
            ok=np.isfinite(v)&base
            p=np.full(BINS,np.nan)
            for k in range(BINS):
                ctl=ok&(bins==k)&~m
                if ctl.sum()>=30: p[k]=v[ctl].mean()
            sel=np.flatnonzero(m&ok&np.isfinite(p[bins]))
            if len(sel)<200: continue
            r_=v[sel]-p[bins[sel]]
            d=day[sel]; o=np.argsort(d); rs,ds=r_[o],d[o]
            ss=np.add.reduceat(rs,np.r_[0,np.flatnonzero(np.diff(ds))+1])
            se=np.sqrt((ss**2).sum())/len(r_)
            rows.append({'id':f['id'],'n':int(len(sel)),'raw_usd':float(v[sel].mean()),
                         'incr_usd':float(r_.mean()),'t':float(r_.mean()/se) if se else None})
        res['parts'][f'{ins} {tag}']=rows
        pos=sum(1 for r in rows if r['incr_usd']>0)
        print(f"{ins} {tag}: семей {len(rows)}, добавка>0 у {pos}, "
              f"средняя добавка ${np.mean([r['incr_usd'] for r in rows]):+.2f}, "
              f"макс |t| {max(abs(r['t']) for r in rows):.2f}")
ov={}
for a,b in (('NQ','ES'),('NQ','YM'),('ES','YM')):
    inter=len(T0[a]&T0[b])
    ov[f'{a}∩{b}']={'shared_t0_minutes':inter,
                    'share_of_'+a:inter/len(T0[a]),'share_of_'+b:inter/len(T0[b])}
res['instrument_overlap_dependence_diagnostic']=ov
print('перекрытие минут T0:',json.dumps(ov,ensure_ascii=False))
open('work/070/frozen_payoff.json','w',encoding='utf-8').write(json.dumps(res,ensure_ascii=False,indent=1))
