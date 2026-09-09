"""ЛОКАЛЬНЫЙ потолок при ФИКСИРОВАННЫХ семьях (не потолок всей процедуры).

Что этот расчёт калибрует: множественность 18 семей прочтения 1 и 18 групп
прочтения 2 при уже выбранных курсоре, определении продолжения, барьерах,
способе сопоставления и статистике. Эти решения уточнялись ПОСЛЕ просмотра
результатов, поэтому путь выбора снят только частично. Отбор семей исходом не
пользовался — это единственная часть, которую расчёт закрывает честно.

Два нуля рядом:
  A. перестановка исхода внутри корзины положения (обмениваемость по фильмам);
  B. кольцевой сдвиг внутри корзины по дням — соседние фильмы одного дня
     остаются рядом, внутридневная зависимость сохраняется.
300 повторов — диагностика, не точная калибровка.
"""
import sys, json
sys.path.insert(0,'research')
import numpy as np
from film_corpus import build
from repetition_map import language, atom_name, Packed
from motif_to_action import barriers, first_touch
from calendar_utils import date_key
import ordinal_events as oe
from relational_stencil import Corpus
from two_readings import signature, keys_of

CUR, BINS, REP = 12, 20, 300
c = build('NQ'); n = c['ohlc'].shape[0]
train = c['t0'] < np.datetime64('2020-01-01','ns').astype('int64')
day_all = date_key(c['t0'])
hi, lo, Cc = barriers(c, CUR)
du, dd = hi-Cc, Cc-lo
ratio = np.divide(du, du+dd, out=np.full(n,np.nan), where=(du+dd)>0)
b = np.minimum(du, dd)
fu, fd = first_touch(c, CUR, Cc+b, Cc-b)
up=(fu>=0)&((fd<0)|(fu<fd)); dn=(fd>=0)&((fu<0)|(fd<fu))
decided=(up^dn)&np.isfinite(ratio)&train
D=np.flatnonzero(decided)
y0=up[D].astype(float)
bins=np.clip(np.digitize(ratio,np.linspace(0,1,BINS+1))-1,0,BINS-1)[D]
days=day_all[D]

r1=json.load(open('work/070/action_NQ_c12.json',encoding='utf-8'))
pts,recs=language(0,CUR); names=[atom_name(x) for x in recs]
P=Packed(c,recs,0,CUR); idx={nm:i for i,nm in enumerate(names)}
memb=[]
for f in r1['families']:
    m=np.ones(n,bool)
    for rel in f['relations']:
        i=idx[rel]; m&=P.bool_block(i,i+1)[:,0]
    memb.append(m[D])
ords=np.arange(0,c['ohlc'].shape[1]); ids=[f'NQ:{int(t)}' for t in c['t0']]
corp=Corpus(c['ohlc'],ords,list(ids),list(ids),
            {'instrument':'NQ','source':{'population':'RIZ'},'anchor':'T0=0',
             'unit_definition':'один T0'})
ev=oe.default_events(corp,start=0,stop=CUR,anchor_ordinal=0)
sig=signature(np.stack([e.values for e in ev])); key=keys_of(sig)
uk,inv,cnt=np.unique(key,return_inverse=True,return_counts=True)
gm=[]
for g in np.argsort(-cnt):
    m=(inv==g)
    if (m&decided).sum()>=400: gm.append(m[D])
gm=gm[:18]

nb=np.bincount(bins,minlength=BINS).astype(float)

class Fam:
    def __init__(self,mask):
        self.ix=np.flatnonzero(mask)
        self.N=len(self.ix)
        self.nfb=np.bincount(bins[self.ix],minlength=BINS).astype(float)
        d=days[self.ix]; o=np.argsort(d)
        self.ix=self.ix[o]; d=d[o]
        self.starts=np.r_[0,np.flatnonzero(np.diff(d))+1]
        nd=len(self.starts)
        self.M=np.zeros((nd,BINS))
        bb=bins[self.ix]
        for k,(s,e) in enumerate(zip(self.starts,np.r_[self.starts[1:],self.N])):
            self.M[k]=np.bincount(bb[s:e],minlength=BINS)
    def t(self,y):
        sb=np.bincount(bins,weights=y,minlength=BINS)
        sfb=np.bincount(bins[self.ix],weights=y[self.ix],minlength=BINS)
        den=nb-self.nfb
        p=np.divide(sb-sfb,den,out=np.zeros(BINS),where=den>=30)
        okb=den>=30
        keep=okb[bins[self.ix]]
        if keep.sum()<200: return 0.0
        res_sum=y[self.ix][keep].sum()-(self.nfb*okb*p).sum()
        N=int(keep.sum())
        yd=np.add.reduceat(np.where(keep,y[self.ix],0.0),self.starts)
        sd=yd-(self.M*(okb*p)).sum(1)
        se=np.sqrt((sd**2).sum())/N
        return abs((res_sum/N)/se) if se else 0.0

F1=[Fam(m) for m in memb]; F2=[Fam(m) for m in gm]
obs1=max(f.t(y0) for f in F1); obs2=max(f.t(y0) for f in F2)
bin_ix=[np.flatnonzero(bins==k) for k in range(BINS)]
bin_ix_day=[ix[np.argsort(days[ix])] for ix in bin_ix]
rng=np.random.default_rng(70)
out={}
for tag in ('A_within_bin','B_day_block_shift'):
    m1,m2=[],[]
    for _ in range(REP):
        yp=y0.copy()
        for k in range(BINS):
            ix=bin_ix[k] if tag=='A_within_bin' else bin_ix_day[k]
            if len(ix)<2: continue
            if tag=='A_within_bin':
                yp[ix]=y0[rng.permutation(ix)]
            else:
                yp[ix]=np.roll(y0[ix],int(rng.integers(1,len(ix))))
        m1.append(max(f.t(yp) for f in F1)); m2.append(max(f.t(yp) for f in F2))
    out[tag]={'reading1':{'p50':float(np.percentile(m1,50)),'p95':float(np.percentile(m1,95)),
                          'max':float(max(m1)),'p_value':float(np.mean(np.array(m1)>=obs1))},
              'reading2':{'p50':float(np.percentile(m2,50)),'p95':float(np.percentile(m2,95)),
                          'max':float(max(m2)),'p_value':float(np.mean(np.array(m2)>=obs2))}}
res={'scope':'локальный потолок при фиксированных семьях; курсор, определение '
             'продолжения, барьеры, сопоставление и статистика уточнялись после '
             'просмотра результатов и этим расчётом НЕ закрыты',
     'observed_max_abs_t_reading1':obs1,'observed_max_abs_t_reading2':obs2,
     'families':len(F1),'signature_groups':len(F2),'repeats':REP,
     'nulls':out}
print(json.dumps(res,ensure_ascii=False,indent=1))
open('work/070/null_max_t.json','w',encoding='utf-8').write(json.dumps(res,ensure_ascii=False,indent=1))
