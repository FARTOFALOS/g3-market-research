#!/usr/bin/env python3
"""Пять реальных хронологий замороженного правила C. Свечи, не пересказ."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE))
from policy_v2 import load
from trading import WIN,LOSS,AMBIG,TIME_EXIT,UNKNOWN,NO_TRADE,COST,PV,sessions,bar_session_map

def et(ns): return pd.Timestamp(ns,tz='UTC').tz_convert('America/New_York').strftime('%Y-%m-%d %H:%M')

def main(inst='NQ',terr='evaluation'):
    c=COST[inst]; f,tr=load(inst,terr,2.0)
    m=ROOT/'data/market'/inst
    o=np.load(m/'open.npy');h=np.load(m/'high.npy');l=np.load(m/'low.npy')
    cl=np.load(m/'close.npy');ts=np.load(m/'close_ts_utc_ns.npy')
    st,cls=sessions(); sess,last=bar_session_map(ts,st,cls)
    cond=(tr.in_window.to_numpy()&(tr.age.to_numpy()>=2)&(tr.d_close.to_numpy()>4*c)
          &(tr.retraced_fraction.to_numpy()>=0.5))
    trig=tr[cond].groupby('film',sort=False).head(1)
    ent=trig[trig.kind.to_numpy()!=NO_TRADE]
    can=trig[trig.kind.to_numpy()==NO_TRADE]
    # upushchennye ozhidaniem: film voobshche ne dozhil do age>=2
    missed=f[f.n_q.to_numpy()<=2]
    picks=[]
    for nm,kind in (('УСПЕШНАЯ',WIN),('НЕУСПЕШНАЯ',LOSS),
                    ('ВЫХОД ПО ВРЕМЕНИ (структурный двойник)',TIME_EXIT),
                    ('НЕОПРЕДЕЛЁННАЯ',UNKNOWN)):
        s=ent[ent.kind.to_numpy()==kind]
        if len(s): picks.append((nm,s.iloc[len(s)//2]))
    if len(can): picks.append(('ОТМЕНЁННОЕ ИСПОЛНЕНИЕ',can.iloc[len(can)//2]))
    NAMES={WIN:'TP',LOSS:'стоп',AMBIG:'оба барьера в одной минуте',
           TIME_EXIT:'закрытие NY',UNKNOWN:'наблюдаемость кончилась',NO_TRADE:'входа нет'}
    for nm,r in picks:
        i=int(r.film); fi=f.iloc[i]; t0=int(fi.t0_spine_pos); a=int(r.age); q=t0+a
        e=float(fi.exit_boundary); up=fi.side=='north'
        print(f'--- {nm} --- riz {fi.riz_id}  ТФ {fi.tf_minutes}  {fi.side}  граница {e:g}')
        print(f'    T0 {et(ts[t0])}  close {cl[t0]:g}  ширина зоны {fi.zone_width:g}')
        for b in range(t0,min(q+1,t0+6)):
            dc=(cl[b]-e) if up else (e-cl[b])
            print(f'    {et(ts[b])}  O{o[b]:g} H{h[b]:g} L{l[b]:g} C{cl[b]:g}   до границы {dc:+.2f}')
        if q>t0+5: print(f'    … ещё {q-t0-5} минут ожидания')
        print(f'    РЕШЕНИЕ на закрытии {et(ts[q])}: d_close {r.d_close:.2f} п. > {4*c:g}, '
              f'откат от максимума ухода {r.retraced_fraction:.3f} >= 0,50 → ENTER')
        if r.kind==NO_TRADE:
            print('    исполнение отменено: next open вне окна, за промежутком либо уже не снаружи границы')
            print(); continue
        j=q+1; g=float(r.gross); stop=(o[j]+2*g) if up else (o[j]-2*g)
        print(f'    вход по open[{et(ts[j])}] = {o[j]:g}; цель {e:g} ({g:.2f} п.); стоп {stop:g} (2×цели)')
        end=j+int(r.dur)-1
        for b in range(j,min(end+1,j+4)):
            print(f'      {et(ts[b])}  O{o[b]:g} H{h[b]:g} L{l[b]:g} C{cl[b]:g}')
        if end>j+3: print(f'      … всего {int(r.dur)} минут в позиции, до {et(ts[end])}')
        res=float(r.lo) if not np.isnan(r.lo) else float("nan")
        print(f'    ИСХОД: {NAMES[int(r.kind)]}; валовое {res:+.2f} п., после расхода '
              f'{res-c:+.2f} п. = ${(res-c)*PV[inst]:+.0f}' if not np.isnan(res)
              else f'    ИСХОД: {NAMES[int(r.kind)]}; P&L не определён, нулём не становится')
        if int(r.kind) in (WIN,AMBIG) and int(r.tp_conf)==0:
            print('    TP НЕ ПОДТВЕРЖДЁН: прохода за границу на тик не наблюдалось')
        print()
    print(f'УПУЩЕННЫЕ ОЖИДАНИЕМ: {len(missed)} фильмов из {len(f)} '
          f'({len(missed)/len(f):.4f}) кончились раньше, чем правилу разрешено смотреть (age>=2)')
    r=missed.iloc[len(missed)//2]; t0=int(r.t0_spine_pos)
    print(f'  пример: riz {r.riz_id}, T0 {et(ts[t0])}, контакт через '
          f'{int(r.first_observed_contact_pos)-t0} бар(а) — решения не было вовсе')

if __name__=='__main__':
    main(*(sys.argv[1:] or ['NQ','evaluation']))
