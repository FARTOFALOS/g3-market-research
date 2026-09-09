import sys, numpy as np
from pathlib import Path
ROOT=Path('C:/Users/Admin/Claude/g3-market-research')
sys.path.insert(0,str(ROOT/'setups/S-09')); import run as s09
MIN=60_000_000_000
rows=[]
for ins in ('NQ','ES','YM'):
    m=s09.market(ins); ts=m['close_ts_utc_ns']
    tr=s09.trades(ins, lo=s09.S0, hi=s09.H1)
    p=np.searchsorted(ts,tr['ts'].to_numpy()); s=tr['side'].to_numpy()
    def col(name,off):
        q=p+off; ok=(q>=0)&(q<len(ts)); v=np.full(len(p),np.nan)
        g=ok.copy(); g[ok]&= ts[q[ok]]==ts[p[ok]]+off*MIN; v[g]=m[name][q[g]]; return v
    tr['al']= s*(col('close',-1)-col('close',-3))>0
    for door in ('utro_0933','den_1338'):
        for terr,mk in (('поиск',~tr['holdout']),('отлож',tr['holdout'])):
            d=tr[(tr['door']==door)&mk]
            a=d[d['al']]['net'].to_numpy(); b=d[~d['al']]['net'].to_numpy()
            rows.append((ins,door,terr,len(a),a.mean(),len(b),b.mean(),a.mean()-b.mean()))
print(f"{'ins':4}{'дверь':11}{'терр':7}{'по ходу n':>10}{'ср$':>9}{'против n':>10}{'ср$':>9}{'разница$':>10}{'':3}знак")
for r in rows:
    print(f"{r[0]:4}{r[1]:11}{r[2]:7}{r[3]:10d}{r[4]:9.1f}{r[5]:10d}{r[6]:9.1f}{r[7]:10.1f}   {'ПО ХОДУ' if r[7]>0 else 'ПРОТИВ'}")
