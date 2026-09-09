import sys, json
sys.path.insert(0,'research')
import numpy as np
from film_corpus import build
from calendar_utils import date_key, minute_of_day, year_of
out={}
for ins in ('ES','NQ','YM'):
    c=build(ins); t0=c['t0']
    d=date_key(t0); mod=minute_of_day(t0)
    # окна: Лондон 02:00-09:30 ET, RTH 09:30-16:00 ET
    lon=(mod>=120)&(mod<570); rth=(mod>=570)&(mod<960)
    days=np.unique(d)
    def prof(m):
        dd=d[m]
        u,cn=np.unique(dd,return_counts=True)
        return {'days_with_any':int(len(u)),
                'share_of_all_days':float(len(u)/len(days)),
                'median_per_day':float(np.median(cn)),
                'p90_per_day':float(np.percentile(cn,90)),
                'max_per_day':int(cn.max())}
    # промежутки без сигнала (в календарных торговых днях корпуса)
    def gaps(m):
        u=np.unique(d[m]); allu=days
        pos=np.searchsorted(allu,u)
        g=np.diff(pos)-1
        return {'max_gap_days':int(g.max()) if len(g) else None,
                'p95_gap_days':float(np.percentile(g,95)) if len(g) else None,
                'share_days_without':float(1-len(u)/len(allu))}
    out[ins]={'films':int(len(t0)),'trading_days_in_corpus':int(len(days)),
              'all_hours':prof(np.ones(len(t0),bool)),
              'london_0200_0930_et':{**prof(lon),**gaps(lon)},
              'rth_0930_1600_et':{**prof(rth),**gaps(rth)}}
print(json.dumps(out,ensure_ascii=False,indent=1))
open('work/070/frequency.json','w',encoding='utf-8').write(json.dumps(out,ensure_ascii=False,indent=1))
