"""S-08 fixed candidate. No Volume; no data at/after 2025-11-01 in calculation."""
from pathlib import Path
import argparse, hashlib, json, platform
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'data/research/S-08'
M = 60_000_000_000
CUT = pd.Timestamp('2025-11-01', tz='UTC').value
KEYS = ['close_ts_utc_ns', 'open', 'high', 'low', 'close']
VARIANTS = [('v1',60,1),('h30',30,1),('h120',120,1),('delay1',60,2)]

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def load():
    m={k:np.load(ROOT/'data/market/NQ'/f'{k}.npy',mmap_mode='r') for k in KEYS}
    # Searching the timestamp index establishes the bound; no future OHLC read.
    n=np.searchsorted(m['close_ts_utc_ns'],CUT)
    m={k:v[:n] for k,v in m.items()}
    cal=pd.read_parquet(ROOT/'setups/S-04/calendar.parquet')
    return m,cal[cal.close_ns<CUT].reset_index(drop=True)

def recognize(m,prev,day):
    ts=m['close_ts_utc_ns']; op,hi,lo,cl=(m[k] for k in KEYS[1:])
    r={'date':str(day.date),'status':'unknown_previous_session'}
    a,b=np.searchsorted(ts,[int(prev.open_ns)+M,int(prev.close_ns)])
    count=(int(prev.close_ns)-int(prev.open_ns))//M
    if b>=len(ts) or b-a+1!=count or not np.array_equal(ts[a:b+1],np.arange(int(prev.open_ns)+M,int(prev.close_ns)+M,M)): return r
    H,L,C=float(hi[a:b+1].max()),float(lo[a:b+1].min()),float(cl[b])
    r.update(prev_high=H,prev_low=L,target=C,prev_date=str(prev.date))
    s=int(np.searchsorted(ts,int(day.open_ns)+M))
    if s>=len(ts) or ts[s]!=int(day.open_ns)+M:
        r['status']='unknown_open'; return r
    r['session_start']=s
    d=-1 if op[s]>H else (1 if op[s]<L else 0)
    r.update(side=d,opening_price=float(op[s]))
    if not d: r['status']='open_inside'; return r
    B=H if d<0 else L
    r['boundary']=B
    for k in range(30):
        p=s+k
        if p>=len(ts) or ts[p]!=int(day.open_ns)+(k+1)*M:
            r['status']='unknown_signal_prefix';return r
        if k<2:
            if (cl[p]-B)*d>=0: r['status']='no_two_external_closes';return r
            continue
        if (cl[p]-B)*d>=0:
            if not L<=cl[p]<=H: r['status']='crossed_entire_range';return r
            stop=float(hi[s:p+1].max()+.25) if d<0 else float(lo[s:p+1].min()-.25)
            r.update(status='signal',p=p,signal_ns=int(ts[p]),signal_close=float(cl[p]),stop=stop,
                     elapsed_open_minutes=k+1,session_end=int(day.close_ns))
            return r
    r['status']='no_reentry_by_1000';return r

def execute(m,r,horizon=60,delay=1,optimistic=False):
    ts=m['close_ts_utc_ns'];op,hi,lo,cl=(m[k] for k in KEYS[1:])
    p=int(r['p']);b=p+delay;d=int(r['side']);stop=r['stop'];target=r['target']
    z={'execution':'unknown_entry','gross_points':None}
    if b>=len(ts) or ts[b]!=ts[p]+delay*M:return z
    entry=float(op[b]);reward=(target-entry)*d;risk=(entry-stop)*d
    z.update(entry=entry,entry_pos=b,reward_points=reward,risk_points=risk)
    if not r['prev_low']<=entry<=r['prev_high'] or reward<=.75 or risk<=0:
        z['execution']='cancelled_open';return z
    end=min(int(ts[p])+horizon*M,int(r['session_end']))
    adverse=0.;favorable=0.
    for j in range(b,len(ts)):
        if ts[j]!=ts[p]+(j-p)*M or ts[j]>end:
            z['execution']='unknown_path';return z
        adverse=max(adverse,(entry-float(lo[j])) if d>0 else (float(hi[j])-entry))
        favorable=max(favorable,(float(hi[j])-entry) if d>0 else (entry-float(lo[j])))
        stop_hit=lo[j]<=stop if d>0 else hi[j]>=stop
        target_hit=hi[j]>=target+.25 if d>0 else lo[j]<=target-.25
        both=bool(stop_hit and target_hit)
        price=None;reason=None
        if (op[j]-stop)*d<=0:price=float(op[j]);reason='stop_gap'
        elif (op[j]-target)*d>=.25:price=target;reason='target_gap'
        elif stop_hit and (not optimistic or not target_hit):price=stop;reason='stop'
        elif target_hit:price=target;reason='target'
        elif ts[j]==end:price=float(cl[j]);reason='time'
        if price is not None:
            z.update(execution=reason,exit=price,exit_pos=j,gross_points=(price-entry)*d,
                     both_barriers=both,held_minutes=(int(ts[j])-int(ts[b])+M)/M,
                     mae_bar_bound=adverse,mfe_bar_bound=favorable)
            return z
    z['execution']='unknown_path';return z

def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['observe','run']);args=ap.parse_args()
    m,cal=load();OUT.mkdir(parents=True,exist_ok=True)
    rows=[recognize(m,cal.iloc[i-1],cal.iloc[i]) for i in range(1,len(cal))]
    rec=pd.DataFrame(rows);rec.to_parquet(OUT/'recognition.parquet',index=False)
    sig=rec[rec.status=='signal']
    if args.mode=='observe':
        # Coordinate chosen, never selected by subsequent profit.
        picks=sig[sig.date>='2024-01-01'].head(3)
        print(picks.to_json(orient='records',indent=2))
        for _,r in picks.iterrows():
            p=int(r.p);s=int(r.session_start)
            print(pd.DataFrame({k:m[k][s:p+1] for k in KEYS}).to_string(index=False))
        return
    tables=[]
    for name,h,delay in VARIANTS:
        for upper in [False,True]:
            for _,r in sig.iterrows():
                tables.append(dict(r,variant=name,bound='upper' if upper else 'lower',**execute(m,r,h,delay,upper)))
    t=pd.DataFrame(tables);t['net_usd']=t.gross_points*20-15
    t.to_parquet(OUT/'trades.parquet',index=False)
    summary=[]
    for (variant,bound),g in t.groupby(['variant','bound']):
        for era,start,stop in [('all','2006','2026'),('2006-2012','2006','2013'),('2013-2019','2013','2020'),('2020-2025','2020','2026')]:
            z=g[(g.date>=start)&(g.date<stop)];v=z[z.gross_points.notna()];net=v.net_usd
            eq=np.r_[0.,net.cumsum().to_numpy()]
            summary.append(dict(variant=variant,bound=bound,era=era,signals=len(z),trades=len(v),
                 unknown=int(z.execution.str.startswith('unknown').sum()),cancelled=int((z.execution=='cancelled_open').sum()),
                 gross_usd=float(v.gross_points.sum()*20),net_usd=float(net.sum()),mean_usd=float(net.mean()),
                 median_usd=float(net.median()),max_drawdown_closed_usd=float((np.maximum.accumulate(eq)-eq).max()),
                 win_fraction=float((net>0).mean()),worst_usd=float(net.min()),best_usd=float(net.max()),
                 median_hold_minutes=float(v.held_minutes.median()),median_risk_points=float(v.risk_points.median()),
                 median_reward_points=float(v.reward_points.median()),ties=int(v.both_barriers.sum())))
    pd.DataFrame(summary).to_csv(HERE/'summary.csv',index=False)
    result={'recognition':rec.status.value_counts().to_dict(),'calendar_days':len(rec),
            'variants':VARIANTS,'summary':summary,'rule_exposure':'exploratory; v1 fixed before this run; repository history not independent',
            'versions':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__},
            'hashes':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'SOURCE_DATA.json',ROOT/'setups/S-04/calendar.parquet',OUT/'recognition.parquet',OUT/'trades.parquet']}}
    (HERE/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'recognition':result['recognition'],'main':[s for s in summary if s['variant']=='v1' and s['bound']=='lower']},indent=2))

if __name__=='__main__':main()
