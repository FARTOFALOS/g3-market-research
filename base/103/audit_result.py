"""Audit real executions, unknown witnesses, source hashes, and readable scenes."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from census import ROOT,load,MINUTE

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def candle(ax,a,lo,hi):
    pos=np.arange(lo,hi+1); o=a['open'][pos]; c=a['close'][pos]
    for i,(p,x,y) in enumerate(zip(pos,o,c)):
        color='#217a54' if y>=x else '#b84740'
        ax.plot([i,i],[a['low'][p],a['high'][p]],color=color,lw=.6)
        ax.add_patch(Rectangle((i-.3,min(x,y)),.6,max(abs(y-x),.03),color=color,lw=.3))
    ix=np.linspace(0,len(pos)-1,min(5,len(pos))).astype(int)
    labels=pd.to_datetime(a['ts'][pos[ix]],utc=True).tz_convert('America/New_York').strftime('%m-%d %H:%M')
    ax.set_xticks(ix,labels,rotation=20,fontsize=7); ax.grid(alpha=.15)

def main():
    a,_,manifest=load(); out=ROOT/'work/103/replay'
    p=pd.read_parquet(ROOT/'work/103/parent.parquet').set_index('riz_id')
    verified=[]
    for v in ('v1','v2'):
        t=pd.read_parquet(out/f'{v}_trades.parquet'); k=t[t.status=='known']
        # Independent vectorized check of every known fill and earliest closed event.
        for r in k.itertuples():
            en=int(r.entry_pos); ex=int(r.exit_pos); event=int(r.event_pos); d=r.direction
            price=a['close'][ex] if r.reason=='session_close' else a['open'][ex]
            assert r.entry_price==a['open'][en] and r.exit_price==price
            assert r.gross==d*(price-r.entry_price) and r.net==r.gross-1
            c=a['close'][en:event]
            assert not np.any(d*(c-r.target)>=0)
            if v=='v1': assert not np.any(d*(c-r.far)<0)
            assert np.all(np.diff(a['ts'][en:ex+1])==MINUTE)
            assert r.q==int(p.loc[r.riz_id,'t0_spine_pos']) and en==r.q+1
            if r.reason=='target_close': assert d*(a['close'][event]-r.target)>=0 and ex==event+1
            if r.reason=='zone_close': assert d*(a['close'][event]-r.far)<0 and ex==event+1
        day=pd.read_parquet(out/f'{v}_daily.parquet')
        sums=k.groupby('date').net.sum()
        for r in day.itertuples():
            assert abs(r.certified_prefix_net-sums.get(r.date,0))<1e-9
            if r.status=='known': assert r.net==r.certified_prefix_net
            else: assert pd.isna(r.net)
        verified.append({'version':v,'all_known_executions_checked':len(k),'all_session_accounting_checked':len(day)})
    t=pd.read_parquet(out/'v1_trades.parquet')
    pairs=pd.read_parquet(out/'v1_pairs.parquet')
    assert set(pairs.riz_id)==set(t[t.reason=='zone_close'].riz_id)
    for r in pairs[pairs.continue_status=='known'].itertuples():
        assert r.delta_exit_minus_continue==r.exit_net-r.continuation_net
    # A concrete source-gap witness that can change the next selected action.
    decisions=pd.read_parquet(out/'v1_decisions.parquet')
    u=decisions[decisions.reason=='prefix_unknown_changes_selection'].iloc[0]
    z=p.loc[u.riz_id]; start=int(z.c3_start); q=int(z.t0_spine_pos)
    gaps=np.flatnonzero((np.diff(a['ts'][start:q+1])!=MINUTE)&
                       (np.diff(a['session_id'][start:q+1])==0))+start+1
    witness={'riz_id':u.riz_id,'date':u.date,'tf':int(z.tf_minutes),
        't0':str(pd.Timestamp(int(a['ts'][q]),tz='UTC')),
        'observed_target_not_certified':float(z.target),'first_span_pos':int(z.first_span_pos),
        'gaps':[{'before':str(pd.Timestamp(int(a['ts'][g-1]),tz='UTC')),
                 'after':str(pd.Timestamp(int(a['ts'][g]),tz='UTC')),
                 'scheduled_pause':bool((a['ts'][g]-a['ts'][g-1]==16*MINUTE) and
                   pd.Timestamp(int(a['ts'][g]),tz='UTC').tz_convert('America/New_York').strftime('%H:%M')=='16:31'),
                 'in_target_interval':bool(g<=z.first_span_pos)} for g in gaps[:20]],
        'meaning':'missing extrema/touches can change admission or E; no valid full-policy bounds were established'}
    (ROOT/'base/103/unknown_witness.json').write_text(json.dumps(witness,indent=2),encoding='utf-8')
    # Exact bytes of inputs (hashing does not expose future market observations).
    hashes={}
    for name in ('open.npy','high.npy','low.npy','close.npy','close_ts_utc_ns.npy','session_id.npy','sessions.npz'):
        h=sha(ROOT/'data/market/NQ'/name); assert h==manifest['array_sha256'][name]; hashes[name]=h
    # Illustrations selected by declared outcome class, not for rule selection.
    rows=[]
    no_recovery=set(pairs[(pairs.continue_status=='known')&(pairs.continuation_net<=0)].riz_id)
    for label,frame in [('target',t[(t.status=='known')&(t.reason=='target_close')]),
                         ('zone_loss',t[(t.status=='known')&(t.reason=='zone_close')&(t.net<0)&t.riz_id.isin(no_recovery)])]:
        rows.append((label,frame.sort_values('entry_pos').iloc[0]))
    recovered=pairs[(pairs.continue_status=='known')&(pairs.recovered_winner==True)].sort_values('event_pos').iloc[0]
    rows.append(('recovered_after_exit',t[t.riz_id==recovered.riz_id].iloc[0]))
    scenes=[]; sceneout=ROOT/'base/103/scenes'; sceneout.mkdir(exist_ok=True)
    for label,r in rows:
        z=p.loc[r.riz_id]; q=int(r.q); en=int(r.entry_pos); ex=int(r.exit_pos)
        end=ex
        if label=='recovered_after_exit': end=int(recovered.continue_exit_pos)
        start=int(z.c3_start)
        for suffix,lo,hi in [('prefix',start,q),('continuation',q+1,end)]:
            pos=np.arange(lo,hi+1)
            df=pd.DataFrame({'spine_pos':pos,'relative_to_t0':pos-q,
                'close_utc':pd.to_datetime(a['ts'][pos],utc=True)})
            for name in ('open','high','low','close'): df[name]=a[name][pos]
            df.to_csv(sceneout/f'{label}_{suffix}.csv',index=False)
        fig,axes=plt.subplots(1,2,figsize=(13,4.3))
        left=max(start,q-59); candle(axes[0],a,left,q); candle(axes[1],a,q+1,end)
        for ax in axes:
            ax.axhline(z.target,color='#315ea3',lw=1,label='Original extreme E')
            ax.axhspan(z.zone_bottom,z.zone_top,color='#bca670',alpha=.2,label='RIZ')
        axes[0].set_title('Closed prefix to T0 (last 60 bars at most)')
        axes[1].set_title('Continuation shown separately')
        axes[1].scatter([0],[r.entry_price],marker='>',color='black',s=32,zorder=5)
        axes[1].scatter([ex-(q+1)],[r.exit_price],marker='x',color='black',s=40,zorder=5)
        axes[1].annotate('v1 exit',(ex-(q+1),r.exit_price),xytext=(5,10),textcoords='offset points',fontsize=8)
        axes[0].legend(fontsize=7)
        fig.suptitle(f'{label} | TF {int(z.tf_minutes)} | {r.date} | known v1 net {r.net:.2f} points')
        fig.tight_layout(); fig.savefig(sceneout/f'{label}.png',dpi=145); plt.close(fig)
        scenes.append({'label':label,'riz_id':r.riz_id,'tf':int(z.tf_minutes),'q':q,
                       'target':float(z.target),'target_pos':int(z.target_pos),'far':float(z.far),
                       'entry':float(r.entry_price),'exit':float(r.exit_price),'net':float(r.net)})
    (sceneout/'index.json').write_text(json.dumps(scenes,indent=2),encoding='utf-8')
    # A close-looking non-trigger: E has already been touched, so the proposed
    # unfinished return does not exist at T0. It is not added to the signal denominator.
    near=p[(p.clock_status=='entry_window')&(p.structural_status=='target_already_visited')].sort_values('t0_spine_pos').iloc[0]
    q=int(near.t0_spine_pos); lo=int(near.c3_start); pos=np.arange(lo,q+1)
    df=pd.DataFrame({'spine_pos':pos,'relative_to_t0':pos-q,'close_utc':pd.to_datetime(a['ts'][pos],utc=True)})
    for name in ('open','high','low','close'): df[name]=a[name][pos]
    df.to_csv(sceneout/'near_miss_prefix.csv',index=False)
    near_meta={'riz_id':near.name,'tf':int(near.tf_minutes),'q':q,'target':float(near.target),
               't0_close':float(near.t0_close),'reason':'original extreme already visited after span1; no v1/v2 signal'}
    (sceneout/'near_miss.json').write_text(json.dumps(near_meta,indent=2),encoding='utf-8')
    result={'status':'pass','verified':verified,'paired_event_denominator_matches_certified_v1_exits':True,
        'source_hashes_verified':hashes,'scope':'known executions/accounting/source identity; full policy unresolved',
        'illustration_selection':'first chronological case in each named outcome class'}
    (ROOT/'base/103/result_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
