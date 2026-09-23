"""S-19 chronological one-contract replay and paired exit/continue comparison.

Unknown source history/quotes are not imputed. Stop certifying a session when
an unknown can change the chosen action; later rows remain policy-unknown.
This preserves a certified prefix, not a fictitious full resolved backtest.
Run only after documented competition rule and v2 declaration.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from census import load,ROOT,MINUTE

def position(a,entry,close_ns,d,target,far,cancel):
    ts=a['ts']; stop=int(np.searchsorted(ts,close_ns))
    result={'entry_pos':entry,'entry_price':float(a['open'][entry]),'direction':d,
            'status':'unknown','reason':'missing_session_close','exit_pos':None,'gross':None}
    close_known=stop<len(ts) and ts[stop]==close_ns
    scan_stop=stop if close_known else min(stop-1,len(ts)-1)
    for k in range(entry,scan_stop+1):
        if k>entry and ts[k]-ts[k-1]!=MINUTE:
            result.update(reason='gap_while_open',unknown_pos=k)
            return result
        why=''
        if close_known and k==stop:
            ep=k; price=a['close'][k]; why='session_close'
        elif d*(a['close'][k]-target)>=0:
            ep=k+1; price=a['open'][ep]; why='target_close'
        elif cancel and d*(a['close'][k]-far)<0:
            ep=k+1; price=a['open'][ep]; why='zone_close'
        if not why:
            continue
        if ep>=len(ts) or (not close_known and ts[ep]>close_ns):
            result.update(reason='missing_session_close',event_pos=k)
            return result
        if ep!=k and ts[ep]-ts[k]!=MINUTE:
            result.update(reason='gap_at_exit',unknown_pos=ep,event_pos=k)
            return result
        lo=float(a['low'][entry:ep].min()) if ep>entry else float(a['low'][entry])
        hi=float(a['high'][entry:ep].max()) if ep>entry else float(a['high'][entry])
        if why=='session_close':
            lo=min(lo,float(a['low'][ep])); hi=max(hi,float(a['high'][ep]))
        else:
            lo=min(lo,float(price)); hi=max(hi,float(price))
        v=result['entry_price']
        result.update(status='known',reason=why,event_pos=k,exit_pos=ep,exit_price=float(price),
            gross=float(d*(price-v)),net=float(d*(price-v)-1),
            held_minutes=float((ts[ep]-ts[entry])/MINUTE+(why=='session_close')),
            mae=float(v-lo if d>0 else hi-v),mfe=float(hi-v if d>0 else v-lo))
        return result
    return result

def replay(parent,a,calendar,cancel):
    decisions=[]; trades=[]; days=[]; pairs=[]
    candidates=parent[(parent.clock_status=='entry_window')&
        parent.structural_status.isin(['structural_signal','prefix_unknown','ancestry_unavailable'])][
        ['riz_id','tf_minutes','t0_spine_pos','session_date','precursor_formed_spine_pos',
         'structural_status','direction','target','far']]
    bydate={k:v for k,v in candidates.groupby('session_date')}
    for day in calendar.itertuples():
        date=day.date; release=-1; unknown=False; cash=0.; n=0
        dayrows=bydate.get(date,pd.DataFrame())
        if len(dayrows):
            dayrows=dayrows.sort_values(['t0_spine_pos','precursor_formed_spine_pos','tf_minutes','riz_id'])
        for q,group in dayrows.groupby('t0_spine_pos') if len(dayrows) else []:
            q=int(q); chosen=False
            for r in group.itertuples():
                rec={'riz_id':r.riz_id,'tf':r.tf_minutes,'q':q,'date':date,'reason':''}
                if unknown:
                    rec['reason']='policy_unknown_after_prior_gap'
                elif chosen:
                    rec['reason']='simultaneous_not_selected'
                elif q<release:
                    rec['reason']='busy'
                elif r.structural_status!='structural_signal':
                    # The oldest possible action is not knowably a signal/non-signal.
                    rec['reason']='prefix_unknown_changes_selection'; unknown=True
                else:
                    chosen=True; entry=q+1
                    if entry>=len(a['ts']) or a['ts'][entry]-a['ts'][q]!=MINUTE:
                        rec['reason']='entry_unknown'; unknown=True
                    elif a['ts'][entry]>day.close_ns:
                        rec['reason']='no_time_to_enter'
                    elif r.direction*(r.target-a['open'][entry])<=0 or r.direction*(a['open'][entry]-r.far)<0:
                        rec['reason']='open_outside_action'
                    else:
                        t=position(a,entry,day.close_ns,r.direction,r.target,r.far,cancel)
                        t.update(riz_id=r.riz_id,tf=r.tf_minutes,q=q,date=date,target=r.target,far=r.far)
                        trades.append(t); n+=1
                        rec['reason']='entered_'+t['status']
                        if t['status']=='known':
                            release=t['exit_pos']; cash+=t['net']
                            if cancel and t['reason']=='zone_close':
                                counter=position(a,entry,day.close_ns,r.direction,r.target,r.far,False)
                                pair={'date':date,'riz_id':r.riz_id,'event_pos':t['event_pos'],
                                      'exit_now_pos':t['exit_pos'],'exit_now_price':t['exit_price'],
                                      'continue_status':counter['status'],'continue_reason':counter['reason'],
                                      'continue_exit_pos':counter['exit_pos'],'delta_exit_minus_continue':None}
                                if counter['status']=='known':
                                    pair.update(delta_exit_minus_continue=t['net']-counter['net'],
                                                continuation_net=counter['net'],exit_net=t['net'],
                                                recovered_winner=counter['net']>0)
                                pairs.append(pair)
                        else:
                            unknown=True
                decisions.append(rec)
        days.append({'date':date,'status':'unknown' if unknown else 'known',
                     'net':None if unknown else cash,'certified_prefix_net':cash,'entries':n})
    return pd.DataFrame(decisions),pd.DataFrame(trades),pd.DataFrame(days),pd.DataFrame(pairs)

def main():
    a,_,_=load(); p=pd.read_parquet(ROOT/'work/103/parent.parquet')
    cal=pd.read_csv(ROOT/'base/103/calendar.csv')
    out=ROOT/'work/103/replay'; out.mkdir(exist_ok=True)
    results={}
    for label,cancel in [('v1',True),('v2',False)]:
        decisions,trades,days,pairs=replay(p,a,cal,cancel)
        for name,data in [('decisions',decisions),('trades',trades),('daily',days),('pairs',pairs)]:
            data.to_parquet(out/f'{label}_{name}.parquet',index=False)
        known=days[days.status=='known']; kt=trades[trades.status=='known']
        results[label]={'sessions':len(days),'known_sessions':len(known),'unknown_sessions':int((days.status=='unknown').sum()),
            'all_session_net_mean':float(days.net.mean()) if len(known)==len(days) else None,
            'resolved_sessions_only_mean_not_population':float(known.net.mean()),
            'certified_entries':len(trades),'known_trade_outcomes':len(kt),
            'known_trades_gross_mean_not_policy':float(kt.gross.mean()),
            'decision_reasons':decisions.reason.value_counts().to_dict(),
            'exit_reasons':trades.reason.value_counts().to_dict(),
            'sign_status':('UNRESOLVED: certified replay stops at shared missing history; '
                           'trader observation channel and joint full-policy bounds not established')
                           if len(known)!=len(days) else 'requires dependence inference'}
        if len(pairs):
            kp=pairs[pairs.continue_status=='known']
            results[label]['pairs']={'events':len(pairs),'known':len(kp),'unknown':len(pairs)-len(kp),
                'delta_mean_known_only':float(kp.delta_exit_minus_continue.mean()),
                'recovered_winners_known':int(kp.recovered_winner.sum())}
    (ROOT/'base/103/replay_summary.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(json.dumps(results,indent=2))

if __name__=='__main__': main()
