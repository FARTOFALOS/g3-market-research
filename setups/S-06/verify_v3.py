"""S-06 v3: fresh confined recognition, independent raw-OHLC reconciliation, all-72 daily check."""
import json
import numpy as np
import pandas as pd
import run as breakout
import reject
import confined
import verify as earlier
from run import ROOT,HERE,OUT,M,POINT,COST,TICK,market,dump,recognition

def confinement(z,m):
    # The v3 condition itself, re-read from raw minutes: closes inside the shared area.
    p=int(z.p);k=int(z.k);a=p-k+1
    lo=m['low'][a:p+1];hi=m['high'][a:p+1];cl=m['close'][a:p+1]
    L=float(lo.max());U=float(hi.min())
    assert L==z.common_low and U==z.common_high and L<U
    assert float(cl.min())>=L and float(cl.max())<=U
    assert z.box_low<=L and U<=z.box_high

def main():
    m=market('NQ');reports=[]
    fresh_pauses,_=recognition('NQ',k=3)
    fresh_rows=confined.gate(fresh_pauses,m,3)
    saved_rows=pd.read_parquet(OUT/'NQ_confined_k3.parquet')
    pd.testing.assert_frame_equal(fresh_rows,saved_rows,check_dtype=False)
    for version,key in [(1,'NQ_k3_stop_h5_confined'),(2,'NQ_k3_reject_target_h5_confined')]:
        module=breakout if version==1 else reject
        config=next(x for x in confined.family() if x['id']==key)
        s=module.decisions(fresh_rows,m,'stop') if version==1 else module.decisions(fresh_rows,m)
        fresh=module.replay(s,m,config).sort_values('q').reset_index(drop=True)
        saved=pd.read_parquet(OUT/'all_decisions_v3.parquet',filters=[('variant','==',key)]).sort_values('q').reset_index(drop=True)
        # The stored ledger keeps both action grammars; columns of the other one stay empty.
        extra=[c for c in saved.columns if c not in fresh.columns]
        assert saved[extra].isna().all().all()
        shared=[c for c in saved.columns if c in fresh.columns]
        pd.testing.assert_frame_equal(fresh[shared],saved[shared],check_dtype=False)
        trades=fresh.loc[fresh.day_known&(fresh.status>0)]
        for z in trades.itertuples():
            confinement(z,m);earlier.direct(z,m,version,version==2,5)
        fresh.to_parquet(OUT/f'illustrative_v3_{version}_replay.parquet',index=False)
        daily=pd.read_parquet(OUT/'daily_v3.parquet').set_index('date')[key]
        years=[]
        for name,start,stop in [('2006-2012','2006-01-01','2012-12-31'),('2013-2019','2013-01-01','2019-12-31'),('2020-2026','2020-01-01','2026-12-31')]:
            d=daily.loc[(daily.index>=start)&(daily.index<=stop)]
            z=trades.loc[(trades.date>=start)&(trades.date<=stop)]
            years.append(dict(period=name,trades=len(z),gross=float(z.gross_dollars.sum()),
                net=float(z.net_dollars.sum()),mean_net=float(z.net_dollars.mean()),
                known_days=int(d.notna().sum()),trade_days=z.date.nunique()))
        risks=trades.initial_risk_points*POINT['NQ'];recent=trades.loc[trades.date>='2020-01-01']
        info=dict(variant=key,full_replay_equal=True,independent_trade_checks=len(trades),
            total=float(trades.net_dollars.sum()),median_risk=float(risks.median()),
            risk_q90=float(risks.quantile(.9)),worst_trade=float(trades.net_dollars.min()),
            average_trades_per_active_day_2020=float(len(recent)/recent.date.nunique()),
            trade_days_2020=recent.date.nunique(),known_days_2020=int(daily.loc[daily.index>='2020-01-01'].notna().sum()),
            exposure_age_native_quantiles=trades.age_native_bars.quantile([.1,.5,.9]).tolist(),epochs=years)
        if version==2:
            optim=module.replay(s,m,config,optimistic=True)
            delay=module.replay(s,m,config,delay=1);delay=delay.loc[delay.day_known&(delay.status>0)]
            info.update(optimistic_total=float(optim.loc[optim.day_known&(optim.status>0),'net_dollars'].sum()),
                one_minute_delay=dict(trades=len(delay),net=float(delay.net_dollars.sum())),
                without_best_ten_days=float(daily.sum()-daily.nlargest(10).sum()))
        reports.append(info)
        print('INDEPENDENT v3',key,len(trades),flush=True)
    summary=pd.read_csv(HERE/'summary_v3.csv')
    daily=pd.read_parquet(OUT/'daily_v3.parquet').set_index('date')
    count=pd.read_parquet(OUT/'counts_v3.parquet').set_index('date')
    for z in summary.itertuples():
        assert daily[z.id].sum()==z.total and count[z.id].sum()==z.trades
    dump(HERE/'verification_v3.json',dict(illustrative_rules=reports,
        fresh_confined_recognition_equal=True,all_72_v3_daily_results_reconciled=True,
        positive_rules=int((summary.total>0).sum()),best_rule=summary.loc[summary.total.idxmax(),'id'],
        best_total=float(summary.total.max())))
    print(json.dumps(reports,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
