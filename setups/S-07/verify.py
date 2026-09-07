"""S-07: устойчивость, стоимость выбора и физическая проверка префикса."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import run as base

ROOT = base.ROOT
HERE = base.HERE
OUT = base.OUT
MIN = base.MIN


def yearly(t):
    g = t.groupby('year').net_dollars.agg(['sum', 'mean', 'size'])
    return {y: dict(total=round(float(r['sum']), 1), mean=round(float(r['mean']), 1), trades=int(r['size']))
            for y, r in g.iterrows()}


def robustness():
    d = pd.read_parquet(OUT / 'trades_v1.parquet')
    d = d.loc[d.status > 0].copy()
    d['year'] = d.date.str.slice(0, 4)
    out = {}
    for key, sel in [('NQ_m3_h60_nostop', (d.instrument == 'NQ') & (d.minute == 3) & (d.horizon == 60) & (d.stop == 0)),
                     ('NQ_m3_h60_stop2', (d.instrument == 'NQ') & (d.minute == 3) & (d.horizon == 60) & (d.stop == 2)),
                     ('NQ_m4_h60_nostop', (d.instrument == 'NQ') & (d.minute == 4) & (d.horizon == 60) & (d.stop == 0)),
                     ('NQ_m3_h30_nostop', (d.instrument == 'NQ') & (d.minute == 3) & (d.horizon == 30) & (d.stop == 0)),
                     ('NQ_m3_h120_nostop', (d.instrument == 'NQ') & (d.minute == 3) & (d.horizon == 120) & (d.stop == 0))]:
        t = d.loc[sel]
        recent = t.loc[t.year >= '2020']
        daily = recent.groupby('date').net_dollars.sum().sort_index()
        eq = np.r_[0, np.cumsum(daily.to_numpy())]
        drop = max(1, int(len(daily) * .05))
        out[key] = dict(trades_all=int(len(t)), mean_all=round(float(t.net_dollars.mean()), 2),
                        recent_trades=int(len(recent)), recent_total=round(float(daily.sum()), 1),
                        recent_mean=round(float(recent.net_dollars.mean()), 1),
                        recent_median=round(float(recent.net_dollars.median()), 1),
                        recent_win=round(float((recent.net_dollars > 0).mean()), 3),
                        max_dd=round(float((np.maximum.accumulate(eq) - eq).max()), 1),
                        without_best_5pct=round(float(daily.sum() - daily.nlargest(drop).sum()), 1),
                        worst_trade=round(float(recent.net_dollars.min()), 1),
                        candles_recent=round(float(recent.gross_candles.mean()), 3),
                        candles_2006_2012=round(float(t.loc[t.year < '2013'].gross_candles.mean()), 3),
                        candles_2013_2019=round(float(t.loc[(t.year >= '2013') & (t.year < '2020')].gross_candles.mean()), 3),
                        years=yearly(recent))
    return out


def selection_cost():
    """Годовой выбор по прошлым трём годам среди всех конфигураций перебора."""
    d = pd.read_parquet(OUT / 'trades_v1.parquet')
    d = d.loc[d.status > 0].copy()
    d['id'] = d.instrument + '_m' + d.minute.astype(str) + '_h' + d.horizon.astype(str) + '_s' + d.stop.astype(int).astype(str)
    daily = d.pivot_table(index='date', columns='id', values='net_dollars', aggfunc='sum')
    counts = d.pivot_table(index='date', columns='id', values='net_dollars', aggfunc='size')
    years = pd.to_datetime(daily.index).year.to_numpy()
    curve = np.full(len(daily), np.nan)
    records = []
    for year in range(2010, 2027):
        train = (years >= year - 3) & (years < year)
        test = years == year
        if not train.sum() or not test.sum():
            continue
        means = daily.loc[train].mean().where(counts.loc[train].sum() >= 100, -np.inf)
        winner = means.idxmax() if means.max() > 0 else None
        curve[test] = daily.loc[test, winner].fillna(0.) if winner else 0.
        records.append(dict(year=int(year), variant=winner,
                            test_trades=int(counts.loc[test, winner].sum()) if winner else 0,
                            test_net=round(float(np.nansum(curve[test])), 1)))
    eq = np.r_[0, np.nancumsum(curve)]
    return dict(rows=records, total=round(float(np.nansum(curve)), 1),
                max_dd=round(float((np.maximum.accumulate(eq) - eq).max()), 1),
                configurations=int(daily.shape[1]))


def prefix_check(ins='NQ', minute=3, cases=200):
    """Физическое обрезание ленты: решение не меняется от будущих минут."""
    m = base.market(ins)
    u = base.unit(m)
    cal = pd.read_parquet(ROOT / 'setups/S-04/calendar.parquet')
    ts = m['close_ts_utc_ns']
    open_pos = np.searchsorted(ts, cal.open_ns.to_numpy())
    p = open_pos + minute
    good = (p < len(ts)) & (ts[np.minimum(p, len(ts) - 1)] == cal.open_ns.to_numpy() + minute * MIN)
    idx = np.where(good)[0]
    decide = p[good].astype(np.int64)
    rng = np.random.default_rng(20260908)
    pick = rng.choice(len(decide), size=min(cases, len(decide)), replace=False)
    trades = pd.read_parquet(OUT / 'trades_v1.parquet')
    saved = trades.loc[(trades.instrument == ins) & (trades.minute == minute) &
                       (trades.horizon == 60) & (trades.stop == 0) & (trades.status > 0)].set_index('date')
    dates = cal.date.to_numpy()[idx]
    checks = 0
    for i in pick:
        date = dates[i]
        if date not in saved.index:
            continue
        pos = int(decide[i])
        cut = pos + 1                       # лента физически заканчивается на минуте входа
        hi = m['high'][:cut + 1]
        lo = m['low'][:cut + 1]
        cl = m['close'][:cut + 1]
        op = m['open'][:cut + 1]
        a = pos - base.WINDOW + 1
        side = 1. if cl[pos] > (hi[a:pos + 1].max() + lo[a:pos + 1].min()) / 2 else -1.
        row = saved.loc[date]
        assert side == row.side, 'сторона зависит от будущего'
        assert op[pos + 1] == row.entry, 'цена входа зависит от будущего'
        checks += 1
    return dict(cases=checks, side_and_entry_stable_under_truncation=True)


def main():
    report = dict(robustness=robustness(), selection=selection_cost(), prefix=prefix_check())
    (HERE / 'verification_v1.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    r = report['robustness']['NQ_m3_h60_nostop']
    print(json.dumps({k: v for k, v in r.items() if k != 'years'}, ensure_ascii=False, indent=2), flush=True)
    print('годы:', json.dumps(r['years'], ensure_ascii=False), flush=True)
    print('стоп 2 свечи:', json.dumps({k: v for k, v in report['robustness']['NQ_m3_h60_stop2'].items() if k != 'years'}, ensure_ascii=False), flush=True)
    print('выбор среди всех конфигураций:', json.dumps(report['selection'], ensure_ascii=False), flush=True)
    print('префикс:', report['prefix'], flush=True)


if __name__ == '__main__':
    main()
