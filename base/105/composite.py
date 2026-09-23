"""Составная политика: общий поток сигналов нескольких ветвей, одна позиция, дневные правила (LOG.md, C0…)."""
import math, numpy as np, pandas as pd
from trade import Tape, simulate, COST, SLIP, LIMIT, DAY_LIMIT, TICK
from v0_batch1 import s07_side

pd.set_option('display.width', 260); pd.set_option('display.max_columns', 40)


def s07_signals(T, cap=18.5):
    rows = []
    for D, ix in T.days.items():
        sg, i33 = s07_side(T, D, ix)
        i1133 = T.at(D, 693)
        if sg == 0 or i1133 < 0 or i33 + 1 >= len(T.o):
            continue
        e = T.o[i33 + 1]
        rows.append(dict(date=D, irec=i33, side=sg, stop=e - sg * cap, target=np.nan, iend=i1133))
    return pd.DataFrame(rows)


def library(T, names):
    parts = []
    for nm in names:
        if nm == 'S07':
            s = simulate(T, s07_signals(T))
        else:
            s = pd.read_csv(f'{nm}_trades.csv')
        s = s.copy(); s['branch'] = nm
        if 'iend' not in s:
            s['iend'] = -1
        parts.append(s[['date', 'irec', 'side', 'stop', 'target', 'iend', 'branch', 'entry', 'exit', 'xtype', 'jexit',
                        'risk', 'plan_loss', 'net', 'fits']])
    return pd.concat(parts, ignore_index=True)


def resim(T, r, newrisk):
    sig = pd.DataFrame([dict(date=r['date'], irec=r['irec'], side=r['side'], stop=r['entry'] - r['side'] * newrisk,
                             target=r['target'], iend=r['iend'])])
    x = simulate(T, sig).iloc[0]
    for k in ('exit', 'xtype', 'jexit', 'net', 'risk', 'plan_loss'):
        r[k] = x[k]
    return r


def run(T, lib, lock=True, min_lim=3.0, y0=20200101, y1=20251231):
    lib = lib[(lib.date >= y0) & (lib.date <= y1) & (lib.xtype > 0)].sort_values(['date', 'irec'])
    out = []
    for D, g in lib.groupby('date', sort=True):
        pnl = 0.0; busy = -1
        for _, r in g.iterrows():
            if r.irec + 1 <= busy or not r.fits:
                continue
            lim = min(LIMIT, DAY_LIMIT + pnl)
            if lock and pnl > 0:
                lim = min(lim, pnl)
            if lim < min_lim:
                continue
            r = r.copy()
            if r.plan_loss > lim:
                r = resim(T, r, lim - COST - SLIP)
                r['cut'] = True
            pnl += r.net; busy = r.jexit
            r['day_after'] = pnl
            out.append(r)
    return pd.DataFrame(out)


def req(tk, y0=20200101, y1=20251231):
    cal = pd.read_csv('calendar_nq.csv')
    terr = cal[(cal.date >= y0) & (cal.date <= y1)]
    elig = terr[terr.status.isin(['regular', 'short']) | ((terr.status == 'special') & terr['last'].notna())]
    x = tk[(tk.date >= y0) & (tk.date <= y1)]
    day = x.groupby('date').agg(net=('net', 'sum'), n=('net', 'size'), low=('day_after', 'min'))
    day = day.reindex(elig.date.values)      # допустимые дни без сделки — видны как NaN
    traded = day.dropna(subset=['net'])
    total = traded.net.sum(); k = math.ceil(0.05 * len(traded))
    return dict(elig=len(elig), cover=round(len(traded) / len(elig), 3), trades=len(x), total=round(total, 1),
                per_day=round(traded.net.mean(), 2), t=round(traded.net.mean() / traded.net.std() * math.sqrt(len(traded)), 2),
                days_ge0=round((traded.net >= -1e-9).mean(), 3), neg_days=int((traded.net < -1e-9).sum()),
                worst=round(traded.net.min(), 1), breach60=int((traded.low < -DAY_LIMIT).sum()),
                over20=int((x.net < -LIMIT).sum()), top5=round(traded.net.nlargest(k).sum() / total, 2) if total > 0 else None), traded


if __name__ == '__main__':
    import sys
    T = Tape()
    names = sys.argv[1].split(',')
    lib = library(T, names)
    for lock in (False, True):
        tk = run(T, lib, lock=lock)
        r, day = req(tk)
        r['lock'] = lock; r['branches'] = '+'.join(names)
        print(r)
        print(tk.groupby('branch').net.agg(['size', 'mean', 'sum']).round(2).to_string())
        tk.to_csv(f'C0_{"lock" if lock else "base"}.csv', index=False)
