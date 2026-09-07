"""S-07 v2: ломает ли живой RIZ симметрию открытия.

Берём решения v1 (сторона по предрыночному положению цены) и расщепляем их
состоянием поля в ту же минуту: что стоит на пути движения и стоит ли цена
внутри непроторгованной области. Поле только читается.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'data/research/S-07'
MIN = 60_000_000_000
MINUTE = 3          # минута решения из профиля v1
HORIZON = 60
NEAR = 3.0          # «близко» — три свечи, объявлено до счёта
EPOCHS = [('2006-2012', '2006', '2012'), ('2013-2019', '2013', '2019'), ('2020-2026', '2020', '2026')]


@njit(cache=False)
def field_state(t0, deletion, top, bottom, exit_side, order, p, price, out):
    """Для каждой минуты решения: ближайшая граница по ходу и против, число живых."""
    n = len(p)
    for i in range(n):
        pos = p[i]
        ahead = 1e18
        behind = 1e18
        inside = 0.
        alive = 0.
        agree = 0.
        for k in range(len(order)):
            j = order[k]
            if t0[j] > pos:
                break
            if deletion[j] <= pos:
                continue
            alive += 1.
            b = top[j] if exit_side[j] > 0 else bottom[j]
            d = b - price[i]
            if bottom[j] <= price[i] <= top[j]:
                inside += 1.
            if exit_side[j] > 0:
                agree += 1.
            if d > 0:
                if d < ahead:
                    ahead = d
            elif -d < behind:
                behind = -d
        out[i, 0] = ahead
        out[i, 1] = behind
        out[i, 2] = inside
        out[i, 3] = alive
        out[i, 4] = agree
    return out


def main(ins='NQ', minute=MINUTE, quiet=False):
    m = {k: np.load(ROOT / 'data/market' / ins / (k + '.npy'))
         for k in ['close_ts_utc_ns', 'open', 'high', 'low', 'close']}
    trades = pd.read_parquet(OUT / 'trades_v1.parquet')
    t = trades.loc[(trades.instrument == ins) & (trades.minute == minute) &
                   (trades.horizon == HORIZON) & (trades.stop == 0) & (trades.status > 0)].copy()
    cal = pd.read_parquet(ROOT / 'setups/S-04/calendar.parquet')
    day = cal.set_index('date').open_ns
    decide = np.searchsorted(m['close_ts_utc_ns'], day.loc[t.date].to_numpy() + minute * MIN)
    t['decide_pos'] = decide
    price = m['close'][decide]

    p = pd.read_parquet(ROOT / f'data/research/S-06/{ins}_objects.parquet')
    order = np.argsort(p.t0_spine_pos.to_numpy())
    out = np.full((len(t), 5), np.nan)
    out = field_state(p.t0_spine_pos.to_numpy(np.int64),
                      p.c1_deletion_spine_pos.fillna(len(m['close'])).to_numpy(np.int64),
                      p.zone_top.to_numpy(), p.zone_bottom.to_numpy(),
                      np.where(p.t0_exit_side.to_numpy() == 'north', 1, -1).astype(np.int64),
                      order.astype(np.int64), decide.astype(np.int64), price, out)
    up, down, inside, alive, agree = [out[:, i] for i in range(5)]
    side = t.side.to_numpy()
    unit = t.unit.to_numpy()
    # по ходу движения и против него, в свечах
    t['ahead_gap'] = np.where(side > 0, up, down) / unit
    t['behind_gap'] = np.where(side > 0, down, up) / unit
    t['inside_zones'] = inside
    t['alive'] = alive
    t['agree_share'] = np.where(alive > 0, np.where(side > 0, agree, alive - agree) / np.maximum(alive, 1), np.nan)
    t['year'] = t.date.str.slice(0, 4)
    t.to_parquet(OUT / f'{ins}_field_v2_m{minute}.parquet', index=False)
    if quiet:
        return t

    rows = []
    for name, mask in [('путь свободен (>3 свечей)', t.ahead_gap > NEAR),
                       ('граница близко (<=3 свечей)', t.ahead_gap <= NEAR),
                       ('цена внутри живой зоны', t.inside_zones > 0),
                       ('цена вне всех живых зон', t.inside_zones == 0)]:
        g = t.loc[mask]
        row = dict(split=name, trades=int(len(g)), mean=float(g.net_dollars.mean()),
                   median=float(g.net_dollars.median()), win=float((g.net_dollars > 0).mean()),
                   candles=float(g.gross_candles.mean()))
        for epoch, a, b in EPOCHS:
            part = g.loc[(g.year >= a) & (g.year <= b)]
            row[f'{epoch}'] = round(float(part.net_dollars.mean()), 1) if len(part) else None
            row[f'n_{epoch}'] = int(len(part))
        recent = g.loc[g.year >= '2020']
        row['recent_years_positive'] = int((recent.groupby('year').net_dollars.mean() > 0).sum())
        row['recent_years'] = int(recent.year.nunique())
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    pd.DataFrame(rows).to_csv(HERE / 'field_split_v2.csv', index=False)

    # тот же раскол внутри современной эпохи по квартилям свободного пути
    recent = t.loc[t.year >= '2020'].copy()
    recent['q'] = pd.qcut(recent.ahead_gap, 4, labels=False, duplicates='drop')
    print(recent.groupby('q').agg(n=('net_dollars', 'size'), mean=('net_dollars', 'mean'),
                                  median=('net_dollars', 'median'), gap=('ahead_gap', 'median'),
                                  candles=('gross_candles', 'mean')).round(2).to_string(), flush=True)


def across(instruments=('NQ', 'ES', 'YM'), minutes=(3, 4)):
    rows = []
    for ins in instruments:
        for minute in minutes:
            t = main(ins, minute, quiet=True)
            t['year'] = t.date.str.slice(0, 4)
            for name, mask in [('inside', t.inside_zones > 0), ('outside', t.inside_zones == 0)]:
                g = t.loc[mask]
                row = dict(instrument=ins, minute=minute, split=name, trades=int(len(g)),
                           candles=float(g.gross_candles.mean()), mean=float(g.net_dollars.mean()))
                for epoch, a, b in EPOCHS:
                    part = g.loc[(g.year >= a) & (g.year <= b)]
                    row[epoch] = round(float(part.gross_candles.mean()), 3) if len(part) else None
                    row[f'usd_{epoch}'] = round(float(part.net_dollars.mean()), 1) if len(part) else None
                rows.append(row)
    d = pd.DataFrame(rows)
    d.to_csv(HERE / 'field_across_v2.csv', index=False)
    pd.set_option('display.width', 250)
    print(d.to_string(index=False), flush=True)
    return d


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'across':
        across()
    else:
        main()
