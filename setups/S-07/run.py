"""S-07: сторона по предрыночному положению цены, решение в первые минуты сессии.

Правило пришло из переданной трейдером заметки как «базовый уровень», не как
кандидат: сторона по положению цены в 30-свечном окне. Здесь оно исполняется
как сделка по минутным OHLC, с расходами, стопом и полным профилем минут.
Поле не читается: нужны только рыночные минуты и календарь. Объёма нет.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'data/research/S-07'
MIN = 60_000_000_000
POINT = {'ES': 50., 'NQ': 20., 'YM': 5.}
TICK = {'ES': .25, 'NQ': .25, 'YM': 1.}
COST = {'ES': 30., 'NQ': 15., 'YM': 15.}
WINDOW = 30                       # окно положения цены и единицы свечи, из заметки
MINUTES = list(range(0, 15))      # профиль минут решения, не перебор победителя
RISK_NOTE = 'стопы расширены после того, как стоп в 2 свечи срезал правый хвост'
HORIZONS = [30, 60, 120]
STOPS = [0., 2., 4., 6., 10.]     # без стопа и внешние стопы в свечах
EPOCHS = [('2006-2012', '2006', '2012'), ('2013-2019', '2013', '2019'), ('2020-2026', '2020', '2026')]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def market(ins):
    return {k: np.load(ROOT / 'data/market' / ins / (k + '.npy'))
            for k in ['close_ts_utc_ns', 'open', 'high', 'low', 'close']}


def unit(m):
    prev = np.r_[np.nan, m['close'][:-1]]
    tr = np.maximum(m['high'], prev) - np.minimum(m['low'], prev)
    ok = np.r_[False, np.diff(m['close_ts_utc_ns']) == MIN]
    return pd.Series(np.where(ok, tr, np.nan)).rolling(WINDOW).median().to_numpy()


@njit(cache=False)
def execute(ts, op, hi, lo, cl, u, decide, session_end, horizon, stop_units, tick, window):
    """status: 1 стоп, 2 время/конец сессии, -1 нет входа, -2 нет префикса."""
    out = np.full((len(decide), 9), np.nan)
    for i in range(len(decide)):
        p = decide[i]
        if p - window + 1 < 0 or ts[p] - ts[p - window + 1] != (window - 1) * MIN:
            out[i, 0] = -2
            continue
        top = hi[p - window + 1]
        bot = lo[p - window + 1]
        for j in range(p - window + 2, p + 1):
            if hi[j] > top:
                top = hi[j]
            if lo[j] < bot:
                bot = lo[j]
        side = 1. if cl[p] > (top + bot) / 2 else -1.
        b = p + 1
        if b >= len(ts) or ts[b] != ts[p] + MIN or ts[b] > session_end[i]:
            out[i, 0] = -1
            continue
        entry = op[b]
        uu = u[p]
        if not np.isfinite(uu) or uu <= 0:
            out[i, 0] = -2
            continue
        stop = entry - side * stop_units * uu if stop_units > 0 else np.nan
        end_ts = min(ts[p] + horizon * MIN, session_end[i])
        exit_price = np.nan
        status = 0.
        j = b
        while j < len(ts) and ts[j] == ts[b] + (j - b) * MIN and ts[j] <= end_ts:
            if stop_units > 0:
                if j > b and (op[j] - stop) * side <= 0:
                    exit_price = op[j]
                    status = 1.
                    break
                hit = lo[j] <= stop if side > 0 else hi[j] >= stop
                if hit:
                    exit_price = stop
                    status = 1.
                    break
            if ts[j] == end_ts:
                exit_price = cl[j]
                status = 2.
                break
            j += 1
        if status == 0.:
            out[i, 0] = -3            # лента оборвалась внутри позиции
            continue
        gross = (exit_price - entry) * side
        out[i] = np.array([status, side, entry, exit_price, gross, j, uu,
                           gross / uu, ts[j] - ts[b]])
    return out


def build(ins):
    m = market(ins)
    u = unit(m)
    cal = pd.read_parquet(ROOT / 'setups/S-04/calendar.parquet')
    ts = m['close_ts_utc_ns']
    rows = []
    open_pos = np.searchsorted(ts, cal.open_ns.to_numpy())
    for minute in MINUTES:
        p = open_pos + minute
        good = (p < len(ts)) & (ts[np.minimum(p, len(ts) - 1)] == cal.open_ns.to_numpy() + minute * MIN)
        idx = np.where(good)[0]
        decide = p[good].astype(np.int64)
        for horizon in HORIZONS:
            for stop in STOPS:
                a = execute(ts, m['open'], m['high'], m['low'], m['close'], u, decide,
                            cal.close_ns.to_numpy()[idx].astype(np.int64), horizon, stop,
                            TICK[ins], WINDOW)
                d = pd.DataFrame(a, columns=['status', 'side', 'entry', 'exit', 'gross_points',
                                             'exit_pos', 'unit', 'gross_candles', 'held_ns'])
                d['date'] = cal.date.to_numpy()[idx]
                d['minute'] = minute
                d['horizon'] = horizon
                d['stop'] = stop
                d['instrument'] = ins
                d['net_dollars'] = d.gross_points * POINT[ins] - COST[ins]
                rows.append(d)
    return pd.concat(rows, ignore_index=True)


def summarise(d):
    out = []
    d = d.loc[d.status > 0].copy()
    d['year'] = d.date.str.slice(0, 4)
    for (ins, minute, horizon, stop), g in d.groupby(['instrument', 'minute', 'horizon', 'stop']):
        row = dict(instrument=ins, minute=minute, horizon=horizon, stop=stop, trades=len(g),
                   net_total=float(g.net_dollars.sum()), net_mean=float(g.net_dollars.mean()),
                   net_median=float(g.net_dollars.median()), win=float((g.net_dollars > 0).mean()),
                   candles_mean=float(g.gross_candles.mean()))
        for epoch, a, b in EPOCHS:
            part = g.loc[(g.year >= a) & (g.year <= b)]
            row[f'mean_{epoch}'] = float(part.net_dollars.mean()) if len(part) else None
            row[f'n_{epoch}'] = int(len(part))
        recent = g.loc[g.year >= '2020']
        daily = recent.groupby('date').net_dollars.sum()
        row['recent_total'] = float(daily.sum())
        row['recent_without_best_5pct'] = float(daily.sum() - daily.nlargest(max(1, int(len(daily) * .05))).sum())
        eq = np.r_[0, np.cumsum(daily.to_numpy())]
        row['recent_max_dd'] = float((np.maximum.accumulate(eq) - eq).max())
        row['recent_years_positive'] = int((recent.groupby('year').net_dollars.sum() > 0).sum())
        row['recent_years'] = int(recent.year.nunique())
        out.append(row)
    return pd.DataFrame(out)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frames = [build(ins) for ins in ['ES', 'NQ', 'YM']]
    d = pd.concat(frames, ignore_index=True)
    d.to_parquet(OUT / 'trades_v1.parquet', index=False)
    s = summarise(d)
    s.to_csv(HERE / 'summary_v1.csv', index=False)
    (HERE / 'run_hashes_v1.json').write_text(json.dumps({'run.py': sha(HERE / 'run.py')}, indent=2), encoding='utf-8')
    pd.set_option('display.width', 250)
    print(s.loc[(s.stop == 0) & (s.horizon == 60)][['instrument', 'minute', 'trades', 'net_mean',
          'net_median', 'win', 'mean_2006-2012', 'mean_2013-2019', 'mean_2020-2026',
          'recent_years_positive', 'recent_years']].to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
