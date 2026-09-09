"""S-09 — две двери дня: полное правило, исполнение и все проверки в одном файле.

Правило v1 целиком:
  NQ, основная сессия XNYS, America/New_York. Два решения в день — в минуты,
  чьи метки закрытия равны 09:33 и 13:38. Требуется 30 подряд идущих минут
  ленты до решающей минуты включительно.
  Сторона: покупка, если закрытие решающей минуты выше середины диапазона
  последних 30 свечей (макс high и мин low этих 30 минут), иначе продажа.
  Вход по open следующей минуты. Выход через 120 минут после входа либо на
  последней минуте основной сессии, что раньше.
  Защитный предел: 12.88 свечи для утренней двери, 6.33 для дневной, где свеча
  это медиана истинного диапазона по 30 контактным минутам к решающей минуте.
  Числа предела сняты с трафарета: это p90 хода против входа у прибыльных дней,
  то есть предел режет 10% выигрышей. Это ограничение убытка, а не ошибка идеи.
  Один контракт на дверь, расход $15 за оборот. Пропуск минут в ленте означает
  неизвестное: сделка не открывается и не заменяется соседней.

Территория поиска 2020-01-01 … 2025-10-31. Кусок 2025-11-01 … 2026-05-04 при
поиске не открывался; его независимость от прежних линий репозитория не
установлена, это ограничение собственного доступа, а не чистый holdout.
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
MIN = 60_000_000_000
W, HOLD = 30, 120
POINT = {'ES': 50., 'NQ': 20., 'YM': 5.}
COST = {'ES': 30., 'NQ': 15., 'YM': 15.}
DOORS = {'utro_0933': (573, 12.88), 'den_1338': (818, 6.33)}
S0 = pd.Timestamp('2020-01-01', tz='UTC').value
S1 = pd.Timestamp('2025-11-01', tz='UTC').value
H1 = pd.Timestamp('2026-05-05', tz='UTC').value


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def market(ins):
    m = {k: np.load(ROOT / 'data/market' / ins / (k + '.npy'))
         for k in ('close_ts_utc_ns', 'open', 'high', 'low', 'close')}
    ts = m['close_ts_utc_ns']
    n = len(ts)
    step = np.r_[False, np.diff(ts) == MIN]
    idx = np.arange(n)
    m['run'] = (idx - np.maximum.accumulate(np.where(~step, idx, 0)) + 1).astype(np.int32)
    prev = np.r_[np.nan, m['close'][:-1]]
    tr = np.maximum(m['high'], prev) - np.minimum(m['low'], prev)
    m['unit'] = pd.Series(np.where(step, tr, np.nan)).rolling(W).median().to_numpy()
    t = pd.DatetimeIndex(pd.to_datetime(ts, unit='ns', utc=True)).tz_convert('America/New_York')
    m['mod'] = (t.hour.values * 60 + t.minute.values).astype(np.int32)
    m['day'] = t.normalize()
    m['year'] = t.year.values
    return m


def session_end(m):
    cal = pd.read_parquet(ROOT / 'setups/S-04/calendar.parquet')
    ts = m['close_ts_utc_ns']
    o = np.searchsorted(ts, cal.open_ns.to_numpy())
    c = np.searchsorted(ts, cal.close_ns.to_numpy())
    end = np.full(len(ts), -1, np.int64)
    for s, e in zip(o, c):
        if s < len(ts) and e < len(ts) and e > s:
            end[s:e + 1] = ts[e]
    return end


@njit(cache=True)
def execute(ts, op, hi, lo, cl, dec, side, stop_pts, cap, sess_end):
    """ishod: 1 predel, 2 vremya, 3 konets sessii."""
    n = len(ts)
    out = np.full((len(dec), 5), np.nan)
    for i in range(len(dec)):
        p = dec[i]
        b = p + 1
        if b >= n or ts[b] != ts[p] + MIN:
            continue
        e = op[b]
        s = side[i]
        stp = e - s * stop_pts[i]
        j = b
        last = b
        res = 0.
        px = np.nan
        while j < n and ts[j] == ts[b] + (j - b) * MIN:
            if ts[j] > sess_end[i]:
                res = 3.
                px = cl[j - 1]
                last = j - 1
                break
            if j - b + 1 > cap:
                res = 2.
                px = cl[j - 1]
                last = j - 1
                break
            hit = (s > 0 and lo[j] <= stp) or (s < 0 and hi[j] >= stp)
            if hit:
                px = stp
                if s > 0 and op[j] < stp:
                    px = op[j]
                if s < 0 and op[j] > stp:
                    px = op[j]
                res = 1.
                last = j
                break
            last = j
            j += 1
        if res == 0.:
            res = 3.
            px = cl[last]
        out[i, 0] = e
        out[i, 1] = px
        out[i, 2] = res
        out[i, 3] = last - b + 1
        out[i, 4] = s
    return out


def trades(ins='NQ', cost=None, lo=S0, hi=H1):
    m = market(ins)
    end = session_end(m)
    ts = m['close_ts_utc_ns']
    hi_roll = pd.Series(m['high']).rolling(W).max().to_numpy()
    lo_roll = pd.Series(m['low']).rolling(W).min().to_numpy()
    side_all = np.where(m['close'] > (hi_roll + lo_roll) / 2, 1., -1.)
    cost = COST[ins] if cost is None else cost
    frames = []
    for name, (minute, lim) in DOORS.items():
        p = np.where((m['mod'] == minute) & (m['run'] >= W) & np.isfinite(m['unit'])
                     & (m['unit'] > 0) & (ts >= lo) & (ts < hi))[0]
        se = np.where(end[p] > 0, end[p], np.iinfo(np.int64).max)
        o = execute(ts, m['open'], m['high'], m['low'], m['close'], p.astype(np.int64),
                    side_all[p].astype(np.float64), (lim * m['unit'][p]).astype(np.float64),
                    HOLD, se.astype(np.int64))
        ok = np.isfinite(o[:, 0])
        q = p[ok]
        gross = o[ok, 4] * (o[ok, 1] - o[ok, 0]) * POINT[ins]
        frames.append(pd.DataFrame({
            'door': name, 'ts': ts[q], 'day': m['day'][q], 'year': m['year'][q],
            'side': o[ok, 4], 'entry': o[ok, 0], 'exit': o[ok, 1],
            'outcome': o[ok, 2].astype(int), 'minutes': o[ok, 3].astype(int),
            'candle': m['unit'][q], 'gross': gross, 'net': gross - cost,
            'holdout': ts[q] >= S1}))
    return pd.concat(frames, ignore_index=True)


def block(d):
    g = d.groupby('day')['net'].sum().sort_index()
    cs = g.cumsum().to_numpy()
    y = d.groupby('year')['net'].sum()
    return {'sdelok': len(d), 'dney': len(g), 'v_den': round(len(d) / len(g), 2),
            'srednyaya': round(d['net'].mean(), 1), 'chistoe': int(d['net'].sum()),
            'valovoe': int(d['gross'].sum()),
            'let_plus': f"{int((y > 0).sum())}/{len(y)}",
            'prosadka': int((np.maximum.accumulate(cs) - cs).max()),
            'hudshiy_den': int(g.min()), 'hudshaya_sdelka': int(d['net'].min()),
            'dney_v_plyuse_pct': round(100 * (g > 0).mean()),
            'zakryto_predelom_pct': round(100 * (d['outcome'] == 1).mean()),
            'mediana_minut': int(d['minutes'].median())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', nargs='?', default='run', choices=['run', 'checks'])
    a = ap.parse_args()
    d = trades('NQ')
    d.to_csv(HERE / 'trades.csv', index=False)
    s, h = d[~d['holdout']], d[d['holdout']]
    res = {'rule': 'S-09 v1', 'territory': 'NQ, XNYS, 2020-01-01..2025-10-31',
           'poisk': block(s), 'otlozhennoe': block(h),
           'po_dveryam': {k: {'poisk': block(s[s['door'] == k]),
                              'otlozhennoe': block(h[h['door'] == k])} for k in DOORS},
           'po_godam': {str(k): int(v) for k, v in s.groupby('year')['net'].sum().items()},
           'cost_30': block(trades('NQ', cost=30.).pipe(lambda t: t[~t['holdout']])),
           'ES': block(trades('ES').pipe(lambda t: t[~t['holdout']])),
           'YM': block(trades('YM').pipe(lambda t: t[~t['holdout']])),
           'input_hashes': {f.name: sha(f) for f in sorted((ROOT / 'data/market/NQ').glob('*.npy'))
                            if f.name in ('close_ts_utc_ns.npy', 'open.npy', 'high.npy',
                                          'low.npy', 'close.npy')},
           'code_sha': sha(Path(__file__))}
    if a.cmd == 'checks':
        m = market('NQ')
        rng = np.random.default_rng(5)
        hi_roll = pd.Series(m['high']).rolling(W).max().to_numpy()
        lo_roll = pd.Series(m['low']).rolling(W).min().to_numpy()
        side_all = np.where(m['close'] > (hi_roll + lo_roll) / 2, 1., -1.)
        idx = np.where(np.isin(m['mod'], [573, 818]) & (m['run'] >= W)
                       & (m['close_ts_utc_ns'] >= S0) & (m['close_ts_utc_ns'] < S1))[0]
        t = rng.choice(idx, 500, replace=False)
        bad = 0
        for p in t:
            side_prefix = 1. if m['close'][p] > (m['high'][p - W + 1:p + 1].max()
                                                 + m['low'][p - W + 1:p + 1].min()) / 2 else -1.
            if side_prefix != side_all[p]:
                bad += 1
        res['prefix_check'] = {'resheniy': 500, 'rashozhdeniy': bad}
        daily = s.groupby('day')['net'].sum().sort_index()
        mm = daily.groupby(pd.PeriodIndex(daily.index, freq='M')).sum().to_numpy()
        r = np.random.default_rng(17)
        boot = np.array([mm[r.integers(0, len(mm), len(mm))].sum() for _ in range(10000)])
        res['interval'] = {'metod': 'mesyachnoe blochnoe peresemplirovanie, 10000 povtorov',
                           'p2.5': int(np.percentile(boot, 2.5)),
                           'p97.5': int(np.percentile(boot, 97.5)),
                           'dolya_nizhe_nulya_pct': round(100 * float((boot < 0).mean()), 2)}
        g = daily.sort_values()
        res['bez_pyati_luchshih_dney'] = int(g.iloc[:-5].sum())
        res['bez_2022'] = int(daily[daily.index.year != 2022].sum())
    (HERE / 'result.json').write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
    s.groupby('day')['net'].sum().to_csv(HERE / 'daily.csv')
    show = ['poisk', 'otlozhennoe'] + (['prefix_check', 'interval', 'bez_pyati_luchshih_dney',
                                        'bez_2022'] if a.cmd == 'checks' else [])
    print(json.dumps({k: res[k] for k in show}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
