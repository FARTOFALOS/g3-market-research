"""line-102 / чтение 3, проверки замороженной v0 (правило band.py не меняется).

Объявлено до счёта, сразу после первого результата v0 (NQ 2020-2026 +$141,6 на сессию, t 3,17):
  1. Что приходится выдержать: ход против позиции внутри сделки (по минутным low/high между отметками),
     квантиль сделок, худшая сделка, серия убыточных сессий.
  2. Стороны: лонг и шорт отдельно, и тот же набор интервалов удержания вслепую в лонг («бета»).
  3. Соседи правила — только диагностика чувствительности, победитель не выбирается:
     окно 7/14/28, множитель 0,75/1/1,25/1,5, шаг отметок 15/30/60 — по одному измерению за раз.
  4. Вместе с S-07 v1r1: просадка суммы против просадки каждой, общие убыточные дни.
  5. Кусок 2025-11-01…2026-05-04, который S-09 и 063 держали отложенным (для меня он не отложен:
     вся лента посчитана разом — это записывается).
  6. Узнавание не смотрит вперёд: для 300 случайных (сессия, отметка) решение пересчитывается по
     массивам, в которых всё после отметки заменено на NaN.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import run as band
from run import ROOT, HERE, MIN, POINT, COST, KMAX

pd.set_option('display.width', 300, 'display.max_columns', 60)


def hl(ins):
    d = ROOT / 'data/market' / ins
    ts = np.load(d / 'close_ts_utc_ns.npy')
    h = np.load(d / 'high.npy')
    l = np.load(d / 'low.npy')
    cal = pd.read_parquet(ROOT / 'setups/S-04/calendar.parquet')
    S = len(cal)
    H = np.full((S, KMAX + 1), np.nan)
    L = np.full((S, KMAX + 1), np.nan)
    lo = np.searchsorted(ts, cal.open_ns.to_numpy(), side='right')
    hi = np.searchsorted(ts, cal.close_ns.to_numpy(), side='right')
    for s in range(S):
        idx = np.arange(lo[s], hi[s])
        k = ((ts[idx] - cal.open_ns.iloc[s]) // MIN).astype(int)
        good = (k >= 1) & (k <= KMAX)
        H[s, k[good]] = h[idx[good]]
        L[s, k[good]] = l[idx[good]]
    return cal, H, L


def stats(x):
    return {'n': len(x), 'средняя': round(x.mean(), 1), 't': round(x.mean() / x.std() * np.sqrt(len(x)), 2),
            'итог': int(x.sum())}


def main():
    ins = 'NQ'
    D, T = band.run(ins)
    ok = D[D.status == 'ok'].copy()
    late = ok[ok.year >= 2020]
    Tl = T[T.date.dt.year >= 2020].copy()

    # 1. что приходится выдержать
    cal, H, L = hl(ins)
    _, K, O, C = band.sessions(ins)
    sidx = {pd.Timestamp(d): i for i, d in enumerate(pd.to_datetime(cal.date))}
    mae, mfe = [], []
    for r in Tl.itertuples():
        s = sidx[r.date]
        entry = O[s, r.k_in]
        lo_ = np.nanmin(L[s, r.k_in:r.k_out + 1])
        hi_ = np.nanmax(H[s, r.k_in:r.k_out + 1])
        adv = (entry - lo_) if r.side > 0 else (hi_ - entry)
        fav = (hi_ - entry) if r.side > 0 else (entry - lo_)
        mae.append(adv * POINT[ins])
        mfe.append(fav * POINT[ins])
    Tl['mae'], Tl['mfe'], Tl['minutes'] = mae, mfe, Tl.k_out - Tl.k_in
    print('=== 1. NQ 2020-2026, сделки v0: что приходится выдержать ===')
    print('сделок', len(Tl), '; доля прибыльных', round((Tl.net > 0).mean(), 3))
    print('квантили net $:', Tl.net.quantile([.01, .05, .25, .5, .75, .95, .99]).round(0).to_dict())
    print('квантили хода против (MAE) $:', Tl.mae.quantile([.5, .75, .9, .95, .99]).round(0).to_dict(), ' худший', round(Tl.mae.max()))
    w = Tl[Tl.net > 0]
    print('MAE у прибыльных сделок $:', w.mae.quantile([.5, .75, .9, .95]).round(0).to_dict())
    print('минут в сделке:', Tl.minutes.quantile([.25, .5, .75]).to_dict())
    print('худшая сделка', round(Tl.net.min()), '; лучшая', round(Tl.net.max()))
    x = late.usd.to_numpy()
    run_, worst_run = 0, 0
    for v in x:
        run_ = run_ + 1 if v < 0 else 0
        worst_run = max(worst_run, run_)
    print('самая длинная серия убыточных сессий:', worst_run, '; доля сессий в плюсе/минусе/ноль:',
          round((x > 0).mean(), 3), round((x < 0).mean(), 3), round((x == 0).mean(), 3))
    by_in = Tl.groupby('k_in').net.agg(['size', 'mean', 'sum']).round(0)
    by_in.index = [f'{9 + (30 + k - 1) // 60:02d}:{(30 + k - 1) % 60:02d}' for k in by_in.index]
    print('по минуте входа:\n', by_in.T.to_string())

    # 2. стороны и бета
    print('\n=== 2. стороны ===')
    for ep, lo_y, hi_y in (('2006-2012', 2006, 2012), ('2013-2019', 2013, 2019), ('2020-2026', 2020, 2026)):
        tt = T[(T.date.dt.year >= lo_y) & (T.date.dt.year <= hi_y)]
        for name, sub in (('лонг', tt[tt.side > 0]), ('шорт', tt[tt.side < 0])):
            print(ep, name, stats(sub.net))
        blind = tt.gross * tt.side          # тот же интервал, всегда лонг
        print(ep, 'тот же интервал вслепую в лонг (валовое):', stats(blind), '; правило валовое:', stats(tt.gross))

    # 3. соседи
    print('\n=== 3. соседи (NQ 2020-2026, после расходов, на сессию) ===')
    rows = []
    for name, kw in ([(f'окно {v}', {'look': v}) for v in (7, 14, 28)]
                     + [(f'множитель {v}', {'mult': v}) for v in (0.75, 1.0, 1.25, 1.5)]
                     + [(f'шаг {v}', {'step': v, 'first': 30}) for v in (15, 30, 60)]):
        d, t = band.run(ins, **kw)
        g = d[(d.status == 'ok') & (d.year >= 2020)]
        e = d[(d.status == 'ok') & (d.year >= 2013) & (d.year <= 2019)]
        yr = g.groupby('year').usd.sum()
        rows.append({'вариант': name, **stats(g.usd), 'сделок/сессию': round(g.n_tr.mean(), 2),
                     'лет+': f'{int((yr > 0).sum())}/{len(yr)}',
                     'просадка': int((g.usd.cumsum().cummax() - g.usd.cumsum()).max()),
                     '2013-2019 средняя': round(e.usd.mean(), 1)})
    print(pd.DataFrame(rows).to_string(index=False))

    # 4. вместе с S-07
    print('\n=== 4. вместе с S-07 v1r1 (2020-2026) ===')
    h = pd.read_csv(ROOT / 'setups/S-17/engine_history_NQ.csv', parse_dates=['date'])
    j = late.merge(h[['date', 's07_usd']], on='date', how='inner').dropna(subset=['s07_usd'])

    def dd(v):
        e = v.cumsum()
        return int((e.cummax() - e).max())
    print('общих сессий', len(j), '; r =', round(j.usd.corr(j.s07_usd), 3))
    print('просадка: правило', dd(j.usd), '; S-07', dd(j.s07_usd), '; сумма', dd(j.usd + j.s07_usd))
    print('итог: правило', int(j.usd.sum()), '; S-07', int(j.s07_usd.sum()), '; сумма', int((j.usd + j.s07_usd).sum()))
    print('худший день суммы', int((j.usd + j.s07_usd).min()), '; дней, где обе в минусе', round(((j.usd < 0) & (j.s07_usd < 0)).mean(), 3))
    same = np.sign(j.usd) == np.sign(j.s07_usd)
    print('t суммы', round((j.usd + j.s07_usd).mean() / (j.usd + j.s07_usd).std() * np.sqrt(len(j)), 2))

    # 5. кусок S-09/063
    print('\n=== 5. 2025-11-01…2026-05-04 ===')
    piece = ok[(ok.date >= '2025-11-01')]
    print(stats(piece.usd), '; по месяцам:', piece.groupby(piece.date.dt.to_period('M')).usd.sum().round(0).to_dict())

    # 6. узнавание не смотрит вперёд
    rng = np.random.default_rng(7)
    op = O[:, 1]
    cl = np.array([C[s, min(K[s], KMAX)] for s in range(len(K))])
    bad = 0
    for _ in range(300):
        s = int(rng.integers(100, len(K)))
        k = int(rng.choice(np.arange(30, 361, 30)))
        if k + 1 > K[s] - 1 or not np.isfinite(op[s]) or not np.isfinite(cl[s - 1]):
            continue
        C2 = C.copy()
        C2[s, k + 1:] = np.nan
        C2[s + 1:, :] = np.nan
        def want(Cm):
            mv = np.abs(Cm[s - 14:s] / op[s - 14:s, None] - 1.0)
            nm = np.nanmean(mv[:, k])
            p = Cm[s, k]
            up = max(op[s], cl[s - 1]) * (1 + nm)
            dn = min(op[s], cl[s - 1]) * (1 - nm)
            return 1 if p > up else (-1 if p < dn else 0)
        if want(C) != want(C2):
            bad += 1
    print('\n=== 6. расхождений решения при обрезанном будущем:', bad)


if __name__ == '__main__':
    main()
