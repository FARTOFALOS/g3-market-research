"""S-18 v0 (line-102 / чтение 3) — день, вышедший за свою обычную область: участие до закрытия.

ОБЪЯВЛЕНО ДО СЧЁТА (2026-09-21). Вторая внешняя гипотеза линии; первая (час прихода Европы)
закрыта собственным правилом остановки, см. LOG.

Откуда вопрос. Источник внешний и назван: Zarattini, Aziz, Barbon, «Beat the Market: An
Effective Intraday Momentum Strategy for S&P500 ETF (SPY)» (SFI 24-97, выборка 2007 — начало
2024). Наше собственное основание: у S-07 всё ожидание сидит примерно в 10 % сессий, которые
не вернулись (097 §5), S-07 выходит в 11:33, а вопрос «как хвостовые сессии добирают результат
дальше» записан открытым (097 §8.4). В переписи 72 конструкций нет ни одной с таким участием:
вход по ненормальному для этого времени дня уходу от открытия, выход по возврату в норму
либо на закрытии. Проверенное рядом и отличающееся: 063 close.py брал только ЗНАК хода дня в
15:00/15:30; 097 не нашёл добавки лучшего закрытия к среднему остатку внутри первых 120 минут
позиции S-07 — это возражение против посылки, оно названо до счёта.

Трейдерская проекция. «Смотрю раз в полчаса. Если цена ушла от открытия дальше, чем обычно
уходит к этому времени дня за последние 14 сессий, — встаю по ходу. Вернулась в обычную
область — выхожу. В закрытие выхожу всегда. Беру день, в котором кто-то вынужден
дорабатывать объём в одну сторону; плачу мелкими возвратами в остальные дни».

Правило — адаптация версии источника с выходом по возврату за ближнюю границу (рис. 5a), без VWAP
(объём исключён словом человека), без размера по волатильности и без подбора. ПОПРАВКА ОПИСАНИЯ
2026-09-21: до сверки с первоисточником здесь стояло «как у источника в базовом виде» — неверно,
базовая модель авторов выходит на противоположной границе; код и числа не менялись:
  норма(k) = среднее по 14 предыдущим сессиям |close(k) / open сессии − 1|, k — минута сессии;
  верх(k) = max(open сегодня, close вчера) * (1 + норма(k));
  низ(k)  = min(open сегодня, close вчера) * (1 − норма(k));
  отметки решения — закрытия минут 10:00, 10:30, …, 15:30; на отметке: выше верха — лонг,
  ниже низа — шорт, внутри — без позиции; смена состояния исполняется по open следующей
  минуты; в закрытие сессии XNYS позиция закрывается по последнему close. Один контракт.
  Расход за оборот NQ $15, ES $30, YM $15; отдельно при удвоенном расходе.
Единица нормы — относительный уход цены от открытия в то же время дня: снята с рынка, не
ATR свечей и не объём. Множитель нормы 1 и окно 14 взяты у источника и не перебираются.

Единица счёта — сессия (сумма сделок дня). Неизвестное: нет open сессии, нет вчерашнего close,
норма по менее чем 10 из 14 сессий (0,7 окна), нет цены на отметке или цены исполнения — сессия НЕИЗВЕСТНА,
не ноль. Территория: ES (ближайший к SPY), NQ (основной), YM; эпохи 2006-2012, 2013-2019,
2020-2026; отдельно 2024-04-01 … 2026-05-04 — после выборки источника.

Что считается до взгляда на деньги: доля сессий со сделкой, сделок на сессию, время в позиции.
Затем: средняя на сессию после расходов, m, t, годы, просадка, худший день, итог без лучших
5 % дней, распределение сделок (доля прибыльных, средний выигрыш и проигрыш), вклад времени
10:00-11:30 против 11:30-16:00, связь с дневным результатом S-07 v1r1.

Что опровергает. (1) NQ 2020-2026: средняя на сессию после расходов <= 0 либо t < 2 —
исторической экономики нет. (2) Знак на братьях и в ранних эпохах В ОТНОСИТЕЛЬНЫХ единицах
(б.п. на сессию до расходов) расходится — это не место рынка. (3) Всё набирается до 11:30 и
дневная корреляция с S-07 выше 0,6 — это S-07 в другой одежде, не ещё одна возможность.
(4) Ошибка самого вопроса: норма задаёт ширину, а не сторону (097: путь несёт знание о размере
хода, не о стороне) — тогда вход по выходу за норму даёт ноль до расходов при любой частоте.
После результата правило не трогаю; соседи (окно, множитель, шаг отметок) — только как
объявленная заранее диагностика чувствительности, победитель по ним не выбирается.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
MIN = 60_000_000_000
POINT = {'ES': 50., 'NQ': 20., 'YM': 5.}
COST = {'ES': 30., 'NQ': 15., 'YM': 15.}
KMAX = 390


def sessions(ins, cal=None):
    d = ROOT / 'data/market' / ins
    ts = np.load(d / 'close_ts_utc_ns.npy')
    o = np.load(d / 'open.npy')
    c = np.load(d / 'close.npy')
    if cal is None:
        cal = pd.read_parquet(ROOT / 'setups/S-04/calendar.parquet')
    S = len(cal)
    O = np.full((S, KMAX + 1), np.nan)
    C = np.full((S, KMAX + 1), np.nan)
    K = ((cal.close_ns - cal.open_ns) // MIN).to_numpy().astype(int)
    lo = np.searchsorted(ts, cal.open_ns.to_numpy(), side='right')
    hi = np.searchsorted(ts, cal.close_ns.to_numpy(), side='right')
    for s in range(S):
        idx = np.arange(lo[s], hi[s])
        k = ((ts[idx] - cal.open_ns.iloc[s]) // MIN).astype(int)
        good = (k >= 1) & (k <= min(K[s], KMAX))
        O[s, k[good]] = o[idx[good]]
        C[s, k[good]] = c[idx[good]]
    return cal, K, O, C


def run(ins, look=14, mult=1.0, step=30, cost_mult=1.0, first=30, cal=None, gap=True):
    cal, K, O, C = sessions(ins, cal)
    S = len(cal)
    op = O[:, 1]
    cl = np.array([C[s, min(K[s], KMAX)] for s in range(S)])
    move = np.abs(C / op[:, None] - 1.0)
    rows, trades = [], []
    for s in range(look, S):
        rec = {'date': cal.date.iloc[s], 'status': 'ok'}
        prev = cl[s - 1] if gap else op[s]
        if not np.isfinite(op[s]) or not np.isfinite(prev) or not np.isfinite(cl[s]):
            rec['status'] = 'unknown:нет open, вчерашнего close или close сессии'
            rows.append(rec)
            continue
        hist = move[s - look:s]
        cnt = np.isfinite(hist).sum(axis=0)
        with np.errstate(invalid='ignore'):
            norm = np.where(cnt >= int(np.ceil(0.7 * look)), np.nanmean(hist, axis=0), np.nan)
        up = max(op[s], prev) * (1 + mult * norm)
        dn = min(op[s], prev) * (1 - mult * norm)
        marks = [k for k in range(first, KMAX, step) if k + 1 <= K[s] - 1]
        pos, entry, ek = 0, np.nan, 0
        usd, gross_d, n_tr, held, bad = 0.0, 0.0, 0, 0, False
        for k in marks:
            p = C[s, k]
            if not np.isfinite(p) or not np.isfinite(up[k]):
                bad = True
                break
            want = 1 if p > up[k] else (-1 if p < dn[k] else 0)
            if want != pos:
                px = O[s, k + 1]
                if not np.isfinite(px):
                    bad = True
                    break
                if pos != 0:
                    g = pos * (px - entry) * POINT[ins]
                    trades.append({'date': rec['date'], 'side': pos, 'k_in': ek, 'k_out': k + 1,
                                   'gross': g, 'net': g - COST[ins] * cost_mult})
                    usd += g - COST[ins] * cost_mult
                    gross_d += g
                    held += (k + 1) - ek
                    n_tr += 1
                pos, entry, ek = want, px, k + 1
        if bad:
            rec['status'] = 'unknown:нет цены на отметке, нормы или цены исполнения'
            rows.append(rec)
            continue
        if pos != 0:
            g = pos * (cl[s] - entry) * POINT[ins]
            trades.append({'date': rec['date'], 'side': pos, 'k_in': ek, 'k_out': K[s],
                           'gross': g, 'net': g - COST[ins] * cost_mult})
            usd += g - COST[ins] * cost_mult
            gross_d += g
            held += K[s] - ek
            n_tr += 1
        rec.update({'usd': usd, 'gross': gross_d, 'n_tr': n_tr, 'held': held, 'open': op[s]})
        rows.append(rec)
    D = pd.DataFrame(rows)
    T = pd.DataFrame(trades)
    D['date'] = pd.to_datetime(D.date)
    D['year'] = D.date.dt.year
    if len(T):
        T['date'] = pd.to_datetime(T.date)
        # вклад по времени: сделки, открытые до 11:30 (k<=120), и позже
        T['early'] = T.k_in <= 120
    return D, T


def epoch(y):
    return np.where(y <= 2012, '2006-2012', np.where(y <= 2019, '2013-2019', '2020-2026'))


def summarize(D, T, ins, tag):
    ok = D[D.status == 'ok'].copy()
    ok['epoch'] = epoch(ok.year.values)
    parts = [(e, g) for e, g in ok.groupby('epoch')]
    parts.append(('после источника 2024-04+', ok[ok.date >= '2024-04-01']))
    out = []
    for e, g in parts:
        x = g.usd
        eq = x.cumsum()
        dd = (eq.cummax() - eq).max()
        thr = x.quantile(0.95)
        yr = g.groupby('year').usd.sum()
        bp = g.gross / (g.open * POINT[ins]) * 1e4
        tt = T[T.date.isin(g.date)] if len(T) else T
        out.append({'ins': ins, 'версия': tag, 'эпоха': e, 'сессий': len(g),
                    'со сделкой': round((g.n_tr > 0).mean(), 3),
                    'сделок/сессию': round(g.n_tr.mean(), 2),
                    'минут в позиции': round(g.held.mean(), 0),
                    'средняя $': round(x.mean(), 1), 'm': round(x.mean() / x.std(), 3),
                    't': round(x.mean() / x.std() * np.sqrt(len(x)), 2),
                    'валовое б.п./сессию': round(bp.mean(), 2), 't валового б.п.': round(bp.mean() / bp.std() * np.sqrt(len(bp)), 2),
                    'итог $': int(x.sum()), 'лет+': f'{int((yr > 0).sum())}/{len(yr)}',
                    'просадка $': int(dd), 'худший день $': int(x.min()),
                    'без лучших 5% $': int(x[x < thr].sum()),
                    'доля приб. сделок': round((tt.net > 0).mean(), 3) if len(tt) else np.nan,
                    'ср. выигрыш $': round(tt.net[tt.net > 0].mean(), 0) if len(tt) else np.nan,
                    'ср. проигрыш $': round(tt.net[tt.net <= 0].mean(), 0) if len(tt) else np.nan,
                    'net откр. до 11:30 $': int(tt.net[tt.early].sum()) if len(tt) else 0,
                    'net откр. позже $': int(tt.net[~tt.early].sum()) if len(tt) else 0})
    return pd.DataFrame(out)


def main():
    pd.set_option('display.width', 300, 'display.max_columns', 50)
    res = []
    for ins in ('NQ', 'ES', 'YM'):
        D, T = run(ins)
        D.to_csv(HERE / f'band_daily_{ins}.csv', index=False)
        T.to_csv(HERE / f'band_trades_{ins}.csv', index=False)
        unk = D[D.status != 'ok'].status.value_counts()
        print(f'\n##### {ins}: сессий {len(D)}, неизвестных {int((D.status != "ok").sum())}')
        print(unk.to_string())
        r = summarize(D, T, ins, 'v0 как у источника')
        res.append(r)
        print(r.T.to_string())
        if ins == 'NQ':
            yr = D[D.status == 'ok'].groupby('year').usd.agg(['sum', 'mean', 'size']).round(0)
            print('\nNQ по годам:\n', yr.T.to_string())
            D2, T2 = run(ins, cost_mult=2.0)
            r2 = summarize(D2, T2, ins, 'расход x2')
            res.append(r2)
            print('\nNQ при удвоенном расходе:\n', r2[['эпоха', 'средняя $', 't', 'итог $', 'лет+']].to_string(index=False))
            # связь с S-07 v1r1
            h = pd.read_csv(ROOT / 'setups/S-17/engine_history_NQ.csv', parse_dates=['date'])
            j = D[D.status == 'ok'].merge(h[['date', 's07_usd', 'd30_usd']], on='date', how='inner').dropna(subset=['s07_usd'])
            for lo_, hi_ in (('2006', '2013'), ('2013', '2020'), ('2020', '2027')):
                jj = j[(j.date >= lo_) & (j.date < hi_)]
                print(f'связь дневных $ с S-07 v1r1 {lo_}-{hi_}: r = {jj.usd.corr(jj.s07_usd):.3f}, общих дней {len(jj)}; '
                      f'оба минус {((jj.usd < 0) & (jj.s07_usd < 0)).mean():.3f}; '
                      f'сумма вместе {int((jj.usd + jj.s07_usd).sum())}, S-07 одна {int(jj.s07_usd.sum())}')
    pd.concat(res).to_csv(HERE / 'band_summary.csv', index=False)


if __name__ == '__main__':
    main()
