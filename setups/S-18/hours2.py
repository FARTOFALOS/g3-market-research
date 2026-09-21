"""line-102 / чтение 2 — тот же час, сторона против вчерашнего давления в закрытие.

ОБЪЯВЛЕНО ДО СЧЁТА (2026-09-21), после результата чтения 1.

Состояние выбора, честно. Чтение 1 сработало по собственному правилу остановки для слепого
лонга: клетка 02:00-03:00 ET воспроизводится на нашей ленте до 2019 года на трёх индексах
(t 2,3…4,5; 1,1…1,6 б.п. за час) и после выборки статьи (2021+) исчезает (NQ +0,28 б.п.,
t 0,6; ES и YM минус); в долларах одного контракта NQ валовое ни в одной эпохе не было выше
расхода, кроме 2020 года. Это чтение — ДОПОЛНИТЕЛЬНЫЙ поиск после остановки, не продолжение
выжившей клетки. Оправдание одно: у источника главным названа не безусловная надбавка, а
знаковая — разворот вчерашнего навеса заявок; безусловная есть её асимметричный остаток.
Нулевая безусловная надбавка не исключает знаковую (после падения плюс, после роста минус).

Трейдерская проекция. «Вчера в последний час продавали — в 02:00 покупаю на час; покупали —
продаю на час. Беру возврат уступки в цене, которую дали за срочность в закрытие».

Инженерная проекция. Единица — один час одной даты. Сторона = минус знак условия по
ПРЕДЫДУЩЕЙ закрытой сессии XNYS (понедельник смотрит на пятницу). Условий два, оба названы
до счёта и больше не добавляются:
  A — ход последнего часа сессии (закрытие минус цена за 60 минут до закрытия);
  B — ход всей основной сессии (закрытие минус цена в момент открытия).
Исход — знаковый ход close-to-close за час, б.п., пункты, $. Заранее названная клетка — час 2.
Рельеф целиком: все часы от 16:00 предыдущего дня до 16:00, три индекса, три эпохи и 2021+.
Отдельно четыре клетки «лонг после падения / лонг после роста» по A и B — источник
предсказывает асимметрию (после падения больше). Диагностика дозы: трети |A| — источник
предсказывает рост разворота с размером вчерашнего давления; по ней ничего не выбирается.

Остановка. Если в 2021+ знаковая средняя в клетке часа 2 не выше нуля с t >= 2 хотя бы на
NQ и не одного знака на братьях — линия часа Европы закрыта, новых условий не ищу.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
MIN = 60_000_000_000
POINT = {'ES': 50., 'NQ': 20., 'YM': 5.}
TOL = 10 * MIN


def px_at(ts, c, marks_ns):
    pos = np.searchsorted(ts, marks_ns, side='right') - 1
    ok = (pos >= 0) & (marks_ns - ts[np.maximum(pos, 0)] <= TOL)
    return np.where(ok, c[np.maximum(pos, 0)], np.nan)


def build(ins):
    d = ROOT / 'data/market' / ins
    ts = np.load(d / 'close_ts_utc_ns.npy')
    c = np.load(d / 'close.npy')
    cal = pd.read_parquet(ROOT / 'setups/S-04/calendar.parquet')
    o_ns, c_ns = cal.open_ns.to_numpy(), cal.close_ns.to_numpy()
    p_open, p_close, p_h = px_at(ts, c, o_ns), px_at(ts, c, c_ns), px_at(ts, c, c_ns - 60 * MIN)
    A = (p_close - p_h) / p_h * 1e4
    B = (p_close - p_open) / p_open * 1e4

    idx = pd.date_range('2006-01-05', '2026-05-05', freq='h', tz='America/New_York')
    m_ns = idx.tz_convert('UTC').as_unit('ns').asi8
    px = px_at(ts, c, m_ns)
    step_ok = np.diff(m_ns) == 60 * MIN
    pts = np.where(step_ok, px[1:] - px[:-1], np.nan)
    bp = pts / px[:-1] * 1e4
    start = m_ns[:-1]
    k = np.searchsorted(c_ns, start, side='right') - 1       # последняя сессия, закрытая к началу часа
    ok = k >= 0
    # условие годно, только если та сессия закрылась не раньше чем за 4 суток (выходные, праздники)
    fresh = ok & (start - c_ns[np.maximum(k, 0)] <= 4 * 1440 * MIN)
    a = np.where(fresh, A[np.maximum(k, 0)], np.nan)
    b = np.where(fresh, B[np.maximum(k, 0)], np.nan)
    df = pd.DataFrame({'hour': idx.hour[:-1], 'year': idx.year[:-1], 'pts': pts, 'bp': bp, 'A': a, 'B': b})
    df = df[df.hour != 17].dropna(subset=['bp'])
    df['usd'] = df.pts * POINT[ins]
    df['ins'] = ins
    df['epoch'] = np.where(df.year <= 2012, '2006-2012', np.where(df.year <= 2019, '2013-2019', '2020-2026'))
    return df


def stat(x, usd=None):
    x = x.dropna()
    n = len(x)
    if n < 100:
        return {'n': n}
    r = {'n': n, 'bp': x.mean(), 'm': x.mean() / x.std(), 't': x.mean() / x.std() * np.sqrt(n)}
    if usd is not None:
        r['usd'] = usd.dropna().mean()
    return r


def main():
    pd.set_option('display.width', 250, 'display.max_rows', 500)
    allrows, cells, dose = [], [], []
    for ins in ('NQ', 'ES', 'YM'):
        df = build(ins)
        if ins == 'NQ':
            df.to_parquet(ROOT / 'work/line-102/hours2_NQ.parquet')
        parts = [(ep, g) for ep, g in df.groupby('epoch')] + [('2021+', df[df.year >= 2021])]
        for ep, g in parts:
            for cond in ('A', 'B'):
                s = -np.sign(g[cond])
                for h, gh in g.assign(s=s).groupby('hour'):
                    gh = gh[gh.s != 0].dropna(subset=['s'])
                    r = stat(gh.s * gh.bp, gh.s * gh.usd)
                    yr = (gh.s * gh.bp).groupby(gh.year).mean()
                    allrows.append({'ins': ins, 'epoch': ep, 'cond': cond, 'hour': h, **r,
                                    'years_pos': f'{int((yr > 0).sum())}/{len(yr)}'})
                g2 = g[g.hour == 2]
                for name, sub in (('после падения', g2[g2[cond] < 0]), ('после роста', g2[g2[cond] > 0])):
                    cells.append({'ins': ins, 'epoch': ep, 'cond': cond, 'вчера': name, **stat(sub.bp, sub.usd)})
            g2 = g[(g.hour == 2)].dropna(subset=['A'])
            if len(g2) > 300:
                q = pd.qcut(g2.A.abs(), 3, labels=['малое', 'среднее', 'большое'])
                for lab, sub in g2.groupby(q, observed=True):
                    dose.append({'ins': ins, 'epoch': ep, '|A|': lab,
                                 **stat(-np.sign(sub.A) * sub.bp, -np.sign(sub.A) * sub.usd)})
    R = pd.DataFrame(allrows)
    R.to_csv(HERE / 'hours2_relief.csv', index=False)
    print('=== заранее названная клетка: час 02:00-03:00 ET, сторона против вчерашнего ===')
    print(R[R.hour == 2].round(3).to_string(index=False))
    print('\n=== лонг в час 2 после падения / после роста ===')
    print(pd.DataFrame(cells).round(3).to_string(index=False))
    print('\n=== доза: трети |A|, знаковый ход часа 2 ===')
    print(pd.DataFrame(dose).round(3).to_string(index=False))
    order = [16, 18, 19, 20, 21, 22, 23] + list(range(0, 16))
    for ep in ('2013-2019', '2020-2026', '2021+'):
        s = R[(R.ins == 'NQ') & (R.cond == 'A') & (R.epoch == ep)].set_index('hour')
        s = s.loc[[h for h in order if h in s.index]]
        print(f'\n=== рельеф NQ, условие A, {ep}: знаковый ход по часам ===')
        print(s[['n', 'bp', 'm', 't', 'usd', 'years_pos']].round(3).to_string())


if __name__ == '__main__':
    main()
