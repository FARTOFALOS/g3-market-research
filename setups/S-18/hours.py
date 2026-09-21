"""line-102 / чтение 1 — часовой рельеф слепого удержания по суткам Globex.

ОБЪЯВЛЕНО ДО СЧЁТА (2026-09-21).

Откуда вопрос. Источник внешний и назван: Boyarchenko, Larsen, Whelan, «The Overnight
Drift» (RFS 2023, выборка 1998-2020): фьючерсы на индексы акций США получают крупный
положительный доход в час прихода Европы, 02:00-03:00 ET; после распродаж в конце
предыдущей сессии он больше, после роста — слабее (плата за принятие навеса заявок,
а не прогноз по свечам). Наш собственный пробел: все часовые проходы G3 (061 clock по
1440 минутам, 063, S-09 по 389 минутам, 097 auction_class) задавали сторону положением
цены в 30-свечном окне; слепое удержание появлялось только как контроль утра. Таблицы
безусловного хода по часам суток в репозитории нет.

Трейдерская проекция. «В 02:00 по Нью-Йорку покупаю и держу час, в 03:00 выхожу. Беру не
направление по свечам, а плату за удержание в час, когда ночной навес разгружается».

Инженерная проекция. Единица — один час одной торговой даты. Окно h: от последнего
закрытия с меткой <= h:00 до последнего закрытия с меткой <= (h+1):00 по America/New_York;
обе опоры обязаны лежать не дальше 10 минут от своей отметки, иначе час НЕИЗВЕСТЕН (не ноль).
Исход — ход close-to-close в базисных пунктах цены (без ATR и без объёма), в пунктах и в $.
Рельеф объявлен целиком: 23 часа суток Globex (18:00 -> 17:00 ET), три инструмента, три
эпохи (2006-2012, 2013-2019, 2020-2026), плюс отдельно 2021+ как годы после выборки статьи.
Заранее названная клетка: 02:00-03:00 ET, лонг. Остальные 22 часа — её плацебо.
Мерка: m = среднее / разброс на час; t; доля положительных лет.

Что опровергает постановку. (1) Клетка 02-03 в 2020-2026 не выделяется из остальных
ночных часов по m и не держит знак по годам -> опубликованного явления на нашей ленте
нет в размере, который видит прибор. (2) Все ночные часы дают одинаковое m и 2022 год
отрицателен -> это бета, а не час. (3) Ошибка самого вопроса: метки минут (close-label),
сдвиг лета/зимы между США и Европой (проверяется вторым якорем 08:00 Europe/Berlin),
склейка контрактов (опоры у отметок исключают воскресный разрыв).

Исполнение, расходы и условие по вчерашнему дню — следующим чтением и только если
клетка выдержит это.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
MIN = 60_000_000_000
POINT = {'ES': 50., 'NQ': 20., 'YM': 5.}
TOL = 10 * MIN


def load(ins):
    d = ROOT / 'data/market' / ins
    ts = np.load(d / 'close_ts_utc_ns.npy')
    c = np.load(d / 'close.npy')
    return ts, c


def marks(ts, tz, start='2006-01-05', stop='2026-05-05'):
    """Все отметки целого часа в зоне tz как UTC-ns, вместе с локальным часом и датой."""
    idx = pd.date_range(start, stop, freq='h', tz=tz)
    return idx


def relief(ins, tz):
    ts, c = load(ins)
    idx = marks(ts, tz)
    m_ns = idx.tz_convert('UTC').as_unit('ns').asi8
    pos = np.searchsorted(ts, m_ns, side='right') - 1          # последняя метка <= отметки
    ok = (pos >= 0) & (m_ns - ts[np.maximum(pos, 0)] <= TOL)
    px = np.where(ok, c[np.maximum(pos, 0)], np.nan)
    # час h: от отметки h:00 до отметки (h+1):00 — соседние элементы idx (шаг ровно час в UTC)
    step_ok = np.diff(m_ns) == 60 * MIN
    r_pts = np.where(step_ok, px[1:] - px[:-1], np.nan)
    r_bp = r_pts / px[:-1] * 1e4
    df = pd.DataFrame({'hour': idx.hour[:-1], 'date': idx.normalize()[:-1].tz_localize(None),
                       'year': idx.year[:-1], 'dow': idx.dayofweek[:-1],
                       'pts': r_pts, 'bp': r_bp})
    df['usd'] = df.pts * POINT[ins]
    df['ins'] = ins
    return df


def epoch(y):
    return np.where(y <= 2012, '2006-2012', np.where(y <= 2019, '2013-2019', '2020-2026'))


def table(df, label):
    rows = []
    for (ep, h), g in df.groupby(['epoch', 'hour']):
        x = g.bp.dropna()
        if len(x) < 200:
            continue
        yr = g.dropna(subset=['bp']).groupby('year').bp.mean()
        rows.append({'label': label, 'epoch': ep, 'hour': h, 'n': len(x),
                     'unknown': int(g.bp.isna().sum()),
                     'mean_bp': x.mean(), 'sd_bp': x.std(), 'm': x.mean() / x.std(),
                     't': x.mean() / x.std() * np.sqrt(len(x)),
                     'mean_usd': g.usd.dropna().mean(),
                     'years_pos': f'{int((yr > 0).sum())}/{len(yr)}'})
    return pd.DataFrame(rows)


def main():
    out = []
    for ins in ('NQ', 'ES', 'YM'):
        df = relief(ins, 'America/New_York')
        df = df[(df.hour != 17)]                      # 17:00-18:00 ET — перерыв Globex
        df['epoch'] = epoch(df.year.values)
        t = table(df, f'{ins} ET')
        out.append(t)
        late = df[df.year >= 2021].copy()
        late['epoch'] = '2021+'
        out.append(table(late, f'{ins} ET'))
        if ins == 'NQ':
            df.to_parquet(ROOT / 'work/line-102/hours_NQ_ET.parquet')
    # второй якорь: тот же час в часах Европы (08:00-09:00 Europe/Berlin)
    for ins in ('NQ', 'ES', 'YM'):
        df = relief(ins, 'Europe/Berlin')
        df['epoch'] = epoch(df.year.values)
        t = table(df[df.hour == 8], f'{ins} Berlin')
        out.append(t)
        late = df[(df.year >= 2021) & (df.hour == 8)].copy()
        late['epoch'] = '2021+'
        out.append(table(late, f'{ins} Berlin'))
    res = pd.concat(out, ignore_index=True)
    res.to_csv(HERE / 'hours_relief.csv', index=False)
    pd.set_option('display.width', 250, 'display.max_rows', 500)
    for ins in ('NQ', 'ES', 'YM'):
        for ep in ('2006-2012', '2013-2019', '2020-2026', '2021+'):
            s = res[(res.label == f'{ins} ET') & (res.epoch == ep)].copy()
            order = [18, 19, 20, 21, 22, 23] + list(range(0, 17))
            s = s.set_index('hour').loc[[h for h in order if h in s.hour.values]]
            print(f'\n=== {ins} {ep} (часы ET, слепой лонг) ===')
            print(s[['n', 'unknown', 'mean_bp', 'sd_bp', 'm', 't', 'mean_usd', 'years_pos']].round(3).to_string())
    print('\n=== якорь 08:00-09:00 Europe/Berlin ===')
    print(res[res.label.str.endswith('Berlin')].round(3).to_string())


if __name__ == '__main__':
    main()
