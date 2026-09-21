"""line-102 / чтение 4 — то же правило v0 в других окнах суток (приём 097: настоящее место
даёт ноль в других местах; общее свойство ленты — нет).

ОБЪЯВЛЕНО ДО СЧЁТА (2026-09-21), после результата v0 на основной сессии.

Правило band.py без изменений (окно 14, множитель 1, отметки каждые 30 минут, выход по возврату
в норму или в конец окна), но якорь и конец окна другие и члена «вчерашний close» нет (верх и низ
строятся от цены якоря):
  L — утро Лондона: якорь 03:00 ET, конец 09:25 ET (до открытия акций США), 385 минут;
  A — Азия: якорь 20:00 ET, конец 02:25 ET, 385 минут. Только плацебо: входы в Азию исключены
      рамкой трейдера, кандидатом это окно не станет при любом результате.
Эстиманд — валовое в б.п. на окно и t по эпохам на трёх индексах; для NQ ещё $ после расхода.
Ожидания названы заранее: если явление принадлежит основной сессии США (кто-то дорабатывает
объём к закрытию акций) — в L и A ноль; если это общее свойство ухода от якоря — плюс и там.
Плюс в L стал бы отдельной догадкой со своей линией; здесь по нему ничего не выбирается.
"""
import numpy as np
import pandas as pd

import run as band
from run import ROOT, POINT

pd.set_option('display.width', 250)


def windows(h0, m0, minutes):
    cal = pd.read_parquet(ROOT / 'setups/S-04/calendar.parquet')
    days = pd.to_datetime(cal.date)
    start = pd.DatetimeIndex([pd.Timestamp(d.year, d.month, d.day, h0, m0) for d in days]).tz_localize(
        'America/New_York', ambiguous='NaT', nonexistent='NaT')
    if h0 >= 18:                                    # вечер относится к предыдущему календарному дню
        start = start - pd.Timedelta(days=1)
    o = start.tz_convert('UTC').as_unit('ns').asi8
    out = pd.DataFrame({'date': cal.date, 'open_ns': o, 'close_ns': o + minutes * band.MIN})
    return out[start.notna()].reset_index(drop=True)


def main():
    rows = []
    for tag, (h0, m0) in (('L 03:00-09:25 ET', (3, 0)), ('A 20:00-02:25 ET', (20, 0))):
        cal = windows(h0, m0, 385)
        for ins in ('NQ', 'ES', 'YM'):
            D, T = band.run(ins, cal=cal, gap=False)
            ok = D[D.status == 'ok'].copy()
            ok['gbp'] = ok.gross / (ok.open * POINT[ins]) * 1e4
            ok['epoch'] = band.epoch(ok.year.values)
            for ep, g in list(ok.groupby('epoch')) + [('2021+', ok[ok.year >= 2021])]:
                rows.append({'окно': tag, 'ins': ins, 'эпоха': ep, 'окон': len(g),
                             'неизвестных': int((D.status != 'ok').sum()),
                             'сделок/окно': round(g.n_tr.mean(), 2),
                             'валовое б.п.': round(g.gbp.mean(), 2),
                             't': round(g.gbp.mean() / g.gbp.std() * np.sqrt(len(g)), 2),
                             '$ после расхода': round(g.usd.mean(), 1)})
    R = pd.DataFrame(rows)
    R.to_csv(band.HERE / 'band_elsewhere.csv', index=False)
    print(R.to_string(index=False))


if __name__ == '__main__':
    main()
