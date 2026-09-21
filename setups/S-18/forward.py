"""S-18 — накопительное последующее наблюдение замороженной v0.

Решение человека 2026-09-21: v0 сохранить замороженной; начать накопительное наблюдение с доступных последующих
сессий, включая 47 сессий 2026-05-04 … 2026-07-10; точки рассмотрения 125, 250 и 500 сессий — расписание обзоров,
а не три попытки получить положительный вердикт; правило закрытия по минусу на 250 сессиях НЕ принято.

Режимы:
  validate — движок на ленте продолжения (её начало байт-в-байт равно канонической) обязан воспроизвести
             band_daily_NQ.csv на всей истории;
  overlap  — совместимость источника продолжения именно для операторов v0: 34 сессии 2026-03-16 … 2026-05-01,
             локальные бары против баров источника на той же сетке минут;
  ledger   — исполнить v0 на сессиях после 2026-05-04 и дописать журнал наблюдения. Вердикта до точки
             рассмотрения нет; строка итога печатается с числом сессий и ближайшей точкой.
Правило исполняет run.run без изменений; здесь только календарь продолжения и выбор каталога ленты.
"""
import sys

import numpy as np
import pandas as pd

import run as band
from run import ROOT, HERE, MIN

INS = 'NQ'
HOLIDAYS = {'2026-05-25', '2026-06-19', '2026-07-03'}
F_FROM, F_TO = '2026-05-04', '2026-07-10'
REVIEW = (125, 250, 500)


def history_calendar():
    return pd.read_parquet(ROOT / 'setups/S-04/calendar.parquet')[['date', 'open_ns', 'close_ns']]


def forward_calendar():
    days = [d for d in pd.bdate_range(F_FROM, F_TO).strftime('%Y-%m-%d') if d not in HOLIDAYS]
    o = [pd.Timestamp(d + ' 09:30', tz='America/New_York').tz_convert('UTC').value for d in days]
    c = [pd.Timestamp(d + ' 16:00', tz='America/New_York').tz_convert('UTC').value for d in days]
    return pd.DataFrame(dict(date=days, open_ns=o, close_ns=c))


def full_calendar():
    h = history_calendar()
    h = h[h.date < F_FROM]
    return pd.concat([h, forward_calendar()], ignore_index=True)


def validate():
    D, _ = band.run(INS, cal=history_calendar(), root=ROOT / 'data/forward/market')
    ref = pd.read_csv(HERE / 'band_daily_NQ.csv', parse_dates=['date'])
    j = ref.merge(D, on='date', suffixes=('_ref', ''))
    j = j[j.date < F_FROM]
    same_status = (j.status_ref == j.status).sum()
    ok = j[(j.status == 'ok') & (j.status_ref == 'ok')]
    print('сессий', len(j), '| статус совпал', int(same_status), '| деньги совпали до цента',
          int(np.isclose(ok.usd_ref, ok.usd).sum()), 'из', len(ok), '| сделок совпало', int((ok.n_tr_ref == ok.n_tr).sum()))


def overlap():
    import pyarrow.parquet as pq
    import shutil, tempfile
    loc = {k: np.load(ROOT / 'data/market' / INS / (k + '.npy')) for k in ('close_ts_utc_ns', 'open', 'high', 'low', 'close')}
    ts = loc['close_ts_utc_ns']
    lo = int(np.searchsorted(ts, pd.Timestamp('2026-03-15 21:00', tz='UTC').value))
    lx = pq.read_table(ROOT / 'data/forward/_incoming/lynx1231_equity_index_minute.parquet').to_pandas()
    g = lx[lx.contract_code == 'NQM26'].copy()
    g['close_ns'] = (pd.to_datetime(g.timestamp, unit='ms') + pd.Timedelta(minutes=1)).dt.tz_localize(
        'America/Chicago', ambiguous='NaT', nonexistent='NaT').dt.tz_convert('UTC').dt.tz_localize(None).to_numpy(
        dtype='datetime64[ns]').astype('int64')
    g = g.set_index('close_ns')
    src = {k: v.copy() for k, v in loc.items()}
    w = ts[lo:]
    has = np.isin(w, g.index.to_numpy())
    gg = g.loc[w[has]]
    for k in ('open', 'high', 'low', 'close'):
        a = src[k]
        a[lo:][has] = gg[k].to_numpy()
    tmp = ROOT / 'work/line-102/_overlap_src'
    (tmp / INS).mkdir(parents=True, exist_ok=True)
    for k, v in src.items():
        np.save(tmp / INS / (k + '.npy'), v)
    cal = history_calendar()
    a, _ = band.run(INS, cal=cal)
    b, _ = band.run(INS, cal=cal, root=tmp)
    shutil.rmtree(tmp)
    a = a[(a.date >= '2026-03-16') & (a.date <= '2026-05-01')].set_index('date')
    b = b[(b.date >= '2026-03-16') & (b.date <= '2026-05-01')].set_index('date')
    both = a[(a.status == 'ok')].index.intersection(b[b.status == 'ok'].index)
    d = (b.loc[both, 'usd'] - a.loc[both, 'usd'])
    print('перекрытие: сессий', len(a), '| статус совпал', int((a.status == b.status).sum()),
          '| число сделок совпало', int((a.loc[both, 'n_tr'] == b.loc[both, 'n_tr']).sum()), 'из', len(both),
          '| деньги равны до цента', int(np.isclose(a.loc[both, 'usd'], b.loc[both, 'usd']).sum()),
          f'| расхождение: среднее {d.mean():+.2f} $, макс |.| {d.abs().max():.0f} $',
          f'| итог local {a.loc[both, "usd"].sum():+.0f} / source {b.loc[both, "usd"].sum():+.0f}')
    pd.concat([a.add_suffix('_local'), b.add_suffix('_source')], axis=1).to_csv(HERE / 'forward_overlap_NQ.csv', float_format='%.4f')


def ledger():
    D, T = band.run(INS, cal=full_calendar(), root=ROOT / 'data/forward/market')
    L = D[D.date >= F_FROM].copy()
    L.to_csv(HERE / 'forward_ledger_NQ.csv', index=False, float_format='%.2f')
    T[T.date >= F_FROM].to_csv(HERE / 'forward_trades_NQ.csv', index=False, float_format='%.2f')
    ok = L[L.status == 'ok']
    n = len(ok)
    nxt = next((r for r in REVIEW if r > n), None)
    print(f'журнал наблюдения: сессий календаря {len(L)}, наблюдённых {n}, неизвестных {len(L) - n}; '
          f'со сделкой {(ok.n_tr > 0).mean():.2f}; накопленный итог после расходов {ok.usd.sum():+.0f} $ '
          f'(валовое {ok.gross.sum():+.0f} $).')
    print(f'Это не точка рассмотрения: ближайшая — {nxt} сессий, до неё вердикта нет.'
          if n not in REVIEW else 'ТОЧКА РАССМОТРЕНИЯ.')


if __name__ == '__main__':
    {'validate': validate, 'overlap': overlap, 'ledger': ledger}[sys.argv[1]]()
