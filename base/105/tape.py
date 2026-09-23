"""Общая загрузка ленты NQ для линии nq-manual (контракт research/NQ_MANUAL_TRADING_RESEARCH_PLAN.md).

Лента: data/forward/market/NQ — канонический префикс байт-в-байт (corpus f077783c…) + продолжение
2026-05-04…2026-07-10 (lynx1231, NQM26/NQU26), паспорт base/091/source_gate_forward_2026-09-21.json.
Время бара: close_ts_utc_ns = закрытие минуты; открытие = close − 60 с. Всё переводится в America/New_York.
Volume не загружается.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SPINE = ROOT / 'data' / 'forward' / 'market' / 'NQ'


def spine(inst):
    return ROOT / 'data' / 'forward' / 'market' / inst


def load_minutes(start='2006-01-01', end='2026-07-11', inst='NQ'):
    a = {k: np.load(spine(inst) / f'{k}.npy', mmap_mode='r') for k in ['close_ts_utc_ns', 'open', 'high', 'low', 'close']}
    ts = np.asarray(a['close_ts_utc_ns'], dtype='int64')
    lo = np.searchsorted(ts, pd.Timestamp(start, tz='America/New_York').value)
    hi = np.searchsorted(ts, pd.Timestamp(end, tz='America/New_York').value)
    ts = ts[lo:hi]
    et_open = pd.to_datetime(ts - 60_000_000_000, utc=True).tz_convert('America/New_York')
    df = pd.DataFrame({
        'o': np.asarray(a['open'][lo:hi], dtype='float64'),
        'h': np.asarray(a['high'][lo:hi], dtype='float64'),
        'l': np.asarray(a['low'][lo:hi], dtype='float64'),
        'c': np.asarray(a['close'][lo:hi], dtype='float64'),
    })
    df['date'] = (et_open.year * 10000 + et_open.month * 100 + et_open.day).to_numpy()
    df['mod'] = (et_open.hour * 60 + et_open.minute).to_numpy()   # минута открытия бара, ET
    df['ts'] = ts
    return df


def window(df, m0=120, m1=960):
    """Бары, чьё открытие в [m0, m1) минут ET: по умолчанию 02:00–16:00."""
    return df[(df['mod'] >= m0) & (df['mod'] < m1)]
