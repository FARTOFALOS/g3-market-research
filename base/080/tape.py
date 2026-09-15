#!/usr/bin/env python3
"""Лента как основание достоверности: где наблюдение непрерывно и где нет.

ТРИ РОДА ПРОМЕЖУТКА, И НИ ОДИН ИЗ НИХ НЕ КАЛЕНДАРЬ
==================================================
Провенанса торгового календаря в репозитории нет: `SOURCE_DATA.json` держит
`feed_origin`, `license`, `continuous_roll_rule` и `source_timezone_metadata`
равными null, а сессии спины нарезаны по соглашению 18:00 ET из наблюдённых
баров, а не из расписания биржи. Поэтому здесь не объявляется ни одного
интервала закрытым рынком. Промежуток получает род:

    0  contiguous            соседние минуты часов, промежутка нет
    1  shared_cause_unknown  минуты отсутствуют у ES, NQ и YM одновременно
    2  instrument_specific   хотя бы одна минута есть у другого инструмента,
                             то есть торговля внутри промежутка показана
    3  outside_common_window минуты лежат вне общего окна трёх лент

Род 2 доказывает, что цена внутри промежутка двигалась ненаблюдаемо здесь.
Род 1 согласуется с биржевой паузой, но три ленты могут делить один источник
и один сбой, поэтому «одновременно отсутствует» — это не «торгов не было».
Род 3 неразрешим по локальным данным.

KNOWN SCHEDULED CLOSED остаётся пустым родом до появления календаря с
провенансом. Что этот запрет стоит, считается отдельно как sensitivity: род 1
переживается или не переживается, и обе цифры называются.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MIN = 60_000_000_000
INSTRUMENTS = ('ES', 'NQ', 'YM')
KIND_NAMES = {0: 'contiguous', 1: 'shared_cause_unknown',
              2: 'instrument_specific', 3: 'outside_common_window'}


def _minute_index(inst):
    t = np.load(ROOT / f'data/market/{inst}/close_ts_utc_ns.npy', mmap_mode='r')
    return np.asarray(t) // MIN


def presence_grid():
    """Булева решётка «минута наблюдалась» по трём лентам в общем окне."""
    mins = {i: _minute_index(i) for i in INSTRUMENTS}
    lo = max(int(mins[i][0]) for i in INSTRUMENTS)
    hi = min(int(mins[i][-1]) for i in INSTRUMENTS)
    n = hi - lo + 1
    grid = {}
    for i in INSTRUMENTS:
        m = np.zeros(n, dtype=bool)
        v = mins[i]
        v = v[(v >= lo) & (v <= hi)] - lo
        m[v] = True
        grid[i] = m
    return grid, lo, hi


def gap_kinds(inst, grid=None, lo=None, hi=None):
    """`kind[j]` — род промежутка НЕПОСРЕДСТВЕННО ПЕРЕД баром j. `kind[0] = 0`."""
    if grid is None:
        grid, lo, hi = presence_grid()
    mt = _minute_index(inst)
    n = mt.size
    kind = np.zeros(n, dtype=np.int8)
    missing = np.zeros(n, dtype=np.int64)
    d = mt[1:] - mt[:-1]
    idx = np.flatnonzero(d > 1) + 1          # бары, перед которыми есть дыра
    missing[idx] = d[idx - 1] - 1
    other = np.zeros(hi - lo + 1, dtype=bool)
    for i in INSTRUMENTS:
        if i != inst:
            other |= grid[i]
    # префиксная сумма «минута есть у другого инструмента» по общему окну
    cum = np.concatenate(([0], np.cumsum(other, dtype=np.int64)))
    a = mt[idx - 1] + 1                       # первая отсутствующая минута
    b = mt[idx] - 1                           # последняя отсутствующая
    inside = (a >= lo) & (b <= hi)
    ca = np.clip(a - lo, 0, hi - lo + 1)
    cb = np.clip(b - lo + 1, 0, hi - lo + 1)
    elsewhere = (cum[cb] - cum[ca]) > 0
    kind[idx] = np.where(~inside, 3, np.where(elsewhere, 2, 1)).astype(np.int8)
    return kind, missing


def next_of_kind(kind, kinds):
    """`nxt[p]` — наименьший j > p, чей род входит в `kinds`; иначе len(kind)."""
    n = kind.size
    want = np.isin(kind, list(kinds))
    nxt = np.full(n + 1, n, dtype=np.int64)
    for j in range(n - 1, -1, -1):
        nxt[j] = j if want[j] else nxt[j + 1]
    return nxt


def next_of_kind_fast(kind, kinds):
    n = kind.size
    want = np.isin(kind, list(kinds))
    pos = np.where(want, np.arange(n), n)
    return np.minimum.accumulate(pos[::-1])[::-1]
