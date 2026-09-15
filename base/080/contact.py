#!/usr/bin/env python3
"""Быстрый ТОЧНЫЙ первый post-T0 контакт с собственной exit boundary.

Определение не меняется: первый наблюдаемый бар j > t0, чей диапазон содержит
уровень, `low[j] <= e <= high[j]`. Ускорение чисто техническое — блок
пропускается, только если ни один его бар не может удовлетворить ОБОИМ
условиям (`blockmin_low > e` или `blockmax_high < e`). Это необходимое условие,
поэтому пропуск не может проскочить контакт.

Поиск идёт до конца ленты. Технического горизонта здесь нет: `-1` означает
«до конца архива наблюдаемого контакта не было», а не «не найдено за бюджет».
Гэп через уровень контактом не считается: он не порождает бара, чей диапазон
содержит уровень.
"""
from __future__ import annotations
import numpy as np
from numba import njit, prange

B = 256


def blocks(low, high, b=B):
    n = low.size
    nb = (n + b - 1) // b
    pad = nb * b - n
    lo = np.concatenate([low, np.full(pad, np.inf)]).reshape(nb, b).min(axis=1)
    hi = np.concatenate([high, np.full(pad, -np.inf)]).reshape(nb, b).max(axis=1)
    return lo, hi


@njit(cache=True)
def _first_contact(low, high, blo, bhi, t0, e, b):
    n = low.size
    j = t0 + 1
    if j >= n:
        return -1
    while j < n:
        blk = j // b
        end = min((blk + 1) * b, n)
        if blo[blk] <= e and bhi[blk] >= e:
            for k in range(j, end):
                if low[k] <= e and high[k] >= e:
                    return k
        j = end
    return -1


@njit(parallel=True, cache=True)
def first_contacts(low, high, blo, bhi, t0s, es, b):
    out = np.empty(t0s.size, dtype=np.int64)
    for i in prange(t0s.size):
        out[i] = _first_contact(low, high, blo, bhi, t0s[i], es[i], b)
    return out
