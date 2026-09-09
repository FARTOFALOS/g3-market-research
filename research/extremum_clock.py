"""Где во времени сидит экстремум фильма после двери S-09.

Только порядковые факты: номер свечи, на которой ставится максимум хода
в сторону сделки и максимум хода против неё, считая от входа (open минуты +1).
Величины не участвуют в ответе, только в определении «где выше/ниже».
"""
import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'setups' / 'S-09'))
import run as s09                                    # noqa: E402

MIN = 60_000_000_000
HOLD = 120


def films(ins, door, holdout):
    m = s09.market(ins)
    ts = m['close_ts_utc_ns']
    tr = s09.trades(ins, lo=s09.S0, hi=s09.H1)
    tr = tr[(tr['door'] == door) & (tr['holdout'] == holdout)]
    p = np.searchsorted(ts, tr['ts'].to_numpy())
    side = tr['side'].to_numpy()
    k = np.arange(1, HOLD + 1)
    hi = np.full((len(p), HOLD), np.nan)
    lo = np.full((len(p), HOLD), np.nan)
    entry = np.full(len(p), np.nan)
    for i, q in enumerate(p):
        want = ts[q] + (k + 0) * MIN
        at = np.searchsorted(ts, want)
        ok = at < len(ts)
        good = ok.copy()
        good[ok] &= ts[at[ok]] == want[ok]
        hi[i, good] = m['high'][at[good]]
        lo[i, good] = m['low'][at[good]]
        if good[0]:
            entry[i] = m['open'][at[0]]
    fav = np.where(side[:, None] > 0, hi - entry[:, None], entry[:, None] - lo)
    adv = np.where(side[:, None] > 0, entry[:, None] - lo, hi - entry[:, None])
    return fav, adv, tr


def clock(x):
    """Номер свечи бегущего экстремума и доля фильмов, где он там и остался."""
    run = np.fmax.accumulate(np.where(np.isnan(x), -np.inf, x), axis=1)
    full = np.isfinite(x).all(axis=1)
    x = x[full]
    arg = np.nanargmax(x, axis=1) + 1
    return arg, run[full], x


def show(tag, fav, adv):
    a_fav, _, xf = clock(fav)
    a_adv, _, xa = clock(adv)
    n = len(xf)
    q = [10, 25, 50, 75, 90]
    print(f'{tag}: полных фильмов {n} из {len(fav)}')
    print(f'  свеча максимума ХОДА ЗА вход   квантили {q}: '
          + ' '.join(f'{v:.0f}' for v in np.percentile(a_fav, q))
          + f'  медиана {np.median(a_fav):.0f}')
    print(f'  свеча максимума ХОДА ПРОТИВ    квантили {q}: '
          + ' '.join(f'{v:.0f}' for v in np.percentile(a_adv, q))
          + f'  медиана {np.median(a_adv):.0f}')
    for lim in (10, 20, 30, 60, 90, 120):
        print(f'    к {lim:3d}-й минуте: максимум ЗА уже поставлен у '
              f'{(a_fav <= lim).mean():5.1%}, максимум ПРОТИВ у {(a_adv <= lim).mean():5.1%}')
    # Порядок: что случилось раньше — лучший ход за вход или худший против.
    print(f'  максимум ЗА раньше максимума ПРОТИВ: {(a_fav < a_adv).mean():5.1%}; '
          f'позже: {(a_fav > a_adv).mean():5.1%}')


def main():
    for door in ('utro_0933', 'den_1338'):
        for holdout, name in ((False, 'поиск'), (True, 'отложенное')):
            fav, adv, tr = films('NQ', door, holdout)
            show(f'NQ {door} {name}', fav, adv)


if __name__ == '__main__':
    main()
