#!/usr/bin/env python3
"""Дорога или локальный масштаб: разделимы ли они на этом чтении.

При фиксированном `d_close` величина `retraced_fraction` есть монотонная функция
`max_departure_distance`: max_dep = d_close / (1 − retraced). Поэтому «как цена
сюда пришла» на этом чтении алгебраически совпадает с «насколько широким был
её размах». Здесь это проверяется прямо и сравнивается с простой локальной σ
(средний диапазон бара по префиксу) при том же точном совпадении положения.

Локальная σ вводится ТОЛЬКО как контроль объяснения, а не как признак правила
(журнал: DECLARE_081A §6 требует названного основания — основание здесь в том,
что без него «дорога» и «масштаб» неразличимы).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
from numba import njit, prange
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from family2 import load                                               # noqa: E402
from control import exact_strata, compare                              # noqa: E402


@njit(parallel=True, cache=True)
def _sig(t0s, Fs, off, high, low, out):
    for i in prange(t0s.size):
        t0 = t0s[i]; F = Fs[i]; base = off[i]; s = 0.0
        for q in range(t0, F + 1):
            s += high[q] - low[q]
            out[base + (q - t0)] = s / (q - t0 + 1)


def main(inst='NQ', terr='discovery'):
    q, f = load(inst, terr)
    d = q[(q.age == 5) & q.usable]
    md = d.d_close.to_numpy() / (1 - d.retraced_fraction.to_numpy())
    sub = d[np.isclose(d.d_close, 2.5)]
    ms = sub.d_close.to_numpy() / (1 - sub.retraced_fraction.to_numpy())
    print('АЛГЕБРА: при фиксированном d_close retraced_fraction и max_departure — одно')
    print(f'  внутри d_close=2.50 (n={len(sub)}): Пирсон '
          f'{np.corrcoef(sub.retraced_fraction, ms)[0,1]:.4f}, '
          f'Спирмен {pd.Series(sub.retraced_fraction.to_numpy()).corr(pd.Series(ms), method="spearman"):.4f}')

    high = np.load(ROOT / f'data/market/{inst}/high.npy')
    low = np.load(ROOT / f'data/market/{inst}/low.npy')
    t0 = f.t0_spine_pos.to_numpy().astype(np.int64)
    F = f.certified_fresh_until_pos_strict.to_numpy().astype(np.int64)
    nq = f.n_q.to_numpy().astype(np.int64)
    off = np.concatenate(([0], np.cumsum(nq)))[:-1]
    sig = np.empty(int(nq.sum()), dtype=np.float32)
    _sig(t0, F, off, high, low, sig)
    q['prefix_sigma'] = sig

    print('\nКОНТРОЛЬ: то же сравнение, группы делятся по простой локальной σ')
    for a in (5, 15):
        s = exact_strata(q, f, a, inst)
        print(f'age {a}:')
        for nm in ('retraced_fraction', 'prefix_sigma'):
            r = compare(s, inst, feat=nm)
            dd = r['delta_high_minus_low']
            print(f"  {nm:>17}: страт {r['strata_kept']:>3} опора {r['support_retained']:.3f} | "
                  f"Δcert {dd['certified']:+.4f} Δloss {dd['loss']:+.4f} "
                  f"Δdur {dd.get('dur_cert_p50', float('nan')):+.2f} "
                  f"Δlower {dd['lower']:+.4f} Δupper {dd['upper']:+.4f}")
        cc = s.groupby('key', observed=True).apply(
            lambda g: g.retraced_fraction.corr(g.prefix_sigma, method='spearman')
            if len(g) > 30 else np.nan, include_groups=False)
        print(f'  Спирмен(retraced, σ) внутри точных страт: медиана {np.nanmedian(cc):.3f}')


if __name__ == '__main__':
    main(*(sys.argv[1:] or ['NQ', 'discovery']))
