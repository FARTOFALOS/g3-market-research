#!/usr/bin/env python3
"""096 часть II — добавляет ли RULE_V1 различимость вилки b <-> за M сверх k.

Правила заморожены в FREEZE_096_PART_II.md ДО запуска этого файла. Здесь только
их исполнение. Прибор вилки — base/086/fork086.py без изменений.

    python -B part2.py

Оператор RULE_V1 (§2 freeze, без свободных параметров):
    renews[i]   бар i поставил новый наружный экстремум на [t0..i]
    last_renew  последний i < q с renews[i]  (на нём стоит M)
    inward(i)   тело против outward: north -> close<open, south -> close>open
    signal      |{i: last_renew < i < q, inward(i)}| + (inward(last_renew) ? 1 : 0)
    isolated    signal == 0,  иначе progressive
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'work/096'
sys.path.insert(0, str(ROOT / 'base/086'))
import fork086 as F                                                          # noqa: E402

PERIOD = 'NQ_search'
STRATA = (('1-2', 1, 2), ('3-4', 3, 4), ('5-6', 5, 6), ('7-12', 7, 12), ('13-50', 13, 10 ** 9))
ISO, PRO = 'isolated', 'progressive'


@njit(cache=True)
def _signal(t0s, qs, north, opn, high, low, close, out):
    for i in range(t0s.size):
        t0, q, up = t0s[i], qs[i], north[i]
        run = -1e18 if up else 1e18
        last_renew = -1
        for p in range(t0, q):
            v = high[p] if up else low[p]
            new = v > run if up else v < run
            if new:
                run = v
                last_renew = p
        if last_renew < 0:
            out[i] = -1                       # экстремум не обновлялся до q
            continue
        s = 0
        for p in range(last_renew + 1, q):
            inw = close[p] < opn[p] if up else close[p] > opn[p]
            if inw:
                s += 1
        if last_renew > t0:
            inw = close[last_renew] < opn[last_renew] if up else close[last_renew] > opn[last_renew]
            if inw:
                s += 1
        out[i] = s


def operator(g, inst):
    opn = np.load(ROOT / 'data/market' / inst / 'open.npy').astype(np.float64)
    high = np.load(ROOT / 'data/market' / inst / 'high.npy').astype(np.float64)
    low = np.load(ROOT / 'data/market' / inst / 'low.npy').astype(np.float64)
    close = np.load(ROOT / 'data/market' / inst / 'close.npy').astype(np.float64)
    sig = np.empty(len(g), np.int64)
    _signal(g.t0_pos.to_numpy(), g.q.to_numpy(), g.north.to_numpy(),
            opn, high, low, close, sig)
    return sig


def arm(g, code, mask, label):
    x_lo, x_hi, g_lo, g_hi = F.per_film(g[mask], code[mask])
    e = F.estimate(x_lo, x_hi, g_lo, g_hi, g[mask].t0_day.to_numpy())
    return {'arm': label, 'N': int(mask.sum()), 'days': e['days'],
            'X': e['X'], 'X_outer_95': e['X_outer_95'], 'G': e['G']}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    g = F.geometry(PERIOD)
    inst = g.attrs['instrument']
    g = g[g.in_window.to_numpy() & np.isfinite(g.ttc.to_numpy()) & (g.sigma.to_numpy() > 0)].reset_index(drop=True)
    code, _ = F.walk(g)
    sig = operator(g, inst)
    iso = (sig == 0)
    res = {'freeze_sha256': F.sha(HERE / 'FREEZE_096_PART_II.md'),
           'fork_freeze_sha256': F.freeze(), 'period': F.PERIODS[PERIOD],
           'N': int(len(g)), 'days': int(g.t0_day.nunique()),
           'no_renewal_before_q': int((sig < 0).sum()),
           'share_isolated': round(float(iso.mean()), 4)}

    print(f'{PERIOD}: N={len(g)} дней={g.t0_day.nunique()}  '
          f'isolated={iso.sum()} ({iso.mean():.3f})  progressive={(~iso).sum()}')
    res['overall'] = [arm(g, code, iso, ISO), arm(g, code, ~iso, PRO)]
    for a in res['overall']:
        print(f"  {a['arm']:<12} N={a['N']:>6} дней={a['days']:>5} "
              f"X={a['X']}  outer95={a['X_outer_95']}")
    dx = [res['overall'][0]['X'][0] - res['overall'][1]['X'][1],
          res['overall'][0]['X'][1] - res['overall'][1]['X'][0]]
    res['delta_X_bounds'] = [round(v, 5) for v in dx]
    print(f'  ΔX (граница-к-границе, с учётом цензуры) = {res["delta_X_bounds"]}')

    print('\nпо слоям k (обязательный контроль, §4 freeze):')
    res['by_k'] = []
    k = g.k.to_numpy()
    for name, lo, hi in STRATA:
        m = (k >= lo) & (k <= hi)
        if m.sum() < 100:
            continue
        a_i = arm(g, code, m & iso, ISO)
        a_p = arm(g, code, m & ~iso, PRO)
        d = [a_i['X'][0] - a_p['X'][1], a_i['X'][1] - a_p['X'][0]]
        row = {'stratum': name, 'N': int(m.sum()), 'share_isolated': round(float(iso[m].mean()), 4),
               'iso': a_i, 'pro': a_p, 'delta_X_bounds': [round(v, 5) for v in d]}
        res['by_k'].append(row)
        print(f"  {name:>6} N={m.sum():>6} iso={iso[m].mean():.3f} | "
              f"X_iso={a_i['X']} X_pro={a_p['X']} | ΔX={row['delta_X_bounds']}")

    cell = ((g.u > 1.125) & (g.k <= 6) & (g.ttc <= 414) & (g.r <= 1.091)).to_numpy()
    print(f'\nsecondary stratum (клетка u+ k- ttc- r-): N={cell.sum()} дней={g[cell].t0_day.nunique()}')
    if cell.sum() > 50:
        a_i = arm(g, code, cell & iso, ISO)
        a_p = arm(g, code, cell & ~iso, PRO)
        res['cell'] = {'N': int(cell.sum()), 'share_isolated': round(float(iso[cell].mean()), 4),
                       'iso': a_i, 'pro': a_p,
                       'delta_X_bounds': [round(a_i['X'][0] - a_p['X'][1], 5),
                                          round(a_i['X'][1] - a_p['X'][0], 5)]}
        print(f"  iso N={a_i['N']} X={a_i['X']} | pro N={a_p['N']} X={a_p['X']} "
              f"| ΔX={res['cell']['delta_X_bounds']}")

    (OUT / 'part2.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print('\n->', OUT / 'part2.json')


if __name__ == '__main__':
    main()
