"""074: do the observable elements of the first Z minute separate win from loss?

Only elements of that minute itself, named in advance, no search:
  depth   how far into the zone its close reached, in zone widths;
  shape   where it closed inside its own range, and whether it closed with the
          direction of the entry;
  moment  which minute after T0 it was;
  ratio   its range against the width of the RIZ.

For each element separately: the outcome of the frozen construction, by epoch.
An element separates only if the order of its buckets holds in every epoch.

Run: python -B base/074/first_z_variants.py
"""
from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from frozen_trade import build, resolve, session_window      # noqa: E402
from first_difference import scenes                          # noqa: E402

OUT = Path(__file__).resolve().parent
EPOCHS = [(2006, 2010), (2011, 2015), (2016, 2020), (2021, 2026)]


def main():
    t0, X, xl, d = scenes()
    sess = session_window(t0)
    bs = [build(xl, sess, d.iloc[k:k + 40000].reset_index(drop=True))
          for k in range(0, len(d), 40000)]
    b = {k: np.concatenate([q[k] for q in bs]) for k in bs[0]}
    r1, p1 = resolve(b, b['t1'])
    r2, p2 = resolve(b, b['t2'])
    n = len(b['i'])
    k = np.arange(n)
    zi = b['zi']
    H, L, C = b['H'][k, zi], b['L'][k, zi], b['C'][k, zi]
    O = xl[d.row.to_numpy()[b['i']], zi, 0]
    w = np.abs(b['t2'] - b['t1'])
    near = b['stop'] + np.where(b['sgn'] > 0, .25, -.25)
    rng_ = np.maximum(H - L, .25)
    scale = np.nanmedian(b['H'][:, :8] - b['L'][:, :8], 1)

    df = pd.DataFrame(dict(
        year=b['year'], tradable=np.isin(r1, ['stop', 'target', 'timeout']),
        win1=(r1 == 'target'), win2=(r2 == 'target'),
        depth=pd.cut(np.abs(C - near) / w, [.5, .65, .8, 1.01],
                     labels=['0,5-0,65', '0,65-0,8', '0,8-1,0']),
        shape=pd.cut(np.where(b['sgn'] > 0, (C - L) / rng_, (H - C) / rng_), [-.01, .5, .8, 1.01],
                     labels=['закрытие в нижней половине хода', 'середина', 'на своём конце']),
        moment=pd.cut(zi, [0, 1, 3, 10, 50], labels=['+1', '+2..+3', '+4..+10', 'позже']),
        ratio=pd.cut(rng_ / w, [0, 1, 2, 99], labels=['уже зоны', '1-2 ширины', 'шире 2']),
        body=pd.cut(np.abs(C - O) / rng_, [-.01, .3, .7, 1.01],
                    labels=['тело <30%', '30-70%', 'тело >70%'])))
    t = df[df.tradable]
    rep = {'n_tradable': int(len(t))}
    print('tradable signals:', len(t))
    for col in ['depth', 'shape', 'moment', 'ratio', 'body']:
        tab = {}
        print(f'\n-- {col}: доля достигших цели (Target 1 / Target 2), по эпохам')
        hdr = f'{"":22s}' + ''.join(f'{lo}-{hi:>4}   ' for lo, hi in EPOCHS) + '  n'
        print(hdr)
        order_ok = []
        for lvl in t[col].cat.categories:
            m = t[col] == lvl
            row, r2s = [], []
            for lo, hi in EPOCHS:
                e = m & (t.year >= lo) & (t.year <= hi)
                row.append(float(t.win1[e].mean()) if e.sum() >= 30 else float('nan'))
                r2s.append(float(t.win2[e].mean()) if e.sum() >= 30 else float('nan'))
            tab[str(lvl)] = dict(n=int(m.sum()), t1_by_epoch=row, t2_by_epoch=r2s,
                                 t1=float(t.win1[m].mean()), t2=float(t.win2[m].mean()))
            order_ok.append(row)
            cells = ''.join(f'{a:.3f}/{c:.3f}  ' if a == a else '   -/-      '
                            for a, c in zip(row, r2s))
            print(f'{str(lvl):22s}{cells}{m.sum():5d}')
        arr = np.array(order_ok, float)
        stable = bool(np.all(np.diff(arr, axis=0) > 0) or np.all(np.diff(arr, axis=0) < 0))
        rep[col] = dict(levels=tab, order_holds_in_every_epoch=stable)
        print(f'   порядок сохраняется во всех эпохах: {stable}')
    (OUT / 'first_z_variants.json').write_text(json.dumps(rep, ensure_ascii=False, indent=2),
                                               encoding='utf-8')


if __name__ == '__main__':
    main()
