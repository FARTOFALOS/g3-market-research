#!/usr/bin/env python3
"""083 — направленный баланс на всей префиксной популяции.

    δ = (boundary_first − mirror_first) / N

`N` — вся объявленная префиксная популяция (`d_close > 0`). `unresolved` и
`lost_observability` входят в знаменатель и вносят в числитель ноль: гонка не
разрешилась — перевеса нет. Ненаблюдённое продолжение исходом не становится.

Границы неопределённости (DECLARE_083 §4):
    уровень 1, внутриминутный порядок   δ ± A/N
    уровень 2, плюс потеря ленты        δ ± (A+L)/N
    справочная полная                   δ ± (A+L+U)/N

CI — блочный бутстрэп по дням `t0_day`, блоки 1 / 5 / 20; публикуется самый
широкий интервал (процедура 082 без изменений).
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = Path(os.environ.get('G3_083_OUT', str(ROOT / 'work/083')))
HS = (5, 15, 60)
AGES = (5, 15, 60)
BLOCKS = (1, 5, 20)
NBOOT = 2000
RNG = 20260915


def counts(o):
    return dict(boundary_first=int((o == 1).sum()), mirror_first=int((o == 2).sum()),
                same_bar_ambiguous=int((o == 3).sum()), unresolved=int((o == 0).sum()),
                lost_observability=int((o == 4).sum()))


def delta_block(o, day, nboot=NBOOT, seed=RNG):
    """δ и самый широкий из блочных CI. Ресэмплятся дни, не строки."""
    ud, inv = np.unique(day, return_inverse=True)
    D = ud.size
    # на каждый день: числитель (B−M) и знаменатель (N)
    num = np.bincount(inv, weights=(o == 1).astype(float) - (o == 2), minlength=D)
    den = np.bincount(inv, minlength=D).astype(float)
    amb = np.bincount(inv, weights=(o == 3).astype(float), minlength=D)
    lst = np.bincount(inv, weights=(o == 4).astype(float), minlength=D)
    unr = np.bincount(inv, weights=(o == 0).astype(float), minlength=D)
    out = {}
    tot = den.sum()
    out['n'] = int(tot)
    out['delta'] = float(num.sum() / tot) if tot else float('nan')
    out['amb_lo'] = float((num.sum() - amb.sum()) / tot) if tot else float('nan')
    out['amb_hi'] = float((num.sum() + amb.sum()) / tot) if tot else float('nan')
    out['obs_lo'] = float((num.sum() - amb.sum() - lst.sum()) / tot) if tot else float('nan')
    out['obs_hi'] = float((num.sum() + amb.sum() + lst.sum()) / tot) if tot else float('nan')
    out['full_lo'] = float((num.sum() - amb.sum() - lst.sum() - unr.sum()) / tot) if tot else float('nan')
    out['full_hi'] = float((num.sum() + amb.sum() + lst.sum() + unr.sum()) / tot) if tot else float('nan')
    if tot == 0:
        out['ci'] = [float('nan'), float('nan')]
        return out
    lo, hi = np.inf, -np.inf
    per = {}
    for b in BLOCKS:
        rng = np.random.default_rng(seed + b)
        nb = int(np.ceil(D / b))
        starts = rng.integers(0, max(D - b + 1, 1), size=(nboot, nb))
        offs = np.arange(b)
        idx = (starts[:, :, None] + offs[None, None, :]).reshape(nboot, -1)[:, :D]
        idx = np.minimum(idx, D - 1)
        s_num = num[idx].sum(1); s_den = den[idx].sum(1)
        est = np.where(s_den > 0, s_num / np.maximum(s_den, 1), np.nan)
        a, c = np.nanpercentile(est, [2.5, 97.5])
        per[b] = [float(a), float(c)]
        lo = min(lo, a); hi = max(hi, c)
    out['ci'] = [float(lo), float(hi)]
    out['ci_by_block'] = per
    # тот же бутстрэп для консервативной границы amb_lo
    lo2, hi2 = np.inf, -np.inf
    for b in BLOCKS:
        rng = np.random.default_rng(seed + 100 + b)
        nb = int(np.ceil(D / b))
        starts = rng.integers(0, max(D - b + 1, 1), size=(nboot, nb))
        offs = np.arange(b)
        idx = (starts[:, :, None] + offs[None, None, :]).reshape(nboot, -1)[:, :D]
        idx = np.minimum(idx, D - 1)
        s = (num[idx] - amb[idx]).sum(1); sd = den[idx].sum(1)
        est = np.where(sd > 0, s / np.maximum(sd, 1), np.nan)
        a, c = np.nanpercentile(est, [2.5, 97.5])
        lo2 = min(lo2, a); hi2 = max(hi2, c)
    out['ci_amb_lo'] = [float(lo2), float(hi2)]
    lo3, hi3 = np.inf, -np.inf
    for b in BLOCKS:
        rng = np.random.default_rng(seed + 200 + b)
        nb = int(np.ceil(D / b))
        starts = rng.integers(0, max(D - b + 1, 1), size=(nboot, nb))
        offs = np.arange(b)
        idx = (starts[:, :, None] + offs[None, None, :]).reshape(nboot, -1)[:, :D]
        idx = np.minimum(idx, D - 1)
        s = (num[idx] + amb[idx]).sum(1); sd = den[idx].sum(1)
        est = np.where(sd > 0, s / np.maximum(sd, 1), np.nan)
        a, c = np.nanpercentile(est, [2.5, 97.5])
        lo3 = min(lo3, a); hi3 = max(hi3, c)
    out['ci_amb_hi'] = [float(lo3), float(hi3)]
    return out


def cell(d, h, sub=None):
    x = d[d.state_readable] if sub is None else d[d.state_readable & sub]
    o = x[f'outcome_{h}'].to_numpy()
    r = counts(o)
    r.update(delta_block(o, x.t0_day.to_numpy()))
    bm = r['boundary_first'] + r['mirror_first']
    r['resolved_only_ratio'] = float(r['boundary_first'] / bm) if bm else float('nan')
    r['resolved_share'] = float((bm + r['same_bar_ambiguous']) / r['n']) if r['n'] else float('nan')
    return r


def qsplit(v, k=4):
    e = np.nanpercentile(v, np.linspace(0, 100, k + 1))
    e[0] -= 1e-9; e[-1] += 1e-9
    return np.clip(np.searchsorted(e, v, 'right') - 1, 0, k - 1)


def run(inst='NQ', terr='discovery'):
    res = {'instrument': inst, 'territory': terr, 'ages': {}}
    for a in AGES:
        p = OUT / f'race_{inst}_{terr}_a{a}.parquet'
        if not p.exists():
            continue
        d = pd.read_parquet(p)
        A = {'eligible': int(len(d)),
             'excluded_d_close_le_0': int((~d.state_readable).sum()),
             'N': int(d.state_readable.sum()),
             'h': {}}
        sr = d.state_readable
        for h in HS:
            c = cell(d, h)
            c['by_side'] = {s: cell(d, h, sr & (d.side == s)) for s in ('north', 'south')}
            c['by_prefix_mirror_touched'] = {
                'touched': cell(d, h, sr & d.prefix_mirror_touched),
                'untouched': cell(d, h, sr & ~d.prefix_mirror_touched)}
            for name, col in (('prefix_sigma', 'prefix_sigma'), ('n_bars', 'n_bars')):
                qq = np.full(len(d), -1)
                qq[sr.to_numpy()] = qsplit(d.loc[sr, col].to_numpy())
                c[f'by_{name}_quartile'] = {
                    str(k): cell(d, h, sr & (pd.Series(qq, index=d.index) == k))
                    for k in range(4)}
            c['gap_through'] = {
                'gap_e_share': float(d.loc[sr, 'gap_e'].mean()),
                'gap_m_share': float(d.loc[sr, 'gap_m'].mean())}
            A['h'][str(h)] = c
        A['prefix_mirror_touched_share'] = float(d.loc[sr, 'prefix_mirror_touched'].mean())
        res['ages'][str(a)] = A
    return res


if __name__ == '__main__':
    inst = sys.argv[1] if len(sys.argv) > 1 else 'NQ'
    terr = sys.argv[2] if len(sys.argv) > 2 else 'discovery'
    r = run(inst, terr)
    p = OUT / f'balance_{inst}_{terr}.json'
    p.write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding='utf-8')
    for a in AGES:
        if str(a) not in r['ages']:
            continue
        A = r['ages'][str(a)]
        print(f"age {a}  N={A['N']}  (вне контраста {A['excluded_d_close_le_0']}), "
              f"m накрыт в префиксе {A['prefix_mirror_touched_share']:.4f}")
        for h in HS:
            c = A['h'][str(h)]
            print(f"  h{h:<3} B {c['boundary_first']:>6} M {c['mirror_first']:>6} "
                  f"A {c['same_bar_ambiguous']:>5} U {c['unresolved']:>6} L {c['lost_observability']:>5} "
                  f"| delta {c['delta']:+.4f} CI [{c['ci'][0]:+.4f};{c['ci'][1]:+.4f}] "
                  f"| amb [{c['amb_lo']:+.4f};{c['amb_hi']:+.4f}] "
                  f"obs [{c['obs_lo']:+.4f};{c['obs_hi']:+.4f}] "
                  f"| resolved {c['resolved_share']:.3f} ratio {c['resolved_only_ratio']:.4f}")
    print(p)
