#!/usr/bin/env python3
"""Строит корпуса фильмов вокруг решающей минуты S-09 для relational_stencil.

Единица — один торговый день одной двери (день:дверь). Фильм — окно минут
относительно решающей минуты; ordinal 0 это сама решающая минута, чьё закрытие
уже известно в момент решения. Отрицательные ordinals — минуты до неё.
Положительные — уже будущее относительно решения, оно используется только
в описательных прогонах и никогда как условие входа.

Сторона берётся из правила S-09 (закрытие выше/ниже середины 30-свечного
диапазона). Север и юг не смешиваются: разные корпуса.
Исход (net) используется ТОЛЬКО для разделения корпус/reference во втором
прогоне; в представление фильма он не входит.
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'setups' / 'S-09'))
sys.path.insert(0, str(ROOT / 'research'))
import run as s09                      # noqa: E402
from relational_stencil import Corpus  # noqa: E402

MIN = 60_000_000_000


def build(door, lo_ord, hi_ord, instrument='NQ'):
    m = s09.market(instrument)
    ts = m['close_ts_utc_ns']
    tr = s09.trades(instrument, lo=s09.S0, hi=s09.H1)
    tr = tr[tr['door'] == door].reset_index(drop=True)
    pos = np.searchsorted(ts, tr['ts'].to_numpy())
    assert np.all(ts[pos] == tr['ts'].to_numpy())
    ords = np.arange(lo_ord, hi_ord + 1, dtype=np.int64)
    films = np.full((len(tr), len(ords), 4), np.nan)
    for k, p in enumerate(pos):
        want = ts[p] + ords * MIN
        at = np.searchsorted(ts, want)
        ok = at < len(ts)
        good = ok.copy(); good[ok] &= ts[at[ok]] == want[ok]
        for f, name in enumerate(('open', 'high', 'low', 'close')):
            films[k, good, f] = m[name][at[good]]
    return films, ords, tr, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--door', default='utro_0933')
    ap.add_argument('--instrument', default='NQ')
    ap.add_argument('--lo', type=int, default=-4)
    ap.add_argument('--hi', type=int, default=0)
    ap.add_argument('--outdir', required=True)
    a = ap.parse_args()
    films, ords, tr, m = build(a.door, a.lo, a.hi, a.instrument)
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    src = {'repository': str(ROOT), 'rule': 'setups/S-09/run.py v1',
           'run_py_sha256': s09.sha(ROOT / 'setups/S-09/run.py'),
           'tape': f'data/market/{a.instrument}/*.npy',
           'door': a.door, 'window_ordinals': [a.lo, a.hi]}
    written = []
    for side, sname in ((1.0, 'north'), (-1.0, 'south')):
        for terr, tmask in (('poisk', ~tr['holdout'].to_numpy()),
                            ('otlozhennoe', tr['holdout'].to_numpy())):
            for oname, omask in (('all', np.ones(len(tr), bool)),
                                 ('win', tr['net'].to_numpy() > 0),
                                 ('loss', tr['net'].to_numpy() <= 0)):
                sel = np.flatnonzero((tr['side'].to_numpy() == side) & tmask & omask)
                if len(sel) == 0:
                    continue
                days = [str(np.datetime64(v, 'D')) for v in tr['day'].to_numpy()[sel]]
                ids = [f'{a.door}:{d}' for d in days]
                c = Corpus(films[sel], ords, ids, list(ids), {
                    'instrument': a.instrument,
                    'source': {**src, 'side': sname, 'territory': terr,
                               'outcome_split': oname,
                               'outcome_role': 'split only; never inside the film'},
                    'anchor': f'S-09 decision minute of door {a.door}; ordinal 0 = that minute',
                    'unit_definition': 'trading day of this door',
                    'unknown_reason': 'minute missing from tape; slot kept as NaN'})
                p = out / f'{a.door}_{sname}_{terr}_{oname}.npz'
                c.save(p)
                written.append({'path': str(p), 'units': c.n})
    print(json.dumps({'films': int(len(tr)), 'ordinals': ords.tolist(),
                      'written': written}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
