#!/usr/bin/env python3
"""096 select — instrument-development corpus: 60 обезличенных сцен close_break.

Правила заморожены в FREEZE_096_INSTRUMENT.md §4-§5 ДО запуска этого файла и до
просмотра любой картинки. Здесь только их исполнение.

Источник      NQ_development (NQ, x-ray территория discovery, T0 в [2006-01-01, 2018-12-25))
Пригодность   in_window=True, ttc определён, sigma>0  (правило популяции карты 086)
Дедупликация  (а) одна сцена на пару (t0_pos, side) — копии разных ТФ дают ту же ленту;
              (б) не более одной сцены на календарный день во всём наборе
Слои по k     1-2, 3-4, 5-6, 7-12, 13-50 — по 12 сцен
Seed          20260921

Исходы (терминал вилки b / за M) здесь НЕ читаются: walk() не вызывается, ни одна
колонка после q не трогается. Отбор prefix-honest по построению.

Выход: work/096/gold60.json (сцены, обезличенные и зеркалённые) и
       work/096/gold60_manifest.json (провенанс + sha256 документа, селектора, рендерера).
"""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'work/096'
sys.path.insert(0, str(ROOT / 'base/086'))
import fork086 as F                                                          # noqa: E402

SEED = 20260921
PERIOD = 'NQ_development'
STRATA = (('1-2', 1, 2), ('3-4', 3, 4), ('5-6', 5, 6), ('7-12', 7, 12), ('13-50', 13, 50))
PER_STRATUM = 12


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def pool():
    """Пригодные сцены NQ_development, без единого взгляда за q."""
    g = F.geometry(PERIOD)
    ok = g.in_window.to_numpy() & np.isfinite(g.ttc.to_numpy()) & (g.sigma.to_numpy() > 0)
    g = g[ok].reset_index(drop=True)
    g['t0_day'] = g.t0_day.to_numpy()
    return g


def dedup_episodes(g, rng):
    """Один представитель на физический эпизод (t0_pos, side), равновероятно."""
    order = rng.permutation(len(g))
    g = g.iloc[order].reset_index(drop=True)
    keep = ~g.duplicated(subset=['t0_pos', 'side'], keep='first')
    return g[keep].reset_index(drop=True)


def choose(g, rng):
    """12 сцен на слой, не более одной сцены на календарный день во всём наборе."""
    used_days, picked = set(), []
    for name, lo, hi in STRATA:
        cand = g[(g.k >= lo) & (g.k <= hi)]
        cand = cand.iloc[rng.permutation(len(cand))]
        n = 0
        for row in cand.itertuples():
            if n == PER_STRATUM:
                break
            if row.t0_day in used_days:
                continue
            used_days.add(row.t0_day)
            picked.append({'stratum': name, 'row': row})
            n += 1
        if n < PER_STRATUM:
            raise RuntimeError(f'слой {name}: набрано {n} из {PER_STRATUM}')
    return picked


def scene(row, opn, high, low, close, tick):
    """Обезличенная зеркалённая сцена T0..q в локально нормированных координатах.

    Зеркало: south -> north, то есть all сцены читаются как уход ВВЕРХ от b.
    Нормировка: y = (price - b) / sigma, знак уже развёрнут. Реальные относительные
    расстояния и пропорции сохраняются; номинальной шкалы нет.
    """
    t0, q, north, b, sig = int(row.t0_pos), int(row.q), bool(row.north), float(row.b), float(row.sigma)
    sl = slice(t0, q + 1)
    o, h, l, c = opn[sl].astype(float), high[sl].astype(float), low[sl].astype(float), close[sl].astype(float)
    sgn = 1.0 if north else -1.0
    y = lambda v: (v - b) * sgn / sig
    yo, yc = y(o), y(c)
    yh = y(h) if north else y(l)      # после зеркала «верх бара» — это away-край
    yl = y(l) if north else y(h)
    M = float(row.M)
    return {'n': int(q - t0 + 1),
            'o': [round(v, 4) for v in yo], 'h': [round(v, 4) for v in yh],
            'l': [round(v, 4) for v in yl], 'c': [round(v, 4) for v in yc],
            'b': 0.0, 'M': round(y(M), 4), 'tick': round(tick / sig, 6)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    g0 = pool()
    g = dedup_episodes(g0, rng)
    picked = choose(g, rng)

    inst = 'NQ'
    opn = np.load(ROOT / 'data/market' / inst / 'open.npy')
    high = np.load(ROOT / 'data/market' / inst / 'high.npy')
    low = np.load(ROOT / 'data/market' / inst / 'low.npy')
    close = np.load(ROOT / 'data/market' / inst / 'close.npy')
    tick = F.TICK[inst]

    scenes, key = [], []
    for i, p in enumerate(picked):
        r = p['row']
        s = scene(r, opn, high, low, close, tick)
        s['id'] = f'S{i + 1:02d}'
        scenes.append(s)
        key.append({'id': s['id'], 'stratum': p['stratum'], 'riz_id': r.riz_id, 'side': r.side,
                    't0_pos': int(r.t0_pos), 'q': int(r.q), 'k': int(r.k), 't0_day': int(r.t0_day)})

    # Порядок показа перемешивается, чтобы слои не читались подряд.
    perm = rng.permutation(len(scenes))
    order = [scenes[i]['id'] for i in perm]

    (OUT / 'gold60.json').write_text(json.dumps(
        {'scenes': scenes, 'order': order}, ensure_ascii=False), encoding='utf-8')
    (OUT / 'gold60_key.json').write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding='utf-8')

    manifest = {
        'frozen_utc': '2026-09-21',
        'freeze_doc': 'base/096/FREEZE_096_INSTRUMENT.md',
        'freeze_sha256': sha(HERE / 'FREEZE_096_INSTRUMENT.md'),
        'selector': 'base/096/select.py', 'selector_sha256': sha(__file__),
        'renderer': 'base/096/render.py',
        'renderer_sha256': sha(HERE / 'render.py') if (HERE / 'render.py').exists() else None,
        'source_period': PERIOD, 'source_bounds': F.PERIODS[PERIOD],
        'eligibility': 'in_window & finite ttc & sigma>0',
        'pool_films': int(len(g0)), 'pool_days': int(g0.t0_day.nunique()),
        'pool_physical_episodes': int(len(g)),
        'dedup_rule': 'one per (t0_pos, side); then at most one scene per calendar day',
        'strata': [{'name': n, 'k_lo': lo, 'k_hi': hi, 'n': PER_STRATUM} for n, lo, hi in STRATA],
        'seed': SEED, 'n_scenes': len(scenes),
        'shown_to_trader': ['candles T0..q', 'b line', 'running M line', 'bar numbers 0..k',
                            'q_event marked'],
        'hidden_from_trader': ['date', 'year', 'instrument', 'absolute price', 'riz_id', 'tf',
                               'k as a label', 'u/ttc/r', 'anything after q_event', 'outcome'],
        'mirrored': 'south -> north',
        'y_scale': 'local: (price - b)*sign/sigma; no nominal axis',
        'outcomes_read_during_selection': False,
        'note': 'instrument-development corpus, NOT untouched market validation; '
                '086 measured X on this epoch earlier, but no 096 outcome is used or shown here.',
    }
    txt = json.dumps(manifest, ensure_ascii=False, indent=1)
    (OUT / 'gold60_manifest.json').write_text(txt, encoding='utf-8')
    # провенанс живёт рядом с карточкой, а не только в work/ (work/ в .gitignore)
    (HERE / 'gold60_manifest.json').write_text(txt, encoding='utf-8')
    print(f'pool {len(g0)} фильмов / {g0.t0_day.nunique()} дней -> {len(g)} физических эпизодов')
    print(f'отобрано {len(scenes)} сцен, дней использовано {len({k["t0_day"] for k in key})}')
    for n, lo, hi in STRATA:
        ks = [k['k'] for k in key if k['stratum'] == n]
        print(f'  слой {n:>5}: n={len(ks)} k={sorted(ks)}')
    print('манифест:', OUT / 'gold60_manifest.json')


if __name__ == '__main__':
    main()
