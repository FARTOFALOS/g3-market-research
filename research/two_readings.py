#!/usr/bin/env python3
"""Два прочтения одного предмета на общем корпусе фильмов RIZ.

ПРОЧТЕНИЕ 1 — точное сочетание отношений на определённых минутах от T0.
Это `repetition_map.py` / `motif_to_action.py`.

ПРОЧТЕНИЕ 2 — последовательность событий с различающимися промежутками.
Здесь. Словарь событий объявлен ДО счёта и взят из существующего
`ordinal_events.default_events` (префикс 0..cursor, якорь 0):

  argmaxH первый/последний, argminL первый/последний, последнее обновление
  максимума и минимума, первое прохождение H[0] тенью и закрытием,
  первое прохождение L[0] тенью и закрытием.

Подпись фильма — НЕ перестановка. Сохраняются:
  * какие события вообще наступили (NONE) и какие неизвестны (UNKNOWN);
  * попарный порядок наступивших событий в трёх значениях {<, =, >}, то есть
    совпадение минут сохраняется как совпадение, а не разводится случайно.
Сами номера минут и промежутки между событиями в подпись НЕ входят: именно это
и делает прочтение 2 нечувствительным к разной длительности.

МОМЕНТ ЗНАНИЯ
=============
Все события объявлены на префиксе 0..cursor и известны не раньше его конца:
адрес экстремума префикса известен только когда префикс закрыт. Опоры H[0] и
L[0] известны с минуты 0. Момент узнавания подписи — та же минута `cursor`,
что и у прочтения 1, поэтому прочтения сравнимы.

ЧТО СРАВНИВАЕТСЯ
================
Одно и то же измеримое различие продолжения: гонка симметричных барьеров
после курсора, с контролем по положению закрытия внутри диапазона префикса.
Вопрос один: переживает ли существенное различие смену представления?
Словарь событий заранее объявлен и под остаток прочтения 1 не подбирался.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from film_corpus import build, projections                       # noqa: E402
from repetition_map import language, atom_name, Packed           # noqa: E402
from motif_to_action import barriers, first_touch                # noqa: E402
from calendar_utils import date_key                              # noqa: E402
from ordinal_events import NONE, UNKNOWN                         # noqa: E402
from relational_stencil import Corpus                            # noqa: E402
import ordinal_events as oe                                      # noqa: E402


def declared_events(corpus, cursor):
    """Ровно `default_events` на префиксе 0..cursor, якорь 0."""
    return oe.default_events(corpus, start=0, stop=cursor, anchor_ordinal=0)


def signature(values):
    """Подпись: наступление + попарный порядок с сохранением совпадений."""
    E, n = values.shape
    occ = np.zeros((n, E), dtype=np.int8)
    occ[(values.T == NONE)] = 1
    occ[(values.T == UNKNOWN)] = 2
    parts = [occ]
    for i in range(E):
        for j in range(i + 1, E):
            a, b = values[i], values[j]
            ok = (a >= 0) & (b >= 0)
            col = np.full(n, 3, dtype=np.int8)          # 3 = сравнение не имеет смысла
            col[ok & (a < b)] = 0
            col[ok & (a == b)] = 1
            col[ok & (a > b)] = 2
            parts.append(col[:, None])
    return np.concatenate(parts, axis=1)


def keys_of(sig):
    v = np.ascontiguousarray(sig)
    return np.array([hash(r.tobytes()) for r in v])


def residual(y, mask, decided, bins, day, n_bins):
    """Разница с контролем той же позиции + кластерная по дням ошибка."""
    m = mask & decided
    if m.sum() < 200:
        return None
    p = np.full(n_bins, np.nan)
    for k in range(n_bins):
        ctl = decided & (bins == k) & ~m
        if ctl.sum() >= 30:
            p[k] = y[ctl].mean()
    ok = m & np.isfinite(p[bins])
    if ok.sum() < 200:
        return None
    res = y[ok] - p[bins[ok]]
    N = len(res)
    d = day[ok]
    order = np.argsort(d)
    rs, ds = res[order], d[order]
    sums = np.add.reduceat(rs, np.r_[0, np.flatnonzero(np.diff(ds)) + 1])
    se = np.sqrt((sums ** 2).sum()) / N
    return {'films_decided': int(N), 'days': int(len(sums)),
            'share_up': float(y[ok].mean()),
            'control_share_up': float(p[bins[ok]].mean()),
            'difference': float(res.mean()),
            't_clustered_by_day': float(res.mean() / se) if se else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--instrument', default='NQ')
    ap.add_argument('--cursor', type=int, default=12)
    ap.add_argument('--split', default='2020-01-01')
    ap.add_argument('--bins', type=int, default=20)
    ap.add_argument('--min-support', type=int, default=400)
    ap.add_argument('--reading1', default='work/070/action_NQ_c12.json')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    c = build(a.instrument)
    n = c['ohlc'].shape[0]
    train = c['t0'] < np.datetime64(a.split, 'ns').astype('int64')
    day = date_key(c['t0'])

    # одно и то же измеримое различие для обоих прочтений
    hi, lo, Cc = barriers(c, a.cursor)
    d_up, d_dn = hi - Cc, Cc - lo
    ratio = np.divide(d_up, d_up + d_dn, out=np.full(n, np.nan),
                      where=(d_up + d_dn) > 0)
    b = np.minimum(d_up, d_dn)
    fu, fd = first_touch(c, a.cursor, Cc + b, Cc - b)
    up = (fu >= 0) & ((fd < 0) | (fu < fd))
    dn = (fd >= 0) & ((fu < 0) | (fd < fu))
    decided = (up ^ dn) & np.isfinite(ratio) & train
    y = up.astype(float)
    bins = np.clip(np.digitize(ratio, np.linspace(0, 1, a.bins + 1)) - 1,
                   0, a.bins - 1)

    # ---- прочтение 2: подписи событий ------------------------------------
    ords = np.arange(0, c['ohlc'].shape[1])
    ids = [f'{a.instrument}:{int(t)}' for t in c['t0']]
    corp = Corpus(c['ohlc'], ords, list(ids), list(ids),
                  {'instrument': a.instrument, 'source': {'population': 'RIZ'},
                   'anchor': 'T0 = 0', 'unit_definition': 'один T0'})
    ev = declared_events(corp, a.cursor)
    names = [e.name for e in ev]
    known_at = [e.known_at for e in ev]
    vals = np.stack([e.values for e in ev])
    sig = signature(vals)
    key = keys_of(sig)
    uk, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    groups = []
    for g in np.argsort(-cnt):
        m = inv == g
        if (m & train).sum() < a.min_support:
            continue
        r = residual(y, m, decided, bins, day, a.bins)
        if r is None:
            continue
        pr = projections(c, m & train)
        groups.append({'group': int(g), 'films_all': int(m.sum()),
                       'films_scout': int((m & train).sum()),
                       'days': pr['days'], 'years_covered': len(pr['years']),
                       'continuation_vs_position_control': r})
    groups.sort(key=lambda z: -abs(z['continuation_vs_position_control']
                                   ['t_clustered_by_day'] or 0))

    # ---- прочтение 1: те же семьи, то же различие -------------------------
    r1 = json.loads(Path(a.reading1).read_text(encoding='utf-8'))
    pts, recs = language(0, a.cursor)
    rel_names = [atom_name(x) for x in recs]
    P = Packed(c, recs, 0, a.cursor)
    idx = {nm: i for i, nm in enumerate(rel_names)}
    fam = []
    for f in r1['families']:
        m = np.ones(n, bool)
        for rel in f['relations']:
            i = idx[rel]
            m &= P.bool_block(i, i + 1)[:, 0]
        m &= train
        r = residual(y, m, decided, bins, day, a.bins)
        # во сколько подписей рассыпается семья
        kk = inv[m & decided]
        u2, c2 = np.unique(kk, return_counts=True)
        fam.append({'id': f['id'], 'relations': f['relations'],
                    'films_scout': int(m.sum()),
                    'continuation_vs_position_control': r,
                    'event_signatures_spanned': int(len(u2)),
                    'largest_signature_share': float(c2.max() / c2.sum())
                    if len(c2) else None})

    res = {'instrument': a.instrument, 'cursor': a.cursor,
           'films': int(n), 'scout_films': int(train.sum()),
           'declared_event_dictionary': [
               {'event': nm, 'known_at': int(k)} for nm, k in zip(names, known_at)],
           'signature_definition': 'наступление события (NONE/UNKNOWN) + '
                                   'попарный порядок с сохранением совпадений; '
                                   'номера минут и промежутки не входят',
           'signatures_total': int(len(uk)),
           'signatures_above_min_support': len(groups),
           'largest_signature_films': int(cnt.max()),
           'measured_difference': 'гонка симметричных барьеров b = min(вверх,вниз) '
                                  'после курсора, контроль по положению закрытия '
                                  'внутри диапазона префикса, ошибка кластерная по дням',
           'reading2_groups': groups[:25],
           'reading1_families': fam,
           'implementation_sha256': hashlib.sha256(
               Path(__file__).read_bytes()).hexdigest()}
    out = Path(a.out or f'work/070/two_readings_{a.instrument}_c{a.cursor}.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str),
                   encoding='utf-8')
    print(f'подписей всего {len(uk)}, с опорой >= {a.min_support}: {len(groups)}; '
          f'крупнейшая {int(cnt.max())} фильмов')
    print('--- прочтение 2, по величине |t|')
    for g in groups[:10]:
        r = g['continuation_vs_position_control']
        print(f"  группа {g['group']:>6} n={g['films_scout']:>6} дн={g['days']:>4} "
              f"разница {r['difference']:+.4f} t={r['t_clustered_by_day']:+.2f}")
    print('--- прочтение 1, те же семьи')
    for f in fam:
        r = f['continuation_vs_position_control']
        if r is None:
            continue
        print(f"  {f['id']:>4} n={f['films_scout']:>6} разница {r['difference']:+.4f} "
              f"t={r['t_clustered_by_day']:+.2f} подписей={f['event_signatures_spanned']} "
              f"крупнейшая доля={f['largest_signature_share']:.3f}")


if __name__ == '__main__':
    main()
