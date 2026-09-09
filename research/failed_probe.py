#!/usr/bin/env python3
"""Отвергнутая вылазка за край сессии: наложение фильмов, затем правило.

РЫНОЧНЫЙ ВОПРОС
===============
Аукцион ищет приёмку. Цена выходит за крайнюю цену сессии, и дальше возможны
две вещи: снаружи её принимают, и вылазка становится ходом, либо не принимают,
и она возвращается внутрь. Сцена этого прохода — **вылазка, отвергнутая в ту же
минуту**: новый крайний уровень сессии поставлен тенью, а закрытие вернулось
внутрь прежнего диапазона. Механизм назван до счёта: неудача найти приёмку
снаружи — свидетельство против стороны вылазки.

Это не улучшение сопровождения 064 и не новая версия S-09. Сцена другая,
сторона выводится из направления отвергнутой вылазки, длительность —
из наложения фильмов, а не назначается.

ТЕРРИТОРИЯ, ОБЪЯВЛЕННАЯ ДО СЧЁТА
================================
NQ, минуты основной сессии XNYS, 2020-01-01 … 2025-10-31; кусок
2025-11-01 … 2026-05-04 при поиске не открывается. Связь с RIZ прямой нет:
поле RIZ не читается. Единица — одна вылазка; повторные вылазки одной сессии
сохраняются и учитываются отдельно (стадия сцены — часть описания).

ПОРЯДОК РАБОТЫ
==============
`--stage overlay` — наложение всех подходящих фильмов по общему якорю: куда
доходит цена от исполнимого входа, на какой минуте ставится экстремум, когда
ожидание перестаёт сбываться. Числа правила читаются отсюда.
`--stage rule` — исторический расчёт составного правила с расходами, правилом
одной позиции, пропусками и календарём.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from candidate_check import POINT, COST                            # noqa: E402
from minute_edge import _session_index, ROOT, MIN                  # noqa: E402


def scenes(m, start_date, stop_date, min_into=5, look=60):
    """Отвергнутые вылазки. Всё известно на закрытии минуты k."""
    ts = m['close_ts_utc_ns']
    O, H, L, C = m['open'], m['high'], m['low'], m['close']
    inside, sess_start = _session_index(ts)
    a = np.datetime64(start_date, 'ns').astype('int64')
    b = np.datetime64(stop_date, 'ns').astype('int64')
    step = np.zeros(len(ts), bool)
    step[1:] = (ts[1:] - ts[:-1]) == MIN
    out = []
    idx = np.flatnonzero(inside & (ts >= a) & (ts < b))
    # границы сессий
    bounds = {}
    for s in np.unique(sess_start[idx]):
        if s < 0:
            continue
        e = s
        while e + 1 < len(ts) and inside[e + 1] and sess_start[e + 1] == s:
            e += 1
        bounds[int(s)] = int(e)
    for k in idx:
        s = int(sess_start[k])
        if s < 0:
            continue
        since = k - s
        if since < min_into:
            continue
        e = bounds[s]
        if k + 1 > e:
            continue
        prev_hi = H[s:k].max()
        prev_lo = L[s:k].min()
        up = (H[k] > prev_hi) and (C[k] < prev_hi)
        dn = (L[k] < prev_lo) and (C[k] > prev_lo)
        if up == dn:
            continue
        if not np.all(step[k - min_into + 1:k + 2]):
            continue
        side = -1 if up else 1          # отвергнутая вылазка вверх → короткая
        stop_at = min(k + look, e)
        out.append({'k': int(k), 'session_start': s, 'session_end': e,
                    'since_open': int(since), 'side': side,
                    'entry_index': int(k + 1), 'limit_index': int(stop_at),
                    'probe_extreme': float(H[k] if up else L[k]),
                    'prev_edge': float(prev_hi if up else prev_lo),
                    'ts': int(ts[k])})
    return out


def overlay(m, rows, instrument, look=60):
    """Наложение фильмов по общему якорю: что происходит после входа."""
    O, H, L, C = m['open'], m['high'], m['low'], m['close']
    p = POINT[instrument]
    rec = []
    for r in rows:
        i0, lim, s = r['entry_index'], r['limit_index'], r['side']
        if i0 > lim:
            continue
        e = float(O[i0])
        span = np.arange(i0, lim + 1)
        fav = s * (H[span] - e) * p if s > 0 else s * (L[span] - e) * p
        adv = s * (L[span] - e) * p if s > 0 else s * (H[span] - e) * p
        mfe_run = np.maximum.accumulate(fav)
        mae_run = np.minimum.accumulate(adv)
        mfe, mae = float(mfe_run[-1]), float(mae_run[-1])
        t_mfe = int(np.argmax(fav)) + 1
        t_mae = int(np.argmin(adv)) + 1
        # возврат к краю, из-за которого вылазка названа отвергнутой
        edge = r['prev_edge']
        back = np.flatnonzero((C[span] < edge) if s < 0 else (C[span] > edge))
        # нарушение ожидания: закрытие за экстремумом вылазки
        ext = r['probe_extreme']
        brk_c = np.flatnonzero((C[span] > ext) if s < 0 else (C[span] < ext))
        brk_w = np.flatnonzero((H[span] > ext) if s < 0 else (L[span] < ext))
        rec.append({'ts': r['ts'], 'side': s, 'since_open': r['since_open'],
                    'minutes_available': len(span),
                    'mfe': mfe, 'mae': mae, 't_mfe': t_mfe, 't_mae': t_mae,
                    'r5': float(s * (O[min(i0 + 5, lim)] - e) * p),
                    'r15': float(s * (O[min(i0 + 15, lim)] - e) * p),
                    'r30': float(s * (O[min(i0 + 30, lim)] - e) * p),
                    'to_limit': float(s * (C[lim] - e) * p),
                    'minute_hold_edge': int(back[0]) + 1 if len(back) else -1,
                    'break_close': int(brk_c[0]) + 1 if len(brk_c) else -1,
                    'break_wick': int(brk_w[0]) + 1 if len(brk_w) else -1})
    return rec


def q(x, ps=(5, 10, 25, 50, 75, 90, 95)):
    if len(x) == 0:
        return {'n': 0}
    v = np.percentile(x, ps)
    d = {'n': int(len(x)), 'mean': float(np.mean(x))}
    d.update({('median' if pp == 50 else f'q{pp:02d}'): float(vv)
              for pp, vv in zip(ps, v)})
    return d


def overlay_report(rec, instrument):
    cost = COST[instrument]
    mfe = np.array([r['mfe'] for r in rec])
    mae = np.array([r['mae'] for r in rec])
    tmf = np.array([r['t_mfe'] for r in rec], float)
    bw = np.array([r['break_wick'] for r in rec])
    bc = np.array([r['break_close'] for r in rec])
    since = np.array([r['since_open'] for r in rec])
    out = {'n': len(rec), 'cost_per_turn': cost,
           'mfe_usd': q(mfe), 'mae_usd': q(mae),
           'minute_of_mfe': q(tmf),
           'share_mfe_over_cost': float((mfe > cost).mean()),
           'share_mfe_over_2cost': float((mfe > 2 * cost).mean()),
           'r5_usd': q(np.array([r['r5'] for r in rec])),
           'r15_usd': q(np.array([r['r15'] for r in rec])),
           'r30_usd': q(np.array([r['r30'] for r in rec])),
           'to_limit_usd': q(np.array([r['to_limit'] for r in rec])),
           'break_by_wick_share': float((bw > 0).mean()),
           'break_by_wick_minute': q(bw[bw > 0].astype(float)),
           'break_by_close_share': float((bc > 0).mean()),
           'break_by_close_minute': q(bc[bc > 0].astype(float)),
           'by_stage': {}}
    for lo, hi, name in ((5, 30, 'первые 30 минут'), (30, 90, '30–90'),
                         (90, 240, '90–240'), (240, 10 ** 6, 'после 240')):
        s = (since >= lo) & (since < hi)
        if s.sum():
            out['by_stage'][name] = {
                'n': int(s.sum()),
                'r5_mean': float(np.mean([r['r5'] for r, ok in zip(rec, s) if ok])),
                'r15_mean': float(np.mean([r['r15'] for r, ok in zip(rec, s) if ok])),
                'r30_mean': float(np.mean([r['r30'] for r, ok in zip(rec, s) if ok])),
                'mfe_median': float(np.median(mfe[s]))}
    # альтернативное прочтение той же сцены: продолжение вместо отвержения
    out['mirror_side_r15_mean'] = float(-np.mean([r['r15'] for r in rec]))
    return out


def run_rule(m, rows, instrument, hold, stop_usd, target_usd, look=60):
    """Составное правило: вход open k+1, защитный предел, цель, предел времени."""
    O, H, L, C = m['open'], m['high'], m['low'], m['close']
    ts = m['close_ts_utc_ns']
    p, cost = POINT[instrument], COST[instrument]
    busy, trades = -1, []
    for r in rows:
        i0, lim, s = r['entry_index'], r['limit_index'], r['side']
        if i0 > lim:
            continue
        if int(ts[i0]) <= busy:
            continue
        e = float(O[i0])
        last = min(i0 + hold - 1, lim)
        why, px, out_i = 'предел времени', float(O[min(last + 1, lim)]), last
        for j in range(i0, last + 1):
            fav = s * (H[j] - e) * p if s > 0 else s * (L[j] - e) * p
            adv = s * (L[j] - e) * p if s > 0 else s * (H[j] - e) * p
            hit_stop = stop_usd is not None and adv <= -stop_usd
            hit_tgt = target_usd is not None and fav >= target_usd
            if hit_stop and hit_tgt:
                why, px, out_i = 'предел и цель в одной минуте', None, j
                break
            if hit_stop:
                why, px, out_i = 'защитный предел', None, j
                break
            if hit_tgt:
                why, px, out_i = 'цель', None, j
                break
        if px is None:
            gross = -stop_usd if why != 'цель' else target_usd
        else:
            gross = float(s * (px - e) * p)
        trades.append({'ts': int(ts[i0]), 'gross': gross, 'why': why,
                       'minutes': out_i - i0 + 1, 'since_open': r['since_open']})
        busy = int(ts[out_i])
    return trades


def summarise(trades, cost):
    if not trades:
        return {'n': 0}
    g = np.array([t['gross'] for t in trades])
    tsx = np.array([t['ts'] for t in trades])
    year = (tsx // MIN) // int(365.2425 * 1440) + 1970
    net = g - cost
    eq = np.cumsum(net)
    sem = float(g.std(ddof=1) / np.sqrt(len(g)))
    by = {int(y): float(net[year == y].sum()) for y in np.unique(year)}
    from collections import Counter
    return {'n': len(g), 'mean_gross': float(g.mean()), 'sem_gross': sem,
            't_gross': float(g.mean() / sem) if sem else None,
            'mean_net': float(net.mean()), 'total_net': float(net.sum()),
            'median_gross': float(np.median(g)),
            'share_positive': float((g > 0).mean()),
            'breakeven_cost_usd': float(g.mean()),
            'drawdown_net': float(np.max(np.maximum.accumulate(eq) - eq)),
            'mean_minutes': float(np.mean([t['minutes'] for t in trades])),
            'by_reason': dict(Counter(t['why'] for t in trades)),
            'by_year_net': by,
            'positive_years': int(sum(v > 0 for v in by.values())),
            'years': len(by)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--instrument', default='NQ')
    ap.add_argument('--start', default='2020-01-01')
    ap.add_argument('--stop', default='2025-11-01')
    ap.add_argument('--look', type=int, default=60)
    ap.add_argument('--stage', choices=['overlay', 'rule'], default='overlay')
    ap.add_argument('--hold', type=int, default=15)
    ap.add_argument('--stop-usd', type=float, default=None)
    ap.add_argument('--target-usd', type=float, default=None)
    ap.add_argument('--variants', default=None,
                    help='JSON-список вариантов правила для стадии rule')
    ap.add_argument('--out', required=True)
    ap.add_argument('--films-csv', default=None)
    a = ap.parse_args()

    m = {k: np.load(ROOT / 'data/market' / a.instrument / (k + '.npy'))
         for k in ('close_ts_utc_ns', 'open', 'high', 'low', 'close')}
    rows = scenes(m, a.start, a.stop, look=a.look)
    head = {'implementation_sha256': hashlib.sha256(
                Path(__file__).read_bytes()).hexdigest(),
            'territory': {'instrument': a.instrument, 'start': a.start,
                          'stop': a.stop,
                          'scene': 'отвергнутая вылазка за край сессии XNYS',
                          'riz_link': 'прямой связи с RIZ нет',
                          'look_minutes': a.look},
            'scene_rule': 'H[k] выше прежнего максимума сессии и C[k] ниже него '
                          '(зеркально для низа); всё известно на закрытии k',
            'side': 'против направления отвергнутой вылазки',
            'scenes_found': len(rows)}
    if a.stage == 'overlay':
        rec = overlay(m, rows, a.instrument, a.look)
        head['overlay'] = overlay_report(rec, a.instrument)
        if a.films_csv:
            with open(a.films_csv, 'w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=list(rec[0]))
                w.writeheader()
                w.writerows(rec)
    else:
        cost = COST[a.instrument]
        variants = (json.loads(a.variants) if a.variants
                    else [{'hold': a.hold, 'stop_usd': a.stop_usd,
                           'target_usd': a.target_usd}])
        head['variants'] = []
        for v in variants:
            tr = run_rule(m, rows, a.instrument, v['hold'], v.get('stop_usd'),
                          v.get('target_usd'), a.look)
            head['variants'].append({'params': v, 'result': summarise(tr, cost)})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(head, ensure_ascii=False, indent=1,
                                      default=str), encoding='utf-8')
    print(json.dumps({k: v for k, v in head.items()
                      if k in ('scenes_found', 'variants')} or {},
                     ensure_ascii=False, indent=1)[:3000])


if __name__ == '__main__':
    main()
