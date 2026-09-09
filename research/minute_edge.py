#!/usr/bin/env python3
"""Ближайшая исполнимая минута: есть ли наблюдаемое состояние с направлением.

РЫНОЧНЫЙ ВОПРОС
===============
Накопленный результат проекта: направление в минутной ленте живёт около
аукционов, а вне их лента ведёт себя как честная игра после расходов. Аукцион
разрешает дисбаланс: цена выходит за границу области, и её либо принимают
снаружи, либо отвергают. Вопрос цикла: **существует ли состояние, наблюдаемое
на закрытии минуты, после которого ближайшая исполнимая минута идёт в одну
сторону настолько, чтобы окупить оборот.**

Это другой вопрос, чем у 064. Там вход оказывался вплотную к опоре, которую
форма же и предсказывала пройти, и различался порядок барьеров, а не наклон
пути. Здесь исход — ход за фиксированное время от исполнимой цены, барьера
рядом нет, и артефакт близости повториться не может.

ТЕРРИТОРИЯ, ОБЪЯВЛЕННАЯ ДО СЧЁТА
================================
NQ, минутная лента, 2020-01-01 … 2025-10-31. Кусок 2025-11-01 … 2026-05-04 при
поиске не открывается и остаётся резервом. Сцена — минуты основной сессии XNYS
(календарь `setups/S-04/calendar.parquet`), рыночное основание выбора: это
единственный участок суток с непрерывным аукционным потоком, и весь прежний
положительный результат проекта лежит в нём. Единица — минута k как момент
возможного решения. Связь с RIZ: **прямой нет.** Поле RIZ здесь не читается,
результат ничего сам по себе не устанавливает о RIZ.

ПРОДОЛЖЕНИЯ, ОБЪЯВЛЕННЫЕ ДО СЧЁТА
=================================
Вход `open k+1`. Исходы: R1 = `open k+2` − `open k+1` (ближайшая исполнимая
минута), R5 = `open k+6` − `open k+1`, R15 = `open k+16` − `open k+1`. Все три
цены исполнимы. Каждый исход ищется отдельно и своим перебором.

СЕМЕЙСТВА УСЛОВИЙ И ПРЕДЕЛ ПОИСКА
=================================
Только порядковые отношения O/H/L/C на окне [k-4..k], доступные на закрытии
минуты k: направление тела, отношения закрытий и экстремумов к чужим свечам,
закрытие против префиксных экстремумов, сравнение длин тела и теней (отношение,
не величина), плюс объявленный контекст стадии сессии. ATR, размеры свечей и
Volume в определение условия не входят. Каждый атом берётся вместе со своим
отрицанием. Перебор — ВСЕ одиночные и ВСЕ парные сочетания, полностью, без
выборки; предел объявляется числом сочетаний в отчёте. Тройки в перебор не
входят — это ограничение покрытия, не знание.

ЧТО ИМЕННО КАЛИБРУЕТСЯ
======================
Вместе с сочетанием подбирается и сторона (знак среднего). Поэтому при
разрушенной связи повторяется ВЕСЬ выбор целиком, включая выбор стороны.
Обмен меток здесь не обоснован содержательно, статус результата — `DIAGNOSTIC`:
это потолок шума перебора, а не проверка гипотезы. Порог опоры меняется
объявленной лесенкой: чем выше опора, тем ниже потолок шума и тем меньше
обнаружимый эффект.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from calendar_utils import year_of, quarter_of, date_key  # noqa: E402

from candidate_check import POINT, COST                            # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MIN = 60_000_000_000
BACK = 4          # окно наблюдения [k-4..k]
AHEAD = 16        # нужны open k+1 … open k+16


# ------------------------------------------------------------ территория ----

def _session_index(ts):
    cal = pd.read_parquet(ROOT / 'setups/S-04/calendar.parquet')
    inside = np.zeros(len(ts), bool)
    start = np.full(len(ts), -1, np.int64)
    o = np.searchsorted(ts, cal.open_ns.to_numpy())
    c = np.searchsorted(ts, cal.close_ns.to_numpy())
    for s, e in zip(o, c):
        if s < len(ts) and e < len(ts) and e > s:
            inside[s:e + 1] = True
            start[s:e + 1] = s
    return inside, start


def load(instrument, start_date, stop_date):
    m = {k: np.load(ROOT / 'data/market' / instrument / (k + '.npy'))
         for k in ('close_ts_utc_ns', 'open', 'high', 'low', 'close')}
    ts = m['close_ts_utc_ns']
    inside, sess_start = _session_index(ts)
    a = np.datetime64(start_date, 'ns').astype('int64')
    b = np.datetime64(stop_date, 'ns').astype('int64')
    in_range = (ts >= a) & (ts < b)
    step_ok = np.zeros(len(ts), bool)
    step_ok[1:] = (ts[1:] - ts[:-1]) == MIN
    run = np.zeros(len(ts), np.int32)
    for i in range(1, len(ts)):
        run[i] = run[i - 1] + 1 if step_ok[i] else 0
    idx = np.arange(len(ts))
    back_ok = np.zeros(len(ts), bool)
    good = (idx >= BACK) & (idx + AHEAD < len(ts))
    back_ok[good] = run[idx[good]] >= BACK
    cont = np.ones(len(ts), bool)
    for j in range(1, AHEAD + 1):
        s = np.zeros(len(ts), bool)
        s[:len(ts) - j] = step_ok[j:len(ts)]
        cont &= s
    sel = back_ok & cont & in_range & inside
    return m, np.flatnonzero(sel), sess_start


def session_stage(sess_start, k):
    """Стадия аукциона — объявленный контекст с рыночным основанием.

    Не отдельная минута суток (сплошной счёт по минутам уже сделан в S-09), а
    крупные стадии сессии: открытие, ранняя часть, середина, закрытие.
    """
    since = k - sess_start[k]
    return {'сессия: первые 5 минут': (since >= 0) & (since < 5),
            'сессия: первые 30 минут': (since >= 0) & (since < 30),
            'сессия: последние 30 минут': since >= 360,
            'сессия: середина': (since >= 60) & (since < 330)}


# ------------------------------------------------------------------ атомы ----

def atoms(m, k, stage=None):
    """Порядковые отношения на [k-4..k]. Возвращает (матрица bool, имена)."""
    O, H, L, C = m['open'], m['high'], m['low'], m['close']
    o = [O[k - i] for i in range(BACK + 1)]
    h = [H[k - i] for i in range(BACK + 1)]
    lo = [L[k - i] for i in range(BACK + 1)]
    c = [C[k - i] for i in range(BACK + 1)]
    cols, names = [], []

    def add(v, name):
        cols.append(np.asarray(v, dtype=bool))
        names.append(name)

    for i in range(BACK + 1):
        add(c[i] > o[i], f'body_up[-{i}]')
    for i in range(BACK):
        for j in range(i + 1, BACK + 1):
            add(c[i] > c[j], f'C[-{i}]>C[-{j}]')
            add(h[i] > h[j], f'H[-{i}]>H[-{j}]')
            add(lo[i] < lo[j], f'L[-{i}]<L[-{j}]')
    for j in range(1, BACK + 1):
        add(c[0] > h[j], f'C[0]>H[-{j}]')
        add(c[0] < lo[j], f'C[0]<L[-{j}]')
    for w in (2, 3, 4):
        ph = np.max(np.stack(h[1:w + 1]), axis=0)
        pl = np.min(np.stack(lo[1:w + 1]), axis=0)
        add(c[0] > ph, f'C[0]>maxH[-{w}..-1]')
        add(c[0] < pl, f'C[0]<minL[-{w}..-1]')
        add(h[0] > ph, f'H[0]>maxH[-{w}..-1]')
        add(lo[0] < pl, f'L[0]<minL[-{w}..-1]')
    for i in (0, 1):
        top = np.maximum(o[i], c[i])
        bot = np.minimum(o[i], c[i])
        up, dn, body = h[i] - top, bot - lo[i], np.abs(c[i] - o[i])
        add(up > body, f'upper_wick>body[-{i}]')
        add(dn > body, f'lower_wick>body[-{i}]')
        add(up > dn, f'upper_wick>lower_wick[-{i}]')
    add(o[0] > c[1], 'O[0]>C[-1]')
    if stage is not None:
        for name, v in stage.items():
            add(v, name)
    return np.stack(cols, axis=1), names


def with_negations(B, names):
    return (np.concatenate([B, ~B], axis=1),
            names + ['не ' + n for n in names])


# ---------------------------------------------------------------- перебор ----

def enumerate_all(Bf, y, min_support):
    """Полный точный перебор одиночных и парных сочетаний. Критерий — |среднее|."""
    yf = y.astype(np.float32)
    cnt1 = Bf.sum(axis=0)
    sum1 = (Bf * yf[:, None]).sum(axis=0)
    cnt2 = Bf.T @ Bf
    sum2 = (Bf * yf[:, None]).T @ Bf
    with np.errstate(invalid='ignore', divide='ignore'):
        m1 = np.where(cnt1 >= min_support, sum1 / np.maximum(cnt1, 1), np.nan)
        m2 = np.where(cnt2 >= min_support, sum2 / np.maximum(cnt2, 1), np.nan)
    iu = np.triu_indices(Bf.shape[1], k=1)
    return m1, cnt1, m2, cnt2, iu


def best_of(m1, cnt1, m2, cnt2, iu):
    a1, a2 = np.abs(m1), np.abs(m2[iu])
    b1 = np.nanmax(a1) if np.isfinite(a1).any() else -np.inf
    b2 = np.nanmax(a2) if np.isfinite(a2).any() else -np.inf
    if b2 >= b1:
        p = int(np.nanargmax(a2))
        i, j = int(iu[0][p]), int(iu[1][p])
        return {'kind': 'pair', 'atoms': (i, j), 'mean': float(m2[i, j]),
                'n': int(cnt2[i, j]), 'score': float(b2)}
    i = int(np.nanargmax(a1))
    return {'kind': 'single', 'atoms': (i,), 'mean': float(m1[i]),
            'n': int(cnt1[i]), 'score': float(b1)}


def calibrate(Bf, y, min_support, repeats, seed):
    """Разрушенная связь: повторяется весь выбор, включая выбор стороны."""
    rng = np.random.default_rng(seed)
    cnt1, cnt2 = Bf.sum(axis=0), Bf.T @ Bf
    iu = np.triu_indices(Bf.shape[1], k=1)
    ok1, ok2 = cnt1 >= min_support, cnt2 >= min_support
    out = []
    for _ in range(repeats):
        z = rng.permutation(y).astype(np.float32)
        s1 = (Bf * z[:, None]).sum(axis=0)
        s2 = (Bf * z[:, None]).T @ Bf
        with np.errstate(invalid='ignore', divide='ignore'):
            v1 = np.abs(np.where(ok1, s1 / np.maximum(cnt1, 1), np.nan))
            v2 = np.abs(np.where(ok2, s2 / np.maximum(cnt2, 1), np.nan))
        f1 = np.nanmax(v1) if np.isfinite(v1).any() else -np.inf
        f2 = np.nanmax(v2[iu]) if np.isfinite(v2[iu]).any() else -np.inf
        out.append(float(max(f1, f2)))
    return out


# ----------------------------------------------------------------- деньги ----

def stream(m, k, mask, side, instrument, hold):
    """Исполнимый поток: вход open k+1, выход open k+1+hold, одна позиция."""
    ts, O = m['close_ts_utc_ns'], m['open']
    p = POINT[instrument]
    busy, trades = -1, []
    for i in k[mask]:
        t_in = int(ts[i + 1])
        if t_in <= busy:
            continue
        trades.append((t_in, float(side * (O[i + 1 + hold] - O[i + 1]) * p)))
        busy = int(ts[i + hold])
    return trades


def summarise(trades, cost):
    if not trades:
        return {'n': 0}
    g = np.array([t[1] for t in trades])
    tsx = np.array([t[0] for t in trades])
    year = year_of(tsx)
    net = g - cost
    eq = np.cumsum(net)
    sem = float(g.std(ddof=1) / np.sqrt(len(g)))
    by = {int(y): float(net[year == y].sum()) for y in np.unique(year)}
    return {'n': len(g), 'mean_gross': float(g.mean()), 'sem_gross': sem,
            't_gross': float(g.mean() / sem) if sem else None,
            'median_gross': float(np.median(g)),
            'share_positive': float((g > 0).mean()),
            'mean_net': float(net.mean()), 'total_net': float(net.sum()),
            'drawdown_net': float(np.max(np.maximum.accumulate(eq) - eq)),
            'breakeven_cost_usd': float(g.mean()),
            'by_year_net': by,
            'positive_years': int(sum(v > 0 for v in by.values()))}


# ------------------------------------------------------------------- main ----

def describe(names, cand):
    return ' & '.join(names[i] for i in cand['atoms'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--instrument', default='NQ')
    ap.add_argument('--start', default='2020-01-01')
    ap.add_argument('--stop', default='2025-11-01')
    ap.add_argument('--supports', nargs='+', type=int,
                    default=[2000, 8000, 25000])
    ap.add_argument('--repeats', type=int, default=30)
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    m, k, sess_start = load(a.instrument, a.start, a.stop)
    O = m['open']
    p, cost = POINT[a.instrument], COST[a.instrument]
    series = {'R1': (O[k + 2] - O[k + 1]) * p,
              'R5': (O[k + 6] - O[k + 1]) * p,
              'R15': (O[k + 16] - O[k + 1]) * p}
    holds = {'R1': 1, 'R5': 5, 'R15': 15}
    B, names = atoms(m, k, session_stage(sess_start, k))
    B, names = with_negations(B, names)
    Bf = B.astype(np.float32)
    A = B.shape[1]
    n_pairs = A * (A - 1) // 2

    scan, champion = [], None
    for label, y in series.items():
        for ms in a.supports:
            m1, c1, m2, c2, iu = enumerate_all(Bf, y, ms)
            best = best_of(m1, c1, m2, c2, iu)
            drawn = calibrate(Bf, y, ms, a.repeats, a.seed)
            beats = sum(1 for v in drawn if v >= best['score'])
            order2 = np.argsort(-np.nan_to_num(np.abs(m2[iu]), nan=-1))
            row = {'outcome': label, 'min_support': ms,
                   'best_text': describe(names, best),
                   'best_mean_gross_usd': best['mean'], 'best_n': best['n'],
                   'best_side': 'long' if best['mean'] > 0 else 'short',
                   'calibration_max': max(drawn),
                   'calibration_median': float(np.median(drawn)),
                   'repeats_not_worse': beats,
                   'survives_calibration': bool(beats == 0),
                   'clears_cost': bool(abs(best['mean']) > cost),
                   'top5_pairs': [
                       {'text': f'{names[int(iu[0][q])]} & {names[int(iu[1][q])]}',
                        'mean_gross': float(m2[int(iu[0][q]), int(iu[1][q])]),
                        'n': int(c2[int(iu[0][q]), int(iu[1][q])])}
                       for q in order2[:5]]}
            scan.append(row)
            if row['survives_calibration'] and row['clears_cost']:
                if champion is None or abs(best['mean']) > abs(champion[1]['mean']):
                    champion = (label, best, ms)

    if champion is None:
        pick = max(scan, key=lambda r: abs(r['best_mean_gross_usd']))
        label, ms = pick['outcome'], pick['min_support']
        m1, c1, m2, c2, iu = enumerate_all(Bf, series[label], ms)
        best = best_of(m1, c1, m2, c2, iu)
    else:
        label, best, ms = champion

    sel = B[:, best['atoms'][0]].copy()
    for i in best['atoms'][1:]:
        sel &= B[:, i]
    side = 1 if best['mean'] > 0 else -1
    runs = {}
    for lbl, hold in holds.items():
        runs[lbl] = summarise(stream(m, k, sel, side, a.instrument, hold), cost)
        runs[lbl]['hold_minutes'] = hold
        runs[lbl]['mean_gross_all_films'] = float((side * series[lbl][sel]).mean())
        runs[lbl]['n_films'] = int(sel.sum())

    result = {
        'implementation_sha256': hashlib.sha256(
            Path(__file__).read_bytes()).hexdigest(),
        'territory': {
            'instrument': a.instrument, 'start': a.start, 'stop': a.stop,
            'scene': 'минуты основной сессии XNYS, календарь setups/S-04',
            'riz_link': 'прямой связи с RIZ нет; поле RIZ не читалось',
            'decision_minutes': int(len(k)),
            'reserve': '2025-11-01 … 2026-05-04 при поиске не открывался'},
        'search_limit': {'atoms_with_negations': int(A), 'singles': int(A),
                         'pairs': int(n_pairs),
                         'mode': 'полный точный перебор, без выборки',
                         'triples': 'в перебор не входят — ограничение покрытия',
                         'min_support_levels': list(a.supports),
                         'total_combinations_per_cell': int(A + n_pairs)},
        'outcomes': {'R1': 'open k+2 − open k+1', 'R5': 'open k+6 − open k+1',
                     'R15': 'open k+16 − open k+1', 'entry': 'open k+1',
                     'point_value': p, 'cost_per_turn': cost},
        'baseline': {'mean_r1_usd': float(series['R1'].mean()),
                     'sd_r1_usd': float(series['R1'].std(ddof=1)),
                     'mean_abs_r1_usd': float(np.abs(series['R1']).mean()),
                     'median_abs_r1_usd': float(np.median(np.abs(series['R1']))),
                     'sd_r15_usd': float(series['R15'].std(ddof=1))},
        'scan': scan,
        'calibration_note': {'status': 'DIAGNOSTIC',
                             'exchangeability_justified': False,
                             'repeats': a.repeats,
                             'what_is_repeated': 'весь выбор целиком, включая '
                                                 'выбор стороны по знаку среднего'},
        'champion_found': champion is not None,
        'shown': {'outcome': label, 'min_support': ms,
                  'text': describe(names, best), 'kind': best['kind'],
                  'mean_gross_usd': best['mean'], 'n': best['n'],
                  'side': 'long' if side > 0 else 'short'},
        'best_streams': runs}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, ensure_ascii=False, indent=1,
                                      default=str), encoding='utf-8')
    print(json.dumps({'minutes': len(k), 'atoms': A, 'pairs': n_pairs,
                      'champion': champion is not None,
                      'scan': [{q: r[q] for q in
                                ('outcome', 'min_support', 'best_mean_gross_usd',
                                 'best_n', 'calibration_max',
                                 'survives_calibration', 'clears_cost')}
                               for r in scan],
                      'shown': result['shown'],
                      'R1': {kk: runs['R1'][kk] for kk in
                             ('n', 'mean_gross', 't_gross', 'mean_net')}},
                     ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
