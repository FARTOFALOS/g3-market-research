#!/usr/bin/env python3
"""Экономическая проверка кандидата, выросшего из порядковой находки.

ЭТО ОТДЕЛЬНЫЙ СЛЕДУЮЩИЙ РАСЧЁТ, А НЕ ЧАСТЬ НАХОДКИ
==================================================
Порядковая находка сформулирована и записана до этого файла. Здесь считаются
фактические цены исполнения, расходы, доступное движение и переживаемый убыток.
Расчёт МОЖЕТ отвергнуть кандидата. Он не имеет права задним числом переписать
порядковую находку так, будто её обнаружили уже с денежным фильтром: ни один
параметр правила здесь не подбирается по деньгам, кроме защитного предела,
который снимается с трафарета хода против входа и так и называется.

ПРАВИЛО ЦЕЛИКОМ
===============
Решение принимается на минуте T0+5 фильма RIZ. Сторона берётся зафиксированной
порядковой конструкцией:
  продажа, если lastupdateH[1..5] < argminL_last[1..5];
  покупка,  если argmaxH_last[1..5] > lastupdateL[1..5].
Если истинны обе или ни одной — решения нет.
Вход по open минуты T0+6. Выход на закрытии T0+30, то есть на границе ОБЪЯВЛЕННОГО
бюджета наблюдения, а не на выбранном горизонте сделки. Пропуск минуты означает
неизвестное: сделка не открывается и не переносится.
Одна позиция одновременно: пересекающиеся окна не удваивают риск.
Защитный предел снимается с трафарета: p90 хода против входа у прибыльных сделок.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from calendar_utils import year_of, quarter_of, date_key  # noqa: E402

from relational_stencil import Corpus                       # noqa: E402
from ordinal_events import extremum_event, last_update_event  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MIN = 60_000_000_000
CURSOR, HORIZON = 5, 30
POINT = {'ES': 50., 'NQ': 20., 'YM': 5.}
COST = {'ES': 30., 'NQ': 15., 'YM': 15.}


def tape(instrument):
    return {k: np.load(ROOT / 'data/market' / instrument / (k + '.npy'))
            for k in ('close_ts_utc_ns', 'open', 'high', 'low', 'close')}


def sides(corpus):
    """Сторона по зафиксированным конструкциям; спорные случаи — без решения."""
    uh = last_update_event(corpus, 1, CURSOR, 'H')
    ul = last_update_event(corpus, 1, CURSOR, 'L')
    mh = extremum_event(corpus, 1, CURSOR, 'H', True)
    ml = extremum_event(corpus, 1, CURSOR, 'L', True)
    known = ((uh.values >= 0) & (ul.values >= 0) &
             (mh.values >= 0) & (ml.values >= 0))
    sell = known & (uh.values < ml.values)
    buy = known & (mh.values > ul.values)
    side = np.zeros(corpus.n, dtype=np.int64)
    side[buy & ~sell] = 1
    side[sell & ~buy] = -1
    return side, known, int((buy & sell).sum())


def run(anchors, m, side, limit_candles=None, instrument='NQ', cost=None):
    """Исполнение с явным стопом и одной позицией одновременно."""
    ts = m['close_ts_utc_ns']
    cost = COST[instrument] if cost is None else cost
    pos = np.searchsorted(ts, anchors)
    order = np.argsort(anchors)
    busy_until = -1
    rows = []
    for i in order:
        if side[i] == 0:
            continue
        p = int(pos[i])
        entry_at = p + CURSOR + 1
        exit_at = p + HORIZON
        if entry_at >= len(ts) or exit_at >= len(ts):
            continue
        want = ts[p] + np.arange(CURSOR + 1, HORIZON + 1) * MIN
        at = np.searchsorted(ts, want)
        if at[-1] >= len(ts) or not np.all(ts[at] == want):
            continue
        if ts[at[0]] <= busy_until:
            continue
        s = int(side[i])
        entry = float(m['open'][at[0]])
        hi = m['high'][at]
        lo = m['low'][at]
        cl = m['close'][at]
        adverse = float(np.max(entry - lo) if s > 0 else np.max(hi - entry))
        exit_price = float(cl[-1])
        outcome = 'бюджет'
        if limit_candles is not None:
            stop = entry - s * limit_candles
            hit = np.flatnonzero((lo <= stop) if s > 0 else (hi >= stop))
            if len(hit):
                k = int(hit[0])
                exit_price = stop
                op = float(m['open'][at[k]])
                if (s > 0 and op < stop) or (s < 0 and op > stop):
                    exit_price = op
                outcome = 'предел'
        gross = s * (exit_price - entry) * POINT[instrument]
        rows.append({'ts': int(ts[at[0]]), 'side': s, 'entry': entry,
                     'exit': exit_price, 'outcome': outcome,
                     'adverse_points': adverse, 'gross': gross,
                     'net': gross - cost})
        busy_until = int(ts[at[-1]])
    return rows


def block(rows, m):
    if not rows:
        return {'n': 0}
    net = np.array([r['net'] for r in rows])
    ts = np.array([r['ts'] for r in rows])
    years = year_of(ts)
    by_year = {}
    for y in np.unique(years):
        by_year[int(y)] = float(net[years == y].sum())
    equity = np.cumsum(net)
    drawdown = float(np.max(np.maximum.accumulate(equity) - equity)) if len(equity) else 0.
    return {'n': len(rows), 'mean_net': float(net.mean()),
            'total_net': float(net.sum()),
            'positive_years': int(sum(v > 0 for v in by_year.values())),
            'years': len(by_year), 'by_year': by_year,
            'drawdown': drawdown,
            'closed_by_limit': sum(r['outcome'] == 'предел' for r in rows) / len(rows),
            'worst': float(net.min())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', nargs='+', required=True)
    ap.add_argument('--instrument', default='NQ')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    m = tape(a.instrument)
    anchors, side_all, ambiguous, units = [], [], 0, 0
    for path in a.inputs:
        c = Corpus.load(path)
        s, known, amb = sides(c)
        anchors.append(np.array([int(u.split(':')[1]) for u in c.unit_ids]))
        side_all.append(s)
        ambiguous += amb
        units += c.n
    anchors = np.concatenate(anchors)
    side_all = np.concatenate(side_all)
    keep = np.unique(anchors, return_index=True)[1]
    anchors, side_all = anchors[keep], side_all[keep]

    # трафарет: ход против входа у прибыльных сделок, БЕЗ предела
    raw = run(anchors, m, side_all, None, a.instrument)
    profitable = [r['adverse_points'] for r in raw if r['net'] > 0]
    limit = float(np.percentile(profitable, 90)) if profitable else None
    limited = run(anchors, m, side_all, limit, a.instrument)

    result = {
        'rule': {'decision_minute': f'T0+{CURSOR}',
                 'entry': f'open T0+{CURSOR + 1}',
                 'exit': f'close T0+{HORIZON} (граница объявленного бюджета)',
                 'side': 'зафиксированные порядковые конструкции',
                 'position': 'одна одновременно',
                 'cost_per_round': COST[a.instrument]},
        'units_seen': units, 'decisions_possible': int((side_all != 0).sum()),
        'ambiguous_both_constructions': ambiguous,
        'protective_limit_points': limit,
        'limit_source': 'p90 хода против входа у прибыльных сделок; это ограничение '
                        'убытка, а не ошибка идеи',
        'without_limit': block(raw, m),
        'with_limit': block(limited, m),
        'not_a_rewrite': 'порядковая находка сформулирована до этого расчёта и им '
                         'не переписывается'}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, ensure_ascii=False, indent=1),
                           encoding='utf-8')
    print(json.dumps({k: result[k] for k in
                      ('units_seen', 'decisions_possible', 'ambiguous_both_constructions',
                       'protective_limit_points', 'without_limit', 'with_limit')},
                     ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
