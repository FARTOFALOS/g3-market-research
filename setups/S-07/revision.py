"""S-07, ревизия v1r1: полное правило v1, исполненное по свечам без поблажек.

Зачем эта ревизия. Правило v1 целиком — катастрофический лимит 16 свечей,
выход на 15-й минуте при минусе, иначе 120 минут — нигде не было исполнено
одним куском с сохранённым итогом: `run.py` считает сетку из 675 конфигураций
с другими стопами, `manage.py` перебирает правила ведения, а ключевые числа v1
жили только в stdout и двумя константами в `freeze.py`. Здесь правило
исполняется целиком, итог вычисляется и сохраняется.

Что изменено против кода v1. `manage.py` при достижении лимита присваивает
результат ровно уровню лимита, даже если минута открылась уже за ним;
`run.py` для своих стопов такой open учитывает. Ревизия берёт худшее из двух:
уровень, а при открытии за уровнем — этот open. Обе модели считаются рядом,
чтобы цена исправления была видна, а не заявлена.

Что не изменено и остаётся допущением: выход на 15-й и на 120-й минуте
исполняется по close той же минуты. Это самое оптимистичное место конструкции;
измерением оно не подтверждено и здесь не улучшается.

    python -B setups/S-07/revision.py           # считает и пишет result_v1r1.json
    python -B setups/S-07/revision.py --check    # повтор в отдельное место и сверка
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run as base

HERE = base.HERE
ROOT = base.ROOT
MIN = base.MIN

INSTRUMENT = 'NQ'
DECISION_MINUTE = 3          # решение на закрытии минуты 3 сессии = 09:33 America/New_York
HOLD = 120                   # минут позиции, считая минуту входа первой
CHECK_MINUTE = 15            # выход, если позиция в минусе по закрытию этой минуты
LIMIT_CANDLES = 16.          # катастрофический предел от фактической цены входа
EPOCHS = [('2006-2012', '2006', '2012'), ('2013-2019', '2013', '2019'), ('2020-2026', '2020', '2026')]

# итог версии v1, записанный в карточке и в FROZEN_v1.json как константа
FROZEN_CLAIM = dict(total=236265.0, drawdown=30960.0, worst_day=-4620.0, trades=1563)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def population(instrument=INSTRUMENT, minute=DECISION_MINUTE, hold=HOLD):
    """Все дни календаря и причины, по которым день не даёт сделку.

    Пропущенный день остаётся неизвестным, а не нулевой сделкой: у него нет ни
    решения, ни входа, ни исхода.
    """
    m = base.market(instrument)
    u = base.unit(m)
    cal = pd.read_parquet(ROOT / 'setups/S-04/calendar.parquet')
    ts = m['close_ts_utc_ns']
    open_ns = cal.open_ns.to_numpy()
    p = np.searchsorted(ts, open_ns) + minute

    has_decision = (p < len(ts)) & (ts[np.minimum(p, len(ts) - 1)] == open_ns + minute * MIN)
    has_room = has_decision & (p + hold + 1 < len(ts))
    idx = np.where(has_room)[0]
    p = p[has_room].astype(np.int64)
    b = p + 1

    grid = (ts[b[:, None] + np.arange(hold)] - ts[b][:, None] == np.arange(hold) * MIN).all(axis=1)
    dropped_grid = int((~grid).sum())
    p, b, idx = p[grid], b[grid], idx[grid]

    unit = u[p]
    has_prefix = np.isfinite(unit) & (unit > 0)
    dropped_prefix = int((~has_prefix).sum())
    p, b, idx, unit = p[has_prefix], b[has_prefix], idx[has_prefix], unit[has_prefix]

    counts = dict(calendar_days=int(len(cal)),
                  with_decision_minute=int(has_decision.sum()),
                  without_decision_minute=int((~has_decision).sum()),
                  dropped_broken_minute_grid=dropped_grid,
                  dropped_missing_30m_prefix=dropped_prefix,
                  entries=int(len(p)))
    return m, cal, ts, p, b, idx, unit, counts


def trades(honest_gap):
    """Исполнение полного правила v1. honest_gap — брать open, если он уже за лимитом."""
    m, cal, ts, p, b, idx, unit, counts = population()
    win_hi = np.maximum.reduce([m['high'][p - j] for j in range(base.WINDOW)])
    win_lo = np.minimum.reduce([m['low'][p - j] for j in range(base.WINDOW)])
    mid = (win_hi + win_lo) / 2
    side = np.where(m['close'][p] > mid, 1., -1.)
    entry = m['open'][b]

    j = b[:, None] + np.arange(HOLD)
    s = side[:, None]
    e = entry[:, None]
    un = unit[:, None]
    close_c = s * (m['close'][j] - e) / un
    open_c = s * (m['open'][j] - e) / un
    adverse_c = s * (np.where(s > 0, m['low'][j], m['high'][j]) - e) / un
    live = ts[j] < cal.close_ns.to_numpy()[idx][:, None]

    n = len(p)
    result = np.full(n, np.nan)
    reason = np.empty(n, dtype=object)
    exit_pos = np.zeros(n, dtype=np.int64)
    for i in range(n):
        for k in range(HOLD):
            if not live[i, k]:
                result[i] = close_c[i, k - 1]
                reason[i] = 'закрытие сессии'
                exit_pos[i] = k - 1
                break
            if adverse_c[i, k] <= -LIMIT_CANDLES:
                result[i] = min(open_c[i, k], -LIMIT_CANDLES) if honest_gap else -LIMIT_CANDLES
                reason[i] = 'катастрофический лимит'
                exit_pos[i] = k
                break
            if k + 1 == CHECK_MINUTE and close_c[i, k] < 0:
                result[i] = close_c[i, k]
                reason[i] = 'минус на 15-й минуте'
                exit_pos[i] = k
                break
            if k + 1 == HOLD:
                result[i] = close_c[i, k]
                reason[i] = '120 минут'
                exit_pos[i] = k
                break

    money = result * unit * base.POINT[INSTRUMENT] - base.COST[INSTRUMENT]
    return pd.DataFrame(dict(date=cal.date.to_numpy()[idx], side=side, entry=entry, unit=unit,
                             candles=result, usd=money, reason=reason,
                             decision_ns=ts[p], entry_bar_close_ns=ts[b],
                             exit_bar_close_ns=ts[b + exit_pos])), counts


def summarise(d):
    d = d.copy()
    d['year'] = d.date.str.slice(0, 4)
    out = {}
    for epoch, a, b in EPOCHS:
        g = d.loc[(d.year >= a) & (d.year <= b)]
        daily = g.groupby('date').usd.sum().sort_index()
        eq = np.r_[0, np.cumsum(daily.to_numpy())]
        drop = max(1, int(len(daily) * .05))
        out[epoch] = dict(
            trades=int(len(g)),
            candles_mean=round(float(g.candles.mean()), 3),
            mean=round(float(g.usd.mean()), 1),
            median=round(float(g.usd.median()), 1),
            win_share=round(float((g.usd > 0).mean()), 3),
            total=round(float(daily.sum()), 0),
            drawdown=round(float((np.maximum.accumulate(eq) - eq).max()), 0),
            worst_day=round(float(daily.min()), 0),
            without_best_5pct=round(float(daily.sum() - daily.nlargest(drop).sum()), 0),
            years_positive=int((g.groupby('year').usd.sum() > 0).sum()),
            years=int(g.year.nunique()),
            exits={k: int(v) for k, v in g.reason.value_counts().items()})
    return out


def ny(ns):
    return str(pd.Timestamp(int(ns), unit='ns', tz='UTC').tz_convert('America/New_York'))


def build():
    honest, counts = trades(honest_gap=True)
    frozen, _ = trades(honest_gap=False)
    same = honest.candles.to_numpy() == frozen.candles.to_numpy()
    changed = honest.loc[~same]
    recent = honest.date.str.slice(0, 4) >= '2020'
    row = honest.iloc[-1]
    market = {f'{INSTRUMENT}/{name}': sha(ROOT / 'data/market' / INSTRUMENT / (name + '.npy'))
              for name in ['close_ts_utc_ns', 'open', 'high', 'low', 'close']}
    frozen_view = summarise(frozen)

    return dict(
        revision='v1r1',
        of_version='v1',
        what_changed='исполнение катастрофического лимита: при открытии минуты за уровнем '
                     'заполнение идёт по этому open, а не по уровню',
        rule=dict(instrument=INSTRUMENT, decision_minute=DECISION_MINUTE,
                  decision_clock='закрытие минуты 3 сессии, 09:33 America/New_York',
                  entry='open следующего минутного интервала; у свечи с close-меткой 09:34 '
                        'этот open относится к 09:33, то есть к самому моменту решения',
                  window_minutes=base.WINDOW, hold_minutes=HOLD,
                  exit_if_negative_at=CHECK_MINUTE, catastrophic_limit_candles=LIMIT_CANDLES,
                  breakeven='никогда', trailing='никогда',
                  unit='медиана истинного диапазона последних 30 закрытых минут, '
                       'зафиксированная в минуту решения'),
        position_model=dict(contracts=1, cost_per_turn_usd=base.COST[INSTRUMENT],
                            point_value_usd=base.POINT[INSTRUMENT],
                            slippage='сверх фиксированного расхода не моделируется'),
        coverage=counts,
        clock_sample=dict(date=row.date, decision_close=ny(row.decision_ns),
                          entry_bar_close=ny(row.entry_bar_close_ns),
                          entry_price_instant=ny(row.entry_bar_close_ns - MIN),
                          exit_bar_close=ny(row.exit_bar_close_ns), exit_reason=row.reason),
        result_v1r1=summarise(honest),
        result_v1_frozen_execution=frozen_view,
        correction_cost=dict(
            trades_changed=int((~same).sum()),
            trades_changed_2020_2026=int((~same & recent.to_numpy()).sum()),
            usd_difference_2020_2026=round(float(honest.usd[recent].sum() - frozen.usd[recent].sum()), 2),
            usd_difference_all=round(float(honest.usd.sum() - frozen.usd.sum()), 2),
            changed_trades=[dict(date=r.date, candles_v1r1=round(float(r.candles), 3))
                            for r in changed.itertuples()],
            note='минута ни разу не открылась строго за пределом 16 свечей внутри сессии, '
                 'поэтому исправление меняет код, а не деньги; на более тесных пределах '
                 'той же ленты такие случаи единичны, но существуют'),
        matches_frozen_claim=dict(
            claim=FROZEN_CLAIM,
            computed=dict(total=frozen_view['2020-2026']['total'],
                          drawdown=frozen_view['2020-2026']['drawdown'],
                          worst_day=frozen_view['2020-2026']['worst_day'],
                          trades=frozen_view['2020-2026']['trades']),
            note='FROZEN_v1.json хранит эти деньги двумя константами, вписанными в freeze.py. '
                 'Здесь они вычислены; совпадение означает, что константы верны, '
                 'а не что они были вычислены тогда.'),
        identity=dict(revision_py=sha(HERE / 'revision.py'), run_py=sha(HERE / 'run.py'),
                      manage_py=sha(HERE / 'manage.py'),
                      pine=sha(HERE / 'S07_opening_drift.pine'),
                      calendar=sha(ROOT / 'setups/S-04/calendar.parquet'), market=market),
        limits=[
            'выход на 15-й и на 120-й минуте исполняется по close той же минуты — '
            'самое оптимистичное допущение конструкции, измерением не подтверждено',
            'внутри одной минуты порядок лимита и закрытия не восстановим; лимит считается первым',
            'проскальзывание сверх фиксированного расхода не моделируется',
            'независимого подтверждения нет: та же лента, тот же корпус, ведение и предел '
            'подобраны на этой же истории',
            'S07_opening_drift.pine — исследовательский перенос; его исполнение '
            'в TradingView не проверялось',
        ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true',
                    help='повтор пишет в data/research и сверяется с сохранённым итогом')
    args = ap.parse_args()

    saved = HERE / 'result_v1r1.json'
    report = build()
    if args.check:
        out = base.OUT / 'recheck'
        out.mkdir(parents=True, exist_ok=True)
        path = out / 'result_v1r1.json'
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        old = json.loads(saved.read_text(encoding='utf-8'))
        keys = ['rule', 'coverage', 'clock_sample', 'result_v1r1',
                'result_v1_frozen_execution', 'correction_cost']
        diff = [k for k in keys if old[k] != report[k]]
        print('повтор записан в', path.relative_to(ROOT))
        print('расхождений с сохранённым итогом нет' if not diff else f'РАСХОЖДЕНИЕ: {diff}')
        return

    saved.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    r = report['result_v1r1']['2020-2026']
    f = report['result_v1_frozen_execution']['2020-2026']
    print('охват:', report['coverage'])
    print('часы:', report['clock_sample'])
    print(f"v1r1 2020-2026: итог {r['total']}, просадка {r['drawdown']}, худший день "
          f"{r['worst_day']}, сделок {r['trades']}, медиана {r['median']}")
    print(f"v1   2020-2026: итог {f['total']}, просадка {f['drawdown']}, худший день "
          f"{f['worst_day']}, сделок {f['trades']}")
    print('цена исправления:', report['correction_cost']['trades_changed'], 'сделок за 21 год,',
          report['correction_cost']['usd_difference_all'], 'долларов')
    for epoch, _, _ in EPOCHS:
        e = report['result_v1r1'][epoch]
        print(f"{epoch}: {e['trades']} сделок, {e['candles_mean']} свечи, итог {e['total']}, "
              f"выходы {e['exits']}")


if __name__ == '__main__':
    main()
