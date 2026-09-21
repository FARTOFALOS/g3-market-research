"""S-17: один ценовой кандидат на отклик открытия NQ (предложение человека от 2026-09-21).

ОБЪЯВЛЕНО ДО СЧЁТА. Наследует историю просмотра S-07 и линии 097; независимым открытием не является.
Сигнал и сторона — дословно S-07 (setups/S-07/run.py, revision.py): календарь XNYS, решение на закрытии
минуты 3 сессии (метка close 09:33 America/New_York), нужны 30 непрерывных закрытых минут (из них последние
четыре — уже основная сессия, остальные 26 — до открытия); сторона = close против середины H/L этих 30 минут;
вход — open следующего минутного интервала (09:33:00). Одна единица, одна попытка на сессию.

D30 (диагностическая версия): выход через 30 минут после фактического входа = open бара b+30 (10:03:00),
без ценового стопа, без доборов, повторных входов, безубытка и трейлинга. Расход NQ $15 за круг (как в S-07).
Число 30 взято из просмотренной истории (097: результат набирается между 10-й и 30-45-й минутой).
В сетке S-07 (summary_v1.csv, minute 3 / horizon 30 / stop 0) это действие уже посчитано с выходом по close
30-й минуты; здесь оно пересчитывается с выходом по следующей доступной цене и сверяется с той строкой.

P (единственная дополнительная версия защиты): D30 + событие C из base/097/s07_invalidation_events.py дословно:
первое закрытие минуты за ДАЛЬНИМ краем первого гэпа, рождённого по стороне сделки после решения
(рождение — правило машины RIZ verbatim, известно на закрытии третьего бара); выход — open следующей минуты.
До рождения гэпа и в сессиях без события действует только выход по времени. Событие проверяется на закрытии
бара j относительно гэпа, известного ДО бара j. Других версий защиты, сетки стопов и фильтров дней нет.
Линейка «закрытие на d свечей против входа» печатается только как ОПИСАНИЕ цены защиты, версией не является.

Территория: первичная NQ 2021-2025; 2026 (до конца канонической ленты 2026-05-04) отдельно; 2006-2012 и
2013-2020 — границы применимости; ES и YM — сравнение. Вся эта история уже экспонирована.
data/forward (2026-05-04…07-10) этим скриптом НЕ читается.

    python -B setups/S-17/run.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'setups' / 'S-07'))
import run as base  # noqa: E402  (семантика рынка, единицы «свеча», расходов — из S-07)

MIN = base.MIN
DECISION_MINUTE = 3
HOLD = 30
TERR = [('NQ 2021-2025 (первичная)', 'NQ', 2021, 2025), ('NQ 2026 неполный', 'NQ', 2026, 2026),
        ('NQ 2013-2020', 'NQ', 2013, 2020), ('NQ 2006-2012', 'NQ', 2006, 2012),
        ('ES 2021-2025', 'ES', 2021, 2025), ('YM 2021-2025', 'YM', 2021, 2025)]
DEPTHS = (2, 4, 6, 8, 12)


def sessions(ins):
    m = base.market(ins)
    u = base.unit(m)
    cal = pd.read_parquet(ROOT / 'setups/S-04/calendar.parquet')
    ts = m['close_ts_utc_ns']
    open_ns = cal.open_ns.to_numpy()
    in_tape = open_ns + DECISION_MINUTE * MIN <= ts[-1]
    p = np.searchsorted(ts, open_ns) + DECISION_MINUTE
    ok_p = in_tape & (p < len(ts)) & (ts[np.minimum(p, len(ts) - 1)] == open_ns + DECISION_MINUTE * MIN)
    rows = []
    why = dict(calendar=int(in_tape.sum()), no_decision_minute=0, no_30m_prefix=0, no_entry_bar=0, broken_grid_in_hold=0,
               no_exit_bar=0)
    for i in np.flatnonzero(in_tape):
        if not ok_p[i]:
            why['no_decision_minute'] += 1
            continue
        pi = int(p[i])
        if not (np.isfinite(u[pi]) and u[pi] > 0) or pi < 31:
            why['no_30m_prefix'] += 1
            continue
        b = pi + 1
        if b >= len(ts) or ts[b] != ts[pi] + MIN:
            why['no_entry_bar'] += 1
            continue
        if b + HOLD - 1 >= len(ts) or ts[b + HOLD - 1] - ts[b] != (HOLD - 1) * MIN:
            why['broken_grid_in_hold'] += 1
            continue
        has_exit = b + HOLD < len(ts) and ts[b + HOLD] == ts[b] + HOLD * MIN
        if not has_exit:
            why['no_exit_bar'] += 1
            continue
        rows.append((i, pi, b))
    return m, u, cal, rows, why


def one_instrument(ins):
    m, u, cal, rows, why = sessions(ins)
    O, H, L, C = m['open'], m['high'], m['low'], m['close']
    out = []
    for i, p, b in rows:
        hi, lo = H[p - 29:p + 1].max(), L[p - 29:p + 1].min()
        s = 1.0 if C[p] > (hi + lo) / 2 else -1.0
        cand = float(u[p])
        e0 = float(O[b])
        sl = slice(p - 1, b + HOLD + 1)                     # бары p-1 … b+30; индекс входа в срезе = 2
        o, h, l, c = (O[sl], H[sl], L[sl], C[sl]) if s > 0 else (-O[sl], -L[sl], -H[sl], -C[sl])
        eo = s * e0
        x_time = float(o[2 + HOLD] - eo)                    # выход по open бара b+30, пункты по стороне
        x_close = float(c[2 + HOLD - 1] - eo)               # справка: выход по close 30-й минуты (как в сетке S-07)
        lows = l[2:2 + HOLD] - eo
        closes = c[2:2 + HOLD] - eo
        highs = h[2:2 + HOLD] - eo
        first = None
        tC = None
        depth_t = {d: None for d in DEPTHS}
        for j in range(2, 2 + HOLD):
            if first is not None and tC is None and c[j] < first[1]:
                tC = j
            for d in DEPTHS:
                if depth_t[d] is None and c[j] < eo - d * cand:
                    depth_t[d] = j
            if l[j] > h[j - 2] and max(o[j - 1], c[j - 1]) > h[j - 2] and min(o[j - 1], c[j - 1]) < l[j]:
                bt1 = max(o[j - 2], c[j - 2]); bt2 = max(o[j - 1], c[j - 1])
                c2bb = min(o[j - 1], c[j - 1]); c3bb = min(o[j], c[j])
                nb_ = bt1 if c2bb > bt1 else h[j - 2]
                nt_ = c3bb if c3bb > bt2 else l[j]
                if nt_ > nb_ and first is None:
                    first = (nt_, nb_, j)
        x_prot = x_time if tC is None else float(o[tC + 1] - eo)
        row = dict(date=cal.date.to_numpy()[i], year=int(str(cal.date.to_numpy()[i])[:4]), side=int(s), entry=e0, candle=cand,
                   d30_pts=x_time, d30_close_pts=x_close, prot_pts=x_prot,
                   mae_wick_pts=float(lows.min()), mae_wick_min=int(lows.argmin()) + 1,
                   mae_close_pts=float(closes.min()), mfe_wick_pts=float(highs.max()), mfe_wick_min=int(highs.argmax()) + 1,
                   gap_born_min=np.nan if first is None else first[2] - 1,
                   gap_far_below_entry_pts=np.nan if first is None else float(eo - first[1]),
                   c_min=np.nan if tC is None else tC - 1)
        for d in DEPTHS:
            row[f'dep{d}_min'] = np.nan if depth_t[d] is None else depth_t[d] - 1
            row[f'dep{d}_pts'] = x_time if depth_t[d] is None else float(o[depth_t[d] + 1] - eo)
        out.append(row)
    return pd.DataFrame(out), why


def drawdown(x):
    eq = np.cumsum(x)
    return float((np.maximum.accumulate(np.r_[0.0, eq])[1:] - eq).max()) if len(x) else 0.0


def account(pts, ins, cost=None, slip_ticks=0):
    """Денежный счёт одной версии: пункты по стороне -> доллары одной единицы после расходов."""
    cost = base.COST[ins] if cost is None else cost
    usd_g = pts * base.POINT[ins]
    usd = usd_g - cost - 2 * slip_ticks * base.TICK[ins] * base.POINT[ins]
    n = len(usd)
    if n == 0:
        return dict(trades=0)
    srt = np.sort(usd)[::-1]
    k5, k10 = max(1, int(round(n * .05))), max(1, int(round(n * .10)))
    win = usd > 0
    return dict(trades=int(n), gross_mean=round(float(usd_g.mean()), 2), net_mean=round(float(usd.mean()), 2),
                net_median=round(float(np.median(usd)), 2), win_share=round(float(win.mean()), 4),
                avg_win=round(float(usd[win].mean()), 2) if win.any() else None,
                avg_loss=round(float(usd[~win].mean()), 2) if (~win).any() else None,
                p05=round(float(np.quantile(usd, .05)), 2), p95=round(float(np.quantile(usd, .95)), 2),
                worst_day=round(float(usd.min()), 2), best_day=round(float(usd.max()), 2),
                net_total=round(float(usd.sum()), 2), max_drawdown=round(drawdown(usd), 2),
                sd=round(float(usd.std(ddof=1)), 2), mean_over_sd=round(float(usd.mean() / usd.std(ddof=1)), 4),
                t_by_session=round(float(usd.mean() / (usd.std(ddof=1) / np.sqrt(n))), 2),
                top5pct_share_of_total=round(float(srt[:k5].sum() / usd.sum()), 3) if usd.sum() != 0 else None,
                total_without_best_5pct=round(float(srt[k5:].sum()), 2),
                total_without_best_10pct=round(float(srt[k10:].sum()), 2))


def main():
    res = dict(declared=__doc__.split('\n\n')[0], hold_minutes=HOLD, decision_minute=DECISION_MINUTE, territories={},
               skipped={}, xray={}, check_against_S07_grid={})
    frames = {}
    for ins in ('NQ', 'ES', 'YM'):
        d, why = one_instrument(ins)
        frames[ins] = d
        res['skipped'][ins] = why | dict(trades=int(len(d)))
        d.to_csv(HERE / f'sessions_{ins}.csv', index=False, float_format='%.4f')

    for name, ins, a, b in TERR:
        d = frames[ins]
        d = d[(d.year >= a) & (d.year <= b)]
        block = dict(D30=account(d.d30_pts.to_numpy(), ins), P=account(d.prot_pts.to_numpy(), ins))
        block['D30_by_year'] = {int(y): dict(n=int(len(g)), net_total=round(float((g.d30_pts * base.POINT[ins] - base.COST[ins]).sum()), 0),
                                             P_net_total=round(float((g.prot_pts * base.POINT[ins] - base.COST[ins]).sum()), 0))
                                for y, g in d.groupby('year')}
        block['sensitivity_D30'] = {
            'выход по close 30-й минуты (как в сетке S-07)': account(d.d30_close_pts.to_numpy(), ins)['net_mean'],
            'расход x2': account(d.d30_pts.to_numpy(), ins, cost=2 * base.COST[ins])['net_mean'],
            'расход x3': account(d.d30_pts.to_numpy(), ins, cost=3 * base.COST[ins])['net_mean'],
            'хуже на 1 тик с каждой стороны': account(d.d30_pts.to_numpy(), ins, slip_ticks=1)['net_mean'],
            'хуже на 4 тика с каждой стороны': account(d.d30_pts.to_numpy(), ins, slip_ticks=4)['net_mean'],
        }
        block['sensitivity_P'] = {
            'расход x2': account(d.prot_pts.to_numpy(), ins, cost=2 * base.COST[ins])['net_mean'],
            'хуже на 4 тика с каждой стороны': account(d.prot_pts.to_numpy(), ins, slip_ticks=4)['net_mean'],
        }
        res['territories'][name] = block

    # сверка с уже посчитанной строкой сетки S-07 (NQ, minute 3, horizon 30, stop 0, эпоха 2020-2026)
    g = frames['NQ']; g = g[(g.year >= 2020) & (g.year <= 2026)]
    grid = pd.read_csv(ROOT / 'setups/S-07/summary_v1.csv')
    row = grid[(grid.instrument == 'NQ') & (grid.minute == 3) & (grid.horizon == 30) & (grid.stop == 0)].iloc[0]
    res['check_against_S07_grid'] = dict(grid_mean_2020_2026=float(row['mean_2020-2026']), grid_n=int(row['n_2020-2026']),
                                         here_close_exit_mean=round(float((g.d30_close_pts * 20 - 15).mean()), 2), here_n=int(len(g)))

    # рентген риска на первичной территории: что происходит после ухода против входа и после события C
    d = frames['NQ']; d = d[(d.year >= 2021) & (d.year <= 2025)].copy()
    usd = d.d30_pts * 20 - 15
    tail = usd >= np.quantile(usd, .90)
    x = dict(sessions=int(len(d)), tail_definition='лучшие 10 % сессий D30 по деньгам — ретроспективная группа, только для отчёта',
             tail_share_of_total=round(float(usd[tail].sum() / usd.sum()), 3),
             mae_wick_candles_quantiles={q: round(float((-d.mae_wick_pts / d.candle).quantile(q)), 2) for q in (.25, .5, .75, .9)},
             mae_wick_candles_of_winners={q: round(float((-d.mae_wick_pts / d.candle)[usd > 0].quantile(q)), 2) for q in (.5, .75, .9)},
             mae_wick_candles_of_tail={q: round(float((-d.mae_wick_pts / d.candle)[tail].quantile(q)), 2) for q in (.5, .75, .9)})
    ruler = {}
    for dep in DEPTHS:
        hit = d[f'dep{dep}_min'].notna()
        rem = (d.d30_pts - d[f'dep{dep}_pts'])[hit] * 20
        man = d[f'dep{dep}_pts'] * 20 - 15
        ruler[f'{dep} свечей по закрытию'] = dict(
            occurs=round(float(hit.mean()), 3), median_minute=None if not hit.any() else float(d[f'dep{dep}_min'][hit].median()),
            remaining_after_usd=None if not hit.any() else round(float(rem.mean()), 1),
            remaining_t=None if hit.sum() < 3 else round(float(rem.mean() / (rem.std(ddof=1) / np.sqrt(hit.sum()))), 2),
            kept_tail_money=round(float(man[tail].sum() / usd[tail].sum()), 3), managed_mean=round(float(man.mean()), 1),
            managed_p05=round(float(man.quantile(.05)), 0), managed_worst=round(float(man.min()), 0))
    hit = d.c_min.notna()
    rem = (d.d30_pts - d.prot_pts)[hit] * 20
    man = d.prot_pts * 20 - 15
    x['event_C'] = dict(gap_born_share=round(float(d.gap_born_min.notna().mean()), 3),
                        gap_born_median_minute=float(d.gap_born_min.median()),
                        far_edge_below_entry_candles_p25_50_75=[round(float(v), 2) for v in (d.gap_far_below_entry_pts / d.candle).quantile([.25, .5, .75])],
                        occurs=round(float(hit.mean()), 3), median_minute=float(d.c_min[hit].median()),
                        remaining_after_usd=round(float(rem.mean()), 1),
                        remaining_t=round(float(rem.mean() / (rem.std(ddof=1) / np.sqrt(hit.sum()))), 2),
                        kept_tail_money=round(float(man[tail].sum() / usd[tail].sum()), 3),
                        winners_cut=round(float(((usd > 0) & hit & (man < usd)).sum() / (usd > 0).sum()), 3),
                        losers_reduced=round(float(((usd <= 0) & hit & (man > usd)).sum() / (usd <= 0).sum()), 3),
                        time_in_position=round(float(np.where(hit, d.c_min, HOLD).mean() / HOLD), 3))
    x['ruler_description_only'] = ruler
    res['xray']['NQ 2021-2025'] = x
    (HERE / 'result.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print('written', HERE / 'result.json')


if __name__ == '__main__':
    main()
