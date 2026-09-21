"""S-17: один совместный прогон трёх ЗАМОРОЖЕННЫХ версий на продолжении ленты (решение человека 2026-09-21).

Версии: D30 (проверяемый кандидат), P (экспериментальное управление), S-07 v1r1 (экономический эталон).
Правила не меняются; движок ниже общий для истории, перекрытия и новой ленты и обязан воспроизводить замороженные
результаты на истории до сессии (режим `validate`). Условия, порядок чтения и обработка пропусков — FREEZE_FORWARD.md.

    python -B setups/S-17/forward.py validate   # движок == sessions_NQ.csv и == setups/S-07/revision.py на истории
    python -B setups/S-17/forward.py overlap    # те же операторы на локальных барах и на барах источника, 2026-03-16…05-01
    python -B setups/S-17/forward.py grid       # только сетка минут новой ленты в окне 09:03–11:33 ET, без цен
    python -B setups/S-17/forward.py open       # ВСКРЫТИЕ: результат на 2026-05-04…07-10 (один раз)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'setups' / 'S-07'))
import run as base  # noqa: E402

MIN = base.MIN
INS = 'NQ'
POINT, COST, TICK = base.POINT[INS], base.COST[INS], base.TICK[INS]
HOLD_D30, HOLD_S07, CHECK_MINUTE, LIMIT = 30, 120, 15, 16.0
HOLIDAYS = {'2026-05-25', '2026-06-19', '2026-07-03'}     # XNYS закрыт; на ленте это короткие сессии по 1 140 минут
F_FROM, F_TO = '2026-05-04', '2026-07-10'


def load(root):
    return {k: np.load(Path(root) / (k + '.npy'), mmap_mode='r') for k in ['close_ts_utc_ns', 'open', 'high', 'low', 'close']}


def forward_calendar():
    days = pd.bdate_range(F_FROM, F_TO).strftime('%Y-%m-%d')
    days = [d for d in days if d not in HOLIDAYS]
    o = [pd.Timestamp(d + ' 09:30', tz='America/New_York').tz_convert('UTC').value for d in days]
    c = [pd.Timestamp(d + ' 16:00', tz='America/New_York').tz_convert('UTC').value for d in days]
    return pd.DataFrame(dict(date=days, open_ns=o, close_ns=c))


def engine(m, cal):
    """Одна сессия — одна попытка. Возвращает строку на КАЖДУЮ сессию календаря со статусом:
    no_entry:<причина>  — войти было нельзя, позиции нет (сессия не наблюдалась);
    ok                  — всё окно наблюдалось;
    entered_hole        — вход состоялся, внутри окна позиции есть пропуски ленты (исход считается по объявленной
                          конвенции: правила читаются на наблюдённых барах, выход по времени — первая доступная цена
                          не раньше срока в той же сессии; если её нет — исход неизвестен)."""
    ts = np.asarray(m['close_ts_utc_ns'])
    O, H, L, C = (np.asarray(m[k]) for k in ('open', 'high', 'low', 'close'))
    u = base.unit(dict(close_ts_utc_ns=ts, open=O, high=H, low=L, close=C))

    def at(t):
        i = int(np.searchsorted(ts, t))
        return i if i < len(ts) and ts[i] == t else -1

    rows = []
    for r in cal.itertuples():
        row = dict(date=r.date, status='ok')
        p = at(r.open_ns + 3 * MIN)
        if p < 0:
            row['status'] = 'no_entry:нет минуты решения'; rows.append(row); continue
        if p < 31 or not (np.isfinite(u[p]) and u[p] > 0):
            row['status'] = 'no_entry:нет 30 непрерывных минут'; rows.append(row); continue
        b = at(ts[p] + MIN)
        if b < 0:
            row['status'] = 'no_entry:нет минуты входа'; rows.append(row); continue
        hi, lo = H[p - 29:p + 1].max(), L[p - 29:p + 1].min()
        s = 1.0 if C[p] > (hi + lo) / 2 else -1.0
        cand, e0 = float(u[p]), float(O[b])
        idx = [at(ts[b] + k * MIN) for k in range(-2, HOLD_S07 + 1)]     # k=-2 это бар p-1, k=0 бар входа
        hole = any(i < 0 for i in idx[:2 + HOLD_S07])
        if hole:
            row['status'] = 'entered_hole'

        def px(i, arr):                                                   # цена по стороне сделки
            return s * (arr[i] - e0)

        def lowside(i):
            return s * ((L[i] if s > 0 else H[i]) - e0)

        def highside(i):
            return s * ((H[i] if s > 0 else L[i]) - e0)

        def first_at_or_after(k):                                         # первая доступная цена не раньше срока
            for kk in range(k, HOLD_S07 + 1):
                i = idx[kk + 2]
                if i >= 0 and ts[i] - MIN < r.close_ns:
                    return i
            return -1

        # ---- D30 и P
        ex = first_at_or_after(HOLD_D30)
        d30 = np.nan if ex < 0 else px(ex, O)
        first, tC = None, None
        for k in range(0, HOLD_D30):
            j, j1, j2 = idx[k + 2], idx[k + 1], idx[k]
            if j < 0:
                continue
            if first is not None and tC is None and px(j, C) < first:
                tC = k
            if j1 >= 0 and j2 >= 0 and first is None:
                o1, c1 = px(j1, O), px(j1, C)
                if lowside(j) > highside(j2) and max(o1, c1) > highside(j2) and min(o1, c1) < lowside(j):
                    bt1 = max(px(j2, O), px(j2, C)); bt2 = max(o1, c1)
                    c2bb = min(o1, c1); c3bb = min(px(j, O), px(j, C))
                    nb_ = bt1 if c2bb > bt1 else highside(j2)
                    nt_ = c3bb if c3bb > bt2 else lowside(j)
                    if nt_ > nb_:
                        first = nb_
        prot = d30
        if tC is not None:
            exc = first_at_or_after(tC + 1)
            prot = np.nan if exc < 0 else px(exc, O)
        # ---- S-07 v1r1 (предел 16 свечей с честным гэпом; минус на 15-й минуте; иначе 120 минут; не позже закрытия сессии)
        res, reason, last = np.nan, 'исход неизвестен', None
        for k in range(HOLD_S07):
            i = idx[k + 2]
            if i < 0:
                continue
            if ts[i] >= r.close_ns:
                res, reason = (np.nan, reason) if last is None else (px(last, C) / cand, 'закрытие сессии'); break
            if lowside(i) / cand <= -LIMIT:
                res, reason = min(px(i, O) / cand, -LIMIT), 'катастрофический лимит'; break
            if k + 1 >= CHECK_MINUTE and (last is None or (ts[last] - ts[b]) // MIN + 1 < CHECK_MINUTE) and px(i, C) < 0:
                res, reason = px(i, C) / cand, 'минус на 15-й минуте'; break
            if k + 1 == HOLD_S07:
                res, reason = px(i, C) / cand, '120 минут'; break
            last = i
        else:
            if last is not None:
                res, reason = px(last, C) / cand, '120 минут (последний наблюдённый бар)'
        row.update(side=int(s), entry=e0, candle=cand, d30_pts=d30, prot_pts=prot, c_min=np.nan if tC is None else tC + 1,
                   s07_candles=res, s07_reason=reason,
                   d30_usd=d30 * POINT - COST, prot_usd=prot * POINT - COST, s07_usd=res * cand * POINT - COST)
        rows.append(row)
    return pd.DataFrame(rows)


def history_calendar():
    return pd.read_parquet(ROOT / 'setups/S-04/calendar.parquet')[['date', 'open_ns', 'close_ns']]


def summary(x):
    x = np.asarray(x, dtype=float)
    eq = np.cumsum(x)
    dd = float((np.maximum.accumulate(np.r_[0.0, eq])[1:] - eq).max()) if len(x) else 0.0
    return dict(n=int(len(x)), total=round(float(x.sum()), 0), mean=round(float(x.mean()), 1), median=round(float(np.median(x)), 1),
                win_share=round(float((x > 0).mean()), 3), worst=round(float(x.min()), 0), best=round(float(x.max()), 0),
                max_drawdown=round(dd, 0), sd=round(float(x.std(ddof=1)), 0))


def validate():
    m = load(ROOT / 'data/market' / INS)
    cal = history_calendar()
    cal = cal[cal.open_ns + 3 * MIN <= int(m['close_ts_utc_ns'][-1])]
    e = engine(m, cal)
    ok = e[e.status == 'ok'].set_index('date')
    ent = e[~e.status.str.startswith('no_entry')].set_index('date')
    ref = pd.read_csv(HERE / 'sessions_NQ.csv').set_index('date')
    j = ent.join(ref[['d30_pts', 'prot_pts', 'c_min']], rsuffix='_ref', how='inner')
    print('D30/P: общих сессий', len(j), '| d30 равны', int(np.isclose(j.d30_pts, j.d30_pts_ref).sum()),
          '| P равны', int(np.isclose(j.prot_pts, j.prot_pts_ref).sum()),
          '| минута C равна', int(((j.c_min == j.c_min_ref) | (j.c_min.isna() & j.c_min_ref.isna())).sum()))
    print('  сессий в sessions_NQ.csv без пары в движке:', len(set(ref.index) - set(ok.index)))
    import revision
    fr, _ = revision.trades(honest_gap=True)
    fr = fr.set_index('date')
    j2 = ok.join(fr[['candles', 'usd', 'reason']], how='inner')
    print('S-07 v1r1: общих сессий', len(j2), '| свечи равны', int(np.isclose(j2.s07_candles, j2.candles).sum()),
          '| деньги равны', int(np.isclose(j2.s07_usd, j2.usd).sum()), '| причина выхода равна', int((j2.s07_reason == j2.reason).sum()),
          '| сессий эталона без пары:', len(set(fr.index) - set(ok.index)))
    st = e.status.value_counts().to_dict()
    print('статусы движка на всей истории:', st)
    hole = e[e.status == 'entered_hole']
    hole5 = hole[(hole.date >= '2021') & (hole.date < '2026')]
    noent5 = e[(e.status.str.startswith('no_entry')) & (e.date >= '2021') & (e.date < '2026')]
    print('2021-2025: войти было нельзя —', len(noent5), list(noent5.date), '| вход состоялся, лента с пропусками —', len(hole5))
    print(hole5[['date', 'side', 'd30_usd', 'prot_usd', 's07_usd', 's07_reason']].to_string(index=False))
    e.to_csv(HERE / 'engine_history_NQ.csv', index=False, float_format='%.4f')


def overlap():
    import pyarrow.parquet as pq
    loc = load(ROOT / 'data/market' / INS)
    ts = np.asarray(loc['close_ts_utc_ns'])
    lo = int(np.searchsorted(ts, pd.Timestamp('2026-03-15 21:00', tz='UTC').value))
    lx = pq.read_table(ROOT / 'data/forward/_incoming/lynx1231_equity_index_minute.parquet').to_pandas()
    g = lx[lx.contract_code == 'NQM26'].copy()
    g['close_ns'] = (pd.to_datetime(g.timestamp, unit='ms') + pd.Timedelta(minutes=1)).dt.tz_localize(
        'America/Chicago', ambiguous='NaT', nonexistent='NaT').dt.tz_convert('UTC').dt.tz_localize(None).to_numpy(
        dtype='datetime64[ns]').astype('int64')
    g = g.set_index('close_ns')
    src = {k: np.array(loc[k]) for k in loc}
    w = ts[lo:]
    has = np.isin(w, g.index.to_numpy())
    gg = g.loc[w[has]]
    for k in ('open', 'high', 'low', 'close'):
        a = src[k]; a[lo:][has] = gg[k].to_numpy()
    cal = history_calendar()
    cal = cal[(cal.date >= '2026-03-16') & (cal.open_ns + 3 * MIN <= int(ts[-1]))]
    a, b = engine(loc, cal).set_index('date'), engine(src, cal).set_index('date')
    print('перекрытие: сессий', len(a), '| статусы совпали', int((a.status == b.status).sum()), '| сторона совпала', int((a.side == b.side).sum()))
    for col in ('d30_usd', 'prot_usd', 's07_usd'):
        d = (b[col] - a[col])
        print(f'  {col}: равны до цента {int(np.isclose(a[col], b[col]).sum())} из {len(a)} | расхождение: среднее {d.mean():+.2f} $, '
              f'макс |.| {d.abs().max():.0f} $ | итог local {a[col].sum():+.0f} / source {b[col].sum():+.0f}')
    print('  причина выхода S-07 совпала', int((a.s07_reason == b.s07_reason).sum()), '| минута C совпала',
          int(((a.c_min == b.c_min) | (a.c_min.isna() & b.c_min.isna())).sum()))
    pd.concat([a.add_suffix('_local'), b.add_suffix('_source')], axis=1).to_csv(HERE / 'overlap_operators_NQ.csv', float_format='%.4f')


def grid():
    m = load(ROOT / 'data/forward/market' / INS)
    ts = np.asarray(m['close_ts_utc_ns'])
    cal = forward_calendar()
    bad = []
    for r in cal.itertuples():
        want = r.open_ns + np.arange(-27, 124) * MIN               # закрытия 09:03 … 11:33 ET
        miss = int((~np.isin(want, ts)).sum())
        if miss:
            bad.append((r.date, miss))
    print('сессий в календаре новой ленты:', len(cal), '| сессий с пропусками минут в окне 09:03–11:33 ET:', len(bad), bad)


def open_forward():
    out = HERE / 'forward_result.json'
    if out.exists():
        raise SystemExit('forward_result.json уже существует: вскрытие было, повторно не открывается')
    m = load(ROOT / 'data/forward/market' / INS)
    cal = forward_calendar()
    e = engine(m, cal)
    e.to_csv(HERE / 'forward_sessions_NQ.csv', index=False, float_format='%.4f')
    taken = e[~e.status.str.startswith('no_entry')]
    known = taken.dropna(subset=['d30_usd', 'prot_usd', 's07_usd'])
    hist = pd.read_csv(HERE / 'engine_history_NQ.csv')
    hist = hist[(hist.status == 'ok') & (hist.date >= '2021') & (hist.date < '2026')]
    res = dict(window=[F_FROM, F_TO], calendar_sessions=int(len(cal)), statuses=e.status.value_counts().to_dict(),
               positions_taken=int(len(taken)), outcomes_known=int(len(known)), versions={}, paired={}, context={})
    n = len(known)
    for name, col in (('D30', 'd30_usd'), ('P', 'prot_usd'), ('S-07 v1r1', 's07_usd')):
        res['versions'][name] = summary(known[col])
        h = hist[col].to_numpy()
        roll = np.convolve(h, np.ones(n), 'valid') if n and len(h) >= n else np.array([])
        res['context'][name] = dict(historical_windows=int(len(roll)),
                                    percentile_of_forward_total=None if not len(roll) else round(float((roll <= known[col].sum()).mean()), 3),
                                    historical_window_total_p05_p50_p95=None if not len(roll) else [round(float(v), 0) for v in np.quantile(roll, [.05, .5, .95])])
        for mult, slip in ((2, 0), (1, 4)):
            adj = known[col] - (mult - 1) * COST - 2 * slip * TICK * POINT
            res['versions'][name][f'mean_cost_x{mult}_slip{slip}t'] = round(float(adj.mean()), 1)
    for name, col in (('P − D30', 'prot_usd'), ('S-07 v1r1 − D30', 's07_usd')):
        d = (known[col] - known.d30_usd).to_numpy()
        res['paired'][name] = dict(sessions_differ=int((np.abs(d) > 1e-9).sum()), total=round(float(d.sum()), 0), mean=round(float(d.mean()), 1),
                                   median=round(float(np.median(d)), 1), worst=round(float(d.min()), 0), best=round(float(d.max()), 0),
                                   better_in=int((d > 0).sum()), worse_in=int((d < 0).sum()))
    res['s07_exit_reasons'] = known.s07_reason.value_counts().to_dict()
    res['event_C_sessions'] = int(known.c_min.notna().sum())
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    {'validate': validate, 'overlap': overlap, 'grid': grid, 'open': open_forward}[sys.argv[1]]()
