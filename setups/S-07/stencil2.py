"""S-07, продолжение трафарета: если свинг не место для стопа, то что он говорит.

Первый проход показал: первый противоположный экстремум стоит на одной и той же
свече у прибыльных и убыточных дней, и его сносят и те и другие. Значит стоп
туда не ставится. Здесь проверяется, несёт ли структура информацию иначе:
глубиной первого свинга, счётом подряд идущих перелоев и состоянием «низ ещё не
снесён» на конкретной минуте. Всё это доступно на префиксе в момент решения.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'setups/S-07'))
import stencil as S        # noqa: E402
import manage as M         # noqa: E402

HERE = Path(__file__).resolve().parent
EPOCHS = M.EPOCHS
K = 3          # свинг по 3 свечам с каждой стороны: самая крупная из проверенных структур
HARD = 20.     # катастрофический предел, лучший из плоских


def per_day(f):
    """Структурные факты каждого дня, все — на префиксе."""
    rows = []
    for i in range(len(f['dates'])):
        live = f['live'][i]
        m = int(live.sum())
        if m < 3 * K + 2:
            continue
        w, b, path = f['worst'][i][:m], f['best'][i][:m], f['path'][i][:m]
        lows, _ = S.swings(w, b, live, K)
        if not lows:
            continue
        j0, lvl0, conf0 = lows[0]
        after = w[conf0 + 1:]
        hit = np.where(after <= lvl0)[0]
        broke_at = int(conf0 + 1 + hit[0]) if len(hit) else None
        # подряд идущие перелои: каждый следующий подтверждённый низ ниже предыдущего
        run, best_run, prev = 0, 0, None
        run_at = None
        reach = {}
        for j, lvl, conf in lows:
            if prev is not None and lvl < prev:
                run += 1
                if run not in reach:
                    reach[run] = conf          # первый момент, когда счёт дошёл до run
                if run > best_run:
                    best_run, run_at = run, conf
            else:
                run = 0
            prev = lvl
        rows.append(dict(date=str(f['dates'][i]), i=i, minutes=m, final=float(path[m - 1]),
                         win=float(path[m - 1]) > 0,
                         first_at=j0, first_depth=lvl0, first_conf=conf0,
                         broke_at=broke_at,
                         broke_by_30=bool(broke_at is not None and broke_at <= 30),
                         lower_low_run=best_run, run_at=run_at,
                         at_run2=reach.get(2), at_run3=reach.get(3), at_run4=reach.get(4),
                         n_lows=len(lows)))
    return pd.DataFrame(rows)


def simulate(f, days, rule):
    """rule(row) -> индекс минуты выхода или None. Плюс жёсткий предел HARD."""
    n = len(f['dates'])
    out = np.full(n, np.nan)
    for _, r in days.iterrows():
        i = int(r.i)
        m = int(r.minutes)
        w, path = f['worst'][i][:m], f['path'][i][:m]
        hard_hit = np.where(w <= -HARD)[0]
        cut = rule(r)
        cut = None if cut is None or (isinstance(cut, float) and np.isnan(cut)) else int(cut)
        if cut is not None and cut < m and (not len(hard_hit) or cut < hard_hit[0]):
            out[i] = path[cut]
        elif len(hard_hit):
            out[i] = -HARD
        else:
            out[i] = path[m - 1]
    left = np.isnan(out)
    for i in np.where(left)[0]:
        m = int(f['live'][i].sum())
        w, path = f['worst'][i][:m], f['path'][i][:m]
        hard_hit = np.where(w <= -HARD)[0]
        out[i] = -HARD if len(hard_hit) else path[m - 1]
    return out


def money(candles, f, sel):
    usd = candles * f['unit'] * f['point'] - f['cost']
    d = pd.Series(usd[sel], index=f['dates'][sel]).groupby(level=0).sum().sort_index()
    eq = np.r_[0, np.cumsum(d.to_numpy())]
    dd = float((np.maximum.accumulate(eq) - eq).max())
    return dict(total=round(float(d.sum())), dd=round(dd),
                rr=round(float(d.sum()) / dd, 2) if dd else None,
                med=round(float(d.median())), worst=round(float(d.min())))


def draw(f, d, out='stencil_overlay.json'):
    """Трафарет: все дни, наложенные по минуте входа, в виде огибающих и выборки путей."""
    sel = d.date.str.slice(0, 4) >= '2020'
    payload = {'unit': 'свечи от входа', 'anchor': 'минута входа', 'epoch': '2020-2026', 'groups': {}}
    for lab, rows in [('win', d.loc[sel & d.win]), ('loss', d.loc[sel & ~d.win])]:
        stack = np.full((len(rows), 120), np.nan)
        for r, (_, row) in enumerate(rows.iterrows()):
            m = int(row.minutes)
            stack[r, :m] = f['path'][int(row.i)][:m]
        env = {str(q): [round(float(v), 3) for v in np.nanpercentile(stack, q, axis=0)]
               for q in (10, 25, 50, 75, 90)}
        step = max(1, len(rows) // 120)
        payload['groups'][lab] = dict(
            n=int(len(rows)), envelope=env,
            first_swing_confirmed_median=float(rows.first_conf.median()),
            first_swing_depth_median=round(float(rows.first_depth.median()), 2),
            sample=[[round(float(v), 2) if np.isfinite(v) else None for v in stack[r]]
                    for r in range(0, len(rows), step)][:120])
    (HERE / out).write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8', newline=chr(10))
    print('трафарет сохранён: setups/S-07/' + out)


def main():
    f = S.films('NQ')
    year = pd.Series(f['dates']).str.slice(0, 4).to_numpy()
    d = per_day(f)
    d['year'] = d.date.str.slice(0, 4)
    report = {}

    print(f'=== 1. Глубина первого подтверждённого свинга (k={K}) против исхода ===')
    for ep, a, b in EPOCHS:
        g = d.loc[(d.year >= a) & (d.year <= b)].copy()
        g['q'] = pd.qcut(g.first_depth, 5, labels=False, duplicates='drop')
        agg = g.groupby('q', observed=True).agg(n=('final', 'size'), depth=('first_depth', 'median'),
                                                final=('final', 'mean'), win=('win', 'mean'))
        print(f'  {ep}: ' + '  '.join(
            f'[{r.depth:6.2f}] ход {r.final:+5.2f} доля+ {r.win:.0%}' for _, r in agg.iterrows()))
    report['depth_quintiles'] = 'см. вывод'

    print()
    print('=== 2. Сколько перелоев подряд успевает сделать день ===')
    for ep, a, b in EPOCHS:
        g = d.loc[(d.year >= a) & (d.year <= b)]
        t = g.groupby('lower_low_run').agg(n=('final', 'size'), final=('final', 'mean'), win=('win', 'mean'))
        t = t.loc[t.n >= 30]
        print(f'  {ep}: ' + '  '.join(
            f'{int(k)} подряд: ход {r.final:+5.2f} доля+ {r.win:.0%} (n={int(r.n)})' for k, r in t.iterrows()))

    print()
    print('=== 3. Снесён ли первый низ к 30-й минуте (состояние, видимое на префиксе) ===')
    for ep, a, b in EPOCHS:
        g = d.loc[(d.year >= a) & (d.year <= b)]
        for lab, sel in [('снесён', g.broke_by_30), ('не снесён', ~g.broke_by_30)]:
            gg = g.loc[sel]
            print(f'  {ep} {lab:<10} n={len(gg):>4} остаток хода {gg.final.mean():+5.2f} доля+ {gg.win.mean():.0%}')

    print()
    print('=== 4. Структурные выходы против плоского стопа 20 свечей ===')
    rules = {
        'ничего, только предел 20': lambda r: None,
        'выход на 2-м перелое подряд': lambda r: r.at_run2,
        'выход на 3-м перелое подряд': lambda r: r.at_run3,
        'выход на 4-м перелое подряд': lambda r: r.at_run4,
        'выход при сносе первого низа': lambda r: r.broke_at,
        'снос первого низа, если он глубже 4 свечей': lambda r: r.broke_at if r.first_depth <= -4 else None,
        'выход при подтверждении первого свинга глубже 8 свечей': lambda r: r.first_conf if r.first_depth <= -8 else None,
    }
    rows = []
    for name, fn in rules.items():
        c = simulate(f, d, fn)
        row = {'правило': name}
        for ep, a, b in EPOCHS:
            s = money(c, f, (year >= a) & (year <= b))
            row[ep] = f"{s['total']:>9,} / {s['rr']}"
        s = money(c, f, year >= '2020')
        row['медиана'] = s['med']
        row['худший день'] = s['worst']
        rows.append(row)
    grid = pd.DataFrame(rows)
    print(grid.to_string(index=False))
    draw(f, d)
    grid.to_csv(HERE / 'stencil2_v1.csv', index=False, lineterminator='\n')
    (HERE / 'stencil2_v1.json').write_text(json.dumps(
        {'what': 'S-07: несёт ли структура фильма информацию, если стоп в неё не ставится',
         'swing_k': K, 'hard_limit_candles': HARD,
         'caveat': 'разбор на уже просмотренной истории; независимым подтверждением не является'},
        ensure_ascii=False, indent=1), encoding='utf-8', newline='\n')


if __name__ == '__main__':
    main()
