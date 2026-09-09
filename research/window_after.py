#!/usr/bin/env python3
"""Что остаётся ВНУТРИ окна после узнавания начала, найденного в 064.

ПРЕДМЕТ
=======
064 установил: начало «первая минута после реализации не закрылась обратно за
закрытие минуты реализации» различает продолжения — доля расширения 0,722
против 0,550 — и различает их рано, уже на m = 1. Удержание до `close m=15`
эту разницу не монетизировало.

Этот проход спрашивает другое: **какое развитие остаётся после узнавания и
связывается ли оно с наблюдаемым исполнимым действием.** Удержание до конца
окна — лишь один способ распорядиться окном; возможности внутри окна прежде
не разбирались.

ЧТО ЗАФИКСИРОВАНО И НЕ ПОДБИРАЕТСЯ
==================================
Состав формы, сторона, популяция, окно и вход взяты из 064 без изменений.
ATR, размер свечи и Volume в определение формы не вводятся. Цены здесь только
измеряют результат и пределы конкретного входа.

ПОРЯДОК ЧТЕНИЯ
==============
1. объект: популяция, форма, сторона, вход, часы, доступность события;
2. траектория: весь путь от входа до конца окна в отношениях;
3. деньги: ход от фактической цены входа, времена, распределения;
4. действие: одна версия, исторический расчёт с расходом и одной позицией;
5. контроль: симметричные барьеры — есть ли наклон пути помимо порядка.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from relational_stencil import Corpus                               # noqa: E402
from anchor_control import sample_control, tape                     # noqa: E402
from candidate_check import POINT, COST, sides                      # noqa: E402
from ordinal_events import (NONE, UNKNOWN, VERSION, crossing_event,  # noqa: E402
                            point_level, prefix_extreme_level)
import after_realisation as AR                                      # noqa: E402

MIN = 60_000_000_000
BACK, AHEAD = AR.BACK, AR.AHEAD
CURSOR = 1                     # курсор узнавания из 064
ENTRY = CURSOR + 1             # доступный вход: open m=2


# ---------------------------------------------------------------- объект ----

def build(instrument, riz_path, controls_per_anchor, seed):
    """Тот же корпус, что в 064: обычные минуты ленты, заякоренные на реализации."""
    m = tape(instrument)
    riz = Corpus.load(riz_path)
    control = sample_control(riz, instrument, controls_per_anchor, seed)
    anchors = np.array([int(u.split(':')[1]) for u in control.unit_ids])
    side, _, _ = sides(control)
    rows = AR.realisation_anchors(anchors, side, m)
    corpus, rows = AR.build_corpus(rows, m, instrument, 'control')
    corpus = AR.orient(corpus, rows)
    classes = AR.continuation_classes(corpus, CURSOR)
    idx = np.flatnonzero(classes['known'])
    sub, sub_rows = AR.subcorpus(corpus, idx, rows)
    return m, sub, sub_rows, int((~classes['known']).sum())


def form_mask(sub):
    """Начало из 064: wick_below(maxH[-5..0])[1..1]=1 и close_below(C[0])[1..1]=none."""
    pre_hi = prefix_extreme_level(sub, -BACK, 0, 'H')
    c0 = point_level(sub, 0, 'C')
    a = crossing_event(sub, 1, CURSOR, 'L', pre_hi, False)
    b = crossing_event(sub, 1, CURSOR, 'C', c0, False)
    known = (a.values != UNKNOWN) & (b.values != UNKNOWN)
    return known & (a.values == 1) & (b.values == NONE), a, b


def object_block(sub, rows, mask, a, b, unfinished):
    """Явное различение: событие впереди, уже произошло или наступает повторно."""
    col = {int(o): k for k, o in enumerate(sub.ordinals)}
    C = sub.ohlc[:, :, 3]
    hi1 = prefix_extreme_level(sub, -BACK, CURSOR, 'H')[0]
    side = np.array([r['side'] for r in rows])
    edge = np.array([r['edge'] for r in rows]) * np.where(side < 0, -1, 1)
    already = int((C[:, col[0]] > edge).sum())
    ahead = int((C[:, col[CURSOR]] <= hi1).sum())
    return {
        'population': 'обычные минуты ленты (control) с раскладкой по времени суток '
                      'якорей RIZ; результат не переносится на RIZ автоматически',
        'units_with_known_outcome': int(sub.n),
        'unfinished_observation': unfinished,
        'form': {
            'rule': 'wick_below(maxH[-5..0])[1..1] = 1 & close_below(C[0])[1..1] = none',
            'plain': 'первая минута после реализации не закрылась обратно '
                     'за закрытие минуты реализации',
            'recognised_at': CURSOR,
            'n_with': int(mask.sum()), 'n_without': int((~mask).sum()),
            'supports': ['maxH[-5..0]', 'C[0]'],
            'part_a_alone': float((a.values == 1).mean()),
            'part_b_alone': float((b.values == NONE).mean())},
        'side': 'сторона взята из конструкции первых пяти минут; юг отражён по цене, '
                'вверх = по ходу конструкции',
        'entry': 'open m=2 (первая минута после узнавания)',
        'clock': 'm — локальные минуты от минуты реализации; T0 и сдвиг реализации '
                 'сохранены в идентификаторе каждой единицы (NQ:<T0 ns>:r+<k>)',
        'event_availability_at_entry': {
            'close_beyond_original_edge_already_happened': already,
            'note_already': 'реализация сама и есть закрытие за краем первых пяти '
                            'минут: относительно ИСХОДНОГО края событие уже '
                            'произошло во всех случаях, и то, что меряется дальше, '
                            'есть ПОВТОРНОЕ наступление события того же рода',
            'close_above_maxH_-5_1_still_ahead': ahead,
            'note_ahead': 'относительно опоры, обновлённой к курсору, событие '
                          'впереди по построению: уровень включает high[1] >= close[1]'},
        'levels_at_entry': {}}


# ------------------------------------------------------------ траектория ----

def events(sub):
    """Отношения оставшегося пути к опорам, доступным на момент входа."""
    hi1 = prefix_extreme_level(sub, -BACK, CURSOR, 'H')
    lo1 = prefix_extreme_level(sub, -BACK, CURSOR, 'L')
    c0 = point_level(sub, 0, 'C')
    l1 = point_level(sub, CURSOR, 'L')
    return {
        'close>maxH1': crossing_event(sub, ENTRY, AHEAD, 'C', hi1, True),
        'close<minL1': crossing_event(sub, ENTRY, AHEAD, 'C', lo1, False),
        'high>maxH1': crossing_event(sub, ENTRY, AHEAD, 'H', hi1, True),
        'low<minL1': crossing_event(sub, ENTRY, AHEAD, 'L', lo1, False),
        'close<C0': crossing_event(sub, ENTRY, AHEAD, 'C', c0, False),
        'low<C0': crossing_event(sub, ENTRY, AHEAD, 'L', c0, False),
        'low<L1': crossing_event(sub, ENTRY, AHEAD, 'L', l1, False)}


def order_signature(ev, i, limit=None):
    """Порядок событий одного фильма. Совпавшие минуты остаются без порядка."""
    seen = [(int(e.values[i]), name) for name, e in ev.items() if e.values[i] >= 0]
    if limit is not None:
        seen = [(mnt, nm) for mnt, nm in seen if mnt <= limit]
    if not seen:
        return 'нет события'
    by = {}
    for mnt, nm in seen:
        by.setdefault(mnt, []).append(nm)
    return ' -> '.join('|'.join(sorted(by[mnt])) + f'@{mnt}' for mnt in sorted(by))


def trajectory_block(sub, ev, mask, top=12):
    """Что повторяется совместно и какие разные продолжения у общего начала."""
    col = {int(o): k for k, o in enumerate(sub.ordinals)}
    C = sub.ohlc[:, :, 3]
    hi1 = prefix_extreme_level(sub, -BACK, CURSOR, 'H')[0]
    seg = C[:, col[ENTRY]:col[AHEAD] + 1]
    above = seg > hi1[:, None]
    flips = (above[:, 1:] != above[:, :-1]).sum(axis=1)

    def group(sel, limit):
        sig = {}
        for i in np.flatnonzero(sel):
            sig.setdefault(order_signature(ev, i, limit), []).append(sub.unit_ids[i])
        total = max(1, int(sel.sum()))
        return [{'signature': s, 'n': len(members), 'share': len(members) / total,
                 'examples': members[:3]}
                for s, members in sorted(sig.items(), key=lambda kv: -len(kv[1]))[:top]]

    first = Counter()
    for i in np.flatnonzero(mask):
        first[order_signature(ev, i).split(' -> ')[0]] += 1

    ext = ev['close>maxH1'].values
    returned = []
    for i in np.flatnonzero(mask):
        if ext[i] >= 0:
            returned.append(int((C[i, col[int(ext[i])]:col[AHEAD] + 1] < hi1[i]).any()))
    absent = {name: {'never_in_window': int((e.values == NONE)[mask].sum()),
                     'unknown_end_of_observation': int((e.values == UNKNOWN)[mask].sum()),
                     'median_minute': (float(np.median(e.values[mask & (e.values >= 0)]))
                                       if (mask & (e.values >= 0)).any() else None)}
              for name, e in ev.items()}
    return {
        'full_path_signatures_with_form': group(mask, None),
        'full_path_signatures_without_form': group(~mask, None),
        'first_event_after_entry_with_form': dict(first.most_common(top)),
        'until_return_slice': {
            'note': 'дополнительный срез по первым четырём минутам после входа; '
                    'дальнейшая история сохранена в полном пути выше',
            'signatures': group(mask, ENTRY + 3)},
        'event_absence': absent,
        'state_changes_close_vs_maxH1': dict(Counter(flips[mask].tolist()).most_common(8)),
        'share_returned_under_maxH1_after_extension':
            float(np.mean(returned)) if returned else None,
        'n_with_extension': len(returned)}


# ---------------------------------------------------------------- деньги ----

def _dist(x, ps=(5, 10, 25, 50, 75, 90, 95)):
    q = np.percentile(x, ps)
    out = {'n': int(len(x)), 'mean': float(np.mean(x))}
    for p, v in zip(ps, q):
        out[f'q{p:02d}' if p != 50 else 'median'] = float(v)
    return out


def money_block(sub, ev, mask, instrument):
    """Ход от фактической цены входа. Верхняя оценка при известном будущем."""
    col = {int(o): k for k, o in enumerate(sub.ordinals)}
    O, H, L = sub.ohlc[:, :, 0], sub.ohlc[:, :, 1], sub.ohlc[:, :, 2]
    p, cost = POINT[instrument], COST[instrument]
    e = O[:, col[ENTRY]]
    a, b = col[ENTRY], col[AHEAD] + 1
    mfe = (H[:, a:b].max(axis=1) - e) * p
    mae = (L[:, a:b].min(axis=1) - e) * p
    t_mfe = np.argmax(H[:, a:b], axis=1) + ENTRY
    t_mae = np.argmin(L[:, a:b], axis=1) + ENTRY
    hi1 = prefix_extreme_level(sub, -BACK, CURSOR, 'H')[0]
    lo1 = prefix_extreme_level(sub, -BACK, CURSOR, 'L')[0]

    out = {'entry_price': 'open m=2', 'point_value': p, 'cost_per_turn': cost,
           'note_mfe': 'максимальный благоприятный ход — оптимистическая верхняя '
                       'оценка при известном будущем; его превышение над расходом '
                       'ещё не означает существования исполнимой сделки',
           'groups': {}}
    for name, sel in (('with_form', mask), ('without_form', ~mask)):
        out['groups'][name] = {
            'mfe_usd': _dist(mfe[sel]), 'mae_usd': _dist(mae[sel]),
            'minute_of_mfe': _dist(t_mfe[sel].astype(float)),
            'minute_of_mae': _dist(t_mae[sel].astype(float)),
            'share_mfe_over_cost': float((mfe[sel] > cost).mean()),
            'share_mfe_over_2cost': float((mfe[sel] > 2 * cost).mean()),
            'share_mae_under_minus_cost': float((mae[sel] < -cost).mean()),
            'share_mfe_at_entry_minute': float((t_mfe[sel] == ENTRY).mean()),
            'distance_to_upper_barrier_usd': _dist((hi1[sel] - e[sel]) * p),
            'distance_to_lower_barrier_usd': _dist((e[sel] - lo1[sel]) * p),
            'share_upper_nearer_than_lower':
                float(((hi1[sel] - e[sel]) < (e[sel] - lo1[sel])).mean())}
    link = {}
    ext, ret = ev['close>maxH1'].values, ev['close<minL1'].values
    for name, sel in (('extension_first', mask & (ext >= 0) & ((ret < 0) | (ret > ext))),
                      ('give_back_first', mask & (ret >= 0) & ((ext < 0) | (ext > ret))),
                      ('neither', mask & (ext < 0) & (ret < 0))):
        if sel.any():
            link[name] = {'n': int(sel.sum()), 'mfe_usd': _dist(mfe[sel]),
                          'mae_usd': _dist(mae[sel])}
    out['by_first_event'] = link
    out['_mfe'], out['_mae'], out['_e'] = mfe, mae, e
    out['_t'] = (t_mfe, t_mae)
    return out


def after_extension(sub, ev, mask, instrument):
    """Что остаётся ПОСЛЕ подтверждения расширения: вход open минуты k+1."""
    col = {int(o): k for k, o in enumerate(sub.ordinals)}
    O, H, L, C = (sub.ohlc[:, :, k] for k in range(4))
    p = POINT[instrument]
    ext, ret = ev['close>maxH1'].values, ev['close<minL1'].values
    rows = []
    for i in np.flatnonzero(mask):
        k = int(ext[i])
        if k < 0 or (ret[i] >= 0 and ret[i] < k) or k + 1 > AHEAD:
            continue
        e = O[i, col[k + 1]]
        s = slice(col[k + 1], col[AHEAD] + 1)
        rows.append((k, (H[i, s].max() - e) * p, (L[i, s].min() - e) * p,
                     (C[i, col[AHEAD]] - e) * p))
    if not rows:
        return {'n': 0}
    r = np.array(rows, dtype=float)
    by_k = {}
    for lo, hi in ((2, 3), (4, 6), (7, 10), (11, 14)):
        s = (r[:, 0] >= lo) & (r[:, 0] <= hi)
        if s.any():
            by_k[f'{lo}..{hi}'] = {'n': int(s.sum()),
                                   'to_close_m15_mean': float(r[s, 3].mean()),
                                   'mfe_median': float(np.median(r[s, 1])),
                                   'mae_median': float(np.median(r[s, 2]))}
    return {'n': len(r), 'entry': 'open минуты k+1 после закрытия выше maxH[-5..1]',
            'minute_k': _dist(r[:, 0], (10, 25, 50, 75, 90)),
            'mfe_usd': _dist(r[:, 1], (10, 25, 50, 75, 90)),
            'mae_usd': _dist(r[:, 2], (10, 25, 50, 75, 90)),
            'to_close_m15_usd': _dist(r[:, 3], (10, 25, 50, 75, 90)),
            'share_positive_to_close': float((r[:, 3] > 0).mean()), 'by_minute_k': by_k}


# -------------------------------------------------------------- действие ----

def action_run(sub, rows, ev, mask, m, instrument, apply_form=True):
    """Историческая прокрутка версии действия. Одна позиция, расход за оборот.

    Вход `open m=2`. Выход — `open` минуты, следующей за минутой события,
    которое завершает или нарушает ожидание. Оба события наблюдаемы на своём
    закрытии; будущий лучший экстремум выход не назначает. Совпадение обоих
    событий на одной минуте разрешается в пользу нарушения.
    """
    col = {int(o): k for k, o in enumerate(sub.ordinals)}
    O, C = sub.ohlc[:, :, 0], sub.ohlc[:, :, 3]
    p, cost = POINT[instrument], COST[instrument]
    ts = m['close_ts_utc_ns']
    ext, ret = ev['close>maxH1'].values, ev['close<minL1'].values
    order = np.argsort([r['r_index'] for r in rows])
    busy, trades = -1, []
    for i in order:
        if apply_form and not mask[i]:
            continue
        base = int(rows[i]['r_index'])
        want = ts[base] + np.arange(ENTRY, AHEAD + 1) * MIN
        at = np.searchsorted(ts, want)
        if at[-1] >= len(ts) or not np.all(ts[at] == want):
            continue
        if int(ts[at[0]]) <= busy:
            continue
        e = O[i, col[ENTRY]]
        cand = []
        if ret[i] >= 0:
            cand.append((int(ret[i]), 'нарушено'))
        if ext[i] >= 0:
            cand.append((int(ext[i]), 'исполнено'))
        if cand:
            k = min(c[0] for c in cand)
            why = ('нарушено' if any(c[1] == 'нарушено' and c[0] == k for c in cand)
                   else 'исполнено')
            out_m = min(k + 1, AHEAD)
            px = O[i, col[out_m]] if k + 1 <= AHEAD else C[i, col[AHEAD]]
        else:
            why, out_m, px = 'конец окна', AHEAD, C[i, col[AHEAD]]
        gross = float((px - e) * p)
        trades.append({'unit': sub.unit_ids[i], 'ts_entry': int(ts[at[0]]),
                       'minute_out': int(out_m), 'why': why,
                       'gross': gross, 'net': gross - cost})
        busy = int(ts[base] + out_m * MIN)
    return trades


def summarise(trades, cost):
    if not trades:
        return {'n': 0}
    net = np.array([t['net'] for t in trades])
    gross = np.array([t['gross'] for t in trades])
    tsx = np.array([t['ts_entry'] for t in trades])
    year = (tsx // MIN) // int(365.2425 * 1440) + 1970
    by = {int(y): float(net[year == y].sum()) for y in np.unique(year)}
    eq = np.cumsum(net)
    sem = float(gross.std(ddof=1) / np.sqrt(len(gross)))
    per_why = {}
    for k in sorted(set(t['why'] for t in trades)):
        s = np.array([t['why'] == k for t in trades])
        per_why[k] = {'n': int(s.sum()), 'mean_gross': float(gross[s].mean()),
                      'share_positive': float((gross[s] > 0).mean())}
    return {'n': len(trades), 'mean_gross': float(gross.mean()),
            'mean_net': float(net.mean()), 'total_net': float(net.sum()),
            'sem_gross': sem, 't_gross': float(gross.mean() / sem) if sem else None,
            'share_positive_gross': float((gross > 0).mean()),
            'median_gross': float(np.median(gross)),
            'drawdown_net': float(np.max(np.maximum.accumulate(eq) - eq)),
            'by_year_net': by, 'by_reason': per_why, 'cost_per_turn': cost}


def symmetric_control(sub, mask, instrument, targets):
    """Есть ли наклон пути помимо порядка? Барьеры равны по расстоянию."""
    col = {int(o): k for k, o in enumerate(sub.ordinals)}
    O, H, L, C = (sub.ohlc[:, :, k] for k in range(4))
    p = POINT[instrument]
    e = O[:, col[ENTRY]]
    a, b = col[ENTRY], col[AHEAD] + 1
    hh = (H[:, a:b] - e[:, None]) * p
    ll = (L[:, a:b] - e[:, None]) * p
    cc = (C[:, a:b] - e[:, None]) * p
    big = 10 ** 6
    out = []
    for X in targets:
        row = {'target_usd': float(X)}
        for name, sel in (('with_form', mask), ('without_form', ~mask)):
            ht, hs = hh[sel] >= X, ll[sel] <= -X
            ft = np.where(ht.any(1), ht.argmax(1), big)
            fs = np.where(hs.any(1), hs.argmax(1), big)
            win, lose = ft < fs, fs <= ft
            none = (ft == big) & (fs == big)
            res = np.where(win, float(X), np.where(lose, -float(X), cc[sel][:, -1]))
            res = np.where(none, cc[sel][:, -1], res)
            row[name] = {'share_target_first': float(win.mean()),
                         'mean_gross': float(res.mean()),
                         'mean_net': float(res.mean() - COST[instrument])}
        row['difference_share'] = (row['with_form']['share_target_first']
                                   - row['without_form']['share_target_first'])
        out.append(row)
    return {'note': 'цель и стоп на равном расстоянии; совпадение на одной минуте '
                    'разрешается в пользу стопа; недостигнутое закрывается на '
                    'close m=15. Уровни X прочитаны с распределения MFE, не назначены',
            'rows': out}


# ------------------------------------------------------------------ main ----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--riz-input', default='work/riz-pass/tf5_south.npz')
    ap.add_argument('--instrument', default='NQ')
    ap.add_argument('--controls-per-anchor', type=int, default=8)
    ap.add_argument('--seed', type=int, default=5)
    ap.add_argument('--out', required=True)
    ap.add_argument('--trades-csv', default=None)
    ap.add_argument('--films-csv', default=None)
    a = ap.parse_args()

    m, sub, rows, unfinished = build(a.instrument, a.riz_input,
                                     a.controls_per_anchor, a.seed)
    mask, ea, eb = form_mask(sub)
    ev = events(sub)
    obj = object_block(sub, rows, mask, ea, eb, unfinished)
    traj = trajectory_block(sub, ev, mask)
    money = money_block(sub, ev, mask, a.instrument)
    obj['levels_at_entry'] = {
        'median_to_upper_barrier_usd':
            money['groups']['with_form']['distance_to_upper_barrier_usd']['median'],
        'median_to_lower_barrier_usd':
            money['groups']['with_form']['distance_to_lower_barrier_usd']['median'],
        'note': 'барьеры, между которыми различает найденная форма, не равны по '
                'расстоянию от цены входа'}

    cost = COST[a.instrument]
    with_form = action_run(sub, rows, ev, mask, m, a.instrument, True)
    all_cases = action_run(sub, rows, ev, mask, m, a.instrument, False)
    g = money['groups']['with_form']['mfe_usd']
    targets = sorted(set([35, 75] + [round(g[k]) for k in ('q25', 'median', 'q75')]))

    result = {
        'version': VERSION,
        'implementation_sha256': hashlib.sha256(
            Path(__file__).read_bytes()).hexdigest(),
        'subject': 'что остаётся внутри окна после узнавания начала из 064',
        'object': obj,
        'trajectory': traj,
        'money': {k: v for k, v in money.items() if not k.startswith('_')},
        'after_extension_confirmation': after_extension(sub, ev, mask, a.instrument),
        'action_version': {
            'seen': 'реализация ближайшего ожидания (m=0) и первая минута без '
                    'отката по закрытию (форма m=1)',
            'expected': 'повторное расширение: закрытие выше maxH[-5..1] раньше, '
                        'чем закрытие ниже minL[-5..1]',
            'completes': 'закрытие выше maxH[-5..1]',
            'breaks': 'закрытие ниже minL[-5..1]; совпадение на одной минуте '
                      'разрешается в пользу нарушения',
            'end_of_observation': 'ни того ни другого к m=15 — выход на close m=15',
            'decide_at': 'закрытие минуты, где событие наступило',
            'execute': 'вход open m=2, выход open следующей минуты после события',
            'grounds_available_then': 'опоры maxH[-5..1] и minL[-5..1] закрыты к m=1; '
                                      'будущий лучший экстремум выход не назначает',
            'runs': {
                'recognised_stream': summarise(with_form, cost),
                'all_admissible_without_recognition': summarise(all_cases, cost)},
            'stream_note': 'колонка «без распознавания» — тот же поток, применённый '
                           'ко ВСЕЙ допустимой популяции, а не к случаям без формы; '
                           'оба потока подчинены правилу одной позиции и потому не '
                           'равны сумме исследованных фильмов'},
        'symmetric_barrier_control': symmetric_control(sub, mask, a.instrument, targets),
        'films_studied': int(sub.n),
        'films_with_form': int(mask.sum()),
        'executable_trades_recognised': len(with_form),
        'note_films_vs_trades': 'множество исследованных фильмов и фактически '
                                'исполнимый поток сделок различны: правило одной '
                                'позиции отсекает перекрывающиеся окна'}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, ensure_ascii=False, indent=1,
                                      default=str), encoding='utf-8')

    if a.trades_csv and with_form:
        with open(a.trades_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(with_form[0]))
            w.writeheader()
            w.writerows(with_form)
    if a.films_csv:
        mfe, mae, e = money['_mfe'], money['_mae'], money['_e']
        t_mfe, t_mae = money['_t']
        with open(a.films_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['unit', 'form', 'signature', 'mfe_usd', 'mae_usd',
                        'minute_mfe', 'minute_mae', 'entry_open_oriented'])
            for i in range(sub.n):
                w.writerow([sub.unit_ids[i], int(mask[i]), order_signature(ev, i),
                            round(float(mfe[i]), 1), round(float(mae[i]), 1),
                            int(t_mfe[i]), int(t_mae[i]), float(e[i])])
    run = result['action_version']['runs']['recognised_stream']
    print(json.dumps({
        'films': result['films_studied'], 'with_form': result['films_with_form'],
        'trades': result['executable_trades_recognised'],
        'action_mean_gross': run['mean_gross'], 'action_mean_net': run['mean_net'],
        'action_t': run['t_gross'],
        'mfe_median': money['groups']['with_form']['mfe_usd']['median'],
        'mae_median': money['groups']['with_form']['mae_usd']['median']},
        ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
