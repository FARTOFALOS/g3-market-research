#!/usr/bin/env python3
"""Проверочные сцены 080A: реальные хронологии, читаемые трейдером.

Каждая сцена — настоящий объект замороженного поля. Хронология всегда одна и та
же: `T0 известен → q доступны → reference executions → contact либо потеря
достоверного наблюдения`. Ничего не изобретается: если тип не найден, так и
сказано.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'research'))
from grid import Film1Index                                          # noqa: E402
from calendar_utils import eastern                                   # noqa: E402

MIN = 60_000_000_000


def et(ns):
    return str(eastern([int(ns)])[0])[:19]


def chronology(idx: Film1Index, riz_id, max_q=6):
    r = idx.row(riz_id)
    qs = idx.q_positions(riz_id)
    c = int(r.first_observed_contact_pos)
    lines = []
    lines.append(f'  RIZ {riz_id}  {idx.instrument} TF{int(r.tf_minutes)}  side={r.side}')
    lines.append(f'  зона [{r.zone_bottom:g} .. {r.zone_top:g}]  '
                 f'exit boundary {r.exit_boundary:g}  far {r.far_boundary:g}')
    lines.append(f'  T0 известен: {et(r.t0_ts_ns)} ET  spine {int(r.t0_spine_pos)}  '
                 f'close {r.t0_close:g}')
    lines.append(f'  q доступны: {qs.size} шт., от spine {int(qs[0])} до {int(qs[-1])}'
                 if qs.size else '  q доступны: нет')
    show = list(qs[:max_q]) + (['...'] if qs.size > max_q * 2 else []) + \
        list(qs[-max_q:]) if qs.size > max_q else list(qs)
    seen = set()
    for q in show:
        if q == '...':
            lines.append('        ...')
            continue
        q = int(q)
        if q in seen:
            continue
        seen.add(q)
        p = idx.prefix(riz_id, q)
        x = idx.execution_reference(riz_id, q)
        dist = ('—' if x.distance_to_tp is None
                else f'{x.distance_to_tp:.2f}')
        lines.append(
            f'        q={q} (+{p.observed_bars_since_t0} бар, '
            f'{p.wall_clock_minutes_since_t0} мин часов, пропущено '
            f'{p.missing_minutes_since_t0}) close {p.close_q:g} | '
            f'reference open {("—" if x.open_price is None else format(x.open_price, "g"))} '
            f'[{x.availability}/{x.gap_kind}, ждать {x.wait_minutes} мин] '
            f'{x.directional_status}, до TP {dist} | {p.lifecycle_state_as_of_q}, '
            f'{p.freshness_as_of_q}')
    if c >= 0:
        lines.append(f'  observed contact: spine {c} = '
                     f'{et(r.first_observed_contact_ts_ns)} ET  '
                     f'(+{int(r.observed_bars_t0_to_end)} наблюдённых бар, '
                     f'{int(r.wall_clock_minutes_t0_to_end)} мин часов)')
    else:
        lines.append('  observed contact: не наблюдался до края архива')
    cont = idx.continuation(riz_id, int(qs[-1]) if qs.size else int(r.t0_spine_pos))
    lines.append(f'  continuation (strict): {cont.continuation_status}; '
                 f'сертифицированный TP '
                 f'{"нет" if cont.certified_tp_pos is None else cont.certified_tp_pos}; '
                 f'остаток {"пуст" if cont.remaining_range is None else cont.remaining_range}')
    lines.append(f'  primacy: {r.first_observed_contact_primacy}   '
                 f'status: {r.film1_status}   '
                 f'(shared-survived: {r.first_observed_contact_primacy_shared_survived} / '
                 f'{r.film1_status_shared_survived})')
    if c >= 0:
        ex = idx.excursion_to_tp(riz_id)
        if ex['kind'] == 'bounded':
            lines.append(
                f'  от reference entry {ex["entry_price"]:g}: до TP '
                f'{ex["distance_entry_to_tp"]:.2f} | против цели ГРАНИЦЫ '
                f'[{ex["adverse_excursion_lower_bound"]:.2f} .. '
                f'{ex["adverse_excursion_upper_bound"]:.2f}], порядок внутри '
                f'касательной свечи неизвестен; её OHLC {ex["touch_candle_ohlc"]}')
        elif ex['kind'] == 'exact':
            lines.append(
                f'  от reference entry {ex["entry_price"]:g}: до TP '
                f'{ex["distance_entry_to_tp"]:.2f} | против цели точно '
                f'{ex["adverse_excursion_to_tp"]:.2f} (касательная свеча своего '
                f'максимума против цели не поднимает)')
        elif ex['kind'] == 'primacy_not_certified':
            lines.append(
                f'  экскурсия до TP НЕ считается: первичность не доказана при '
                f'{ex["certification"]}. Достоверное наблюдение кончается на '
                f'{ex["certified_fresh_until_pos"]}, наблюдавшийся позже контакт '
                f'на {ex["observed_contact_pos"]} первым касанием не является — '
                f'оно могло случиться внутри пропуска.')
        else:
            lines.append(f'  экскурсия: {ex["kind"]}')
    lines.append(f'  жизненный цикл: deletion {int(r.c1_deletion_spine_pos)}, '
                 f'blue_end {int(r.blue_eligibility_end_spine_pos)}, '
                 f'native_conf {int(r.native_blue_confirmation_spine_pos)}, '
                 f'passport censored={bool(r.censored)}')
    return '\n'.join(lines)


def pick(d, mask, n=1, seed=0):
    v = np.flatnonzero(mask)
    if not v.size:
        return []
    rng = np.random.default_rng(seed)
    return [str(x) for x in d.riz_id.to_numpy()[rng.choice(v, size=min(n, v.size),
                                                           replace=False)]]


def main(inst='NQ'):
    idx = Film1Index(inst)
    d = idx.df
    c = d.first_observed_contact_pos.to_numpy()
    t0 = d.t0_spine_pos.to_numpy()
    has = c >= 0
    lag = d.observed_bars_t0_to_end.to_numpy()
    prim = d.first_observed_contact_primacy.to_numpy()
    dele = d.c1_deletion_spine_pos.to_numpy()
    e = d.exit_boundary.to_numpy()
    o = d.t0_next_open.to_numpy()
    north = d.side.to_numpy() == 'north'
    s = np.where(north, o - e, e - o)

    # «разные RIZ одной минуты T0 с разными границами»
    g = d.groupby('t0_ts_ns').exit_boundary.nunique()
    multi_ts = g[g >= 2].index.to_numpy()
    same_minute = []
    if multi_ts.size:
        # берём минуту с наибольшим расхождением первого контакта
        sub = d[d.t0_ts_ns.isin(multi_ts[:4000])]
        spread = sub.groupby('t0_ts_ns').first_observed_contact_pos.agg(
            lambda x: x.max() - x.min())
        best = spread.idxmax()
        same_minute = d[d.t0_ts_ns == best].riz_id.tolist()[:3]

    # касательная свеча с неоднозначной экскурсией: против цели на самой свече
    tang = []
    cc = np.flatnonzero(prim == 'certified')
    for i in cc[:60000]:
        ex = idx.excursion_to_tp(str(d.riz_id.iloc[i]))
        if (ex['kind'] == 'bounded'
                and ex['adverse_excursion_upper_bound']
                > ex['adverse_excursion_lower_bound']):
            tang.append(str(d.riz_id.iloc[i]))
            break

    # Сцена, которую архитектор обязан увидеть целиком: состояние уже
    # распознаваемо, reference open ещё actionable — а оставшегося движения до
    # естественной цели почти нет, и контакт приходит на следующей же минуте.
    tick = {'ES': 0.25, 'NQ': 0.25, 'YM': 1.0}[inst]
    tiny = pick(d, has & (lag == 1) & (d.t0_next_bar_kind.to_numpy() == 0)
                & (s > 0) & (s <= 2 * tick), 2, 10)

    types = {
        'РЕШАЮЩАЯ: T0 известен → next open ещё actionable → запас до цели '
        f'<= 2 тика ({2*tick:g}) → contact на T0+1': tiny,
        'contact на T0+1': pick(d, has & (lag == 1), 1, 1),
        'несколько минут дальнейшего удаления': pick(d, has & (lag >= 5) & (lag <= 25), 1, 2),
        # Разделено намеренно: под strict сертифицированных фильмов через смену
        # сессии НЕТ НИ ОДНОГО на всей территории (0 из 56 075 наблюдаемых).
        # Каждая граница сессии несёт промежуток, и сертификация встаёт там.
        'длинный фильм через смену дня, сертифицированный (strict)': pick(
            d, has & (d.sessions_spanned.to_numpy() >= 3) & (prim == 'certified'), 1, 3),
        'длинный фильм через смену дня, наблюдаемый (первичность не доказана)': pick(
            d, has & (d.sessions_spanned.to_numpy() >= 3), 1, 3),
        'разные RIZ одной минуты T0 (разные границы)': same_minute,
        'flicker (нет нативного подтверждения)': pick(
            d, has & (d.native_blue_confirmation_spine_pos.to_numpy() < 0), 1, 4),
        'deletion до контакта': pick(d, has & (dele >= 0) & (dele < c), 1, 5),
        'потеря достоверности / missing tape до контакта': pick(d, prim == 'unknown', 1, 6),
        'контакт не наблюдался до края архива': pick(d, ~has, 1, 7),
        'reference open ровно на boundary': pick(d, (s == 0) & np.isfinite(o), 1, 8),
        'reference open за boundary': pick(d, (s < 0) & np.isfinite(o), 1, 9),
        'неоднозначная экскурсия внутри касательной свечи': tang,
    }
    print(f'############ ПРОВЕРОЧНЫЕ СЦЕНЫ — {inst} ############\n')
    for name, ids in types.items():
        print(f'### {name}')
        if not ids:
            print('  ТИП НЕ НАЙДЕН на этой территории (пример не изобретается).\n')
            continue
        for rid in ids:
            print(chronology(idx, rid))
            print()
    return types


if __name__ == '__main__':
    main(*(sys.argv[1:] or []))
