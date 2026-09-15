#!/usr/bin/env python3
"""Покрытие strict-корпуса как функция ВОЗРАСТА Film-1, и reference на каждом q.

ЗАЧЕМ ЭТО ОТДЕЛЬНО
==================
При strict все 2 287 фильмов без наблюдавшегося контакта теряют сертификацию на
промежутке, а не на краю архива. Значит strict-корпус изучает ровно ту часть
Film-1, которая успела развиться внутри непрерывного наблюдаемого участка. На
малых возрастах это почти вся популяция; на больших остаётся подвыборка,
отобранная по непрерывности ленты. 28 млн q без этой кривой читались бы как
полноценная временная поверхность.

Риск-множество на возрасте `a` объявлено до счёта: фильмы, чей Film-1 ещё НЕ
закрыт наблюдавшимся контактом на возрасте <= a. Числитель — те из них, чья
сертифицированная свежесть ещё дотягивает до `a`. Ось возраста — наблюдённые
бары ленты, не минуты часов: это разные величины, и вторая здесь не годится,
потому что решение принимается на закрытии бара.

НА КАЖДОМ q ДВА РАЗНЫХ ВОПРОСА
==============================
Доступность reference (род следующего бара) и направленная пригодность цены
хранятся раздельно, плюс знаковое исполнимое расстояние до boundary.

q НЕПОСРЕДСТВЕННО ПЕРЕД КОНТАКТОМ — ЗАКОННОЕ РЕШЕНИЕ
====================================================
Его reference — open самой касательной свечи. Если этот open ещё снаружи
границы, условный вход существует, хотя внутри той же свечи позже происходит
контакт. Поэтому случай считается отдельной строкой переписи, а не растворяется
в общей доле.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

import os
#: Где лежит компактный индекс. По умолчанию рядом с кодом; переопределяется
#: `G3_080A_INDEX`, чтобы несколько копий кода читали один и тот же индекс и не
#: плодили 120 МБ на каждую.
INDEX = Path(os.environ.get('G3_080A_INDEX', str(HERE / 'index')))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'research'))
from tape import presence_grid, gap_kinds                            # noqa: E402

INSTRUMENTS = ('ES', 'NQ', 'YM')
AGES = [0, 1, 2, 3, 5, 8, 15, 30, 60, 120, 240, 480, 1024, 2048, 4096,
        8192, 16384, 65536, 262144]
TF_BANDS = ([0, 1, 2, 5, 15, 60, 240, 1440],
            ['1', '2', '3-5', '6-15', '16-60', '61-240', '241-1440'])
# границы гистограммы исполнимого расстояния в ширинах зоны
WBINS = np.array([0.0, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, np.inf])


@njit(cache=True)
def scan_q(t0s, fresh, e, north, width, opn, kind, last, nbin, wbins):
    """Счётчики и гистограмма по всем q. Ни одна строка не сохраняется."""
    n = t0s.size
    cnt = np.zeros((n, 6), dtype=np.int64)
    dsum = np.zeros(n, dtype=np.float64)
    wsum = np.zeros(n, dtype=np.float64)
    hist = np.zeros(nbin, dtype=np.int64)
    for i in range(n):
        lvl, up, w = e[i], north[i], width[i]
        for q in range(t0s[i], fresh[i] + 1):
            cnt[i, 0] += 1
            nxt = q + 1
            if nxt > last:
                cnt[i, 5] += 1                       # reference за краем архива
                continue
            if kind[nxt] != 0:
                cnt[i, 4] += 1                       # в strict неизвестна
                continue
            cnt[i, 1] += 1                           # contiguous: определена
            s = (opn[nxt] - lvl) if up else (lvl - opn[nxt])
            if s > 0.0:
                cnt[i, 2] += 1                       # снаружи
                dsum[i] += s
                r = s / w if w > 0.0 else 0.0
                wsum[i] += r
                for b in range(nbin):
                    if r < wbins[b + 1]:
                        hist[b] += 1
                        break
            elif s == 0.0:
                cnt[i, 3] += 1                       # ровно на границе
    return cnt, dsum, wsum, hist


@njit(cache=True)
def age_curve(t0s, fresh, contact, ages):
    """Для каждого возраста: риск-множество и сколько из него сертифицировано."""
    n = t0s.size
    m = ages.size
    at_risk = np.zeros(m, dtype=np.int64)
    certified = np.zeros(m, dtype=np.int64)
    for i in range(n):
        lag = (contact[i] - t0s[i]) if contact[i] >= 0 else (1 << 60)
        cov = fresh[i] - t0s[i]
        for k in range(m):
            a = ages[k]
            if lag > a:                       # Film-1 ещё не закрыт контактом
                at_risk[k] += 1
                if cov >= a:
                    certified[k] += 1
    return at_risk, certified


def one(inst, kind):
    d = pd.read_parquet(INDEX / f'film1_{inst}.parquet')
    opn = np.asarray(np.load(ROOT / f'data/market/{inst}/open.npy'))
    last = opn.size - 1
    t0 = d.t0_spine_pos.to_numpy().astype(np.int64)
    fresh = d.certified_fresh_until_pos_strict.to_numpy().astype(np.int64)
    contact = d.first_observed_contact_pos.to_numpy().astype(np.int64)
    e = d.exit_boundary.to_numpy().astype(np.float64)
    north = (d.side.to_numpy() == 'north')
    width = (d.zone_top.to_numpy() - d.zone_bottom.to_numpy()).astype(np.float64)

    cnt, dsum, wsum, hist = scan_q(t0, fresh, e, north, width, opn, kind, last,
                                   WBINS.size - 1, WBINS)
    q_all, q_ref, q_out, q_on, q_gap, q_edge = (cnt[:, k] for k in range(6))
    beyond = q_ref - q_out - q_on

    ages = np.array(AGES, dtype=np.int64)
    at_risk, cert = age_curve(t0, fresh, contact, ages)
    curve = {str(int(a)): {'at_risk': int(r), 'certified': int(c),
                           'share': (round(float(c / r), 4) if r else None)}
             for a, r, c in zip(ages, at_risk, cert)}

    # та же кривая по нативным ТФ-полосам
    band = pd.cut(d.tf_minutes, TF_BANDS[0], labels=TF_BANDS[1])
    by_band = {}
    for lab in TF_BANDS[1]:
        m = (band == lab).to_numpy()
        if not m.any():
            continue
        ar, ce = age_curve(t0[m], fresh[m], contact[m], ages)
        by_band[lab] = {str(int(a)): (round(float(c / r), 4) if r else None)
                        for a, r, c in zip(ages, ar, ce)}
        by_band[lab]['_films'] = int(m.sum())
        by_band[lab]['_at_risk_at_60'] = int(ar[list(ages).index(60)])

    # q непосредственно перед сертифицированным контактом
    cc = d.first_observed_contact_primacy.to_numpy() == 'certified'
    cpos = contact[cc]
    open_at_contact = opn[cpos]
    e_cc, n_cc = e[cc], north[cc]
    s = np.where(n_cc, open_at_contact - e_cc, e_cc - open_at_contact)
    kind_at_contact = kind[cpos]
    r = {
        'films': int(len(d)),
        'q_certified_strict': int(q_all.sum()),
        'q_next_bar_contiguous_reference_defined': int(q_ref.sum()),
        'q_gap_before_next_bar_reference_unknown_strict': int(q_gap.sum()),
        'q_next_bar_past_archive_edge': int(q_edge.sum()),
        'q_open_outside_boundary': int(q_out.sum()),
        'q_open_exactly_on_boundary': int(q_on.sum()),
        'q_open_beyond_boundary': int(beyond.sum()),
        'share_q_reference_defined': round(float(q_ref.sum() / q_all.sum()), 5),
        'share_q_open_outside': round(float(q_out.sum() / q_all.sum()), 5),
        'signed_executable_distance_points_mean': round(
            float(dsum.sum() / max(q_out.sum(), 1)), 3),
        'signed_executable_distance_zone_widths_mean': round(
            float(wsum.sum() / max(q_out.sum(), 1)), 4),
        'signed_executable_distance_zone_width_histogram': {
            f'{WBINS[i]:g}-{WBINS[i+1]:g}': int(hist[i]) for i in range(WBINS.size - 1)},
        'films_with_no_usable_q': int((q_out == 0).sum()),
        'age_coverage_observed_bars': curve,
        'age_coverage_by_native_tf_band': by_band,
        'last_q_before_certified_contact': {
            'films': int(cc.sum()),
            'reference_defined_contiguous': int((kind_at_contact == 0).sum()),
            'open_of_contact_candle_still_outside': int(((s > 0) & (kind_at_contact == 0)).sum()),
            'open_exactly_on_boundary': int(((s == 0) & (kind_at_contact == 0)).sum()),
            'open_already_beyond': int(((s < 0) & (kind_at_contact == 0)).sum()),
            'reference_through_gap': int((kind_at_contact != 0).sum()),
            'note': 'контакт внутри этой же свечи не отменяет условный вход по её open',
        },
    }
    # то же для подмножества «контакт на T0+1» — самый острый случай
    plus1 = cc & (contact - t0 == 1)
    cp = contact[plus1]
    s1 = np.where(north[plus1], opn[cp] - e[plus1], e[plus1] - opn[cp])
    r['contact_at_t0_plus_1'] = {
        'films': int(plus1.sum()),
        'next_open_contiguous': int((kind[cp] == 0).sum()),
        'open_still_outside_boundary': int(((s1 > 0) & (kind[cp] == 0)).sum()),
        'open_exactly_on_boundary': int(((s1 == 0) & (kind[cp] == 0)).sum()),
        'open_already_beyond': int(((s1 < 0) & (kind[cp] == 0)).sum()),
    }
    return r


if __name__ == '__main__':
    grid, lo, hi = presence_grid()
    out = {}
    for inst in INSTRUMENTS:
        kind, _ = gap_kinds(inst, grid, lo, hi)
        out[inst] = one(inst, kind)
        print(inst, 'done', flush=True)
    txt = json.dumps(out, ensure_ascii=False, indent=1)
    (HERE / 'census_age.json').write_text(txt, encoding='utf-8', newline='\n')
    print('written', HERE / 'census_age.json')
