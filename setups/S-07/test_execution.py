"""Адресные проверки исполнения S-07. Не общий слой и не покрытие исследований.

Это пять разрывов, найденных при сверке слов карточки с кодом: цена следующего
входа, гэп за предел, момент узнавания, часы удержания и сходимость с
сохранённым итогом. Всё остальное про S-07 они не проверяют.

    python -B -m pytest setups/S-07/test_execution.py -q

Проверки, которым нужны локальные `data/market`, пропускаются без них.
Отсутствие данных не считается пустым результатом.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

MIN = 60_000_000_000
needs_market = pytest.mark.skipif(
    not (ROOT / 'data/market/NQ/manifest.json').exists(),
    reason='локальные data/market не подключены; клон байтов не несёт')


@pytest.fixture(scope='module')
def rev():
    return pytest.importorskip('revision')


@pytest.fixture(scope='module')
def saved():
    path = HERE / 'result_v1r1.json'
    if not path.exists():
        pytest.skip('result_v1r1.json ещё не посчитан')
    return json.loads(path.read_text(encoding='utf-8'))


# --- гэп за предел: чистая логика, данные не нужны -------------------------

def fill_at_limit(open_candles, limit=-16.0, honest_gap=True):
    """Та же арифметика, что в revision.trades: худшее из уровня и open минуты."""
    return min(open_candles, limit) if honest_gap else limit


def test_open_beyond_limit_fills_at_that_open():
    """Минута открылась на 19 свечей против — заполнение по 19, а не по 16."""
    assert fill_at_limit(-19.0) == -19.0
    assert fill_at_limit(-19.0, honest_gap=False) == -16.0


def test_open_inside_limit_fills_at_the_limit():
    """Минута открылась внутри предела и лишь потом дошла до него — по уровню."""
    assert fill_at_limit(-3.0) == -16.0
    assert fill_at_limit(-16.0) == -16.0


# --- цена следующего входа и момент узнавания ------------------------------

@needs_market
def test_entry_price_is_the_open_of_the_minute_after_the_decision(rev):
    """Вход — open интервала, начинающегося в момент решения, а не close решения."""
    m, cal, ts, p, b, idx, unit, _ = rev.population()
    entry = m['open'][b]
    assert (ts[b] == ts[p] + MIN).all(), 'вход должен идти по следующей минуте ленты'
    # Именно поэтому «вход по open минуты 09:34» нельзя читать как исполнение в 09:34:
    # у свечи с close-меткой 09:34 open относится к 09:33.
    coincide = float((entry == m['close'][p]).mean())
    assert 0.0 < coincide < 1.0, 'open следующей минуты и close решения — разные числа'


@needs_market
def test_decision_minute_is_the_close_of_0933_new_york(rev):
    m, cal, ts, p, b, idx, unit, _ = rev.population()
    ny = pd.to_datetime(ts[p], utc=True).tz_convert('America/New_York')
    assert set(ny.strftime('%H:%M')) == {'09:33'}
    assert set(pd.to_datetime(ts[b], utc=True).tz_convert('America/New_York')
               .strftime('%H:%M')) == {'09:34'}


@needs_market
def test_side_and_entry_do_not_move_when_the_future_is_cut_off(rev):
    """Узнавание на префиксе: лента, оборванная на минуте входа, даёт то же решение."""
    m, cal, ts, p, b, idx, unit, _ = rev.population()
    win_hi = np.maximum.reduce([m['high'][p - j] for j in range(rev.base.WINDOW)])
    win_lo = np.minimum.reduce([m['low'][p - j] for j in range(rev.base.WINDOW)])
    side = np.where(m['close'][p] > (win_hi + win_lo) / 2, 1., -1.)
    rng = np.random.default_rng(20260908)
    for i in rng.choice(len(p), size=200, replace=False):
        cut = int(b[i]) + 1
        hi, lo, cl, op = m['high'][:cut], m['low'][:cut], m['close'][:cut], m['open'][:cut]
        a = int(p[i]) - rev.base.WINDOW + 1
        truncated = 1. if cl[p[i]] > (hi[a:p[i] + 1].max() + lo[a:p[i] + 1].min()) / 2 else -1.
        assert truncated == side[i]
        assert op[b[i]] == m['open'][b[i]]


# --- часы удержания --------------------------------------------------------

@needs_market
def test_the_fifteenth_and_hundred_twentieth_minute_are_counted_from_the_entry_bar(rev):
    """15-я минута позиции — 09:48, 120-я — 11:33. Минута входа считается первой."""
    honest, _ = rev.trades(honest_gap=True)
    ny = pd.to_datetime(honest.exit_bar_close_ns.to_numpy(), utc=True).tz_convert('America/New_York')
    clock = pd.Series(ny.strftime('%H:%M'), index=honest.index)
    assert set(clock[honest.reason == 'минус на 15-й минуте']) == {'09:48'}
    assert set(clock[honest.reason == '120 минут']) == {'11:33'}


@needs_market
@pytest.mark.parametrize('date,reason,exit_clock', [
    ('2024-01-02', 'катастрофический лимит', '09:41'),
    ('2024-01-03', 'минус на 15-й минуте', '09:48'),
    ('2024-01-05', '120 минут', '11:33'),
])
def test_named_days_reconstruct_by_hand_from_the_raw_tape(rev, date, reason, exit_clock):
    """Три конкретных дня, пересчитанных прямо по массивам, а не по матрицам ревизии.

    По одному на каждый исход. Проверяется цепочка целиком: цена входа, уровень
    предела от неё, первая минута, дошедшая до уровня, момент выхода и его цена.
    """
    honest, _ = rev.trades(honest_gap=True)
    row = honest.loc[honest.date == date].iloc[0]
    assert row.reason == reason

    m = rev.base.market('NQ')
    ts = m['close_ts_utc_ns']
    p = int(np.searchsorted(ts, row.decision_ns))
    assert ts[p] == row.decision_ns
    b = p + 1

    # вход — open следующей минуты, а не close минуты решения
    assert m['open'][b] == row.entry
    side = row.side
    level = row.entry - side * rev.LIMIT_CANDLES * row.unit

    exit_ns = int(row.exit_bar_close_ns)
    j = int(np.searchsorted(ts, exit_ns))
    assert ts[j] == exit_ns
    assert (pd.Timestamp(exit_ns, unit='ns', tz='UTC')
            .tz_convert('America/New_York').strftime('%H:%M')) == exit_clock

    reached = (m['low'] <= level) if side > 0 else (m['high'] >= level)
    if reason == 'катастрофический лимит':
        assert reached[j], 'минута выхода обязана дойти до уровня'
        assert not reached[b:j].any(), 'выход должен быть на ПЕРВОЙ дошедшей минуте'
        assert m['open'][j] * side >= level * side, 'здесь гэпа нет: заполнение по уровню'
        assert row.candles == pytest.approx(-rev.LIMIT_CANDLES)
    else:
        assert not reached[b:j + 1].any(), 'предел не должен был сработать раньше'
        assert row.candles == pytest.approx(side * (m['close'][j] - row.entry) / row.unit)
        if reason == 'минус на 15-й минуте':
            assert j - b == rev.CHECK_MINUTE - 1, 'минута входа считается первой'
            assert row.candles < 0
        else:
            assert j - b == rev.HOLD - 1


@needs_market
def test_the_catastrophic_limit_never_gaps_inside_the_session(rev, saved):
    """Измеренный факт, а не допущение: за 21 год ни одного открытия за пределом."""
    assert saved['correction_cost']['trades_changed'] == 0
    assert saved['correction_cost']['usd_difference_all'] == 0.0


# --- сходимость с сохранённым итогом ---------------------------------------

@needs_market
def test_recomputation_matches_the_saved_result(rev, saved):
    fresh = rev.build()
    for key in ['rule', 'coverage', 'clock_sample', 'result_v1r1',
                'result_v1_frozen_execution', 'correction_cost']:
        assert fresh[key] == saved[key], f'ревизия разошлась с сохранённым итогом: {key}'


@needs_market
def test_saved_result_reproduces_the_frozen_v1_money(saved):
    """Константы FROZEN_v1.json проверяются вычислением, а не переписыванием."""
    claim = saved['matches_frozen_claim']
    assert claim['computed'] == claim['claim']


@needs_market
def test_missing_days_stay_unknown_and_are_not_counted_as_zero(saved):
    c = saved['coverage']
    assert c['entries'] < c['calendar_days']
    assert (c['entries'] + c['without_decision_minute'] + c['dropped_broken_minute_grid']
            + c['dropped_missing_30m_prefix']) == c['calendar_days']
