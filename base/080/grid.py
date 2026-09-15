#!/usr/bin/env python3
"""Ленивый вид на сетку решений. Три слоя, разделённые технически, не на словах.

Свечи хранятся один раз — в замороженной спине. Здесь нет ни одной копии бара
на `(riz_id, q)`: слои возвращают АДРЕСА (диапазоны позиций спины) и читают
ленту по требованию.

    decision_prefix        сведения, доступные не позже закрытия q
    execution_reference    следующий open; появляется только при наблюдении
    continuation           остаток пути после q

РАЗДЕЛЕНИЕ ПРОВЕРЯЕМО, А НЕ ОБЕЩАНО
===================================
`PrefixSource` физически не может вернуть бар позже курсора: он владеет срезом
ленты, обрезанным по q, и индексирует внутри него. Подмена полного источника на
такой срез обязана давать тот же prefix-ответ — это и проверяет `qa_prefix.py`.
`ExecutionReference` и `Continuation` берут ПОЛНЫЙ источник и поэтому никогда
не передаются построителю признаков.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

import os
#: Где лежит компактный индекс. По умолчанию рядом с кодом; переопределяется
#: `G3_080A_INDEX`, чтобы несколько копий кода читали один и тот же индекс и не
#: плодили 120 МБ на каждую.
INDEX = Path(os.environ.get('G3_080A_INDEX', str(HERE / 'index')))
MIN = 60_000_000_000
COLS = ('open', 'high', 'low', 'close')


class Tape:
    """Полная замороженная лента одного инструмента. Только чтение."""

    def __init__(self, inst):
        m = ROOT / 'data/market' / inst
        self.instrument = inst
        for k in COLS + ('close_ts_utc_ns', 'session_id'):
            setattr(self, k, np.load(m / f'{k}.npy', mmap_mode='r'))
        self.n = self.close.size

    def slice(self, a, b):
        """OHLC на позициях [a, b] включительно плюс часы и сессия."""
        b = min(b, self.n - 1)
        out = {k: np.asarray(getattr(self, k)[a:b + 1]) for k in COLS}
        out['close_ts_utc_ns'] = np.asarray(self.close_ts_utc_ns[a:b + 1])
        out['session_id'] = np.asarray(self.session_id[a:b + 1])
        out['spine_pos'] = np.arange(a, b + 1)
        return out


class PrefixSource:
    """Источник, физически не содержащий будущего за курсором.

    Не «источник, который обещает не читать дальше», а срез, в котором позже
    курсора ничего нет. Обращение за баром правее курсора — исключение.
    """

    def __init__(self, tape: Tape, cursor: int, start: int = 0):
        self.instrument = tape.instrument
        self.start = int(start)
        self.cursor = int(cursor)
        self._d = tape.slice(self.start, self.cursor)
        self.n = self._d['spine_pos'].size

    def slice(self, a, b):
        if b > self.cursor:
            raise IndexError(f'позиция {b} лежит за курсором {self.cursor}')
        if a < self.start:
            raise IndexError(f'позиция {a} лежит до начала среза {self.start}')
        i, j = a - self.start, b - self.start + 1
        out = {k: self._d[k][i:j] for k in self._d}
        return out


@dataclass(frozen=True)
class DecisionPrefix:
    """Слой 1. Только то, что известно не позже закрытия q."""
    riz_id: str
    instrument: str
    tf_minutes: int
    side: str
    zone_top: float
    zone_bottom: float
    exit_boundary: float
    far_boundary: float
    t0_spine_pos: int
    q_spine_pos: int
    prefix_range: tuple            # адрес в ленте, не копия свечей
    observed_bars_since_t0: int
    wall_clock_minutes_since_t0: int
    missing_minutes_since_t0: int
    sessions_since_t0: int
    lifecycle_state_as_of_q: str
    close_q: float
    d_close_to_exit_boundary: float
    price_beyond_exit_boundary: bool
    freshness_as_of_q: str


@dataclass(frozen=True)
class ExecutionReference:
    """Слой 2. Следующий open. На закрытии q его ещё нет."""
    riz_id: str
    q_spine_pos: int
    reference_spine_pos: int | None
    reference_ts_ns: int | None
    open_price: float | None
    wait_minutes: int | None
    gap_kind: str
    availability: str              # observed_contiguous | through_gap | none
    distance_to_tp: float | None   # НЕзнаковая величина; знак — в directional_status
    directional_status: str        # outside | on_boundary | beyond | unknown


@dataclass(frozen=True)
class Continuation:
    """Слой 3. Будущее. Построителю признаков не передаётся.

    Слой ЗНАЕТ о сертификации, и это не украшение. Наблюдавшийся контакт после
    неизвестного промежутка остаётся фактом наблюдения, но TP выбранной
    сертификации он не является: первое касание могло случиться внутри
    пропуска. Поэтому `certified_tp_pos` пуст там, где первичность не доказана,
    а `observed_contact_pos` заполнен всегда — это разные поля про разное.
    """
    riz_id: str
    q_spine_pos: int
    certification: str
    remaining_range: tuple | None     # до сертифицированного конца; None — пусто
    certified_tp_pos: int | None      # None, если первичность не доказана
    observed_contact_pos: int | None  # факт наблюдения, независимо от первичности
    observed_contact_primacy: str
    film1_status: str
    continuation_status: str
    censor_reason: str | None


class Film1Index:
    """Компактный индекс + ленивый вид. Ничего построчного не материализует."""

    def __init__(self, instrument, kind=None):
        self.instrument = instrument
        self.tape = Tape(instrument)
        self.df = pd.read_parquet(INDEX / f'film1_{instrument}.parquet')
        self.by_id = {r: i for i, r in enumerate(self.df.riz_id.to_numpy())}
        if kind is None:
            import sys
            sys.path.insert(0, str(HERE))
            from tape import presence_grid, gap_kinds
            grid, lo, hi = presence_grid()
            kind, _ = gap_kinds(instrument, grid, lo, hi)
        self.kind = kind
        self.kind_name = {0: 'contiguous', 1: 'shared_cause_unknown',
                          2: 'instrument_specific', 3: 'outside_common_window'}

    def row(self, riz_id):
        return self.df.iloc[self.by_id[riz_id]]

    def q_positions(self, riz_id, certification='strict'):
        """Все допустимые q. Членство определяется доступным прошлым."""
        r = self.row(riz_id)
        t0 = int(r.t0_spine_pos)
        fresh = int(r[f'certified_fresh_until_pos_{certification}'])
        return np.arange(t0, fresh + 1)

    # ---------- слой 1 ----------
    def prefix(self, riz_id, q, source=None, certification='strict'):
        r = self.row(riz_id)
        t0 = int(r.t0_spine_pos)
        src = source if source is not None else self.tape
        d = src.slice(t0, q)
        ts = d['close_ts_utc_ns']
        wall = int((ts[-1] - ts[0]) // MIN)
        bars = int(d['spine_pos'].size - 1) if 'spine_pos' in d else q - t0
        dele = int(r.c1_deletion_spine_pos)
        blue = int(r.blue_eligibility_end_spine_pos)
        conf = int(r.native_blue_confirmation_spine_pos)
        state = 'alive'
        if 0 <= conf <= q:
            state = 'native_blue_confirmed'
        if 0 <= blue <= q:
            state = 'blue_eligibility_ended'
        if 0 <= dele <= q:
            state = 'deleted'
        e = float(r.exit_boundary)
        close_q = float(d['close'][-1])
        north = r.side == 'north'
        # свежесть сверяется с границей ВЫБРАННОЙ сертификации, иначе
        # sensitivity-сетка из `q_positions` читалась бы strict-мерой
        fresh_s = int(r[f'certified_fresh_until_pos_{certification}'])
        return DecisionPrefix(
            riz_id=riz_id, instrument=self.instrument, tf_minutes=int(r.tf_minutes),
            side=str(r.side), zone_top=float(r.zone_top), zone_bottom=float(r.zone_bottom),
            exit_boundary=e, far_boundary=float(r.far_boundary),
            t0_spine_pos=t0, q_spine_pos=int(q), prefix_range=(t0, int(q)),
            observed_bars_since_t0=bars, wall_clock_minutes_since_t0=wall,
            missing_minutes_since_t0=wall - bars,
            sessions_since_t0=int(d['session_id'][-1] - d['session_id'][0]),
            lifecycle_state_as_of_q=state, close_q=close_q,
            d_close_to_exit_boundary=close_q - e,
            price_beyond_exit_boundary=bool(close_q > e if north else close_q < e),
            freshness_as_of_q=('certified_fresh' if q <= fresh_s else 'unknown'))

    # ---------- слой 2 ----------
    def execution_reference(self, riz_id, q, certification='strict'):
        r = self.row(riz_id)
        nxt = int(q) + 1
        if nxt >= self.tape.n:
            return ExecutionReference(riz_id, int(q), None, None, None, None,
                                      'none', 'none', None, 'unknown')
        k = int(self.kind[nxt])
        name = self.kind_name[k]
        allowed = (k == 0) if certification == 'strict' else (k in (0, 1))
        op = float(self.tape.open[nxt])
        ts0 = int(self.tape.close_ts_utc_ns[int(q)])
        ts1 = int(self.tape.close_ts_utc_ns[nxt])
        e = float(r.exit_boundary)
        north = r.side == 'north'
        signed = (op - e) if north else (e - op)     # >0: снаружи, сделка цела
        status = 'outside' if signed > 0 else ('on_boundary' if signed == 0 else 'beyond')
        return ExecutionReference(
            riz_id=riz_id, q_spine_pos=int(q), reference_spine_pos=nxt,
            reference_ts_ns=ts1, open_price=op, wait_minutes=int((ts1 - ts0) // MIN),
            gap_kind=name,
            availability='observed_contiguous' if k == 0 else 'through_gap',
            distance_to_tp=abs(signed) if allowed else None,
            directional_status=status if allowed else 'unknown')

    # ---------- слой 3 ----------
    def _primacy(self, r, certification):
        col = ('first_observed_contact_primacy' if certification == 'strict'
               else 'first_observed_contact_primacy_shared_survived')
        return str(r[col])

    def continuation(self, riz_id, q, certification='strict'):
        """Остаток пути. Поздний контакт за промежутком TP не объявляется."""
        r = self.row(riz_id)
        c = int(r.first_observed_contact_pos)
        fresh = int(r[f'certified_fresh_until_pos_{certification}'])
        primacy = self._primacy(r, certification)
        observed = c if c >= 0 else None
        if primacy == 'certified':
            # конец Film-1 установлен: остаток ведёт ровно до него
            end, tp, status, reason = c, c, 'certified_contact', None
        elif observed is not None:
            # контакт наблюдался, но за неизвестным промежутком. Достоверное
            # наблюдение кончается ПЕРЕД промежутком, и TP здесь нет.
            end, tp = fresh, None
            status = 'primacy_unknown_observed_contact_later'
            reason = 'freshness_lost_before_observed_contact'
        else:
            end, tp = fresh, None
            if str(r.film1_status) == 'no_contact_through_archive':
                status, reason = 'archive_edge_no_contact', 'archive_edge'
            else:
                status, reason = 'freshness_lost_no_contact', 'freshness_lost'
        # q на самой границе сертификации оставляет остаток ПУСТЫМ. Вывернутый
        # кортеж (q+1, end < q+1) читался бы как диапазон и молча дал бы срез.
        rng = (int(q) + 1, end) if end >= int(q) + 1 else None
        return Continuation(
            riz_id=riz_id, q_spine_pos=int(q), certification=certification,
            remaining_range=rng, certified_tp_pos=tp,
            observed_contact_pos=observed, observed_contact_primacy=primacy,
            film1_status=str(r.film1_status), continuation_status=status,
            censor_reason=reason)

    # ---------- касательная свеча ----------
    def excursion_to_tp(self, riz_id, q_from=None, certification='strict'):
        """Экскурсия до TP с честным обращением с касательной свечой.

        Полный high/low минуты контакта может содержать движение ПОСЛЕ первого
        касания. Поэтому возвращается либо точное значение, либо границы, либо
        прямое признание неизвестности порядка.

        TP считается ТОЛЬКО когда первичность контакта доказана под выбранной
        сертификацией. Иначе функция отказывается считать «до TP»: поздний
        наблюдавшийся контакт за неизвестным промежутком первым касанием не
        является, и подставлять его сюда — ровно та подмена, против которой
        построен весь слой сертификации.
        """
        r = self.row(riz_id)
        c = int(r.first_observed_contact_pos)
        t0 = int(r.t0_spine_pos)
        # начало отсчёта — бар reference entry, а не сама минута T0
        a = int(q_from) + 1 if q_from is not None else t0 + 1
        if c < 0:
            return {'kind': 'no_contact', 'anchor': a}
        primacy = self._primacy(r, certification)
        if primacy != 'certified':
            return {'kind': 'primacy_not_certified', 'anchor': a,
                    'certification': certification,
                    'observed_contact_pos': c,
                    'certified_fresh_until_pos': int(
                        r[f'certified_fresh_until_pos_{certification}']),
                    'why': 'первое касание могло случиться внутри пропуска; '
                           'наблюдавшийся позже контакт TP не является'}
        if a > c:
            return {'kind': 'entry_after_contact', 'anchor': a, 'contact_pos': c}
        north = r.side == 'north'
        e = float(r.exit_boundary)
        d = self.tape.slice(a, c)
        entry = float(d['open'][0])          # reference entry, начало отсчёта
        # ход против цели на барах СТРОГО ДО касательного известен точно
        pre = (d['high'][:-1] if north else d['low'][:-1])
        if pre.size:
            worst_pre = float(pre.max() if north else pre.min())
        else:
            worst_pre = entry
        worst_pre = max(worst_pre, entry) if north else min(worst_pre, entry)
        touch = float(d['high'][-1] if north else d['low'][-1])
        # Порядок внутри касательной свечи не восстановим, но это меняет ответ
        # только если её собственный экстремум ПРЕВЫШАЕТ уже набранный максимум.
        raises = (touch > worst_pre) if north else (touch < worst_pre)
        adverse_exact = abs(worst_pre - entry)
        out = {'anchor_spine_pos': a, 'anchor': 'reference entry open',
               'contact_pos': c, 'entry_price': entry,
               'distance_entry_to_tp': abs(entry - e),
               'adverse_excursion_before_touch_candle': adverse_exact,
               'touch_candle_ohlc': {k: float(d[k][-1]) for k in COLS}}
        if raises:
            out['kind'] = 'bounded'
            out['adverse_excursion_lower_bound'] = adverse_exact
            out['adverse_excursion_upper_bound'] = abs(touch - entry)
            out['order_within_touch_candle'] = 'unknown'
        else:
            out['kind'] = 'exact'
            out['adverse_excursion_to_tp'] = adverse_exact
        return out
