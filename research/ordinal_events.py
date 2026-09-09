#!/usr/bin/env python3
"""Язык разворачивания событий над трафаретом отношений.

ЧТО ЭТО И ЧЕМ НЕ ЯВЛЯЕТСЯ
=========================
Это НАДСТРОЙКА над `relational_stencil.py`, а не его замена. Точное ядро
отношений, префикс, знаменатели и происхождение примеров берутся оттуда без
изменений: `Corpus`, `Space`, `bits`, `indices`. Ни один оператор здесь не
трогает цену иначе, чем сравнением <, =, > — длины, ATR, размах, объём, доход
и стоп не участвуют ни в представлении, ни в отборе, ни в ранжировании.

Слой отвечает на разрыв, найденный на поле: между «парой точек» и «целым
фильмом» у прибора не было ничего. «H на минуте k есть максимум по 1..30» —
это конъюнкция из 29 отношений, недостижимая при max_relations 2 или 3 не по
бюджету, а по устройству словаря. Здесь такие факты становятся атомами.

СОБЫТИЕ
=======
Событие — порядковая величина на единице наблюдения: номер свечи, либо
`none` (в объявленном окне не наступило), либо неизвестно. Виды:

  argmaxH_first[a..b], argmaxH_last[a..b]   адрес максимума H на участке
  argminL_first[a..b], argminL_last[a..b]   адрес минимума L на участке
  wick_above(level)[a..b]                   первое прохождение уровня тенью
  close_above(level)[a..b]                  первое закрытие за уровнем
  wick_below(level)[a..b], close_below(level)[a..b]
  updates_H_last[a..b], updates_L_last[a..b]  последнее обновление опорного
                                              экстремума (адрес)

Тень и закрытие РАЗДЕЛЕНЫ намеренно: `H[k] > level` и `C[k] > level` — разные
события, и их расхождение и есть предмет. Уровень задаётся адресом точки
фильма (например `H[0]`), а не числом.

Равные экстремумы: даются обе крайности, `first` и `last`. Их совпадение или
расхождение выражается атомом порядка, то есть остаётся наблюдаемым фактом,
а не решением кода.

ПРОПУСКИ — ГДЕ ПРОХОДИТ ГРАНИЦА НЕИЗВЕСТНОГО
============================================
* Адрес экстремума на [a..b] неизвестен, если пропущена ЛЮБАЯ минута участка:
  максимум законченного окна известен только после окончания окна.
* Первое прохождение: если до наблюдаемого пробоя есть пропущенная минута,
  событие НЕИЗВЕСТНО. «Первый увиденный» первым произошедшим не назначается.
* `none` (не наступило) законно только когда весь участок известен.
* Неизвестен уровень — неизвестно и событие.

ВРЕМЯ ДОСТУПНОСТИ
=================
`known_at` атома консервативен: для `E = k` это max(k, доступность уровня);
для `E = none` и для адресов экстремума это конец объявленного участка; для
атомов порядка и расстояния — максимум концов обоих участков. Консервативность
означает, что атом никогда не объявляется доступным раньше, чем это верно на
каждой единице корпуса.

ЧИСЛА ПРИХОДЯТ ИЗ ИСТОРИИ
=========================
Никакой адрес не назначается заранее. Перебираются ВСЕ допустимые значения
k в объявленном бюджете и ВСЕ допустимые расстояния d; какие из них часты,
говорит корпус. Поясняющий пример в этом файле искомой конструкцией не является.
Окно наблюдения — явно названный бюджет, а не горизонт сделки.

ТРИ РАЗНЫЕ ВЕЩИ, КОТОРЫЕ ПРИБОР БОЛЬШЕ НЕ ПУТАЕТ
================================================
* `logically_impossible` — противоречит анатомии свечи. Из перебора исключено.
* `constant_on_corpus` — истинно (или ложно) на ВСЕХ известных случаях этого
  корпуса. Это свойство корпуса, а не аксиома. `H > L` попадает сюда: плоская
  свеча допустима, просто не встретилась. Вынесено в свою витрину.
* самостоятельное отношение — всё остальное.

Равенства сохраняются отдельным доступным слоем. Их не запрещают: наблюдаемые
цены дискретны, и совпадение уровней может иметь смысл.

СИНОНИМЫ И ИСКОМАЯ СОВМЕСТНОСТЬ
===============================
Совпадение носителей двух отношений НЕ является поводом выбросить сочетание:
именно такую согласованность мы и ищем. Различаются три случая:
  * логическая эквивалентность записи — канонизируется ядром (`Space.atom`),
    все адреса сохраняются;
  * условная эквивалентность через сцепку соседних close/open — помечается
    вместе с ИЗМЕРЕННОЙ долей выполнения условия, зазоры сохраняются;
  * совпадение только по примерам корпуса — остаётся предметом исследования
    и показывается как нулевой добавочный вклад, а не как приговор.

`joint_excess` главным способом обнаруживать содержательность больше не
является. Вместо него для каждой конструкции показывается её состав, совместная
частота и вклад каждого отношения при его исключении.

КАЛИБРОВКА
==========
`calibrate` повторяет ВЕСЬ зависящий от меток выбор — поиск и отбор верхушек —
на разрушенной связи. Не зависящие от меток носители считаются один раз.
Результат помечается `DIAGNOSTIC`, пока обоснованность перестановок не
объявлена явно: перестановка должна сохранять релевантную зависимость
наблюдений и сопоставимость обстоятельств. Малое число повторов показывает
серьёзность проблемы, но потолка не устанавливает.

Три утверждения не смешиваются: конструкция ПОВТОРЯЕТСЯ; конструкция
СПЕЦИФИЧНА для якоря; конструкция ПОМОГАЕТ предсказать продолжение. Калибровка
третьего не запрещает видеть первое.

ПОЛНОТА ДОСТУПА
===============
Журнал больше не текстовый. Сохраняются: входы, версия вычисления, параметры,
seed, порядок перебора, границы завершённых блоков, знаменатели и по каждому
просмотренному кандидату его адреса и счётчики в двоичных массивах. Любой
кандидат восстанавливается по индексу вместе с примерами, включая не попавший
ни в одну витрину. Хранить только прошедшее калибровку нельзя.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np

from relational_stencil import (FIELDS, SIGNS, Corpus, Space, bits, indices,
                                anatomy_possible, continuity_context,
                                logically_impossible, logically_redundant)

VERSION = 'ordinal-events-1'
NONE = -1          # событие не наступило в объявленном бюджете
UNKNOWN = -2       # событие неизвестно (пропуск, неизвестный уровень)


# ---------------------------------------------------------------- события

class Event:
    """Порядковая величина на единице наблюдения плюс её происхождение.

    `values[u]` — номер свечи, NONE или UNKNOWN. `witness(u)` возвращает
    конкретные свечи и сравнения, которыми значение получено, чтобы любой
    результат раскрывался до исходных OHLC.
    """

    def __init__(self, name, kind, segment, values, known_at, level=None):
        self.name = name
        self.kind = kind
        self.segment = segment            # (a, b) в ordinals, включительно
        self.values = np.asarray(values, dtype=np.int64)
        self.known_at = int(known_at)
        self.level = level                # (ordinal, field) либо None

    def describe(self):
        return {'event': self.name, 'kind': self.kind,
                'segment': [int(self.segment[0]), int(self.segment[1])],
                'level': {'available_from': self.level[0],
                          'name': self.level[1]} if self.level else None,
                'known_at': self.known_at}


def _finite(corpus):
    return np.isfinite(corpus.ohlc).all(axis=2)


def _column(corpus, ordinal, field):
    j = int(np.flatnonzero(corpus.ordinals == ordinal)[0])
    return corpus.ohlc[:, j, FIELDS.index(field)]


def _slice(corpus, a, b):
    lo = int(np.flatnonzero(corpus.ordinals == a)[0])
    hi = int(np.flatnonzero(corpus.ordinals == b)[0])
    return lo, hi


def extremum_event(corpus, a, b, field, take_last, name=None):
    """Адрес экстремума на объявленном участке. Пропуск в участке = неизвестно.

    Совпадает по определению с бегущим экстремумом из `extremum_clock.py`:
    там же адрес максимума считается argmax по полностью известному отрезку.
    """
    lo, hi = _slice(corpus, a, b)
    k = FIELDS.index(field)
    x = corpus.ohlc[:, lo:hi + 1, k]
    ok = _finite(corpus)[:, lo:hi + 1]
    full = ok.all(axis=1)
    values = np.full(len(x), UNKNOWN, dtype=np.int64)
    if full.any():
        y = x[full]
        best = y.max(axis=1) if field == 'H' else y.min(axis=1)
        hit = y == best[:, None]
        pos = (hit.shape[1] - 1 - hit[:, ::-1].argmax(axis=1)) if take_last else hit.argmax(axis=1)
        values[full] = corpus.ordinals[lo] + pos
    edge = 'last' if take_last else 'first'
    label = name or f'arg{"max" if field == "H" else "min"}{field}_{edge}[{a}..{b}]'
    return Event(label, f'extremum_{field}_{edge}', (a, b), values, b)


def point_level(corpus, ordinal, field):
    """Уровень — точка фильма. Доступен с той минуты, которой принадлежит."""
    v = _column(corpus, ordinal, field)
    return v, np.isfinite(v), int(ordinal), f'{field}[{ordinal}]'


def prefix_extreme_level(corpus, a, b, field):
    """Уровень, ПОСЧИТАННЫЙ ПО ПРЕФИКСУ. Законен: он из доступного прошлого.

    Неизвестен, если в участке есть пропуск: экстремум законченного участка
    известен только после его окончания.
    """
    lo, hi = _slice(corpus, a, b)
    k = FIELDS.index(field)
    x = corpus.ohlc[:, lo:hi + 1, k]
    ok = _finite(corpus)[:, lo:hi + 1]
    full = ok.all(axis=1)
    values = np.full(len(x), np.nan)
    if full.any():
        y = x[full]
        values[full] = y.max(axis=1) if field == 'H' else y.min(axis=1)
    name = f'{"max" if field == "H" else "min"}{field}[{a}..{b}]'
    return values, full, int(b), name


def crossing_event(corpus, a, b, field, level, above, name=None):
    """Первое прохождение уровня. Тень и закрытие — РАЗНЫЕ события.

    `level` — либо (ordinal, field) для точки фильма, либо готовая четвёрка
    (значения, известность, время доступности, имя) от `prefix_extreme_level`.
    Уровень, рассчитанный по префиксу, законен; незаконно засчитывать уже
    состоявшееся достижение как последующее — это решает окно [a..b] и контракт.

    Пропуск раньше наблюдённого пробоя делает событие неизвестным: первый
    увиденный пробой первым произошедшим не назначается.
    """
    if isinstance(level, tuple) and len(level) == 2:
        level_values, level_ok, level_clock, level_name = point_level(corpus, *level)
    else:
        level_values, level_ok, level_clock, level_name = level
    lo, hi = _slice(corpus, a, b)
    k = FIELDS.index(field)
    x = corpus.ohlc[:, lo:hi + 1, k]
    ok = _finite(corpus)[:, lo:hi + 1]
    beyond = (x > level_values[:, None]) if above else (x < level_values[:, None])
    beyond &= ok
    n, width = x.shape
    values = np.full(n, UNKNOWN, dtype=np.int64)
    any_hit = beyond.any(axis=1)
    first_hit = np.where(any_hit, beyond.argmax(axis=1), width)
    missing = ~ok
    gap_before = np.zeros(n, dtype=bool)
    for i in range(n):
        gap_before[i] = missing[i, :first_hit[i]].any()
    hit = any_hit & ~gap_before & level_ok
    miss = (~any_hit) & ok.all(axis=1) & level_ok
    values[hit] = corpus.ordinals[lo] + first_hit[hit]
    values[miss] = NONE
    side = 'above' if above else 'below'
    what = 'wick' if field in ('H', 'L') else 'close'
    label = name or f'{what}_{side}({level_name})[{a}..{b}]'
    return Event(label, f'{what}_{side}', (a, b), values, b,
                 level=(level_clock, level_name))


def last_update_event(corpus, a, b, field, name=None):
    """Адрес последнего обновления опорного экстремума на участке.

    Строгое обновление бегущего экстремума. Это сжатая запись
    последовательности обновлений; сама последовательность доступна в witness.
    """
    lo, hi = _slice(corpus, a, b)
    k = FIELDS.index(field)
    x = corpus.ohlc[:, lo:hi + 1, k]
    ok = _finite(corpus)[:, lo:hi + 1]
    full = ok.all(axis=1)
    values = np.full(len(x), UNKNOWN, dtype=np.int64)
    if full.any():
        y = x[full]
        run = np.maximum.accumulate(y, axis=1) if field == 'H' else np.minimum.accumulate(y, axis=1)
        upd = np.empty_like(y, dtype=bool)
        upd[:, 0] = True
        upd[:, 1:] = (y[:, 1:] > run[:, :-1]) if field == 'H' else (y[:, 1:] < run[:, :-1])
        pos = upd.shape[1] - 1 - upd[:, ::-1].argmax(axis=1)
        values[full] = corpus.ordinals[lo] + pos
    label = name or f'lastupdate{field}[{a}..{b}]'
    return Event(label, f'last_update_{field}', (a, b), values, b)


def update_trace(corpus, unit, a, b, field):
    """Полная последовательность обновлений для раскрытия результата."""
    lo, hi = _slice(corpus, a, b)
    k = FIELDS.index(field)
    y = corpus.ohlc[unit, lo:hi + 1, k]
    out, best = [], None
    for j, v in enumerate(y):
        if not np.isfinite(v):
            out.append({'minute': int(corpus.ordinals[lo] + j), 'value': None,
                        'update': None})
            continue
        better = best is None or (v > best if field == 'H' else v < best)
        if better:
            best = v
        out.append({'minute': int(corpus.ordinals[lo] + j), 'value': float(v),
                    'update': bool(better)})
    return out


# ---------------------------------------------------------------- линза

class Lens:
    """Единое пространство атомов: точные отношения ядра плюс события.

    Носители — те же битовые множества, что и в `Space`, поэтому знаменатели,
    пересечения и префикс работают ровно как в ядре.
    """

    def __init__(self, corpus, events, base_space=None, min_known=1):
        self.corpus = corpus
        self.n = corpus.n
        self.all = (1 << self.n) - 1
        self.space = base_space if base_space is not None else Space(corpus)
        self.events = list(events)
        self.atoms = []          # список описаний
        self.masks = []
        self.knowns = []
        self._build_event_atoms(min_known)

    # --- построение

    def _add(self, description, truth, known):
        self.atoms.append(description)
        self.masks.append(bits(truth))
        self.knowns.append(bits(known))

    def _build_event_atoms(self, min_known):
        ev = self.events
        for e in ev:
            known = e.values != UNKNOWN
            present = np.unique(e.values[known])
            for v in present:
                truth = known & (e.values == v)
                label = 'none' if v == NONE else int(v)
                known_at = self._value_clock(e, v)
                self._add({'kind': 'event_value', 'event': e.name, 'value': label,
                           'text': f'{e.name} = {label}', 'known_at': int(known_at),
                           'events': [e.name]}, truth, known)
            # «не наступило» может быть возможным, но не встретиться ни разу.
            if NONE not in present:
                truth = np.zeros(self.n, dtype=bool)
                self._add({'kind': 'event_value', 'event': e.name, 'value': 'none',
                           'text': f'{e.name} = none', 'known_at': e.known_at,
                           'events': [e.name]}, truth, known)
        for i, j in itertools.combinations(range(len(ev)), 2):
            a, b = ev[i], ev[j]
            ka = (a.values >= 0) & (b.values >= 0)   # порядок определён только
            clock = max(a.known_at, b.known_at)      # для наступивших событий
            for sign in SIGNS:
                if sign == '<':
                    truth = ka & (a.values < b.values)
                elif sign == '=':
                    truth = ka & (a.values == b.values)
                else:
                    truth = ka & (a.values > b.values)
                self._add({'kind': 'event_order', 'text': f'{a.name} {sign} {b.name}',
                           'op': sign, 'known_at': int(clock),
                           'events': [a.name, b.name]}, truth, ka)
            gap = np.where(ka, b.values - a.values, 0)
            for d in np.unique(gap[ka]):
                truth = ka & (gap == int(d))
                self._add({'kind': 'event_distance',
                           'text': f'{b.name} - {a.name} = {int(d)}',
                           'distance': int(d), 'known_at': int(clock),
                           'events': [a.name, b.name]}, truth, ka)

    def _value_clock(self, event, value):
        """Когда утверждение «E = value» становится проверяемым.

        Адрес экстремума и последнего обновления известен ТОЛЬКО после конца
        участка: максимум законченного окна известен после окончания окна.
        Первое прохождение уровня известно на самой минуте пробоя, но не раньше
        доступности уровня. «Не наступило» — только к концу участка.
        """
        if value == NONE or event.kind.startswith(('extremum', 'last_update')):
            return event.known_at
        level_clock = event.level[0] if event.level else event.segment[0]
        return max(int(value), int(level_clock))

    # --- доступ

    def n_event_atoms(self):
        return len(self.atoms)

    def n_base_atoms(self):
        return len(self.space.masks)

    def describe(self, atom):
        if atom < len(self.atoms):
            return dict(self.atoms[atom])
        d = self.space.describe(atom - len(self.atoms))
        d['kind'] = 'relation'
        return d

    def mask(self, atom):
        return self.masks[atom] if atom < len(self.atoms) else self.space.masks[atom - len(self.atoms)]

    def known(self, atom):
        if atom < len(self.atoms):
            return self.knowns[atom]
        return self.space.known[(atom - len(self.atoms)) // 3]

    def intersection(self, atoms):
        truth, known = self.all, self.all
        for a in atoms:
            truth &= self.mask(a)
            known &= self.known(a)
        return truth, known

    def statistics(self, atoms):
        truth, known = self.intersection(atoms)
        n, count = known.bit_count(), truth.bit_count()
        return {'count': count, 'known': n, 'unknown': self.n - n,
                'frequency': count / n if n else None}

    # --- классификация одиночного атома

    def classify(self, atom, min_known=1):
        d = self.describe(atom)
        stat = self.statistics((atom,))
        out = {**d, **stat}
        if d['kind'] == 'relation':
            base = atom - len(self.atoms)
            out['logically_impossible'] = not anatomy_possible(self.space, base)
            out['is_equality'] = SIGNS[base % 3] == '='
        else:
            out['logically_impossible'] = False
            out['is_equality'] = d.get('op') == '=' or d['kind'] == 'event_distance'
        enough = stat['known'] >= min_known
        out['constant_on_corpus'] = bool(
            enough and stat['frequency'] is not None and stat['frequency'] in (0.0, 1.0))
        out['constancy_is_corpus_property_not_axiom'] = True
        return out


# ---------------------------------------------------------------- разбор конструкции

def dissect(lens, candidate):
    """Состав, совместная частота и вклад каждого отношения при исключении.

    Нулевой добавочный вклад означает зависимость НА ЭТОМ КОРПУСЕ, а не
    приговор конструкции: два разных отношения могут постоянно встречаться
    вместе, и именно такую согласованность мы ищем.
    """
    joint = lens.statistics(candidate)
    parts = []
    for a in candidate:
        rest = tuple(x for x in candidate if x != a)
        without = lens.statistics(rest) if rest else {'count': lens.n, 'known': lens.n,
                                                      'frequency': 1.0}
        alone = lens.statistics((a,))
        conditional = (joint['count'] / without['count']) if without['count'] else None
        parts.append({
            'atom': int(a), 'text': lens.describe(a)['text'],
            'alone_frequency': alone['frequency'],
            'support_without_it': without['count'],
            'conditional_given_rest': conditional,
            'adds_no_restriction': joint['count'] == without['count'],
            'restriction_note': 'нулевой добавочный вклад = зависимость на этом '
                                'корпусе, не приговор конструкции'})
    return {'joint': joint, 'parts': parts}


def equivalence_class(lens, candidate):
    """Какого рода совпадение носителей перед нами."""
    base = [a - len(lens.atoms) for a in candidate if lens.describe(a)['kind'] == 'relation']
    out = {'logical_equivalence_canonicalised_by_core': True,
           'conditional_equivalence': None, 'empirical_coincidence': None}
    if len(base) == len(candidate) and len(base) >= 2:
        out['implied_by_other_relations'] = logically_redundant(lens.space, base)
        out['conditional_equivalence'] = continuity_context(lens.space, base)
    supports = [lens.mask(a) & lens.known(a) for a in candidate]
    same = all(s == supports[0] for s in supports)
    out['empirical_coincidence'] = bool(same)
    out['empirical_coincidence_note'] = ('совпали только по примерам корпуса — '
                                         'остаётся предметом исследования')
    return out


# ---------------------------------------------------------------- перебор

def enumerate_candidates(atoms, size, budget, mode, rng):
    """Детерминированный порядок перебора; индекс кандидата восстановим."""
    total = math.comb(len(atoms), size)
    if mode == 'exact':
        quota = min(total, budget)
        for index, combo in enumerate(itertools.islice(itertools.combinations(atoms, size), quota)):
            yield index, combo
    else:
        seen = set()
        while len(seen) < min(budget, total):
            pick = tuple(sorted(atoms[int(v)] for v in rng.choice(len(atoms), size, replace=False)))
            if pick not in seen:
                seen.add(pick)
                yield len(seen) - 1, pick


def survey(lens, *, sizes, budget, mode='exact', seed=0, top=40, min_known=1,
           min_cell=30, reference=None, labels=None, journal=None,
           include_constants=False, include_equalities=False, atom_pool=None):
    """Один проход поиска. Витрины — навигация, не приговор.

    `labels` — булев вектор по единицам; когда он задан, корпус делится на две
    части и появляются витрины различий. Именно этот выбор целиком повторяет
    `calibrate`.
    """
    rng = np.random.default_rng(seed)
    pool = atom_pool if atom_pool is not None else list(range(
        len(lens.atoms) + len(lens.space.masks)))
    marks = {a: lens.classify(a, min_known) for a in pool}
    eligible = [a for a in pool if not marks[a]['logically_impossible']]
    searchable = [a for a in eligible
                  if (include_constants or not marks[a]['constant_on_corpus'])
                  and (include_equalities or not marks[a]['is_equality'])]
    label_mask = None
    if labels is not None:
        label_mask = bits(np.asarray(labels, dtype=bool))
    heaps = {'frequent': [], 'absent_but_possible': [], 'constant_on_corpus': [],
             'equality_layer': []}
    if labels is not None:
        heaps.update({'more_in_group': [], 'less_in_group': []})
    if reference is not None:
        heaps.update({'more_than_reference': [], 'less_than_reference': []})
    records, tested, by_size = [], 0, {}
    start = time.monotonic()
    for size in sizes:
        possible = math.comb(len(searchable), size)
        seen_here = 0
        for index, candidate in enumerate_candidates(searchable, size, budget, mode, rng):
            tested += 1
            seen_here += 1
            truth, known = lens.intersection(candidate)
            n_known, count = known.bit_count(), truth.bit_count()
            row = {'index': index, 'size': size, 'atoms': candidate,
                   'count': count, 'known': n_known}
            if label_mask is not None:
                inside = (known & label_mask)
                outside = known & ~label_mask & lens.all
                row['in_known'] = inside.bit_count()
                row['in_count'] = (truth & inside).bit_count()
                row['out_known'] = outside.bit_count()
                row['out_count'] = (truth & outside).bit_count()
            if reference is not None:
                rs = reference.statistics(candidate)
                row['ref_count'], row['ref_known'] = rs['count'], rs['known']
            if journal is not None:
                journal(row)
            freq = count / n_known if n_known else None
            if freq is None:
                continue
            scores = {}
            if all(marks[a]['constant_on_corpus'] for a in candidate):
                scores['constant_on_corpus'] = freq
            elif any(marks[a]['is_equality'] for a in candidate):
                scores['equality_layer'] = freq
            else:
                scores['frequent'] = freq
                if count == 0 and n_known >= min_known:
                    scores['absent_but_possible'] = float(n_known)
            # Обе ячейки должны иметь опору: иначе витрина разностей заполняется
            # атомами с крошечным знаменателем, где разность близка к единице
            # по построению, а не по рынку.
            if (label_mask is not None and row['in_known'] >= min_cell
                    and row['out_known'] >= min_cell):
                delta = row['in_count'] / row['in_known'] - row['out_count'] / row['out_known']
                scores['more_in_group'] = delta
                scores['less_in_group'] = -delta
            if reference is not None and row['ref_known']:
                delta = freq - row['ref_count'] / row['ref_known']
                scores['more_than_reference'] = delta
                scores['less_than_reference'] = -delta
            for name, score in scores.items():
                item = (float(score), count, candidate)
                heap = heaps[name]
                if len(heap) < top:
                    heap.append(item)
                    heap.sort()
                elif item > heap[0]:
                    heap[0] = item
                    heap.sort()
        by_size[str(size)] = {'possible': possible, 'tested': seen_here,
                              'complete': seen_here == possible}
    # Постоянность и равенство — свойства ОТДЕЛЬНОГО отношения, а не находки
    # перебора. Их витрины наполняются напрямую из разметки, поэтому остаются
    # доступны даже когда такие атомы исключены из перебора ради скорости.
    for atom in eligible:
        mark = marks[atom]
        if mark['frequency'] is None:
            continue
        if mark['constant_on_corpus']:
            heaps['constant_on_corpus'].append((mark['frequency'], mark['count'], (atom,)))
        if mark['is_equality']:
            heaps['equality_layer'].append((mark['frequency'], mark['count'], (atom,)))
    for name in ('constant_on_corpus', 'equality_layer'):
        heaps[name] = sorted(heaps[name], reverse=True)[:top]
    report = {
        'version': VERSION, 'mode': mode, 'seed': seed, 'sizes': list(sizes),
        'units': lens.n, 'event_atoms': len(lens.atoms),
        'relation_atoms': len(lens.space.masks),
        'atoms_in_pool': len(pool), 'searchable_atoms': len(searchable),
        'excluded_impossible': len(pool) - len(eligible),
        'excluded_constant': sum(1 for a in eligible if marks[a]['constant_on_corpus']),
        'excluded_equality': sum(1 for a in eligible if marks[a]['is_equality']),
        'exclusions_are_view_filters_not_axioms': True,
        'tested': tested, 'by_size': by_size, 'min_cell': min_cell,
        'min_cell_role': 'минимальная опора В КАЖДОЙ ячейке витрины разностей; '
                         'без неё максимум набирают атомы с крошечным знаменателем',
        'complete': all(v['complete'] for v in by_size.values()),
        'seconds': time.monotonic() - start,
        'views_are_navigation_not_verdicts': True,
        'views': {name: [{'atoms': list(item[2]), 'display_value': item[0],
                          'count': item[1]}
                         for item in sorted(heap, reverse=True)]
                  for name, heap in heaps.items()}}
    return report, marks


# ---------------------------------------------------------------- калибровка

def calibrate(lens, *, labels, repeats, sizes, budget, mode, seed, top,
              exchangeability_justified=False, blocks=None, **kw):
    """Повторяет ВЕСЬ зависящий от меток выбор при разрушенной связи.

    Носители атомов от меток не зависят и считаются один раз — переиспользуется
    та же `lens`. Перемешиваются только метки. `blocks` (если заданы) держат
    перестановку внутри сопоставимых обстоятельств.
    """
    rng = np.random.default_rng(seed + 1_000_003)
    labels = np.asarray(labels, dtype=bool)
    watched = ('more_in_group', 'less_in_group')
    draws = {name: [] for name in watched}
    for r in range(repeats):
        if blocks is None:
            shuffled = rng.permutation(labels)
        else:
            shuffled = labels.copy()
            for b in np.unique(blocks):
                idx = np.flatnonzero(blocks == b)
                shuffled[idx] = rng.permutation(labels[idx])
        rep, _ = survey(lens, sizes=sizes, budget=budget, mode=mode, seed=seed,
                        top=top, labels=shuffled, **kw)
        for name in watched:
            values = [v['display_value'] for v in rep['views'].get(name, [])]
            draws[name].append(max(values) if values else None)
    return {
        'repeats': repeats,
        'best_under_broken_link': draws,
        'exchangeability_justified': bool(exchangeability_justified),
        'blocks_used': blocks is not None,
        'status': 'CALIBRATED' if exchangeability_justified else 'DIAGNOSTIC',
        'meaning': 'какие лучшие различия способен произвести ЭТОТ ЖЕ поиск '
                   'при разрушенной проверяемой связи',
        'not_a_ceiling': 'малое число повторов показывает серьёзность проблемы, '
                         'но надёжного потолка не устанавливает',
        'scope': 'калибрует только утверждение «конструкция помогает предсказать '
                 'метку»; повторяемость и специфичность якоря ею не запрещены'}


# ---------------------------------------------------------------- негативное пространство

def negative_space(lens, candidate, *, reference=None, min_known=1):
    """Различает невозможное, ненаблюдавшееся и более редкое, чем сравнение."""
    stat = lens.statistics(candidate)
    marks = [lens.classify(a, min_known) for a in candidate]
    impossible = any(m['logically_impossible'] for m in marks)
    out = {'joint': stat, 'logically_impossible': impossible,
           'contains_equality': any(m['is_equality'] for m in marks),
           'unknown_units': lens.n - stat['known'],
           'bare_zero_note': 'голый ноль среди миллионов сочетаний сам по себе '
                             'ничего не выделяет'}
    if impossible:
        out['class'] = 'logically_impossible'
    elif stat['known'] < min_known:
        out['class'] = 'not_enough_observations'
    elif stat['count'] == 0:
        out['class'] = 'possible_but_unobserved'
    else:
        out['class'] = 'observed'
    if reference is not None:
        rs = reference.statistics(candidate)
        out['reference'] = rs
        if rs['known'] and stat['known']:
            out['rarer_than_comparison'] = (stat['frequency'] or 0) < (rs['frequency'] or 0)
    return out


def vanishing_continuation(lens, prefix, continuation_atoms, *, min_known=1):
    """Какое продолжение почти исчезает ПОСЛЕ уже наблюдаемой конструкции.

    Знаменатель — все единицы, достигшие того же наблюдаемого префикса,
    включая те, где продолжение не наступило, и отдельно неизвестные исходы.
    """
    p_truth, p_known = lens.intersection(prefix)
    c_truth, c_known = lens.intersection(continuation_atoms)
    reached = p_truth & c_known
    others = p_known & ~p_truth & c_known & lens.all
    def rate(mask):
        n = mask.bit_count()
        hit = (c_truth & mask).bit_count()
        return {'known': n, 'count': hit, 'frequency': hit / n if n else None}
    return {'after_prefix': rate(reached), 'elsewhere': rate(others),
            'prefix_reached_outcome_unknown': (p_truth & ~c_known & lens.all).bit_count(),
            'denominator_note': 'знаменатель — ВСЕ достигшие того же префикса, '
                                'включая невозвраты и неизвестные исходы'}


# ---------------------------------------------------------------- журнал

class CompactJournal:
    """Адреса и счётчики в двоичных массивах вместо строки на кандидата."""

    def __init__(self, size_hint=1 << 12):
        self.index, self.size, self.atoms = [], [], []
        self.count, self.known = [], []
        self.extra = {}

    def __call__(self, row):
        self.index.append(row['index'])
        self.size.append(row['size'])
        self.atoms.append(row['atoms'])
        self.count.append(row['count'])
        self.known.append(row['known'])
        for key in ('in_count', 'in_known', 'out_count', 'out_known',
                    'ref_count', 'ref_known'):
            if key in row:
                self.extra.setdefault(key, []).append(row[key])

    def save(self, path, meta):
        width = max((len(a) for a in self.atoms), default=1)
        atoms = np.full((len(self.atoms), width), -1, dtype=np.int32)
        for i, a in enumerate(self.atoms):
            atoms[i, :len(a)] = a
        data = {'index': np.asarray(self.index, dtype=np.int64),
                'size': np.asarray(self.size, dtype=np.int16),
                'atoms': atoms,
                'count': np.asarray(self.count, dtype=np.int32),
                'known': np.asarray(self.known, dtype=np.int32),
                'meta_json': np.array(json.dumps(meta, ensure_ascii=False))}
        for key, values in self.extra.items():
            data[key] = np.asarray(values, dtype=np.int32)
        with Path(path).open('xb') as stream:
            np.savez_compressed(stream, **data)


def restore(journal_path, position):
    """Любой просмотренный кандидат восстановим — попал он в витрину или нет."""
    with np.load(journal_path, allow_pickle=False) as z:
        atoms = z['atoms'][position]
        row = {'index': int(z['index'][position]), 'size': int(z['size'][position]),
               'atoms': [int(v) for v in atoms if v >= 0],
               'count': int(z['count'][position]), 'known': int(z['known'][position]),
               'meta': json.loads(str(z['meta_json'].item()))}
        for key in ('in_count', 'in_known', 'out_count', 'out_known',
                    'ref_count', 'ref_known'):
            if key in z:
                row[key] = int(z[key][position])
    return row


def expand(lens, candidate, unit, *, examples_of=None):
    """Раскрывает атом до конкретных свечей и сравнений на одной единице."""
    out = []
    for a in candidate:
        d = lens.describe(a)
        item = {'text': d['text'], 'kind': d['kind']}
        if d['kind'] == 'relation':
            base = a - len(lens.atoms)
            pair = base // 3
            left, right = int(lens.space.left[pair]), int(lens.space.right[pair])
            flat = lens.corpus.ohlc.reshape(lens.n, -1)
            item['left'] = {'minute': int(lens.corpus.ordinals[left // 4]),
                            'field': FIELDS[left % 4], 'price': float(flat[unit, left])}
            item['right'] = {'minute': int(lens.corpus.ordinals[right // 4]),
                             'field': FIELDS[right % 4], 'price': float(flat[unit, right])}
        else:
            for name in d['events']:
                e = next(x for x in lens.events if x.name == name)
                value = int(e.values[unit])
                item.setdefault('events', []).append({
                    'event': name, **e.describe(),
                    'value': ('unknown' if value == UNKNOWN else
                              'none' if value == NONE else value)})
                if e.kind.startswith('last_update'):
                    item['events'][-1]['trace'] = update_trace(
                        lens.corpus, unit, e.segment[0], e.segment[1], e.kind[-1])
        out.append(item)
    return out


# ---------------------------------------------------------------- стандартный словарь

def default_events(corpus, *, start, stop, anchor_ordinal=0):
    """Компактный словарь событий на ЯВНО названном бюджете наблюдения.

    Бюджет — аргумент, а не горизонт сделки. Ни один адрес не назначен заранее:
    перебираются все допустимые значения, частоту говорит корпус.
    """
    ev = [
        extremum_event(corpus, start, stop, 'H', False),
        extremum_event(corpus, start, stop, 'H', True),
        extremum_event(corpus, start, stop, 'L', False),
        extremum_event(corpus, start, stop, 'L', True),
        last_update_event(corpus, start, stop, 'H'),
        last_update_event(corpus, start, stop, 'L'),
        crossing_event(corpus, start, stop, 'H', (anchor_ordinal, 'H'), True),
        crossing_event(corpus, start, stop, 'C', (anchor_ordinal, 'H'), True),
        crossing_event(corpus, start, stop, 'L', (anchor_ordinal, 'L'), False),
        crossing_event(corpus, start, stop, 'C', (anchor_ordinal, 'L'), False),
    ]
    return ev


# ------------------------------------------------- временной контракт

LEVELS = ('H0', 'L0', 'maxH_prefix', 'minL_prefix')


def resolve_level(corpus, name, cursor):
    """Опора. Уровень, рассчитанный по префиксу, законен и здесь назван."""
    if name == 'H0':
        return point_level(corpus, 0, 'H')
    if name == 'L0':
        return point_level(corpus, 0, 'L')
    if name == 'maxH_prefix':
        return prefix_extreme_level(corpus, 1, cursor, 'H')
    if name == 'minL_prefix':
        return prefix_extreme_level(corpus, 1, cursor, 'L')
    raise ValueError(f'неизвестная опора {name!r}')


class Contract:
    """Объявленный временной контракт проверки продолжения.

    Определяет смысл задачи, а не набор исключений: курсор, опоры, популяцию
    решения по одному только доступному прошлому, будущее событие, границу
    наблюдения и обращение с неизвестными исходами.

    `achievement` различает три РАЗНЫЕ задачи, а не три способа фильтрации:
      first_ever          — первое достижение за весь эпизод; ранее достигшие
                            исключаются, потому что у них этого вопроса нет;
      next_after_cursor   — следующее достижение после курсора независимо от
                            прошлого; прошлые достижения остаются в популяции;
      repeat_after_return — повторное прохождение после возврата; для него
                            прошлое достижение СОСТАВЛЯЕТ ПРЕДМЕТ и требуется.

    Незаконно засчитывать уже состоявшееся достижение как последующее. Поэтому
    измеряемое продолжение всегда начинается строго после курсора. Запрета на
    пересечение окон нет: прошлое необходимо для построения опор.
    """

    def __init__(self, *, cursor, horizon, side, field, level, above,
                 achievement='first_ever', supports=LEVELS):
        if not 0 <= cursor < horizon:
            raise ValueError('курсор должен лежать строго внутри границы наблюдения')
        if achievement not in ('first_ever', 'next_after_cursor',
                               'repeat_after_return'):
            raise ValueError('неизвестный режим достижения')
        self.cursor, self.horizon, self.side = int(cursor), int(horizon), side
        self.field, self.level, self.above = field, level, bool(above)
        self.achievement = achievement
        self.supports = tuple(supports)

    def describe(self):
        return {'cursor': self.cursor, 'horizon': self.horizon, 'side': self.side,
                'future_event': {'field': self.field, 'level': self.level,
                                 'direction': 'above' if self.above else 'below',
                                 'window': [self.cursor + 1, self.horizon]},
                'achievement': self.achievement,
                'supports_available_at_cursor': list(self.supports),
                'rule': 'измеряемое продолжение начинается строго после курсора; '
                        'опоры считаются по префиксу и доступны на курсоре'}


def _returned(corpus, cursor, level_values, above, first_hit):
    """Вернулась ли цена за опору обратно к курсору после пройденного пробоя."""
    lo, hi = _slice(corpus, 1, cursor)
    close = corpus.ohlc[:, lo:hi + 1, FIELDS.index('C')]
    ok = _finite(corpus)[:, lo:hi + 1]
    back = (close < level_values[:, None]) if above else (close > level_values[:, None])
    back &= ok
    out = np.zeros(len(close), dtype=bool)
    base = int(corpus.ordinals[lo])
    for i in range(len(close)):
        k = int(first_hit[i])
        if k < 0:
            continue
        j = k - base
        if 0 <= j < back.shape[1]:
            out[i] = bool(back[i, j + 1:].any())
    return out


def apply_contract(corpus, contract):
    """Популяция решения, метка продолжения и полный аудит исключений."""
    c = contract
    level = resolve_level(corpus, c.level, c.cursor)
    if level[2] > c.cursor:
        raise ValueError('опора недоступна к курсору')
    before = crossing_event(corpus, 1, c.cursor, c.field, level, c.above)
    after = crossing_event(corpus, c.cursor + 1, c.horizon, c.field, level, c.above)
    happened_before = before.values >= 0
    prefix_known = before.values != UNKNOWN
    future_known = after.values != UNKNOWN
    if c.achievement == 'first_ever':
        population = prefix_known & ~happened_before
        reason = 'исключены единицы, где событие уже случилось к курсору'
    elif c.achievement == 'next_after_cursor':
        population = prefix_known
        reason = 'прошлые достижения оставлены в популяции'
    else:
        returned = _returned(corpus, c.cursor, level[0], c.above, before.values)
        population = prefix_known & happened_before & returned
        reason = 'оставлены только вернувшиеся после пройденного пробоя'
    usable = population & future_known
    labels = after.values >= 0
    return {
        'contract': c.describe(),
        'level_name': level[3], 'level_available_from': level[2],
        'population': population, 'usable': usable,
        'labels': labels, 'future_known': future_known,
        'before': before, 'after': after,
        'audit': {
            'units_total': int(corpus.n),
            'prefix_unknown': int((~prefix_known).sum()),
            'already_happened_by_cursor': int(happened_before.sum()),
            'in_decision_population': int(population.sum()),
            'population_rule': reason,
            'future_unknown_inside_population': int((population & ~future_known).sum()),
            'usable': int(usable.sum()),
            'continuation_happened': int((usable & labels).sum()),
            'note': 'событие ещё не наступило к курсору — законный факт префикса, '
                    'а не сведение о будущем'}}


# ------------------------------------------------- сопоставимое состояние

def state_key(corpus, cursor, supports=LEVELS):
    """Минимальное порядковое описание состояния на минуте решения.

    Положение закрытия относительно доступных опор и уже совершённые
    прохождения этих опор. НЕ содержит проверяемой конструкции: иначе
    различие исчезло бы по определению.
    """
    close = _column(corpus, cursor, 'C')
    known = np.isfinite(close)
    parts, names = [], []
    for name in supports:
        values, ok, clock, label = resolve_level(corpus, name, cursor)
        known &= ok
        sign = np.where(close > values, 1, np.where(close < values, -1, 0))
        parts.append(sign)
        names.append(f'C[{cursor}] против {label}')
    for name, field, above in (('H0', 'H', True), ('L0', 'L', False)):
        if name not in supports:
            continue
        level = resolve_level(corpus, name, cursor)
        ev = crossing_event(corpus, 1, cursor, field, level, above)
        known &= ev.values != UNKNOWN
        parts.append((ev.values >= 0).astype(np.int64))
        names.append(f'тень за {level[3]} уже к минуте {cursor}')
    keys = np.stack(parts, axis=1)
    return keys, known, names


def stratified_compare(keys, key_known, group, group_known, labels, usable,
                       min_per_cell=5):
    """Добавка конструкции внутри сопоставимых порядковых состояний.

    Страты, где встретилась только одна группа, — ОТСУТСТВИЕ СРАВНЕНИЯ,
    а не нулевой эффект. Они перечисляются отдельно и в свод не входят.
    Контроль уравнивает доступное порядковое положение; метрическую близость
    к уровню он не уравнивает и этого не обещает.
    """
    ok = key_known & group_known & usable
    rows, no_compare, thin = [], [], []
    view, g, y = keys[ok], group[ok], labels[ok]
    seen = {}
    for i in range(len(view)):
        seen.setdefault(tuple(int(v) for v in view[i]), []).append(i)
    pooled_num = pooled_den = 0.0
    compared_units = 0
    for key, idx in sorted(seen.items()):
        idx = np.asarray(idx)
        gi, yi = g[idx], y[idx]
        n1, n0 = int(gi.sum()), int((~gi).sum())
        row = {'state': list(key), 'n_with': n1, 'n_without': n0,
               'hit_with': int(yi[gi].sum()), 'hit_without': int(yi[~gi].sum())}
        if n1 == 0 or n0 == 0:
            row['status'] = 'нет сравнения: встретилась только одна группа'
            no_compare.append(row)
            continue
        row['freq_with'] = row['hit_with'] / n1
        row['freq_without'] = row['hit_without'] / n0
        row['difference'] = row['freq_with'] - row['freq_without']
        row['status'] = ('сравнение есть, но опора тонкая'
                         if n1 < min_per_cell or n0 < min_per_cell else 'сравнимо')
        (thin if row['status'].startswith('сравнение есть') else rows).append(row)
        w = n1 * n0 / (n1 + n0)
        pooled_num += w * row['difference']
        pooled_den += w
        compared_units += n1 + n0
    lost = int(sum(r['n_with'] + r['n_without'] for r in no_compare))
    return {
        'strata_compared': rows, 'strata_thin': thin,
        'strata_no_comparison': no_compare,
        'pooled_difference': pooled_num / pooled_den if pooled_den else None,
        'pooled_note': 'вес n1*n0/(n1+n0); страты без сравнения в свод не входят',
        'units_in_comparison': compared_units, 'units_without_comparison': lost,
        'coverage': (compared_units / (compared_units + lost)
                     if compared_units + lost else None),
        'controls': 'доступное порядковое положение на минуте решения',
        'does_not_control': 'метрическую близость к уровню; одинаковый порядок цен '
                            'совместим с разными расстояниями между ними'}


# ------------------------------------------------- будущее как фильм

def unfold_future(corpus, cursor, horizon, supports=LEVELS):
    """Будущее — снова фильм, а не одна метка достижения уровня."""
    ev = []
    for name in supports:
        level = resolve_level(corpus, name, cursor)
        for field, above in (('H', True), ('L', False)):
            ev.append(crossing_event(corpus, cursor + 1, horizon, field, level, above))
        for above in (True, False):
            ev.append(crossing_event(corpus, cursor + 1, horizon, 'C', level, above))
    for field in ('H', 'L'):
        ev.append(extremum_event(corpus, cursor + 1, horizon, field, False))
        ev.append(last_update_event(corpus, cursor + 1, horizon, field))
    return ev


def order_picture(a, b, population):
    """Полная картина популяции, а не порядок среди дошедших.

    Условный порядок считается на совместном наступлении, но рядом всегда
    стоят: наступило только одно, не наступило ни одного, исход неизвестен.
    """
    pop = np.asarray(population, dtype=bool)
    ka, kb = a.values != UNKNOWN, b.values != UNKNOWN
    known = pop & ka & kb
    ha, hb = a.values >= 0, b.values >= 0
    both = known & ha & hb
    counts = {
        'a_before_b': int((both & (a.values < b.values)).sum()),
        'b_before_a': int((both & (a.values > b.values)).sum()),
        'same_minute': int((both & (a.values == b.values)).sum()),
        'only_a': int((known & ha & ~hb).sum()),
        'only_b': int((known & ~ha & hb).sum()),
        'neither': int((known & ~ha & ~hb).sum()),
        'unknown': int((pop & ~(ka & kb)).sum())}
    total = int(pop.sum())
    pair = counts['a_before_b'] + counts['b_before_a']
    return {'a': a.name, 'b': b.name, 'population': total, **counts,
            'both_happened': int(both.sum()),
            'conditional_a_before_b_given_both': (counts['a_before_b'] / pair
                                                  if pair else None),
            'conditional_share_of_population': (int(both.sum()) / total
                                                if total else None),
            'warning': 'условный порядок описывает только дошедших до обоих событий'}


def event_profile(events, population, labels=None):
    """Устройство продолжения: что наступает, когда и что перестаёт наступать."""
    pop = np.asarray(population, dtype=bool)
    rows = []
    for e in events:
        known = pop & (e.values != UNKNOWN)
        happened = known & (e.values >= 0)
        row = {'event': e.name, 'known': int(known.sum()),
               'happened': int(happened.sum()),
               'rate': int(happened.sum()) / int(known.sum()) if known.sum() else None,
               'median_minute': (float(np.median(e.values[happened]))
                                 if happened.any() else None)}
        if labels is not None:
            lab = np.asarray(labels, dtype=bool)
            for tag, mask in (('with', known & lab), ('without', known & ~lab)):
                n = int(mask.sum())
                hit = int((mask & happened).sum())
                row[f'{tag}_n'] = n
                row[f'{tag}_rate'] = hit / n if n else None
                row[f'{tag}_median_minute'] = (float(np.median(e.values[mask & happened]))
                                               if (mask & happened).any() else None)
            if row['with_rate'] is not None and row['without_rate'] is not None:
                row['rate_difference'] = row['with_rate'] - row['without_rate']
        rows.append(row)
    return rows
