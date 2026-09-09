#!/usr/bin/env python3
"""Трафарет отношений свечей — исполняемая реализация и вход для следующего агента.

ЗАМЫСЕЛ ТРЕЙДЕРА, уточнённый в разговоре 2026-09-09
=================================================
Каждый пример — минутный фильм. T0 имеет номер 0; следующие минуты 1, 2, ...
Условные 50 минут означают 51 свечу вместе с T0, НЕ выбранный срок сделки.
Предмет исследования — взаимное положение открытий, закрытий, high и low
ВСЕХ свечей фильма. Например, H20 > H18 и C20 < H18 отличает проход тенью
от C20 > H18. Числа 18/20 здесь поясняют язык, не задают искомый сетап.
Цена нужна только для точных сравнений <, =, >. Длины, ATR, размах, Volume,
доход и стоп не участвуют в представлении, отборе или ранжировании.

Полный фильм сохраняет также связи с последующими свечами. Найденный рисунок
может соединять удалённые минуты; его остальные связи могут различаться
в разных примерах. При применении в моменте доступна только часть отношений,
обе свечи которых уже закрылись. Остальное — историческое продолжение.
Это описание OHLC, а не порядок тиков внутри минуты и не доказательство BOS
в любом внешнем торговом словаре. Здесь «закрылось выше» означает C_i > H_j.

ОТЛИЧИЕ ОТ ПРЕЖНИХ ПОПЫТОК
==========================
work/064/alphabet.py теряет дальние связи и положение C относительно чужого H.
work/064/xray.py усредняет каждый из 18 заданных признаков отдельно. Здесь
сохраняется адрес КАЖДОГО отношения и множество примеров, где оно истинно.
Совместная плотность считается пересечением этих множеств, а не произведением
отдельных частот. Один фильм никогда не собирается из свечей разных примеров.
Узор на 4..7 и сходный на 9..12 имеют разные адреса; время от T0 сохраняется.
Готовый паттерн или список свингов на вход майнеру не передаются.

ЧТО ДЕЛАЕТ КОД И ГДЕ ЕГО ГРАНИЦА
================================
1. Space строит ВСЕ сравнения всех точек OHLC: верхний треугольник без потерь.
   Обратное сравнение получается сменой знака, диагональ равна самой себе.
   Включены O/C одной свечи; неизменные анатомические отношения не выдаются
   за находку. Формируются частоты по адресам и битовые множества примеров.
2. discover ищет совместные сочетания отношений без заранее заданного рисунка.
   exact — полный перебор сочетаний от 2 до max_relations включительно.
   sampled — равномерные случайные предложения сочетаний на каждом размере;
   их частоты затем считаются ТОЧНО на ВСЕХ примерах, без подвыборки фильмов.
   Все допустимые знаки участвуют, включая не встречавшиеся: иначе прибор
   потерял бы негативное пространство. min_count только помечает малую опору.
3. Поиск экспоненциален. max_relations, min_count, budget, seed и top — явные
   вычислительные параметры, НЕ рыночные правила. sampled обычно неполон;
   exact с исчерпанным бюджетом тоже неполон. «Ничего не нашли» относится
   только к фактически просмотренному пространству. Для 51 свечи имеется
   20 706 неупорядоченных пар точек, а сочетаний намного больше. Этот файл
   не обещает перебрать их все быстро или обязательно обнаружить сетап.
4. Единого рейтинга нет: отдельные виды по частоте, редкости/отсутствию,
   совместности и, при reference, положительной/отрицательной разности частот.
   top ограничивает ТОЛЬКО каждую из этих витрин; candidates.jsonl сохраняет каждый
   просмотренный кандидат, включая нули, слабые опоры и логические отказы.
   inspect достаёт любой из них и показывает поддержавшие/противоречащие свечи.
   Произведение частот — отдельная описательная диагностика, НЕ рыночный null,
   p-value или преимущество. Reference не является обязательным пропуском
   к обнаружению. Его смысл определяется реальным способом отбора корпуса.
5. Выход содержит формулы, номера свечей, частоты, неизвестные случаи,
   точные ID поддержавших примеров и время доступности. matching(cursor)
   отдельно показывает уже известные отношения и ещё не наступившую часть.

ЕДИНИЦА, ПРОПУСКИ, ПРОИСХОЖДЕНИЕ
==============================
Corpus принимает массив (пример, минута, OHLC) с полями строго O,H,L,C.
Отсутствующая минута — четыре NaN; она НЕ сдвигает нумерацию. В сравнении
с ней исход неизвестен. Доля всегда имеет собственный явный знаменатель.
unit_id задаёт единицу плотности. Повторные строки одной единицы схлопываются
ТОЛЬКО при полностью одинаковых OHLC/пропусках; все исходные IDs сохраняются.
Разные пути под одним unit_id вызывают ошибку. unit_id='riz_id' считает ризы,
unit_id='instrument:T0' считает одинаковые ценовые окна один раз. Это разные
вопросы, а не автоматическое утверждение независимости. Перекрывающиеся окна
и общие дни могут оставаться зависимыми при любом из этих вариантов.

NPZ-вход (без pickle): ohlc, ordinals, ids, unit_ids, metadata_json.
metadata_json — JSON-строка с instrument, source, anchor, unit_definition.
Один корпус содержит один инструмент, один смысл якоря и один отбор стороны.
Север/юг не отражаются и не смешиваются автоматически. source называет
источник, календарь и выборку; SHA256 NPZ добавляется при загрузке.

ГОТОВЫЕ КОМАНДЫ И API
====================
Зависимость: numpy (уже входит в pyproject.toml). Запуск из корня репозитория:

  python -B research/relational_stencil.py selftest
  python -B research/relational_stencil.py export-field --instrument NQ \
      --start 2020-01-01 --stop 2020-02-01 --side south --unit t0 \
      --after 50 --out work/relational-stencil/input.npz
  python -B research/relational_stencil.py discover \
      --input work/relational-stencil/input.npz --mode sampled \
      --max-relations 3 --min-count 20 --budget 200000 --seed 9 --top 40 \
      --out work/relational-stencil/result

Это примеры команд, а не уже проведённое исследование и не выбранная
территория. export-field читает существующие паспорта и OHLC, не пересоздаёт
RIZ и не переопределяет T0. Он экспортирует фиксированное окно; контакт или
удаление не обрезают его молча. Первое/второе посещение линии — отдельный
смысловой отбор по проверенным событиям; эта команда его не подменяет.
Для такого корпуса подготовь Corpus из настоящих сцен, сохрани ordinals от
исходного T0 и опиши anchor/отбор/причины пропусков в metadata. Разные смыслы
якоря и границы наблюдения исследуются отдельными запусками.

Результат: density.jsonl (все отдельные отношения, включая константы),
candidates.jsonl (ВСЕ просмотренные сочетания, независимо от витрин),
motifs.json (найденные совместные рисунки), members.npz (точные множества),
run.json (входы, параметры, бюджет, ограничения, статистика просмотренного).
started.json без run.json означает незаконченный запуск, а не пустой рынок.
Перезапись существующего результата запрещена: новый запуск — новый путь.

  python -B research/relational_stencil.py inspect \
      --result work/relational-stencil/result --sequence 42 --examples 2

Программно: space = Space(corpus); space.compare(20,'C',18,'H') возвращает
для каждого случая -1/0/+1 и known. discover(space, ...) возвращает рисунки.
matching(space, motif, cursor=20) читает только закрытые к +20 свечи.
continuation(space, motif, cursor=20) отдельно измеряет историческую связь
наблюдаемой части с продолжением; это ещё не проверка прогноза на новых данных.

ЛИНЗЫ, ПРИСЛАННЫЕ ТРЕЙДЕРОМ ВО ВРЕМЯ РЕАЛИЗАЦИИ
================================================
Прибор называет результат точно: «сочетание отношений OHLC на этих минутах».
Одна связь с ранним high — расположение/достижение уровня, не вся форма.
Целый рисунок — более содержательная конструкция; такое имя надо оправдать
исходными свечами, которые inspect оставляет человеку. Решение о торговом
действии, смысле сцены или важности находки не выдаётся по одному рейтингу.

«Часто» имеет явный знаменатель. Разность с обычными минутами спрашивает о
специфике якоря, разность между обстоятельствами — о контексте; ни одна сама
по себе не проверяет преимущество. Цена может быть непрерывной без равенства
close/open соседних минут (между последней и первой сделкой бывает зазор).
Поэтому жёсткий страж проверяет только логические противоречия/следствия
OHLC. Отдельно показывается, что стало бы избыточным ПРИ C[i]=O[i+1], и
измеряется, на какой доле доступных примеров это условие действительно верно.
Это не отделяет все свойства случайной непрерывной цены от свойств рынка.

Часы T0 сохраняются намеренно. Это координата распознавания машины, не
доказанная естественная фаза цены. Иной якорь/ритм требует отдельно описанного
корпуса и сопоставления; код не переопределяет T0 и не растягивает фильмы.
Выбор времени дня, эпохи и сцены остаётся явно названным прицелом во входах;
пустой выход не способен отличить неверный прицел от отсутствия явления.

Не встречавшееся сочетание сохраняется вместе с числом наблюдений, пропусками
и логической допустимостью. Сильное отсутствие требует осмысленного сравнения;
ноль в ограниченном корпусе не становится рыночным запретом. Тесты включают
известный внедрённый повтор и исчезнувшее сочетание при одинаковых отдельных
частотах. Они проверяют точный оператор и полный малый поиск; чувствительность
случайного бюджетного поиска на всей истории ещё НЕ установлена. После этой
проверки следующий полезный шаг — небольшой предметный просмотр сцен, а не
стройка универсального движка ради самого движка.

ГДЕ ТЕПЕРЬ ПУБЛИЧНЫЙ ВХОД ИССЛЕДОВАНИЯ
=====================================
Единственный действующий путь обнаружения содержательных конструкций —
`research/ordinal_events.py`. Там исправлены витрины, разделены логическая
невозможность, постоянность на корпусе и самостоятельное отношение, добавлен
язык разворачивания событий и временной контракт проверки продолжения.

`discover` ниже СОХРАНЁН для чтения прежних результатов и вызова низкоуровневых
отношений, но его витрины больше не являются способом обнаружения:
  frequent      забивается анатомическими постоянствами (H > L на всех случаях);
  rare          забивается вырожденными равенствами непрерывной цены;
  joint_...     забивается синонимами: joint_excess максимален, когда два атома
                описывают одно событие с p около 0,5.
Это свойства ранжирования, а не рынка. Точный оператор, префикс, знаменатели
и происхождение примеров остаются верными и используются слоем как есть.

ПЕРЕД ПРОДОЛЖЕНИЕМ
=================
Прочитай AGENTS.md и reference/CONTRACT.md перед рыночным расчётом. Файл
сохраняет смысл беседы и исполняемый метод, но не создаёт новых полномочий.
Поле заморожено; его идентичности и T0 берутся как есть. Исследования и торговые
выводы сейчас НЕ выполнены. Проверен selftest на искусственных OHLC; это
проверка инструмента, не внешнее подтверждение рыночного явления.
Не запускай субагентов: пользователь попросил работать самостоятельно.
Не заменяй этот предмет ATR, буквами соседних свечей, списком назначенных
паттернов или независимыми частотами. Числа найденного узора должны прийти
из поиска, а не из поясняющих примеров этого текста.
"""
from __future__ import annotations

import argparse
import hashlib
import heapq
import itertools
import json
import math
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FIELDS = ('O', 'H', 'L', 'C')
SIGNS = ('<', '=', '>')
VERSION = 'relational-stencil-2'


def bits(values):
    return int.from_bytes(np.packbits(values, bitorder='little').tobytes(), 'little')


def indices(mask, n):
    return np.flatnonzero(np.unpackbits(
        np.frombuffer(mask.to_bytes((n + 7) // 8, 'little'), dtype=np.uint8),
        bitorder='little')[:n]).tolist()


@dataclass
class Corpus:
    ohlc: np.ndarray
    ordinals: np.ndarray
    ids: list[str]
    unit_ids: list[str]
    metadata: dict

    def __post_init__(self):
        x = np.asarray(self.ohlc, dtype=np.float64)
        ords = np.asarray(self.ordinals)
        if x.ndim != 3 or x.shape[2] != 4 or x.shape[0] == 0:
            raise ValueError('ohlc must be a nonempty (cases, candles, 4) array')
        if ords.shape != (x.shape[1],) or not np.issubdtype(ords.dtype, np.integer):
            raise ValueError('ordinals must be an integer vector matching candle axis')
        if not np.all(np.diff(ords) == 1):
            raise ValueError('keep every minute slot; missing minutes must be NaN')
        if len(self.ids) != len(x) or len(self.unit_ids) != len(x):
            raise ValueError('each input row needs id and unit_id')
        if len(set(self.ids)) != len(self.ids):
            raise ValueError('input ids must be unique')
        if any(not isinstance(v, str) or not v for v in self.ids + self.unit_ids):
            raise ValueError('ids and unit_ids must be nonempty strings')
        for key in ('instrument', 'source', 'anchor', 'unit_definition'):
            if not self.metadata.get(key):
                raise ValueError(f'metadata requires {key}')
        known = np.isfinite(x).all(axis=2)
        missing = np.isnan(x).all(axis=2)
        if not np.all(known | missing):
            raise ValueError('a candle must have four finite prices or four NaN')
        o, h, l, c = (x[:, :, k] for k in range(4))
        if np.any(known & ((l > np.minimum(o, c)) | (h < np.maximum(o, c)) | (l > h))):
            raise ValueError('invalid OHLC ordering')
        # Collapse repeated measurements, retaining every original RIZ address.
        positions, members, keep, units = {}, [], [], []
        for i, (identifier, unit) in enumerate(zip(self.ids, self.unit_ids)):
            if unit in positions:
                k = positions[unit]
                if not np.array_equal(x[keep[k]], x[i], equal_nan=True):
                    raise ValueError(f'different films under unit_id {unit!r}')
                members[k].append(identifier)
            else:
                positions[unit] = len(keep)
                keep.append(i); units.append(unit); members.append([identifier])
        self.input_rows = len(x)
        self.members = members
        self.ohlc = x[keep]
        self.ordinals = ords.astype(np.int64)
        self.ids = [self.ids[i] for i in keep]
        self.unit_ids = units
        self.metadata = dict(self.metadata)

    @property
    def n(self):
        return len(self.ohlc)

    @classmethod
    def load(cls, path):
        path = Path(path)
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z['metadata_json'].item()))
            result = cls(z['ohlc'], z['ordinals'], z['ids'].tolist(),
                         z['unit_ids'].tolist(), meta)
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        result.metadata['input_path'] = str(path.resolve())
        result.metadata['input_sha256'] = digest.hexdigest()
        return result

    def save(self, path):
        path = Path(path)
        if path.suffix != '.npz':
            raise ValueError('corpus output must end in .npz')
        path.parent.mkdir(parents=True, exist_ok=True)
        # Expand aliases so a reload preserves the original population addresses.
        rows, ids, units = [], [], []
        for i, group in enumerate(self.members):
            for identifier in group:
                rows.append(i); ids.append(identifier); units.append(self.unit_ids[i])
        with path.open('xb') as stream:
            np.savez_compressed(stream, ohlc=self.ohlc[rows], ordinals=self.ordinals,
                                ids=np.array(ids), unit_ids=np.array(units),
                                metadata_json=json.dumps(self.metadata, ensure_ascii=False))


class Space:
    """Exact order geometry. One bit per retained analysis unit, not per flag sum."""

    def __init__(self, corpus, chunk=128):
        self.corpus = corpus
        self.n = corpus.n
        self.all = (1 << self.n) - 1
        flat = corpus.ohlc.reshape(self.n, -1)
        self.left, self.right = np.triu_indices(flat.shape[1], 1)
        self.lookup = {(int(a), int(b)): k for k, (a, b) in
                       enumerate(zip(self.left, self.right))}
        self.known = []
        self.masks = []                 # pair*3 + {-1,0,+1}+1
        for start in range(0, len(self.left), chunk):
            a = flat[:, self.left[start:start + chunk]]
            b = flat[:, self.right[start:start + chunk]]
            valid = np.isfinite(a) & np.isfinite(b)
            for j in range(a.shape[1]):
                self.known.append(bits(valid[:, j]))
                for relation in (a[:, j] < b[:, j], a[:, j] == b[:, j], a[:, j] > b[:, j]):
                    self.masks.append(bits(relation & valid[:, j]))

    def point(self, ordinal, field):
        where = np.flatnonzero(self.corpus.ordinals == ordinal)
        if len(where) != 1 or field not in FIELDS:
            raise ValueError(f'unknown candle/field: {ordinal}/{field}')
        return int(where[0]) * 4 + FIELDS.index(field)

    def atom(self, i, a, sign, j, b):
        left, right = self.point(i, a), self.point(j, b)
        if left == right or sign not in SIGNS:
            raise ValueError('atom needs two different points and <, =, or >')
        state = SIGNS.index(sign)
        if left > right:
            left, right, state = right, left, 2 - state
        return self.lookup[left, right] * 3 + state

    def compare(self, i, a, j, b):
        left, right = self.point(i, a), self.point(j, b)
        x = self.corpus.ohlc.reshape(self.n, -1)
        known = np.isfinite(x[:, left]) & np.isfinite(x[:, right])
        signs = (x[:, left] > x[:, right]).astype(np.int8) - (x[:, left] < x[:, right])
        return signs, known             # unknown signs have no meaning

    def describe(self, atom):
        pair, state = divmod(int(atom), 3)
        a, b = int(self.left[pair]), int(self.right[pair])
        i, j = int(self.corpus.ordinals[a // 4]), int(self.corpus.ordinals[b // 4])
        return {'atom': int(atom), 'left': [i, FIELDS[a % 4]],
                'op': SIGNS[state], 'right': [j, FIELDS[b % 4]],
                'text': f'{FIELDS[a % 4]}[{i}] {SIGNS[state]} {FIELDS[b % 4]}[{j}]',
                'known_at': max(i, j)}

    def intersection(self, atoms):
        truth, known = self.all, self.all
        for atom in atoms:
            truth &= self.masks[atom]
            known &= self.known[atom // 3]
        return truth, known

    def statistics(self, atoms):
        truth, known = self.intersection(atoms)
        n, count = known.bit_count(), truth.bit_count()
        p = count / n if n else None
        independent = (math.prod((self.masks[a] & known).bit_count() / n for a in atoms)
                       if n else None)
        return {'count': count, 'known': n, 'unknown': self.n - n,
                'frequency': p, 'marginal_product': independent,
                'joint_excess': p - independent if n else None}


def matching(space, motif, cursor):
    """Actual prefix mask; no future truth, availability or outcome filters used."""
    atoms = motif['atoms'] if isinstance(motif, dict) else motif
    seen = [a for a in atoms if space.describe(a)['known_at'] <= cursor]
    later = [a for a in atoms if space.describe(a)['known_at'] > cursor]
    truth, known = space.intersection(seen)
    return {'observed': [space.describe(a) for a in seen],
            'pending': [space.describe(a) for a in later],
            'matching_units': [space.corpus.unit_ids[i] for i in indices(truth, space.n)],
            'unknown_units': [space.corpus.unit_ids[i] for i in indices(space.all ^ known, space.n)],
            'fully_known_by_clock': not later}


def logically_redundant(space, candidate, gapless=False):
    """Drop only relations implied by other relations + OHLC anatomy.

    Equal support masks are NOT redundancy: their co-occurrence is precisely
    something we want to discover. No empirical equality is made a logical axiom.
    """
    constraints = []
    for atom in candidate:
        pair, state = divmod(atom, 3)
        a, b = int(space.left[pair]), int(space.right[pair])
        constraints.append((a, b, state))
    candles = {v // 4 for a, b, _ in constraints for v in (a, b)}
    points = {4 * candle + field for candle in candles for field in range(4)}
    native = {a: [(b, False) for b in points if a != b and a // 4 == b // 4
                  and (a % 4 == 2 or b % 4 == 1)] for a in points}
    if gapless:
        # This is a CONDITIONAL axiom, never silently an assumption about data.
        for candle in candles:
            if candle + 1 in candles:
                a, b = 4 * candle + 3, 4 * (candle + 1)
                native[a].append((b, False)); native[b].append((a, False))
    for excluded, (left, right, state) in enumerate(constraints):
        graph = {a: edges.copy() for a, edges in native.items()}
        for k, (a, b, relation) in enumerate(constraints):
            if k == excluded:
                continue
            if relation == 0:
                graph[a].append((b, True))
            elif relation == 2:
                graph[b].append((a, True))
            else:
                graph[a].append((b, False)); graph[b].append((a, False))
        def reachable(start, end, require_strict):
            pending, seen = [(start, False)], set()
            while pending:
                vertex, strict = pending.pop()
                if vertex == end and (strict or not require_strict):
                    return True
                if (vertex, strict) not in seen:
                    seen.add((vertex, strict))
                    pending.extend((child, strict or edge) for child, edge in graph[vertex])
            return False
        if ((state == 0 and reachable(left, right, True)) or
            (state == 2 and reachable(right, left, True)) or
            (state == 1 and reachable(left, right, False) and reachable(right, left, False))):
            return True
    return False


def anatomy_possible(space, atom):
    pair, state = divmod(atom, 3)
    a, b = int(space.left[pair]), int(space.right[pair])
    if a // 4 != b // 4:
        return True
    # Exact weak inequalities from L <= O,C <= H. Equality remains possible.
    return state in {(0, 1): (0, 1), (0, 2): (1, 2), (0, 3): (0, 1, 2),
                     (1, 2): (1, 2), (1, 3): (1, 2), (2, 3): (0, 1)}[a % 4, b % 4]


def logically_impossible(space, candidate):
    """A positive strict-order cycle, including native L<=O,C<=H constraints."""
    edges, candles = [], set()
    for atom in candidate:
        pair, state = divmod(atom, 3)
        a, b = int(space.left[pair]), int(space.right[pair])
        candles.update((a // 4, b // 4))
        if state == 0:
            edges.append((a, b, 1))
        elif state == 2:
            edges.append((b, a, 1))
        else:
            edges.extend(((a, b, 0), (b, a, 0)))
    for c in candles:
        o, h, l, close = [4 * c + k for k in range(4)]
        edges.extend(((l, o, 0), (l, close, 0), (o, h, 0), (close, h, 0)))
    distance = {4 * c + k: 0 for c in candles for k in range(4)}
    for _ in range(len(distance)):
        changed = False
        for a, b, strict in edges:
            if distance[b] < distance[a] + strict:
                distance[b] = distance[a] + strict
                changed = True
        if not changed:
            return False
    return True


def continuity_context(space, candidate):
    """Measure the exact adjacent C=O condition; do not call continuity a fact."""
    candles = sorted({int(p) // 4 for atom in candidate for p in
                      (space.left[atom // 3], space.right[atom // 3])})
    edges = [c for c in candles if c + 1 in candles]
    if not edges:
        return None
    _, eligible = space.intersection(candidate)
    exact, available = eligible, eligible
    for candle in edges:
        pair = space.lookup[4 * candle + 3, 4 * (candle + 1)]
        exact &= space.masks[3 * pair + 1]
        available &= space.known[pair]
    n = available.bit_count()
    return {'adjacent_slots': [[int(space.corpus.ordinals[c]),
                                int(space.corpus.ordinals[c + 1])] for c in edges],
            'exact_close_open_count': exact.bit_count(), 'known': n,
            'unknown': eligible.bit_count() - n,
            'fraction': exact.bit_count() / n if n else None,
            'redundant_IF_adjacent_opens_equal_previous_closes':
                logically_redundant(space, candidate, gapless=True)}


def continuation(space, motif, cursor):
    """Retrospective prefix/continuation association; NOT prediction validation."""
    atoms = motif['atoms'] if isinstance(motif, dict) else motif
    past = [a for a in atoms if space.describe(a)['known_at'] <= cursor]
    future = [a for a in atoms if space.describe(a)['known_at'] > cursor]
    if not past or not future:
        return {'status': 'NEEDS_BOTH_OBSERVED_PREFIX_AND_PENDING_RELATIONS'}
    p, pk = space.intersection(past)
    f, fk = space.intersection(future)
    eligible = pk & fk
    group, other = p & eligible, eligible & (space.all ^ p)
    def describe(mask):
        n = mask.bit_count()
        return {'known': n, 'count': (f & mask).bit_count(),
                'frequency': (f & mask).bit_count() / n if n else None}
    return {'status': 'RETROSPECTIVE_ASSOCIATION_ONLY', 'cursor': cursor,
            'prefix_true': describe(group), 'prefix_false': describe(other),
            'prefix_known_but_continuation_unknown': (p & (space.all ^ fk)).bit_count(),
            'future_relations': [space.describe(a) for a in future]}


def discover(space, *, max_relations, min_count, budget, mode, seed=0, top=40,
             reference=None, progress=None, journal=None):
    """Exact supports; either exhaustive or explicitly bounded random pattern search.

    A sampled proposal is an unordered set of atom addresses, NOT a candle motif
    chosen by the researcher. All surviving atoms are eligible regardless of
    their marginal prevalence. Sampling is without replacement at each size.
    """
    if mode not in ('exact', 'sampled') or max_relations < 2 or min_count < 1 or budget < 1 or top < 1:
        raise ValueError('invalid search parameters')
    warnings.warn(
        'discover(): витрины этого входа не являются способом обнаружения '
        'содержательных конструкций (постоянства, равенства, синонимы). '
        'Публичный путь исследования — research/ordinal_events.py; здесь '
        'сохранены точный оператор, префикс и знаменатели.',
        DeprecationWarning, stacklevel=2)
    if reference is not None:
        if not np.array_equal(space.corpus.ordinals, reference.corpus.ordinals):
            raise ValueError('reference needs the same candle coordinate axis')
        if space.corpus.metadata['instrument'] != reference.corpus.metadata['instrument']:
            raise ValueError('do not mix instruments')
    # Do NOT discard absent/rare atoms: absence can be the question. Only native
    # impossibilities and addresses with no observations in either corpus go.
    atoms = [a for a in range(len(space.masks)) if anatomy_possible(space, a) and
             (space.known[a // 3] or (reference and reference.known[a // 3]))]
    max_size = min(max_relations, len(atoms))
    rng = np.random.default_rng(seed)
    heaps = {name: [] for name in ('frequent', 'rare_observed_or_absent',
                                   'joint_concentration_diagnostic')}
    if reference:
        heaps.update({'more_than_reference': [], 'less_than_reference': []})
    tested, supported, minimal, by_size = 0, 0, 0, {}
    start = last_progress = time.monotonic()
    total_possible = sum(math.comb(len(atoms), k) for k in range(2, max_size + 1))
    for size in range(2, max_size + 1):
        possible = math.comb(len(atoms), size)
        remaining_sizes = max_size - size + 1
        quota = min(possible, max(0, (budget - tested) // remaining_sizes))
        if mode == 'exact':
            quota = min(possible, budget - tested)
        count_size = 0
        if mode == 'exact' or quota == possible:
            proposals = itertools.islice(itertools.combinations(atoms, size), quota)
        else:
            def draw(q=quota, k=size):
                seen = set()
                while len(seen) < q:
                    candidate = tuple(sorted(atoms[int(v)] for v in rng.choice(len(atoms), k, replace=False)))
                    if candidate not in seen:
                        seen.add(candidate)
                        yield candidate
            proposals = draw()
        for candidate in proposals:
            tested += 1; count_size += 1
            now = time.monotonic()
            if progress and now - last_progress >= 10:
                progress({'tested': tested, 'budget': budget, 'size': size})
                last_progress = now
            truth, eligible = space.intersection(candidate)
            stat = space.statistics(candidate)
            ref = reference.statistics(candidate) if reference else None
            contradictory = len({a // 3 for a in candidate}) != size
            impossible = contradictory or logically_impossible(space, candidate)
            redundant = logically_redundant(space, candidate) if not impossible else False
            supported += stat['count'] >= min_count
            minimal += not redundant and not impossible
            delta = (stat['frequency'] - ref['frequency'] if ref and
                     stat['frequency'] is not None and ref['frequency'] is not None else None)
            record = {'sequence': tested, 'atoms': candidate, **stat,
                      'reference': ref, 'reference_difference': delta,
                      'below_min_count': stat['count'] < min_count,
                      'logically_impossible': impossible, 'logically_redundant': redundant}
            if journal:
                journal(record)       # EVERY inspected conjunction, including zero/low counts.
            if impossible or redundant or stat['frequency'] is None:
                continue
            scores = {'frequent': stat['frequency'],
                      'rare_observed_or_absent': -stat['frequency'],
                      'joint_concentration_diagnostic': stat['joint_excess']}
            if delta is not None:
                scores.update({'more_than_reference': delta, 'less_than_reference': -delta})
            for name, score in scores.items():
                item = (float(score), stat['count'], candidate)
                if len(heaps[name]) < top:
                    heapq.heappush(heaps[name], item)
                elif item > heaps[name][0]:
                    heapq.heapreplace(heaps[name], item)
        by_size[str(size)] = {'possible': possible, 'tested': count_size,
                              'complete': count_size == possible}
        if mode == 'exact' and tested >= budget:
            break
    selected = sorted({item[2] for heap in heaps.values() for item in heap})
    result_ids = {candidate: i for i, candidate in enumerate(selected)}
    results = []
    for candidate in selected:
        details = [space.describe(a) for a in candidate]
        row = {'atoms': list(candidate), 'relations': details,
               **space.statistics(candidate), 'object': 'indexed_OHLC_relation_fragment',
               'known_at': max(d['known_at'] for d in details),
               'continuity_context': continuity_context(space, candidate)}
        if reference:
            row['reference'] = reference.statistics(candidate)
            if row['frequency'] is not None and row['reference']['frequency'] is not None:
                row['reference_difference'] = row['frequency'] - row['reference']['frequency']
        results.append(row)
    complete = tested == total_possible
    report = {'version': VERSION, 'mode': mode, 'seed': seed, 'max_relations': max_relations,
              'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'numpy_version': np.__version__,
              'min_count': min_count, 'budget': budget, 'top': top,
              'min_count_role': 'annotation only; low and zero counts remain searchable and logged',
              'units': space.n, 'input_rows': space.corpus.input_rows,
              'point_pairs': len(space.known), 'eligible_atoms': len(atoms),
              'possible_combinations': total_possible, 'tested': tested,
              'support_passing': supported, 'irredundant': minimal, 'by_size': by_size,
              'complete_within_declared_size_and_support': complete,
              'status': 'DECLARED_SPACE_ENUMERATED' if complete else 'INCOMPLETE_SEARCH',
              'seconds': time.monotonic() - start,
              'candidate_log_complete_for_examined_space': journal is not None,
              'views_are_navigation_not_verdicts': True,
              'views_superseded_by': 'research/ordinal_events.py survey()',
              'view_caveats': {
                  'frequent': 'наполняется анатомическими постоянствами корпуса',
                  'rare_observed_or_absent': 'наполняется вырожденными равенствами',
                  'joint_concentration_diagnostic':
                      'наполняется синонимами; joint_excess максимален на них'},
              'views': {name: [{'motif': result_ids[item[2]], 'display_value': item[0]}
                               for item in sorted(heap, reverse=True)] for name, heap in heaps.items()},
              'view_filter': 'logically possible, nonredundant conjunctions with known cases; all others remain in candidates.jsonl',
              'atom_scope': 'all anatomically possible signs with at least one observed pair in either corpus',
              'aim': 'indexed relational fragments; neither earnings nor anchor causality',
              'claim': 'DESCRIPTIVE_ONLY; no profitability or predictive validation',
              'input': space.corpus.metadata,
              'reference': reference.corpus.metadata if reference else None}
    return results, report


def write_results(path, space, motifs, report, prepared=False, reference=None):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=prepared)
    with (path / 'density.jsonl').open('x', encoding='utf-8') as stream:
        for atom in range(len(space.masks)):
            row = {**space.describe(atom), **space.statistics((atom,)),
                   'anatomically_possible': anatomy_possible(space, atom)}
            if reference:
                row['reference'] = reference.statistics((atom,))
            stream.write(json.dumps(row,
                                    ensure_ascii=False) + '\n')
    # Exact memberships for returned motifs, plus original IDs behind each unit.
    data = {'unit_ids': np.array(space.corpus.unit_ids),
            'original_ids_json': np.array(json.dumps(space.corpus.members)),
            'all_observed_unit_count': np.array(space.n)}
    for i, motif in enumerate(motifs):
        truth, known = space.intersection(motif['atoms'])
        data[f'motif_{i}_support'] = np.array(indices(truth, space.n), dtype=np.int64)
        data[f'motif_{i}_known'] = np.array(indices(known, space.n), dtype=np.int64)
        motif['membership_key'] = f'motif_{i}'
    with (path / 'members.npz').open('xb') as stream:
        np.savez_compressed(stream, **data)
    for name, value in (('motifs.json', motifs), ('run.json', report)):
        with (path / name).open('x', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)


def inspect_candidate(path, sequence, examples=3):
    """Read ANY examined combination, even if absent from all navigation views."""
    path = Path(path)
    report = json.loads((path / 'run.json').read_text(encoding='utf-8'))
    record = None
    with (path / 'candidates.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            value = json.loads(line)
            if value['sequence'] == sequence:
                record = value
                break
    if record is None:
        raise ValueError('sequence was not examined in this run')
    descriptions = {}
    with (path / 'density.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            value = json.loads(line)
            if value['atom'] in record['atoms']:
                descriptions[value['atom']] = value
    corpus = Corpus.load(report['input']['input_path'])
    if corpus.metadata['input_sha256'] != report['input']['input_sha256']:
        raise ValueError('input bytes changed since the run')
    truth, known = np.ones(corpus.n, bool), np.ones(corpus.n, bool)
    for atom in record['atoms']:
        d = descriptions[atom]
        i, field_a = d['left']; j, field_b = d['right']
        a = corpus.ohlc[:, i - corpus.ordinals[0], FIELDS.index(field_a)]
        b = corpus.ohlc[:, j - corpus.ordinals[0], FIELDS.index(field_b)]
        valid = np.isfinite(a) & np.isfinite(b)
        known &= valid
        truth &= valid & {'<': np.less, '=': np.equal, '>': np.greater}[d['op']](a, b)
    out = {'record': record, 'relations': [descriptions[a] for a in record['atoms']],
           'matching_unit_ids': [corpus.unit_ids[i] for i in np.flatnonzero(truth)],
           'unknown_unit_ids': [corpus.unit_ids[i] for i in np.flatnonzero(~known)], 'examples': []}
    for label, selection in (('supports', truth), ('does_not_support', known & ~truth), ('unknown', ~known)):
        for i in np.flatnonzero(selection)[:examples]:
            candles = [[int(ordinal)] + [float(p) if np.isfinite(p) else None for p in bar]
                       for ordinal, bar in zip(corpus.ordinals, corpus.ohlc[i])]
            out['examples'].append({'group': label, 'unit': corpus.unit_ids[i],
                                    'original_ids': corpus.members[i], 'columns': ['minute', *FIELDS],
                                    'candles': candles})
    return out


def export_field(args):
    """Read existing canonical T0 and minute OHLC, preserving archive gaps as NaN."""
    import sys
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / 'src'))
    from g3riz.query import Field
    def stamp(value):
        return int(np.datetime64(value, 'ns').astype(np.int64))
    field = Field(root, args.instrument)
    table = field.passports(tf=args.tf, t0_start_ns=stamp(args.start), t0_stop_ns=stamp(args.stop))
    rows = [r for r in table.select(['riz_id', 't0_spine_pos', 't0_exit_side']).to_pylist()
            if r['t0_exit_side'] == args.side]
    if not rows or args.after < 1:
        raise ValueError('empty selection or invalid observation budget')
    ts = np.asarray(field.market.close_ts_utc_ns)
    ords = np.arange(args.after + 1, dtype=np.int64)
    # Slice unique anchors first, preserving every RIZ-to-window membership.
    anchors = sorted({int(r['t0_spine_pos']) for r in rows})
    positions = {a: k for k, a in enumerate(anchors)}
    values = np.full((len(anchors), len(ords), 4), np.nan)
    for k, a in enumerate(anchors):
        expected = ts[a] + ords * 60_000_000_000
        at = np.searchsorted(ts, expected)
        in_range = at < len(ts)
        good = in_range.copy()
        good[in_range] &= ts[at[in_range]] == expected[in_range]
        for p, name in enumerate(('open', 'high', 'low', 'close')):
            values[k, good, p] = np.asarray(getattr(field.market, name))[at[good]]
    row_positions = [positions[int(r['t0_spine_pos'])] for r in rows]
    ids = [str(r['riz_id']) for r in rows]
    units = (ids if args.unit == 'riz' else
             [f'{args.instrument}:{int(ts[int(r["t0_spine_pos"])])}' for r in rows])
    source_path = root / 'SOURCE_DATA.json'
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest() if source_path.exists() else None
    corpus = Corpus(values[row_positions], ords, ids, units, {
        'instrument': args.instrument, 'source': {'repository': str(root),
        'SOURCE_DATA_sha256': source_hash, 'start_inclusive': args.start, 'stop_exclusive': args.stop,
        'tf': args.tf, 'side': args.side, 'read_surface': 'Field.passports + market OHLC'},
        'anchor': 'canonical t0_spine_pos; fixed minute window including T0; no event truncation',
        'unit_definition': args.unit, 'after_minutes': args.after,
        'unknown_reason': 'minute timestamp missing or archive ended; keep original slot'})
    corpus.save(args.out)
    print(json.dumps({'out': str(Path(args.out).resolve()), 'rows': len(rows), 'units': corpus.n}))


def selftest():
    """Meaningful counterexamples; temporary files only, no historical/field reads."""
    meta = {'instrument': 'SYNTHETIC', 'source': 'artificial fixtures, not market data',
            'anchor': 'synthetic T0', 'unit_definition': 'synthetic case'}
    def make(x, units=None):
        return Corpus(x, np.arange(x.shape[1]), [f'case-{i}' for i in range(len(x))],
                      units or [f'unit-{i}' for i in range(len(x))], meta)
    # The exact disagreement the old 064 representation did not distinguish.
    x = np.tile([100., 110., 90., 102.], (2, 51, 1))
    x[:, 18, 1] = 105
    x[:, 20] = [103, 110, 100, 104]
    x[1, 20, 3] = 106
    s = Space(make(x))
    assert len(s.known) == 20706
    assert s.compare(20, 'C', 18, 'H')[0].tolist() == [-1, 1]
    assert s.compare(20, 'H', 18, 'H')[0].tolist() == [1, 1]
    assert s.compare(18, 'H', 20, 'C')[0].tolist() == [1, -1]
    # Every field against every field, including self/symmetry, checks raw prices.
    for i, j in ((0, 50), (18, 20), (20, 18), (20, 20)):
        for a, b in itertools.product(FIELDS, repeat=2):
            actual, known = s.compare(i, a, j, b)
            expected = np.sign(x[:, i, FIELDS.index(a)] - x[:, j, FIELDS.index(b)])
            assert known.all() and np.array_equal(actual, expected)
    # Strictly increasing price deformation preserves ALL order relations.
    warped = Space(make(np.exp(x / 100)))
    assert s.masks == warped.masks and s.known == warped.known
    # Equal marginals, different joint density: halves carry both A and B,
    # reference quarters carry all four combinations. No motif passed to miner.
    # Only three distant candles are observed in this synthetic corpus. Missing
    # slots preserve their real ordinal addresses and do not become features.
    z = np.full((8, 51, 4), np.nan)
    z[:, [5, 18, 20], :] = [0., 10., -10., 0.]
    z[:, 18, 3] = [-1] * 4 + [1] * 4
    z[:, 20, 3] = [-2] * 4 + [2] * 4
    ref = z.copy(); ref[:, 20, 3] = [-2, -2, 2, 2] * 2
    joint, baseline = Space(make(z)), Space(make(ref))
    a = joint.atom(18, 'C', '>', 5, 'C')
    b = joint.atom(20, 'C', '>', 18, 'C')
    assert joint.statistics([a])['frequency'] == baseline.statistics([a])['frequency'] == .5
    assert joint.statistics([b])['frequency'] == baseline.statistics([b])['frequency'] == .5
    assert joint.statistics([a, b])['frequency'] == .5
    assert baseline.statistics([a, b])['frequency'] == .25
    examined = []
    motifs, report = discover(joint, max_relations=2, min_count=2, budget=100000,
                              mode='exact', top=100000, reference=baseline, journal=examined.append)
    target = next(m for m in motifs if set(m['atoms']) == {a, b})
    assert report['complete_within_declared_size_and_support']
    assert target['reference_difference'] == .25
    assert len(examined) == report['tested']
    negative_b = joint.atom(20, 'C', '<', 18, 'C')
    absence = next(m for m in motifs if set(m['atoms']) == {a, negative_b})
    assert absence['count'] == 0 and absence['reference']['count'] == 2
    assert absence['reference_difference'] == -.25
    cycle = [joint.atom(5, 'C', '<', 18, 'C'), joint.atom(18, 'C', '<', 20, 'C'),
             joint.atom(20, 'C', '<', 5, 'C')]
    assert logically_impossible(joint, cycle)
    assert continuation(joint, target, 18)['prefix_true']['frequency'] == 1
    assert continuation(baseline, target, 18)['prefix_true']['frequency'] == .5
    link = [s.atom(0, 'H', '>', 0, 'C'), s.atom(0, 'H', '>', 1, 'O')]
    assert not logically_redundant(s, link)
    assert logically_redundant(s, link, gapless=True)
    adjacent = x.copy(); adjacent[0, 1, 0] = adjacent[0, 0, 3]
    assert continuity_context(Space(make(adjacent)), link)['fraction'] == .5
    # At +18, the future half is pending and cannot select the input examples.
    before = matching(baseline, target, cursor=18)
    altered = ref.copy(); altered[:, 19:] = np.nan
    after = matching(Space(make(altered)), target, cursor=18)
    assert before == after and len(before['pending']) == 1
    # Missing candles reduce denominators, never become false comparisons.
    missing = z.copy(); missing[0, 20] = np.nan
    ms = Space(make(missing)).statistics([a, b])
    assert ms['known'] == 7 and ms['unknown'] == 1 and ms['count'] == 4
    duplicated = make(np.concatenate([z, z[:1]]), [f'u{i}' for i in range(8)] + ['u0'])
    assert duplicated.n == 8 and duplicated.input_rows == 9 and len(duplicated.members[0]) == 2
    bad = z.copy(); bad[1, 18, 3] = 2
    try:
        make(bad, ['same', 'same'] + [f'u{i}' for i in range(6)])
        raise AssertionError('inconsistent duplicate accepted')
    except ValueError:
        pass
    _, limited = discover(joint, max_relations=3, min_count=2, budget=7,
                           mode='sampled', seed=9, top=3)
    assert limited['status'] == 'INCOMPLETE_SEARCH' and limited['tested'] == 7
    # Corpus/output roundtrip preserves aliases, formulas, support and unknowns.
    import tempfile
    with tempfile.TemporaryDirectory(prefix='stencil-selftest-', dir=Path(__file__).parent) as directory:
        folder = Path(directory)
        assert folder.resolve().is_relative_to(Path(__file__).resolve().parent)
        duplicated.save(folder / 'input.npz')
        loaded = Corpus.load(folder / 'input.npz')
        assert loaded.members == duplicated.members and loaded.input_rows == 9
        write_results(folder / 'result', joint, [dict(target)], report)
        saved = json.loads((folder / 'result/motifs.json').read_text(encoding='utf-8'))
        assert saved[0]['atoms'] == target['atoms']
        with np.load(folder / 'result/members.npz', allow_pickle=False) as archive:
            assert archive['motif_0_support'].tolist() == [4, 5, 6, 7]
        try:
            loaded.save(folder / 'input.npz')
            raise AssertionError('silently overwrote input')
        except FileExistsError:
            pass
        # Exercise the real CLI, full candidate journal and access to a candidate
        # outside the navigation previews, rather than just helper functions.
        import subprocess
        import sys
        completed = subprocess.run([
            sys.executable, '-B', str(Path(__file__).resolve()), 'discover',
            '--input', str(folder / 'input.npz'), '--mode', 'sampled',
            '--max-relations', '2', '--min-count', '20', '--budget', '30',
            '--top', '1', '--seed', '9', '--out', str(folder / 'cli')],
            capture_output=True, text=True, check=True)
        assert json.loads(completed.stdout)['tested'] == 30
        cli_report = json.loads((folder / 'cli/run.json').read_text(encoding='utf-8'))
        ledger = [json.loads(line) for line in
                  (folder / 'cli/candidates.jsonl').read_text(encoding='utf-8').splitlines()]
        previews = json.loads((folder / 'cli/motifs.json').read_text(encoding='utf-8'))
        visible = {tuple(row['atoms']) for row in previews}
        hidden_from_preview = next(row for row in ledger if tuple(row['atoms']) not in visible)
        inspected = inspect_candidate(folder / 'cli', hidden_from_preview['sequence'], examples=1)
        assert inspected['record'] == hidden_from_preview
        assert len(inspected['matching_unit_ids']) == hidden_from_preview['count']
        assert cli_report['candidate_log_complete_for_examined_space'] and len(ledger) == 30
        assert all(row['below_min_count'] for row in ledger)  # threshold did not hide them
    print(json.dumps({'status': 'PASS', 'point_pairs_for_51_candles': len(s.known),
                      'joint_motif_discovered_without_query': target['relations'],
                      'checked': ['body versus wick', 'all OHLC pairs and symmetry',
                      'monotone price invariance', 'equal marginals different joint density',
                      'automatic exhaustive discovery', 'prefix excludes future',
                      'absent but possible combination', 'strict-order contradiction',
                      'conditional adjacent close/open redundancy', 'continuation separated from description',
                      'unknown denominators', 'unit deduplication', 'explicit search budget',
                      'NPZ/JSON output roundtrip and overwrite protection',
                      'real CLI candidate ledger and inspection beyond previews'],
                      'market_research': 'NOT_RUN'}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('selftest')
    export = sub.add_parser('export-field')
    export.add_argument('--instrument', choices=['ES', 'NQ', 'YM'], required=True)
    export.add_argument('--start', required=True)
    export.add_argument('--stop', required=True)
    export.add_argument('--side', choices=['north', 'south'], required=True)
    export.add_argument('--unit', choices=['t0', 'riz'], required=True)
    export.add_argument('--tf', type=int)
    export.add_argument('--after', type=int, default=50)
    export.add_argument('--out', required=True)
    run = sub.add_parser('discover')
    run.add_argument('--input', required=True)
    run.add_argument('--reference')
    run.add_argument('--mode', choices=['exact', 'sampled'], required=True)
    run.add_argument('--max-relations', type=int, required=True)
    run.add_argument('--min-count', type=int, required=True)
    run.add_argument('--budget', type=int, required=True)
    run.add_argument('--seed', type=int, default=0)
    run.add_argument('--top', type=int, default=40)
    run.add_argument('--out', required=True)
    inspect = sub.add_parser('inspect')
    inspect.add_argument('--result', required=True)
    inspect.add_argument('--sequence', type=int, required=True)
    inspect.add_argument('--examples', type=int, default=3)
    args = parser.parse_args()
    if args.command == 'selftest':
        selftest()
    elif args.command == 'export-field':
        export_field(args)
    elif args.command == 'inspect':
        print(json.dumps(inspect_candidate(args.result, args.sequence, args.examples), ensure_ascii=False))
    else:
        if Path(args.out).exists():
            raise FileExistsError('use a new result directory')
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=False)
        with (out / 'started.json').open('x', encoding='utf-8') as stream:
            json.dump({'state': 'RUNNING_OR_INTERRUPTED_UNTIL_run.json_EXISTS',
                       'parameters': vars(args), 'version': VERSION}, stream, indent=2)
        space = Space(Corpus.load(args.input))
        reference = Space(Corpus.load(args.reference)) if args.reference else None
        with (out / 'candidates.jsonl').open('x', encoding='utf-8') as stream:
            def record(row):
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            def progress(row):
                stream.flush()
                print(json.dumps(row), flush=True)
            motifs, report = discover(space, max_relations=args.max_relations,
                                     min_count=args.min_count, budget=args.budget,
                                     mode=args.mode, seed=args.seed, top=args.top,
                                     reference=reference, progress=progress, journal=record)
        write_results(args.out, space, motifs, report, prepared=True, reference=reference)
        print(json.dumps({'out': str(Path(args.out).resolve()), 'status': report['status'],
                          'tested': report['tested'], 'motifs': len(motifs)}))


if __name__ == '__main__':
    main()
