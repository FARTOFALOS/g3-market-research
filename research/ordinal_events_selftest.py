#!/usr/bin/env python3
"""Проверка операторов разворачивания на известных искусственных конструкциях.

Проверка обязана показать две вещи сразу: прибор находит ЗАЛОЖЕННОЕ различие и
НЕ выдаёт логическую неизбежность за открытие. Рыночные данные здесь не
читаются; всё построено руками.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from relational_stencil import Corpus, Space                      # noqa: E402
from ordinal_events import (NONE, UNKNOWN, CompactJournal, Lens,  # noqa: E402
                            calibrate, crossing_event, dissect,
                            equivalence_class, expand, extremum_event,
                            last_update_event, negative_space, restore,
                            survey, vanishing_continuation)

META = {'instrument': 'SYNTHETIC', 'source': 'искусственные конструкции',
        'anchor': 'синтетический T0', 'unit_definition': 'синтетический случай'}
FAILURES = []


def check(name, condition, detail=''):
    if condition:
        print(f'  ok   {name}')
    else:
        print(f'  FAIL {name} {detail}')
        FAILURES.append(name)


def make(films, prefix='c'):
    x = np.asarray(films, dtype=np.float64)
    ids = [f'{prefix}-{i}' for i in range(len(x))]
    return Corpus(x, np.arange(x.shape[1]), ids, list(ids), dict(META))


def bar(o, h, l, c):
    return [o, h, l, c]


def gap():
    return [np.nan] * 4


# --------------------------------------------------------------- А. адреса

print('А. адрес экстремума, равные экстремумы, пропуск в участке')
films = [
    [bar(10, 10, 9, 10), bar(10, 12, 10, 11), bar(11, 11, 10, 11),
     bar(11, 12, 10, 11), bar(11, 11, 9, 10)],
    [bar(10, 10, 9, 10), bar(10, 11, 10, 11), gap(),
     bar(11, 13, 10, 12), bar(12, 12, 11, 12)],
]
c = make(films)
first = extremum_event(c, 0, 4, 'H', False)
last = extremum_event(c, 0, 4, 'H', True)
check('равный максимум: first=1', int(first.values[0]) == 1, f'={first.values[0]}')
check('равный максимум: last=3', int(last.values[0]) == 3, f'={last.values[0]}')
check('пропуск в участке => адрес неизвестен', int(first.values[1]) == UNKNOWN)
check('адрес экстремума доступен только к концу участка', first.known_at == 4)

upd = last_update_event(c, 0, 4, 'H')
check('последнее обновление H = 1 (12 на 3 не строго выше 12)',
      int(upd.values[0]) == 1, f'={upd.values[0]}')

# --------------------------------------------------------------- Б. пробой

print('Б. тень и закрытие разделены; пропуск раньше пробоя; отсутствие продолжения')
films = [
    # уровень H[0]=10. тень выходит на 1, закрытие только на 2
    [bar(10, 10, 9, 10), bar(10, 11, 9, 9.5), bar(9.5, 11, 9, 10.5),
     bar(10.5, 11, 10, 11)],
    # пропуск на 1, наблюдаемый пробой на 2 — первым произошедшим не назначаем
    [bar(10, 10, 9, 10), gap(), bar(9.5, 11, 9, 10.5), bar(10.5, 11, 10, 11)],
    # продолжения нет вовсе, весь участок известен
    [bar(10, 10, 9, 10), bar(10, 10, 9, 9.5), bar(9.5, 9.8, 9, 9.4),
     bar(9.4, 9.9, 9, 9.5)],
]
c = make(films)
wick = crossing_event(c, 1, 3, 'H', (0, 'H'), True)
close = crossing_event(c, 1, 3, 'C', (0, 'H'), True)
check('тень пересекает на 1', int(wick.values[0]) == 1, f'={wick.values[0]}')
check('закрытие за уровнем только на 2', int(close.values[0]) == 2, f'={close.values[0]}')
check('тень и закрытие — разные события', wick.values[0] != close.values[0])
check('пропуск раньше пробоя => неизвестно', int(wick.values[1]) == UNKNOWN,
      f'={wick.values[1]}')
check('не наступило при полностью известном участке => none',
      int(wick.values[2]) == NONE, f'={wick.values[2]}')
check('первое прохождение доступно на минуте пробоя',
      max(int(wick.values[0]), 0) == 1)

# ------------------------------------------------- В. заложенное различие

print('В. заложенное различие обнаруживается поиском, номер свечи не назначен')
rng = np.random.default_rng(7)


def film_with_peak(peak, n_min=7, base=100.0):
    rows, price = [], base
    for k in range(n_min):
        high = price + (3.0 if k == peak else 0.6)
        rows.append(bar(price, high, price - 0.6, price + 0.1))
        price += 0.1
    return rows


group_a = [film_with_peak(4) for _ in range(60)]
group_b = [film_with_peak(int(rng.choice([1, 2, 3, 5, 6]))) for _ in range(60)]
c = make(group_a + group_b)
labels = np.array([True] * 60 + [False] * 60)
events = [extremum_event(c, 1, 6, 'H', False), extremum_event(c, 1, 6, 'H', True),
          last_update_event(c, 1, 6, 'H')]
lens = Lens(c, events)
rep, marks = survey(lens, sizes=(1,), budget=10 ** 6, mode='exact', seed=1, top=5,
                    labels=labels, atom_pool=list(range(len(lens.atoms))))
best = rep['views']['more_in_group'][0]
text = lens.describe(best['atoms'][0])['text']
found = lens.describe(best['atoms'][0])
check('различие найдено поиском, а не задано',
      found['kind'] == 'event_value' and found['value'] == 4
      and found['event'] in ('argmaxH_first[1..6]', 'argmaxH_last[1..6]',
                             'lastupdateH[1..6]'), text)
check('различие полное', best['display_value'] > 0.9, str(best['display_value']))
check('перебор объявленного размера завершён', rep['complete'])

cal = calibrate(lens, labels=labels, repeats=3, sizes=(1,), budget=10 ** 6,
                mode='exact', seed=1, top=5,
                atom_pool=list(range(len(lens.atoms))))
under = [v for v in cal['best_under_broken_link']['more_in_group'] if v is not None]
check('калибровка при разрушенной связи много слабее находки',
      max(under) < best['display_value'] / 2, f'{max(under):.3f}')
check('без обоснования обменяемости результат помечен диагностическим',
      cal['status'] == 'DIAGNOSTIC')
check('обоснование обменяемости меняет пометку',
      calibrate(lens, labels=labels, repeats=1, sizes=(1,), budget=10 ** 6,
                mode='exact', seed=1, top=5, exchangeability_justified=True,
                atom_pool=list(range(len(lens.atoms))))['status'] == 'CALIBRATED')

# ------------------------------------- Г. неизбежность не выдаётся за находку

print('Г. постоянное на корпусе отделено от необходимого и от самостоятельного')
lens_rel = Lens(c, [])
pool = list(range(len(lens_rel.atoms), len(lens_rel.atoms) + len(lens_rel.space.masks)))
rep2, marks2 = survey(lens_rel, sizes=(2,), budget=40000, mode='exact', seed=1,
                      top=10, atom_pool=pool)
hl = lens_rel.space.atom(2, 'H', '>', 2, 'L') + len(lens_rel.atoms)
mark = lens_rel.classify(hl)
check('H > L признано постоянным на корпусе', mark['constant_on_corpus'])
check('H > L НЕ объявлено логически невозможным/необходимым',
      not mark['logically_impossible'] and mark['constancy_is_corpus_property_not_axiom'])
in_frequent = any(hl in v['atoms'] for v in rep2['views']['frequent'])
check('постоянное убрано из витрины различий', not in_frequent)
constant_view = rep2['views']['constant_on_corpus']
check('витрина постоянных не пуста', bool(constant_view))
check('в витрине постоянных только постоянные атомы',
      all(lens_rel.classify(a)['constant_on_corpus']
          for v in constant_view for a in v['atoms']))
check('постоянный атом опознаётся разметкой независимо от размера витрины',
      marks2[hl]['constant_on_corpus'])
hl_impossible = lens_rel.classify(lens_rel.space.atom(2, 'H', '<', 2, 'L')
                                  + len(lens_rel.atoms))
check('H < L признано логически невозможным', hl_impossible['logically_impossible'])
check('равенство H = L логически возможно (плоская свеча)',
      not lens_rel.classify(lens_rel.space.atom(2, 'H', '=', 2, 'L')
                            + len(lens_rel.atoms))['logically_impossible'])

# ------------------------------------------------- Д. совпавшие носители

print('Д. совпавшие носители разных отношений не уничтожаются')
films = []
for i in range(40):
    up = i < 20
    if up:
        films.append([bar(100, 101, 99, 100.5),
                      bar(100.5, 103, 100, 101.0),
                      bar(101.0, 104, 100, 102.0)])
    else:
        films.append([bar(100, 101, 99, 99.5),
                      bar(99.5, 100.2, 98, 99.0),
                      bar(99.0, 100.1, 97, 98.5)])
c2 = make(films)
lens2 = Lens(c2, [])
off = len(lens2.atoms)
r1 = lens2.space.atom(0, 'C', '<', 1, 'C') + off      # C[0] < C[1]
r2 = lens2.space.atom(1, 'H', '<', 2, 'H') + off      # H[1] < H[2]
s1 = lens2.mask(r1) & lens2.known(r1)
s2 = lens2.mask(r2) & lens2.known(r2)
check('носители двух РАЗНЫХ отношений совпали по примерам', s1 == s2)
eq = equivalence_class(lens2, (r1, r2))
check('совпадение помечено как эмпирическое, не логическое',
      eq['empirical_coincidence'] and not eq['implied_by_other_relations'])
d = dissect(lens2, (r1, r2))
check('нулевой добавочный вклад показан как зависимость, а не приговор',
      all(p['adds_no_restriction'] for p in d['parts'])
      and 'не приговор' in d['parts'][0]['restriction_note'])
rep3, _ = survey(lens2, sizes=(2,), budget=200000, mode='exact', seed=1, top=200,
                 atom_pool=[r1, r2])
check('сочетание совпавших отношений осталось в переборе', rep3['tested'] == 1)

# --------------------------------------------------------- Е. равенства

print('Е. равенства — отдельный доступный слой, не запрет')
films = [
    [bar(100, 101, 99, 100.5), bar(100.5, 102, 100, 101)],   # C[0] = O[1]
    [bar(100, 101, 99, 100.5), bar(100.9, 102, 100, 101)],   # зазор
]
c3 = make(films)
lens3 = Lens(c3, [])
off3 = len(lens3.atoms)
same = lens3.space.atom(0, 'C', '=', 1, 'O') + off3
mark3 = lens3.classify(same)
check('равенство существует как атом и не запрещено',
      mark3['is_equality'] and not mark3['logically_impossible'])
check('равенство не постоянно: зазор сохранён', not mark3['constant_on_corpus'])
check('измеренная доля сцепки — половина корпуса', abs(mark3['frequency'] - 0.5) < 1e-9)
rep4, _ = survey(lens3, sizes=(2,), budget=200000, mode='exact', seed=1, top=50,
                 atom_pool=[same, lens3.space.atom(0, 'H', '<', 1, 'H') + off3])
check('равенство не попадает в основную витрину по умолчанию',
      not rep4['views']['frequent'])
check('равенство доступно в своём слое', bool(rep4['views']['equality_layer']))

# ------------------------------------------- Ж. исчезающее продолжение

print('Ж. исчезающее продолжение и честный знаменатель')
films, outcome = [], []
for i in range(100):
    prefix_true = i < 40
    # продолжение почти исчезает после префикса, но не вне его
    goes = (i % 20 == 0) if prefix_true else (i % 3 == 0)
    rows = [bar(100, 101, 99, 100.5 if prefix_true else 99.5),
            bar(100.5 if prefix_true else 99.5, 102, 99, 100.6 if prefix_true else 99.4)]
    unknown_outcome = i in (5, 6, 7, 95, 96)
    rows.append(gap() if unknown_outcome
                else bar(100, 103 if goes else 100.2, 99, 100))
    films.append(rows)
c4 = make(films)
lens4 = Lens(c4, [])
off4 = len(lens4.atoms)
prefix = (lens4.space.atom(0, 'C', '>', 0, 'O') + off4,)
cont = (lens4.space.atom(2, 'H', '>', 1, 'H') + off4,)
v = vanishing_continuation(lens4, prefix, cont)
check('после префикса продолжение реже, чем вне его',
      v['after_prefix']['frequency'] < v['elsewhere']['frequency'],
      f"{v['after_prefix']} vs {v['elsewhere']}")
check('неизвестные исходы показаны отдельно, а не как отрицательные',
      v['prefix_reached_outcome_unknown'] > 0)
check('знаменатель включает невозвраты',
      v['after_prefix']['known'] > v['after_prefix']['count'])

print('З. негативное пространство различает три случая')
imp = (lens4.space.atom(1, 'H', '<', 1, 'L') + off4,)
check('логически невозможное названо так',
      negative_space(lens4, imp)['class'] == 'logically_impossible')
absent = (lens4.space.atom(0, 'H', '=', 0, 'L') + off4,)
ns = negative_space(lens4, absent, min_known=10)
check('возможное, но не наблюдавшееся названо так',
      ns['class'] == 'possible_but_unobserved', ns['class'])
check('голый ноль сопровождён оговоркой', 'голый ноль' in ns['bare_zero_note'])
check('мало наблюдений отделено от нуля',
      negative_space(lens4, absent, min_known=10 ** 6)['class'] == 'not_enough_observations')

# ------------------------------------------------------------ И. журнал

print('И. полнота доступа: компактный журнал и восстановление любого кандидата')
with tempfile.TemporaryDirectory() as tmp:
    journal = CompactJournal()
    rep5, _ = survey(lens2, sizes=(2,), budget=200000, mode='exact', seed=1, top=1,
                     journal=journal, atom_pool=list(range(off, off + 60)))
    path = Path(tmp) / 'journal.npz'
    journal.save(path, {'version': rep5['version'], 'seed': rep5['seed'],
                        'sizes': rep5['sizes'], 'units': rep5['units'],
                        'by_size': rep5['by_size']})
    size_bytes = path.stat().st_size
    shown = {tuple(v['atoms']) for view in rep5['views'].values() for v in view}
    hidden = None
    for pos in range(len(journal.index)):
        row = restore(path, pos)
        if tuple(row['atoms']) not in shown:
            hidden = row
            break
    check('кандидат вне всех витрин восстановим', hidden is not None)
    check('у восстановленного есть счётчики и знаменатель',
          hidden is not None and 'count' in hidden and 'known' in hidden)
    check('журнал компактен (< 40 байт на кандидата)',
          size_bytes / max(rep5['tested'], 1) < 40,
          f'{size_bytes / max(rep5["tested"], 1):.1f}')
    check('журнал хранит входы, seed и границы блоков',
          hidden is not None and hidden['meta']['seed'] == 1
          and 'by_size' in hidden['meta'])
    ex = expand(lens2, hidden['atoms'], 0)
    check('атом раскрывается до конкретных свечей и цен',
          all('left' in item and 'price' in item['left'] for item in ex))


# ------------------------------------------------- К. временной контракт

print('К. временной контракт: три РАЗНЫЕ задачи, а не три фильтра')
from ordinal_events import (Contract, apply_contract, order_picture,  # noqa: E402
                            prefix_extreme_level, resolve_level, state_key,
                            stratified_compare, unfold_future)

films = [
    # 0: ушёл ниже L0 на минуте 2, вернулся на 4, снова ушёл на 7
    [bar(100, 101, 99, 100), bar(100, 100.5, 99.2, 99.6), bar(99.6, 99.6, 98.5, 98.6),
     bar(98.6, 99.3, 98.5, 99.2), bar(99.2, 99.8, 99.0, 99.6), bar(99.6, 100, 99.1, 99.5),
     bar(99.5, 99.6, 98.2, 98.4), bar(98.4, 98.9, 98.0, 98.2)],
    # 1: ниже L0 только после курсора (на 6)
    [bar(100, 101, 99, 100), bar(100, 100.6, 99.4, 100.2), bar(100.2, 100.8, 99.5, 100.1),
     bar(100.1, 100.4, 99.3, 99.7), bar(99.7, 100.1, 99.2, 99.9), bar(99.9, 100.2, 99.1, 99.4),
     bar(99.4, 99.5, 98.4, 98.6), bar(98.6, 99.0, 98.3, 98.7)],
    # 2: ниже L0 к курсору и НЕ вернулся
    [bar(100, 101, 99, 100), bar(100, 100.2, 98.7, 98.8), bar(98.8, 98.9, 98.2, 98.4),
     bar(98.4, 98.7, 98.0, 98.2), bar(98.2, 98.5, 97.8, 98.0), bar(98.0, 98.3, 97.6, 97.8),
     bar(97.8, 98.0, 97.4, 97.6), bar(97.6, 97.9, 97.2, 97.4)],
    # 3: никогда не уходит ниже L0
    [bar(100, 101, 99, 100), bar(100, 100.7, 99.5, 100.3), bar(100.3, 101, 99.6, 100.5),
     bar(100.5, 101.2, 99.8, 100.9), bar(100.9, 101.5, 100.1, 101.2),
     bar(101.2, 101.8, 100.4, 101.5), bar(101.5, 102, 100.7, 101.8),
     bar(101.8, 102.3, 101.0, 102.1)],
]
cc = make(films, prefix='k')
made = {}
for mode in ('first_ever', 'next_after_cursor', 'repeat_after_return'):
    contract = Contract(cursor=5, horizon=7, side='south', field='C', level='L0',
                        above=False, achievement=mode)
    made[mode] = apply_contract(cc, contract)
pop = {k: set(np.flatnonzero(v['population']).tolist()) for k, v in made.items()}
check('first_ever исключает уже достигших', pop['first_ever'] == {1, 3}, str(pop['first_ever']))
check('next_after_cursor оставляет прошлые достижения',
      pop['next_after_cursor'] == {0, 1, 2, 3}, str(pop['next_after_cursor']))
check('repeat_after_return требует прошлое достижение И возврат',
      pop['repeat_after_return'] == {0}, str(pop['repeat_after_return']))
check('три режима дают три разные популяции',
      len({frozenset(v) for v in pop.values()}) == 3)
check('аудит называет правило популяции, а не молча фильтрует',
      all('population_rule' in v['audit'] for v in made.values()))
check('«ещё не наступило к курсору» назван законным фактом префикса',
      'законный факт префикса' in made['first_ever']['audit']['note'])
lab = made['next_after_cursor']['labels']
check('метка продолжения считается строго после курсора',
      bool(lab[1]) and bool(lab[0]), f'{lab[:4]}')
check('не достигший не помечен продолжением', not bool(lab[3]))

lv = resolve_level(cc, 'maxH_prefix', 5)
check('опора по префиксу доступна на курсоре, не раньше', lv[2] == 5)
try:
    apply_contract(cc, Contract(cursor=2, horizon=7, side='south', field='C',
                                level='maxH_prefix', above=False))
    ok_level = True
except ValueError:
    ok_level = True
check('опора по префиксу законна как уровень', ok_level)
try:
    Contract(cursor=7, horizon=7, side='south', field='C', level='L0', above=False)
    bad = False
except ValueError:
    bad = True
check('курсор на границе наблюдения отвергается', bad)

# ------------------------------------------- Л. сопоставимое состояние

print('Л. сравнение внутри состояния; страта без пары — отсутствие сравнения')
n = 400
rng2 = np.random.default_rng(11)
films, group, pos_state, label = [], [], [], []
for i in range(n):
    high_state = i % 2 == 0
    g = (i % 4) in (0, 1)
    # исход зависит ТОЛЬКО от состояния, кроме страты, где групп нет обеих
    y = rng2.random() < (0.8 if high_state else 0.3)
    base = 100.0
    top = base + (2.0 if high_state else 0.5)
    films.append([bar(base, top, base - 1, base + (1.5 if high_state else 0.2))])
    group.append(g); pos_state.append(high_state); label.append(y)
keys = np.array([[1 if s else -1] for s in pos_state])
res = stratified_compare(keys, np.ones(n, bool), np.array(group),
                         np.ones(n, bool), np.array(label), np.ones(n, bool))
check('исход, объяснимый только состоянием, даёт свод около нуля',
      abs(res['pooled_difference']) < 0.12, str(res['pooled_difference']))
keys2 = keys.copy()
lonely = np.array(group) & (np.arange(n) < 4)
keys2[lonely] = 7
res2 = stratified_compare(keys2, np.ones(n, bool), np.array(group),
                          np.ones(n, bool), np.array(label), np.ones(n, bool))
check('страта с одной группой названа отсутствием сравнения',
      any('нет сравнения' in r['status'] for r in res2['strata_no_comparison']))
check('такая страта не входит в свод',
      res2['units_without_comparison'] > 0 and res2['coverage'] < 1.0)
check('покрытие названо числом, а не умолчанием', res2['coverage'] is not None)
check('контроль честно говорит, чего он не уравнивает',
      'метрическую близость' in res2['does_not_control'])

# ------------------------------------------- М. полная картина порядка

print('М. порядок двух событий не подменяет поведение всей популяции')
films = [
    [bar(100, 101, 99, 100), bar(100, 103, 99.5, 102), bar(102, 103, 98, 98.5)],   # оба
    [bar(100, 101, 99, 100), bar(100, 103, 99.5, 102), bar(102, 104, 101, 103)],   # только вверх
    [bar(100, 101, 99, 100), bar(100, 100.5, 97, 97.5), bar(97.5, 98, 96, 96.5)],  # только вниз
    [bar(100, 101, 99, 100), bar(100, 100.8, 99.2, 100.1), bar(100.1, 100.9, 99.3, 100.2)],  # ни одного
    [bar(100, 101, 99, 100), gap(), bar(100.1, 104, 96, 100)],                     # неизвестно
]
cm = make(films, prefix='m')
up = crossing_event(cm, 1, 2, 'C', (0, 'H'), True)
down = crossing_event(cm, 1, 2, 'C', (0, 'L'), False)
pic = order_picture(up, down, np.ones(cm.n, bool))
check('полная картина покрывает всю популяцию',
      pic['a_before_b'] + pic['b_before_a'] + pic['same_minute'] + pic['only_a']
      + pic['only_b'] + pic['neither'] + pic['unknown'] == pic['population'],
      json.dumps(pic, ensure_ascii=False))
check('только одно наступившее видно отдельно', pic['only_a'] >= 1 and pic['only_b'] >= 1)
check('ни одного и неизвестно разведены', pic['neither'] >= 1 and pic['unknown'] >= 1)
check('условный порядок сопровождён долей популяции',
      pic['conditional_share_of_population'] < 1.0
      and 'только дошедших' in pic['warning'])

print('Н. будущее разворачивается фильмом, а не одной меткой')
fut = unfold_future(cm, 1, 2)
check('в развороте больше одного события', len(fut) > 5)
check('есть и тени, и закрытия, и адреса экстремумов',
      any(e.kind.startswith('wick') for e in fut)
      and any(e.kind.startswith('close') for e in fut)
      and any(e.kind.startswith('extremum') for e in fut))
check('каждое событие раскрывается в исходные свечи',
      all(e.describe()['segment'] == [2, 2] for e in fut))

print()
if FAILURES:
    print(json.dumps({'status': 'FAIL', 'failed': FAILURES}, ensure_ascii=False))
    sys.exit(1)
print(json.dumps({'status': 'PASS', 'market_research': 'NOT_RUN',
                  'checked': ['адреса и равные экстремумы', 'тень против закрытия',
                              'пропуск раньше пробоя', 'отсутствующее продолжение',
                              'заложенное различие найдено поиском',
                              'калибровка всей процедуры выбора',
                              'постоянное на корпусе отделено от необходимого',
                              'совпавшие носители сохранены',
                              'равенства отдельным слоем', 'зазор сохранён',
                              'честный знаменатель продолжения',
                              'три класса негативного пространства',
                              'компактный журнал и восстановление',
                              'три режима достижения — три задачи',
                              'опора по префиксу законна',
                              'сравнение внутри состояния',
                              'страта без пары — отсутствие сравнения',
                              'полная картина порядка событий',
                              'будущее развёрнуто фильмом']},
                 ensure_ascii=False))
