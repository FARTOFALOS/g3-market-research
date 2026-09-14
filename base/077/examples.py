"""077: open the exact collisions into candles and write EXAMPLES.md.

Selection rule, declared before looking: every collision in which two prefixes
carry the same multiset of per-minute readings but a different order (there are
six, so all are shown), then the first three NQ tf54 collisions of the other kind
in chronological order. No Y exists in this pass, so nothing in the choice can be
outcome-driven.

Run: python -B base/077/examples.py
"""
import collections
from pathlib import Path
from fibre import load, dedup, fibre_key, label, CELLS, NAME

OUT = Path(__file__).resolve().parent / 'EXAMPLES.md'
lines = []


def w(s=''):
    lines.append(s)


def show(s):
    w('%s  %s  `%s`' % (s['cell'], s['p_utc'], s['riz_id']))
    w('зона %.2f..%.2f · сторона %s · тип %s · dur %d · линейка %.2f · position %+.2f · bulk %.2f'
      % (s['zone'][0], s['zone'][1], s['side'], s['kind'], s['dur'], s['u60'], s['pos'], s['bulk']))
    w('H4: серий %d/%d, касаний выходной %d/%d, дальней %d/%d, закрытий внутри %d/%d'
      % (s['n_runs'], s['dur'], s['e'], s['dur'], s['f'], s['dur'], s['l0'], s['dur']))
    w()
    w('| # | UTC | O | H | L | C | касание | закрытие |')
    w('|---:|---|---:|---:|---:|---:|---|---|')
    for i, (r, o) in enumerate(zip(s['seq'], s['raw_ohlc'])):
        w('| %d | %s | %.2f | %.2f | %.2f | %.2f | %s%s | %s |'
          % (i, s['utc_minutes'][i][11:], o[0], o[1], o[2], o[3],
             'выходная' if r[0] else '', ' дальняя' if r[1] else '', NAME[r[2]]))
    w()


pure, other = [], []
for inst, tf, _ in CELLS:
    _, S = load(inst, tf)
    G = collections.defaultdict(list)
    for s in S:
        G[fibre_key(s)].append(s)
    for k, v in G.items():
        v = dedup(v)
        for i, a in enumerate(v):
            for b in v[i + 1:]:
                if a['seq'] != b['seq']:
                    (pure if label(a, b) == 'pure order' else other).append((a, b))

w('# 077 — раскрытие точных столкновений H4 в свечи')
w()
w('Сгенерировано `base/077/examples.py`. Оба префикса в каждой паре получают одно')
w('и то же компактное описание дороги при одинаковой длительности.')
w()
w('## Все столкновения «то же содержание, другой порядок» (%d)' % len(pure))
for a, b in pure:
    diff = [i for i in range(a['dur']) if a['seq'][i] != b['seq'][i]]
    w()
    w('### %s, dur %d, различие на минутах %s' % (a['cell'], a['dur'], diff))
    w()
    show(a); show(b)

other.sort(key=lambda t: t[0]['p_utc'])
nq = [p for p in other if p[0]['cell'] == 'NQ tf54'][:3]
w('## Первые три столкновения другого рода на NQ tf54 (всего в ячейке %d)'
  % sum(1 for p in other if p[0]['cell'] == 'NQ tf54'))
for a, b in nq:
    w()
    w('### %s' % label(a, b))
    w()
    show(a); show(b)

OUT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
print('wrote %s : %d pure-order pairs, %d other pairs shown' % (OUT, len(pure), len(nq)))
