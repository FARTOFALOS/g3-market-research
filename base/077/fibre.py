"""077: the exact fibre of the 076 road-compression on the observed corpus.

Compression under test, exactly as base/076/portability.py and neighbourhood.py
compute it:

    H4 = (n_runs/dur, e/dur, f/dur, inside/dur)

Duration is an S coordinate, so it is known beside H4 and the fibre is taken at
fixed dur: key = (side, kind, dur, n_runs, e, f, inside). With dur equal the four
shares are then equal bit for bit. No tolerance, no closeness threshold, no new
feature, no Y, no Volume.

Eligibility is the frozen one: contiguous native prefix up to p = T0+2
(support_probe.py) plus a clean 60-minute ruler strictly before the interval
(portability.py). The resulting n per cell is asserted against FROZEN_S_H.md.

Independent unit: one state per physical scene, then one per session day.

Run: python -B base/077/fibre.py   (needs base/077/_states, see rich.py)
"""
import json, collections, functools
from pathlib import Path
import numpy as np

CACHE = Path(__file__).resolve().parent / '_states'
CELLS = [('NQ', 54, 857), ('NQ', 30, 1401), ('NQ', 90, 606), ('ES', 54, 694), ('YM', 54, 834)]
NAME = {-2: 'beyond-far', -1: 'on-far', 0: 'inside', 1: 'on-exit', 2: 'outside'}


def load(inst, tf):
    raw = json.loads((CACHE / ('states_%s_tf%d.json' % (inst, tf))).read_text(encoding='utf-8'))
    out = []
    for s in raw:
        if not s['u60']:
            continue
        s['seq'] = tuple(tuple(r) for r in s['readings'])
        s['dur'] = int(s['g0'][10]); s['side'] = s['g0'][0]; s['kind'] = s['g0'][1]
        s['n_runs'] = len(s['h_order'])
        e, f, l_2, l_1, l0, l1, l2 = s['counts']
        s['e'], s['f'], s['l0'] = e, f, l0
        s['loc'] = (l_2, l_1, l0, l1, l2)
        s['multiset'] = tuple(sorted(collections.Counter(s['seq']).items()))
        s['pairing'] = tuple(sorted(collections.Counter((x[0], x[1]) for x in s['seq']).items()))
        s['year'] = int(s['p_utc'][:4]); s['cell'] = '%s tf%d' % (inst, tf)
        s['pos'] = s['g0'][9] / s['u60']; s['bulk'] = s['g0'][2] / s['u60']
        out.append(s)
    return len(raw), out


def dedup(v):
    by_scene, by_day = {}, {}
    for s in sorted(v, key=lambda x: x['p']):
        by_scene.setdefault(s['scene'], s)
    for s in sorted(by_scene.values(), key=lambda x: x['p']):
        by_day.setdefault(s['session_id'], s)
    return list(by_day.values())


def fibre_key(s):
    return (s['side'], s['kind'], s['dur'], s['n_runs'], s['e'], s['f'], s['l0'])


def label(a, b):
    if a['multiset'] == b['multiset']:
        return 'pure order'
    lo, pa = a['loc'] != b['loc'], a['pairing'] != b['pairing']
    if lo and pa:
        return 'location split + e/f pairing'
    if lo:
        return 'location split'
    if pa:
        return 'e/f pairing'
    return 'same marginals, other joint difference'


def arrangements(counts, runs, budget=2_000_000):
    """orderings of the same per-minute content carrying the same run count"""
    sp = 1
    for c in counts:
        sp *= (c + 1)
    if sp * len(counts) > budget:
        return None

    @functools.lru_cache(maxsize=None)
    def go(rem, last, r):
        if r > runs:
            return 0
        if not any(rem):
            return 1 if r == runs else 0
        t = 0
        for i, c in enumerate(rem):
            if not c:
                continue
            n = list(rem); n[i] -= 1
            t += go(tuple(n), i, r + (0 if i == last else 1))
        return t
    v = go(tuple(counts), -1, 0); go.cache_clear(); return v


if __name__ == '__main__':
    POOL, PAIRS = {}, []
    print('=' * 104)
    print('1. EXACT FIBRES OF H4 AT KNOWN DURATION')
    print('=' * 104)
    print('%-9s %7s %7s %7s %9s %7s %8s %8s %8s %7s %6s'
          % ('cell', 'prefix', 'elig', 'frozen', 'distinct', 'fibres', 'size>=2',
             'seq diff', 'states', 'days', 'pure'))
    for inst, tf, frozen_n in CELLS:
        prefix_n, S = load(inst, tf)
        assert len(S) == frozen_n, (inst, tf, len(S), frozen_n)
        POOL[(inst, tf)] = S
        G = collections.defaultdict(list)
        for s in S:
            G[fibre_key(s)].append(s)
        big = {k: dedup(v) for k, v in G.items()}
        big = {k: v for k, v in big.items() if len(v) > 1}
        seqdif, pure = {}, set()
        for k, v in big.items():
            loc = [(a, b) for i, a in enumerate(v) for b in v[i + 1:] if a['seq'] != b['seq']]
            if loc:
                seqdif[k] = v; PAIRS += loc
                if any(label(a, b) == 'pure order' for a, b in loc):
                    pure.add(k)
        st = [s for v in seqdif.values() for s in v]
        print('%-9s %7d %7d %7d %9d %7d %8d %8d %8d %7d %6d'
              % ('%s tf%d' % (inst, tf), prefix_n, len(S), frozen_n, len({s['seq'] for s in S}),
                 len(G), len(big), len(seqdif), len(st),
                 len({s['session_id'] for s in st}), len(pure)))

    print('\npooled collision pairs: %d' % len(PAIRS))
    for k, v in collections.Counter(label(a, b) for a, b in PAIRS).most_common():
        print('   %-42s %4d' % (k, v))

    print('\n' + '=' * 104)
    print('2. DOES THE COLLISION SURVIVE THE JOINT REPRESENTATION S + H4')
    print('=' * 104)
    bad = 0
    for a, b in PAIRS:
        for s in (a, b):
            pos, bulk = s['pos'], s['bulk']
            loc = -2 if pos < -bulk else (-1 if pos == -bulk else (0 if pos < 0 else (1 if pos == 0 else 2)))
            bad += (loc != s['seq'][-1][2])
    print('last-minute location reproduced from (position, bulk): %d mismatches on %d states'
          % (bad, 2 * len(PAIRS)))
    surv = [(a, b) for a, b in PAIRS if a['seq'][-1][2] == b['seq'][-1][2]]
    print('separated by S through that location : %d pairs' % (len(PAIRS) - len(surv)))
    print('surviving S + H4                     : %d pairs, %d states, %d session days, %d..%d'
          % (len(surv), len({s['riz_id'] for p in surv for s in p}),
             len({s['session_id'] for p in surv for s in p}),
             min(s['year'] for p in surv for s in p), max(s['year'] for p in surv for s in p)))
    for k, v in collections.Counter(label(a, b) for a, b in surv).most_common():
        print('   %-42s %4d' % (k, v))
    print('   by cell: %s' % collections.Counter(a['cell'] for a, b in surv).most_common())
    dp = np.array([abs(a['pos'] - b['pos']) for a, b in surv])
    db = np.array([abs(a['bulk'] - b['bulk']) for a, b in surv])
    ur = np.array([max(a['u60'] / b['u60'], b['u60'] / a['u60']) for a, b in surv])
    print('   descriptive only, not a filter: |d position| p10/med/p90 %.2f/%.2f/%.2f ; '
          '|d bulk| %.2f/%.2f/%.2f ; ruler ratio %.2f/%.2f/%.2f'
          % (*np.percentile(dp, [10, 50, 90]), *np.percentile(db, [10, 50, 90]),
             *np.percentile(ur, [10, 50, 90])))

    print('\n' + '=' * 104)
    print('3. THE POPULATION THAT COULD EVER CARRY "SAME CONTENT, DIFFERENT ORDER"')
    print('=' * 104)
    tot = collections.Counter()
    for inst, tf, _ in CELLS:
        S = POOL[(inst, tf)]
        line = []
        for withruns in (False, True):
            G = collections.defaultdict(list)
            for s in S:
                k = (s['side'], s['kind'], s['dur'], s['multiset']) + ((s['n_runs'],) if withruns else ())
                G[k].append(s)
            groups = [dedup(v) for v in G.values()]
            groups = [v for v in groups if len(v) > 1]
            dif = [v for v in groups if len({s['seq'] for s in v}) > 1]
            line += [len(groups), len(dif)]
            tot['g%d' % withruns] += len(groups); tot['d%d' % withruns] += len(dif)
        print('%-9s content repeats in %3d groups (order differs in %d) | with the run count '
              'also equal %3d (%d)' % ('%s tf%d' % (inst, tf), *line))
    print('pooled: %d groups repeat the content, %d also the run count; the order differs in %d and %d'
          % (tot['g0'], tot['g1'], tot['d0'], tot['d1']))

    forced = free = shown = 0
    for inst, tf, _ in CELLS:
        G = collections.defaultdict(list)
        for s in POOL[(inst, tf)]:
            G[(s['side'], s['kind'], s['dur'], s['multiset'], s['n_runs'])].append(s)
        for k, v in G.items():
            v = dedup(v)
            if len(v) < 2:
                continue
            c = tuple(sorted((n for _, n in k[3]), reverse=True))
            a = arrangements(c, k[4])
            if a == 1:
                forced += 1
            elif a:
                free += 1; shown += len({s['seq'] for s in v}) > 1
    print('of those, %d groups admitted only one arrangement (no contrast possible) and %d admitted '
          'more; %d of the latter actually show a different order' % (forced, free, shown))

    print('\n' + '=' * 104)
    print('4. EVERY "SAME CONTENT, DIFFERENT ORDER" COLLISION')
    print('=' * 104)
    for a, b in PAIRS:
        if label(a, b) != 'pure order':
            continue
        da = [i for i in range(a['dur']) if a['seq'][i] != b['seq'][i]]
        same_loc = a['seq'][-1][2] == b['seq'][-1][2]
        tail = 'separated by S (last minute %s vs %s)' % (NAME[a['seq'][-1][2]], NAME[b['seq'][-1][2]])
        if same_loc:
            tail = ('hidden from S + H4' if max(da) < a['dur'] - 1
                    else 'difference only in the last minute contact flags')
        print('%-9s dur=%-3d runs=%d exit=%d far=%d inside=%d | %s vs %s | minutes %s | %s'
              % (a['cell'], a['dur'], a['n_runs'], a['e'], a['f'], a['l0'],
                 a['p_utc'], b['p_utc'], da, tail))

    print('\n' + '=' * 104)
    print('5. WHERE THE NON-CONSTANT MINUTES SIT (p = T0+2, so index dur-3 is T0)')
    print('=' * 104)
    for inst, tf, _ in CELLS[:3]:
        S = POOL[(inst, tf)]
        tail, dens = [], collections.Counter()
        for s in S:
            ix = [i for i, r in enumerate(s['seq']) if r[0] or r[1]]
            if ix:
                tail.append(all(i >= s['dur'] - 3 for i in ix))
                for i in ix:
                    dens[s['dur'] - 1 - i] += 1
        n = sum(dens.values())
        print('%-9s every contact inside T0..T0+2 : %.3f | density by distance from the cursor %s'
              % ('%s tf%d' % (inst, tf), float(np.mean(tail)),
                 ' '.join('%d:%.3f' % (k, dens[k] / n) for k in range(6))))
