"""077: how large the fibre of the compression is IN PRINCIPLE.

For every eligible prefix: how many orderings of its own per-minute content carry
the same run count, i.e. how many records would have received the identical H4 at
the identical duration. Exact enumeration where the state space fits the budget;
for the rest only the multinomial upper bound is reported.

Standalone on purpose (the counter is repeated from fibre.py, not imported, so
that neither script runs the other's report).

Slow: a few minutes per cell. Run: python -B base/077/fibre_size.py
"""
import json, collections, functools, math
from pathlib import Path
import numpy as np

CACHE = Path(__file__).resolve().parent / '_states'


def arrangements(counts, runs, budget=400_000):
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


for inst, tf in [('NQ', 54), ('NQ', 30)]:
    S = [s for s in json.loads((CACHE / ('states_%s_tf%d.json' % (inst, tf))).read_text(encoding='utf-8'))
         if s['u60']]
    exact, big, alpha, dur = [], [], [], []
    for s in S:
        seq = [tuple(r) for r in s['readings']]
        c = sorted(collections.Counter(seq).values(), reverse=True)
        alpha.append(len(c)); dur.append(len(seq))
        v = arrangements(tuple(c), len(s['h_order']))
        if v is None:
            m = math.factorial(sum(c))
            for x in c:
                m //= math.factorial(x)
            big.append(m)
        else:
            exact.append(v)
    e = np.array(exact, float)
    print('%s tf%d  n=%d  distinct readings per prefix: median %d max %d ; dur median %d max %d'
          % (inst, tf, len(S), int(np.median(alpha)), max(alpha), int(np.median(dur)), max(dur)))
    print('   enumerated exactly for %d states; %d too large (multinomial median 10^%.1f)'
          % (len(exact), len(big), float(np.median([math.log10(x) for x in big])) if big else 0))
    print('   only one arrangement possible: %.3f of states' % float((e == 1).mean()))
    for q in [50, 75, 90, 99, 100]:
        v = float(np.percentile(e, q))
        print('   p%-4d %20.0f   (10^%.1f)' % (q, v, math.log10(v) if v > 0 else 0))
