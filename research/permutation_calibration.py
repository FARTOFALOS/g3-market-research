"""Калибровка: лучшая разность частот среди 148 240 сочетаний при СЛУЧАЙНОЙ метке."""
import sys, json, numpy as np
from pathlib import Path
ROOT=Path('C:/Users/Admin/Claude/g3-market-research'); sys.path.insert(0,str(ROOT/'research'))
from relational_stencil import Corpus, Space, discover
w=Corpus.load(ROOT/'work/relational-stencil/utro-pre5/utro_0933_north_poisk_win.npz')
l=Corpus.load(ROOT/'work/relational-stencil/utro-pre5/utro_0933_north_poisk_loss.npz')
x=np.concatenate([w.ohlc,l.ohlc]); n1=w.n
rng=np.random.default_rng(2026)
for k in range(3):
    perm=rng.permutation(len(x))
    def mk(idx,tag):
        return Corpus(x[idx], w.ordinals, [f'{tag}-{i}' for i in range(len(idx))],
                      [f'{tag}-{i}' for i in range(len(idx))], dict(w.metadata))
    a=mk(perm[:n1],'a'); b=mk(perm[n1:],'b')
    sa, sb = Space(a), Space(b)
    _, rep = discover(sa, max_relations=2, min_count=20, budget=300000, mode='exact',
                      seed=9, top=5, reference=sb)
    best=max(v['display_value'] for v in rep['views']['more_than_reference'])
    worst=max(v['display_value'] for v in rep['views']['less_than_reference'])
    print(json.dumps({'перестановка':k,'лучшая_разность':round(best,4),
                      'лучшая_обратная':round(worst,4),'просмотрено':rep['tested']}))
