#!/usr/bin/env python3
"""Разложение фильмов без пригодного q по причинам, и асимметрия T0 против позже.

`films_with_no_usable_q` одним числом скрывает разные вещи. Пригодность q здесь
означает ровно одно: reference определена (следующий бар contiguous) И её open
ещё снаружи собственной exit boundary, то есть сделка исходного направления к
этой цели сохраняет заданный смысл.

Отдельно считается то, что легко спутать: «нет пригодного входа на T0» и «нет
пригодного входа нигде» — разные утверждения, и обе асимметрии измерены.
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
INDEX = Path(os.environ.get('G3_080A_INDEX', str(HERE / 'index')))
sys.path.insert(0, str(HERE))
from tape import presence_grid, gap_kinds                            # noqa: E402

INSTRUMENTS = ('ES', 'NQ', 'YM')


@njit(cache=True)
def per_film(t0s, fresh, e, north, opn, kind, last):
    """На фильм: состав его q и пригодность первого q."""
    n = t0s.size
    out = np.zeros((n, 6), dtype=np.int64)   # q, gap, outside, on, beyond, edge
    first = np.zeros(n, dtype=np.int8)       # 0 gap, 1 outside, 2 on, 3 beyond, 4 edge
    first_usable_age = np.full(n, -1, dtype=np.int64)
    for i in range(n):
        lvl, up = e[i], north[i]
        for q in range(t0s[i], fresh[i] + 1):
            col = 0
            nxt = q + 1
            if nxt > last:
                out[i, 5] += 1
                col = 4
            elif kind[nxt] != 0:
                out[i, 1] += 1
                col = 0
            else:
                s = (opn[nxt] - lvl) if up else (lvl - opn[nxt])
                if s > 0.0:
                    out[i, 2] += 1
                    col = 1
                    if first_usable_age[i] < 0:
                        first_usable_age[i] = q - t0s[i]
                elif s == 0.0:
                    out[i, 3] += 1
                    col = 2
                else:
                    out[i, 4] += 1
                    col = 3
            out[i, 0] += 1
            if q == t0s[i]:
                first[i] = col
    return out, first, first_usable_age


def one(inst, kind):
    d = pd.read_parquet(INDEX / f'film1_{inst}.parquet')
    opn = np.asarray(np.load(ROOT / f'data/market/{inst}/open.npy'))
    m, first, fua = per_film(
        d.t0_spine_pos.to_numpy().astype(np.int64),
        d.certified_fresh_until_pos_strict.to_numpy().astype(np.int64),
        d.exit_boundary.to_numpy().astype(np.float64),
        (d.side.to_numpy() == 'north'), opn, kind, opn.size - 1)
    nq, gap, out_, on, beyond, edge = (m[:, k] for k in range(6))
    none = out_ == 0                      # ни одного пригодного q
    single = nq == 1

    def reason(mask):
        """Причина непригодности: чем был занят весь состав q этого фильма."""
        only_gap = mask & (gap == nq)
        only_on = mask & (on == nq)
        only_beyond = mask & (beyond == nq)
        only_edge = mask & (edge == nq)
        mixed = mask & ~(only_gap | only_on | only_beyond | only_edge)
        mixed_has_gap = mixed & (gap > 0)
        return {
            'все q потеряны на промежутке (reference не определена)': int(only_gap.sum()),
            'все q: open ровно на boundary': int(only_on.sum()),
            'все q: open уже за boundary': int(only_beyond.sum()),
            'все q: следующего бара нет (край архива)': int(only_edge.sum()),
            'смесь причин, с участием промежутка': int(mixed_has_gap.sum()),
            'смесь причин, без промежутка (on + beyond)': int((mixed & ~(gap > 0)).sum()),
        }

    r = {
        'films': int(len(d)),
        'films_with_no_usable_q': int(none.sum()),
        'no_usable_q_by_reason': reason(none),
        'no_usable_q_among_single_q_films': int((none & single).sum()),
        'no_usable_q_among_multi_q_films': int((none & ~single).sum()),
        'no_usable_q_by_reason_single_q_films': reason(none & single),
        'no_usable_q_by_reason_multi_q_films': reason(none & ~single),
        # асимметрия, которую нельзя сворачивать в одно число
        'usable_at_first_q': int((first == 1).sum()),
        'not_usable_at_first_q': int((first != 1).sum()),
        'not_usable_at_first_q_but_usable_later': int(((first != 1) & (out_ > 0)).sum()),
        'not_usable_at_first_q_and_never': int(((first != 1) & none).sum()),
        'first_q_reason_when_not_usable': {
            'промежуток перед следующим баром': int((first == 0).sum()),
            'open ровно на boundary': int((first == 2).sum()),
            'open уже за boundary': int((first == 3).sum()),
            'следующего бара нет': int((first == 4).sum()),
        },
        'age_of_first_usable_q_when_not_first': {
            str(q): int(np.quantile(fua[(first != 1) & (out_ > 0)], q))
            for q in (0.5, 0.9, 0.99, 1.0)} if ((first != 1) & (out_ > 0)).any() else {},
        'usable_q_count_quantiles': {str(q): int(np.quantile(out_, q))
                                     for q in (0.5, 0.9, 0.99, 1.0)},
    }
    return r


if __name__ == '__main__':
    grid, lo, hi = presence_grid()
    out = {}
    for inst in INSTRUMENTS:
        kind, _ = gap_kinds(inst, grid, lo, hi)
        out[inst] = one(inst, kind)
        print(inst, 'done', flush=True)
    keys = ('films', 'films_with_no_usable_q', 'usable_at_first_q',
            'not_usable_at_first_q', 'not_usable_at_first_q_but_usable_later',
            'not_usable_at_first_q_and_never', 'no_usable_q_among_single_q_films',
            'no_usable_q_among_multi_q_films')
    out['TOTAL'] = {k: sum(out[i][k] for i in INSTRUMENTS) for k in keys}
    for sub in ('no_usable_q_by_reason', 'first_q_reason_when_not_usable'):
        out['TOTAL'][sub] = {k: sum(out[i][sub][k] for i in INSTRUMENTS)
                             for k in out['NQ'][sub]}
    txt = json.dumps(out, ensure_ascii=False, indent=1)
    (Path(os.environ.get('G3_080A_OUT', str(HERE))) / 'census_unusable.json'
     ).write_text(txt, encoding='utf-8', newline='\n')
    print(json.dumps(out['TOTAL'], ensure_ascii=False, indent=1))
