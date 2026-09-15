#!/usr/bin/env python3
"""Сверка эталона с каноническим предикатом репозитория.

Проверяется не «похожий ответ», а тождество: маска
`lenses.interaction.exit_boundary_touch_v1` на построенном Film и мой прямой
читатель ленты должны совпасть поэлементно, а `first_exit_contact_v1` — вернуть
тот же бар, что эталон, внутри того же окна.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from g3riz.query import Field                                   # noqa: E402
from g3riz.film import build_film                               # noqa: E402
from g3riz.lenses.interaction import (exit_boundary_touch_v1,   # noqa: E402
                                      first_exit_contact_v1)
from reference import spine, passports, exit_boundary, first_contact_reference  # noqa: E402

W = 5000  # окно сверки в барах спины; за ним эталон сам себе определение


def main(inst='NQ', tf=54, n=400, seed=7):
    field = Field(ROOT, inst)
    s = spine(inst)
    p = passports(inst, tf)
    e = exit_boundary(p)
    rng = np.random.default_rng(seed)
    n_bars = len(s['low'])

    # обычная случайная выборка + adversarial: граница ровно на high/low,
    # самые длинные фильмы, самые короткие, гэпы через уровень
    ref_all = pd.read_parquet(ROOT / 'work/080a/ref_NQ_54.parquet') if (inst, tf) == ('NQ', 54) else None
    idx = set(rng.choice(len(p), size=min(n, len(p)), replace=False).tolist())
    if ref_all is not None:
        lag = ref_all.contact_pos.to_numpy() - ref_all.t0_spine_pos.to_numpy()
        no = ref_all.contact_pos.to_numpy() < 0
        idx |= set(np.argsort(-np.where(no, -1, lag))[:40].tolist())   # длинные
        idx |= set(np.flatnonzero(lag == 1)[:40].tolist())             # +1
        idx |= set(np.flatnonzero(no)[:40].tolist())                   # без контакта
    idx = sorted(idx)

    rows = p.iloc[idx].to_dict('records')
    bad_mask = bad_first = 0
    exact_touch = 0
    for k, row in zip(idx, rows):
        t0 = int(row['t0_spine_pos'])
        end = min(n_bars - 1, t0 + W)
        film = build_film(field.market, row, end_position=end, end_reason='qa_window')
        b = float(e[k])
        assert film.exit_boundary == b, 'exit_boundary расходится с паспортом'
        mask_canon = exit_boundary_touch_v1(film)
        mask_mine = (s['low'][t0:end + 1] <= b) & (s['high'][t0:end + 1] >= b)
        if not np.array_equal(mask_canon, mask_mine):
            bad_mask += 1
        exact_touch += int(((s['low'][t0:end+1] == b) | (s['high'][t0:end+1] == b)).sum())
        canon = first_exit_contact_v1(film)
        mine = first_contact_reference(s['low'], s['high'], t0, b)
        mine_in_window = (mine - t0) if (0 <= mine <= end) else None
        if canon != mine_in_window:
            bad_first += 1
            print('MISMATCH', row['riz_id'], 'canon', canon, 'mine', mine_in_window)
    print(f'{inst} TF{tf}: сверено {len(idx)} RIZ в окне {W} баров')
    print('расхождений маски:', bad_mask, '| расхождений первого контакта:', bad_first)
    print('баров, где граница ровно равна high или low:', exact_touch)
    return bad_mask + bad_first


if __name__ == '__main__':
    raise SystemExit(1 if main() else 0)
