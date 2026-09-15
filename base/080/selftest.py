#!/usr/bin/env python3
"""Инварианты индекса Film-1. Любое нарушение — стоп, а не предупреждение."""
from __future__ import annotations
import sys
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
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'src'))
from reference import spine, passports, exit_boundary, first_contact_reference  # noqa: E402
from g3riz.query import Field                                                   # noqa: E402
from g3riz.film import build_film                                               # noqa: E402
from g3riz.lenses.interaction import first_exit_contact_v1                      # noqa: E402

FAIL = []


def check(name, ok, detail=''):
    print(('OK  ' if ok else 'FAIL') + '  ' + name + (('  | ' + detail) if detail else ''))
    if not ok:
        FAIL.append(name)


def main():
    idx = {i: pd.read_parquet(INDEX / f'film1_{i}.parquet') for i in ('ES', 'NQ', 'YM')}

    for inst, d in idx.items():
        t0 = d.t0_spine_pos.to_numpy()
        c = d.first_observed_contact_pos.to_numpy()
        has = c >= 0
        fs = d.certified_fresh_until_pos_strict.to_numpy()
        prim = d.first_observed_contact_primacy.to_numpy()
        stat = d.film1_status.to_numpy()

        check(f'{inst}: контакт строго после T0',
              bool((c[has] > t0[has]).all()))
        check(f'{inst}: сертифицированный контакт даёт fresh_until = c-1',
              bool((fs[prim == 'certified'] == c[prim == 'certified'] - 1).all()))
        check(f'{inst}: fresh_until не раньше самой T0',
              bool((fs >= t0).all()))
        check(f'{inst}: q_certified = fresh_until - t0 + 1',
              bool((d.q_certified_strict.to_numpy() == fs - t0 + 1).all()))
        m = (prim == 'certified') & (c == t0 + 1)
        check(f'{inst}: контакт на T0+1 оставляет сеткой одну минуту T0',
              bool((d.q_certified_strict.to_numpy()[m] == 1).all()),
              f'{int(m.sum())} таких RIZ')
        check(f'{inst}: q_certified <= q_observed',
              bool((d.q_certified_strict.to_numpy() <= d.q_observed.to_numpy()).all()))
        check(f'{inst}: strict не мягче shared_survived',
              bool((d.q_certified_strict.to_numpy()
                    <= d.q_certified_shared_survived.to_numpy()).all()))
        check(f'{inst}: primacy=not_applicable ровно там, где контакта нет',
              bool(((prim == 'not_applicable') == ~has).all()))
        check(f'{inst}: статус contact_* ровно там, где контакт есть',
              bool((np.isin(stat, ['contact_certified',
                                   'contact_seen_primacy_unknown']) == has).all()))
        check(f'{inst}: наблюдаемых баров не больше, чем минут по часам',
              bool((d.observed_bars_t0_to_end.to_numpy()
                    <= d.wall_clock_minutes_t0_to_end.to_numpy()).all()))
        check(f'{inst}: минуты по часам = наблюдённые бары + пропущенные минуты',
              bool((d.wall_clock_minutes_t0_to_end.to_numpy()
                    == d.observed_bars_t0_to_end.to_numpy()
                    + d.missing_minutes_inside.to_numpy()).all()))
        check(f'{inst}: deletion не обрезает Film-1',
              bool((c[has & (d.c1_deletion_spine_pos.to_numpy() >= 0)]
                    >= 0).all()))

    # индекс против эталона на NQ TF54 построчно. Эталон — производный
    # артефакт `reference.py`; если его рядом нет, он строится заново, чтобы
    # проверка не зависела от чужого рабочего каталога.
    ref_path = INDEX / 'ref_NQ_54.parquet'
    if not ref_path.exists():
        from reference import run as ref_run
        ref_run('NQ', 54).to_parquet(ref_path, index=False)
    ref = pd.read_parquet(ref_path)
    sub = idx['NQ'][idx['NQ'].tf_minutes == 54].set_index('riz_id')
    ref = ref.set_index('riz_id')
    j = ref.join(sub[['first_observed_contact_pos', 'exit_boundary', 't0_spine_pos']],
                 rsuffix='_idx')
    check('NQ TF54: индекс воспроизводит эталонный contact_pos построчно',
          bool((j.contact_pos == j.first_observed_contact_pos).all()),
          f'{len(j)} строк')
    check('NQ TF54: индекс воспроизводит exit_boundary паспорта',
          bool((j.exit_boundary == j.exit_boundary_idx).all()))
    check('NQ TF54: индекс воспроизводит t0_spine_pos',
          bool((j.t0_spine_pos == j.t0_spine_pos_idx).all()))

    # свежая случайная сверка с каноническим `first_exit_contact_v1`
    rng = np.random.default_rng(11)
    field = Field(ROOT, 'NQ')
    s = spine('NQ')
    bad = 0
    tested = 0
    for tf in rng.choice(np.arange(1, 1441), size=12, replace=False):
        tf = int(tf)
        p = passports('NQ', tf)
        if not len(p):
            continue
        e = exit_boundary(p)
        take = rng.choice(len(p), size=min(20, len(p)), replace=False)
        want = idx['NQ'][idx['NQ'].tf_minutes == tf].set_index('riz_id')
        for k in take:
            row = p.iloc[k].to_dict()
            t0 = int(row['t0_spine_pos'])
            end = min(len(s['low']) - 1, t0 + 3000)
            film = build_film(field.market, row, end_position=end, end_reason='qa_window')
            canon = first_exit_contact_v1(film)
            got = int(want.loc[row['riz_id'], 'first_observed_contact_pos'])
            mine = (got - t0) if (0 <= got <= end) else None
            tested += 1
            if canon != mine:
                bad += 1
    check('NQ: индекс совпадает с каноническим first_exit_contact_v1',
          bad == 0, f'{tested} RIZ на 12 случайных ТФ, расхождений {bad}')

    # слой continuation обязан знать о сертификации: поздний наблюдавшийся
    # контакт за неизвестным промежутком не имеет права стать TP ни через
    # `continuation`, ни через `excursion_to_tp`
    from grid import Film1Index
    fi = Film1Index('NQ')
    bad_tp = bad_exc = 0
    checked = 0
    for cert in ('strict', 'shared_survived'):
        col = ('first_observed_contact_primacy' if cert == 'strict'
               else 'first_observed_contact_primacy_shared_survived')
        pool = fi.df[fi.df[col] != 'certified']
        for rid in pool.riz_id.to_numpy()[:150]:
            rid = str(rid)
            qs = fi.q_positions(rid, cert)
            c = fi.continuation(rid, int(qs[-1]), certification=cert)
            e = fi.excursion_to_tp(rid, certification=cert)
            checked += 1
            if c.certified_tp_pos is not None:
                bad_tp += 1
            if e['kind'] not in ('primacy_not_certified', 'no_contact'):
                bad_exc += 1
    check('continuation не выдаёт TP при недоказанной первичности', bad_tp == 0,
          f'{checked} проверок в обеих сертификациях')
    check('excursion_to_tp отказывается считать TP при недоказанной первичности',
          bad_exc == 0)
    # и наоборот: где первичность доказана, TP обязан быть наблюдавшимся контактом
    ok = fi.df[fi.df.first_observed_contact_primacy == 'certified'].head(150)
    bad_ok = 0
    for _, row in ok.iterrows():
        c = fi.continuation(str(row.riz_id), int(row.certified_fresh_until_pos_strict))
        if c.certified_tp_pos != int(row.first_observed_contact_pos):
            bad_ok += 1
    check('при доказанной первичности TP — это наблюдавшийся контакт', bad_ok == 0)

    print('\nПРОВАЛЕНО:' if FAIL else '\nВсе инварианты держатся.')
    for f in FAIL:
        print(' -', f)
    return 1 if FAIL else 0


if __name__ == '__main__':
    raise SystemExit(main())
