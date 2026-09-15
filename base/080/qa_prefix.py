#!/usr/bin/env python3
"""Проверка информационной границы. Обязательная, а не иллюстративная.

Наблюдение строится дважды: из полного источника через штатный prefix-интерфейс
и из независимого представления, в котором за q физически ничего нет. Совпасть
должны не только OHLC, но и доступность lifecycle-полей.

Усечение делается на срезе в памяти. Ни замороженное поле, ни лента не меняются:
`Tape` открывает `.npy` в режиме `mmap_mode='r'`.
"""
from __future__ import annotations
import sys
from dataclasses import asdict
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from grid import Film1Index, PrefixSource, Tape                      # noqa: E402

FAIL = []


def check(name, ok, detail=''):
    print(('OK  ' if ok else 'FAIL') + '  ' + name + (('  | ' + detail) if detail else ''))
    if not ok:
        FAIL.append(name)


class TruncatedTape:
    """Независимое представление ленты, физически обрезанное по курсору.

    Массивы скопированы и обрезаны: бара за курсором в этом объекте нет, и
    получить его неоткуда. Это не тот же объект с флагом, а другой источник.
    """

    def __init__(self, tape: Tape, cursor: int):
        self.instrument = tape.instrument
        self.cursor = int(cursor)
        self.n = self.cursor + 1
        for k in ('open', 'high', 'low', 'close', 'close_ts_utc_ns', 'session_id'):
            setattr(self, k, np.array(getattr(tape, k)[:self.n]))

    def slice(self, a, b):
        if b > self.cursor:
            raise IndexError('за курсором в этом источнике баров нет')
        out = {k: np.asarray(getattr(self, k)[a:b + 1])
               for k in ('open', 'high', 'low', 'close', 'close_ts_utc_ns', 'session_id')}
        out['spine_pos'] = np.arange(a, b + 1)
        return out


def main(inst='NQ', n_riz=120, seed=5):
    idx = Film1Index(inst)
    d = idx.df
    rng = np.random.default_rng(seed)

    # выборка с намеренным перекосом в трудные случаи
    has = d.first_observed_contact_pos.to_numpy() >= 0
    lag = d.observed_bars_t0_to_end.to_numpy()
    pools = {
        'contact_t0_plus_1': np.flatnonzero(has & (lag == 1)),
        'few_minutes_away': np.flatnonzero(has & (lag >= 3) & (lag <= 20)),
        'long_across_days': np.flatnonzero(has & (d.sessions_spanned.to_numpy() > 1)),
        'primacy_unknown': np.flatnonzero(
            d.first_observed_contact_primacy.to_numpy() == 'unknown'),
        'no_contact': np.flatnonzero(~has),
        'deletion_before_contact': np.flatnonzero(
            has & (d.c1_deletion_spine_pos.to_numpy() >= 0)
            & (d.c1_deletion_spine_pos.to_numpy() < d.first_observed_contact_pos.to_numpy())),
        'flicker': np.flatnonzero(d.native_blue_confirmation_spine_pos.to_numpy() < 0),
        'archive_censored': np.flatnonzero(d.censored.to_numpy()),
    }
    take = []
    for k, v in pools.items():
        if v.size:
            take.append(rng.choice(v, size=min(n_riz // len(pools) + 2, v.size),
                                   replace=False))
    take = np.unique(np.concatenate(take))

    bad_prefix = bad_life = 0
    tested = 0
    for i in take:
        r = d.iloc[i]
        rid = str(r.riz_id)
        qs = idx.q_positions(rid)
        if qs.size == 0:
            continue
        pick = qs[rng.choice(qs.size, size=min(3, qs.size), replace=False)]
        for q in pick:
            q = int(q)
            a = idx.prefix(rid, q)                                   # полный источник
            trunc = TruncatedTape(idx.tape, q)
            b = idx.prefix(rid, q, source=trunc)                     # обрезанный
            tested += 1
            if asdict(a) != asdict(b):
                bad_prefix += 1
                diff = {k: (v, asdict(b)[k]) for k, v in asdict(a).items()
                        if v != asdict(b)[k]}
                print('  PREFIX DIFF', rid, q, diff)
            # lifecycle as-of q не имеет права смотреть вперёд
            dele = int(r.c1_deletion_spine_pos)
            expect_deleted = 0 <= dele <= q
            if (a.lifecycle_state_as_of_q == 'deleted') != expect_deleted:
                bad_life += 1

    check('prefix из полного источника == prefix из физически обрезанного',
          bad_prefix == 0, f'{tested} пар (riz_id, q) на {take.size} RIZ')
    check('lifecycle as-of q не опережает записанное событие', bad_life == 0)

    # --- следующий open недостижим из prefix ---
    rid = str(d.iloc[int(take[0])].riz_id)
    q = int(idx.q_positions(rid)[0])
    trunc = TruncatedTape(idx.tape, q)
    try:
        trunc.slice(q + 1, q + 1)
        got = True
    except IndexError:
        got = False
    check('следующий open из prefix-источника недостижим', not got)
    ps = PrefixSource(idx.tape, cursor=q, start=q - 0)
    try:
        ps.slice(q, q + 1)
        got2 = True
    except IndexError:
        got2 = False
    check('PrefixSource отказывает в баре за курсором', not got2)

    # --- q на самой минуте T0 ---
    ok_t0 = all(int(idx.q_positions(str(d.iloc[int(i)].riz_id))[0])
                == int(d.iloc[int(i)].t0_spine_pos)
                for i in take[:40] if idx.q_positions(str(d.iloc[int(i)].riz_id)).size)
    check('первый q — сама минута T0', ok_t0)

    # --- касательная минута нового q не создаёт ---
    cert = d[d.first_observed_contact_primacy == 'certified'].head(2000)
    last_q = cert.certified_fresh_until_pos_strict.to_numpy()
    contact = cert.first_observed_contact_pos.to_numpy()
    check('на закрытии касательной минуты нового q нет',
          bool((last_q == contact - 1).all()),
          f'{len(cert)} сертифицированных фильмов')

    # --- ранние q переживают позднейшую цензуру ---
    lost = d[d.film1_status == 'freshness_lost_before_contact'].head(300)
    keep = [(int(r.certified_fresh_until_pos_strict) >= int(r.t0_spine_pos))
            for _, r in lost.iterrows()]
    check('при позднейшей потере достоверности ранние q сохраняются',
          all(keep), f'{len(lost)} фильмов, минимум по одному q (сама T0)')

    # --- T0 переживает позднейший failure/flicker ---
    fl = d[d.native_blue_confirmation_spine_pos < 0]
    check('T0 сохраняется при отсутствии нативного подтверждения (flicker)',
          bool((fl.t0_spine_pos.to_numpy() >= 0).all()
               and (fl.q_certified_strict.to_numpy() >= 1).all()),
          f'{len(fl)} RIZ без нативного подтверждения')

    # --- deletion не обрезает Film-1 ---
    db = d[(d.c1_deletion_spine_pos >= 0)
           & (d.first_observed_contact_pos > d.c1_deletion_spine_pos)]
    check('deletion до контакта не подменяет конец Film-1',
          bool((db.first_observed_contact_pos.to_numpy()
                > db.c1_deletion_spine_pos.to_numpy()).all()),
          f'{len(db)} фильмов, контакт наблюдается после удаления')

    print('\nПРОВАЛЕНО:' if FAIL else '\nИнформационная граница держится.')
    for f in FAIL:
        print(' -', f)
    return 1 if FAIL else 0


if __name__ == '__main__':
    raise SystemExit(main(*(sys.argv[1:] or [])))
