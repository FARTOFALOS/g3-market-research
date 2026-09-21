#!/usr/bin/env python3
"""093 ARCH — архитектура завершённого возврата: сколько свингов до/после экстремума.

Слово трейдера: у выборки, что дошла до `b`, посмотреть от главного экстремума,
сколько свингов чаще всего происходит — до экстремума (свинг-хаи на пути вверх у
north) и после (свинг-лоу на пути вниз к `b`), — найти самую частую
архитектурную повторяемость, и где конструкция обычно ломается (там логичен стоп).

Описательный X-ray по завершённым возвратам (contact_certified Film-1). Полное
будущее используется как рентген (границы фильма известны); это НЕ prefix-правило
входа — для действия нужен отдельный rewind.

Свинг = фрактальный пивот полуширины w=3 (строгий локальный max/min по [p-w,p+w]).
north-фильм: цена выше `b`, ушла вверх, вернулась вниз к `b`; главный экстремум —
максимум high на [T0, contact]; до него считаем свинг-хаи, после — свинг-лоу.
south — зеркально.
"""
from __future__ import annotations
import os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit, prange

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = Path(os.environ.get('G3_093_OUT', str(ROOT / 'work/093')))
sys.path.insert(0, str(ROOT / 'base/081'))
sys.path.insert(0, str(ROOT / 'base/080'))
from trading import sessions, bar_session_map                          # noqa: E402
from paths import load_films                                           # noqa: E402

TICK = {'NQ': 0.25, 'ES': 0.25, 'YM': 1.0}


@njit(parallel=True, cache=True)
def _arch(t0s, cs, es, north, w, high, low, ts, sess, close_ns,
          o_next, o_nbef, o_naft, o_extexc, o_extfrac, o_len, o_ttc, o_break):
    for i in prange(t0s.size):
        t0 = t0s[i]; c = cs[i]; e = es[i]; up = north[i]
        n = c - t0 + 1
        o_len[i] = n
        if n < 2 * w + 3:
            o_next[i] = -1
            continue
        # главный экстремум на [t0, c]
        ext_p = t0; ext_v = high[t0] if up else low[t0]
        for p in range(t0, c + 1):
            v = high[p] if up else low[p]
            if up:
                if v > ext_v:
                    ext_v = v; ext_p = p
            else:
                if v < ext_v:
                    ext_v = v; ext_p = p
        o_extexc[i] = (ext_v - e) if up else (e - ext_v)
        o_extfrac[i] = (ext_p - t0) / max(n - 1, 1)
        sq = sess[t0]
        if sq >= 0:
            o_ttc[i] = (close_ns[sq] - ts[t0]) / 6.0e10
        # свинги до экстремума (одноимённые главному) и после (противоположные)
        nb = 0; naf = 0
        for p in range(t0 + w, c - w + 1):
            is_hi = True; is_lo = True
            for k in range(1, w + 1):
                if not (high[p] > high[p - k] and high[p] > high[p + k]):
                    is_hi = False
                if not (low[p] < low[p - k] and low[p] < low[p + k]):
                    is_lo = False
            if up:
                if is_hi and p < ext_p:
                    nb += 1
                if is_lo and p > ext_p:
                    naf += 1
            else:
                if is_lo and p < ext_p:
                    nb += 1
                if is_hi and p > ext_p:
                    naf += 1
        o_nbef[i] = nb; o_naft[i] = naf
        # где ломается возврат: на пути от экстремума к c — первый бар, делающий
        # ПРОТИВ-ход выше предыдущего противо-свинга (higher-high на спуске = слом).
        # считаем число «пробоев структуры» на возврате (сколько раз спуск делает
        # higher-high над последним swing-high; для south — lower-low).
        brk = 0
        last_cross = -np.inf if up else np.inf
        prev_sw = -np.inf if up else np.inf   # последний противо-свинг на возврате
        for p in range(ext_p + w, c - w + 1):
            is_hi = True; is_lo = True
            for k in range(1, w + 1):
                if not (high[p] > high[p - k] and high[p] > high[p + k]):
                    is_hi = False
                if not (low[p] < low[p - k] and low[p] < low[p + k]):
                    is_lo = False
            if up:
                # возврат вниз: противо-свинг = swing-high (откат); слом = выше прошлого
                if is_hi:
                    if high[p] > prev_sw and prev_sw > -np.inf:
                        brk += 1
                    prev_sw = high[p]
            else:
                if is_lo:
                    if low[p] < prev_sw and prev_sw < np.inf:
                        brk += 1
                    prev_sw = low[p]
        o_break[i] = brk
        o_next[i] = 1


def build(inst='NQ', terr='discovery', w=3):
    m = ROOT / 'data/market' / inst
    high = np.load(m / 'high.npy'); low = np.load(m / 'low.npy')
    ts = np.load(m / 'close_ts_utc_ns.npy')
    st, cl = sessions(); cl = cl.astype(np.int64)
    sess, _ = bar_session_map(ts, st, cl)
    f = load_films(inst, terr)
    f = f[f.film1_status.to_numpy() == 'contact_certified'].reset_index(drop=True)
    t0 = f.t0_spine_pos.to_numpy().astype(np.int64)
    c = f.first_observed_contact_pos.to_numpy().astype(np.int64)
    e = f.exit_boundary.to_numpy().astype(np.float64)
    north = (f.side.to_numpy() == 'north')
    n = len(f)
    z = lambda dt, v=0: np.full(n, v, dtype=dt)
    o = dict(nxt=z(np.int8, 0), nbef=z(np.int16), naft=z(np.int16), extexc=z(np.float32, np.nan),
             extfrac=z(np.float32, np.nan), length=z(np.int32), ttc=z(np.float32, np.nan),
             brk=z(np.int16))
    t = time.time()
    _arch(t0, c, e, north, w, high, low, ts, sess, cl,
          o['nxt'], o['nbef'], o['naft'], o['extexc'], o['extfrac'], o['length'], o['ttc'], o['brk'])
    secs = round(time.time() - t, 2)
    r = pd.DataFrame({'nbef': o['nbef'], 'naft': o['naft'], 'ext_exc': o['extexc'],
        'ext_frac': o['extfrac'], 'length': o['length'], 'ttc': o['ttc'], 'brk': o['brk'],
        'ok': o['nxt'], 'tf': f.tf_minutes.to_numpy().astype(np.int32),
        't0_ns': f.t0_ts_ns.to_numpy()})
    r.attrs['seconds'] = secs
    return r


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    terr = sys.argv[1] if len(sys.argv) > 1 else 'discovery'
    inst = sys.argv[2] if len(sys.argv) > 2 else 'NQ'
    w = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    r = build(inst, terr, w)
    r.to_parquet(OUT / f'arch_{inst}_{terr}_w{w}.parquet', index=False)
    a = r[r.ok == 1].copy()
    print(f'{inst}/{terr}/w{w}: {len(r)} certified films, аналитических {len(a)} (длина>={2*w+3}), walk {r.attrs["seconds"]}s')
    def dist(s, lbl):
        vc = s.value_counts(normalize=True).sort_index()
        top = {int(k): round(float(v), 3) for k, v in vc.head(8).items()}
        print(f'  {lbl}: {top}  (median={int(s.median())}, mean={s.mean():.2f})')
    print('ВСЕ завершённые возвраты:')
    dist(a.nbef.clip(upper=6), 'свингов ДО экстремума (0..6+)')
    dist(a.naft.clip(upper=6), 'свингов ПОСЛЕ экстремума до b (0..6+)')
    # самая частая пара
    pair = a.assign(nb=a.nbef.clip(upper=5), na=a.naft.clip(upper=5)).groupby(['nb', 'na']).size()
    pair = (pair / pair.sum()).sort_values(ascending=False)
    print('  ТОП-8 архитектур (nbef,naft): ' + ', '.join(f'({b},{aa})={v:.3f}' for (b, aa), v in pair.head(8).items()))
    print('  медиана длины фильма (баров):', int(a.length.median()), ' экстремум в среднем на', round(a.ext_frac.mean(), 2), 'пути')
    far = a[a.ext_exc >= 8.0]
    print(f'\nДАЛЬНИЕ возвраты (экстремум >=8 пт от b): {len(far)} ({len(far)/len(a):.3f} от всех)')
    dist(far.nbef.clip(upper=6), 'свингов ДО экстремума')
    dist(far.naft.clip(upper=6), 'свингов ПОСЛЕ до b')
    pairf = far.assign(nb=far.nbef.clip(upper=5), na=far.naft.clip(upper=5)).groupby(['nb', 'na']).size()
    pairf = (pairf / pairf.sum()).sort_values(ascending=False)
    print('  ТОП-8 архитектур дальних: ' + ', '.join(f'({b},{aa})={v:.3f}' for (b, aa), v in pairf.head(8).items()))
    print('  слом структуры на возврате (higher-high на спуске), доля возвратов с 0 сломов:',
          round((far.brk == 0).mean(), 3), ' median сломов:', int(far.brk.median()))
