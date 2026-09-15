#!/usr/bin/env python3
"""Слой 082: популяция среза и полностью наблюдённое окно после него.

ПОПУЛЯЦИЯ — ТОЛЬКО ПРЕФИКС
==========================
`certified_fresh_until_pos_strict` определением популяции служить не может: у
сертифицированного фильма он равен `c − 1`, то есть построен из будущего
контакта. Проверка `FINDINGS_SEMANTICS.md` показала, что он и не эквивалентен
префиксному критерию — он строго консервативнее ровно на фильмах, у которых
промежуток ленты лежит перед самим баром T0 (дефект `nxt_any[t0]` вместо
`nxt_any[t0+1]`, 1 154 / 720 / 290 фильмов на возрастах 5 / 15 / 60).

Поэтому пригодность считается прямым просмотром и ничем иным:

    eligible(q) ⇔ q <= last
                  ∧ kind[j] == 0                для всех j ∈ (t0, q]
                  ∧ ¬(low[j] <= e <= high[j])   для всех j ∈ (t0, q]

Ординал 0 исключён, как в `first_exit_contact_v1`.

ЗАЧЕМ ОКНО, А НЕ ДЛИТЕЛЬНОСТЬ ДО КОНТАКТА
=========================================
В `strict` фильм теряет сертификацию на промежутке ленты — прежде всего на
границе сессии. Это цензура по часам, а не по рынку, и независимой считаться
не имеет права. 082 её не моделирует, а устраняет:

    at_risk(q, h) ⇔ kind[q+1 … q+h] ≡ 0

Внутри непрерывного окна вопрос «случился ли первый post-T0 контакт» разрешён
ТОЧНО для каждого фильма: цензуры в такой популяции нет. Цена конструкции —
смещённый состав, и он меряется отдельно, а не прячется.

ЧТО СЧИТАЕТСЯ И ОТ ЧЕГО
=======================
Всё будущее меряется от `close[q]` — опоры самого среза. `open[q+1]`, `gross`,
`mae_*` и семейство стопов сюда не входят: это слой исполнения, и нормировать
им описание рынка значит внести в него чужую entry-policy (DECLARE_082).

    contact_h   первый post-T0 контакт лежит в барах q+1 … q+h
    away_h      max хода ОТ границы относительно close[q] по окну
    toward_h    max хода К границе относительно close[q] по окну

Оба хода обрываются на баре контакта включительно: после конца Film-1
продолжения нет.

D_CLOSE <= 0 СОХРАНЯЕТСЯ И ПОМЕЧАЕТСЯ
=====================================
Живой Film-1 с закрытием по другую сторону границы существует (гэп через
уровень внутри непрерывной ленты, 0,63 % срезов возраста 5). Объект их
сохраняет; отказывают ЧТЕНИЯ состояния. Строка остаётся в слое с флагом
`state_readable`, а исключение происходит в сравнении и публикуется там.
"""
from __future__ import annotations
import os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit, prange

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = Path(os.environ.get('G3_082_OUT', str(ROOT / 'work/082')))
P081 = Path(os.environ.get('G3_081A_OUT', str(ROOT / 'work/081a')))
sys.path.insert(0, str(ROOT / 'base/080'))
from tape import presence_grid, gap_kinds, next_of_kind_fast            # noqa: E402

AGES = (5, 15, 60)
HS = (5, 15, 60)
NS_HOUR = 3_600_000_000_000
#: контрольные численности из `semantics.py` (независимый просмотр), NQ discovery
EXPECT_NQ_DISCOVERY = {5: 65269, 15: 39122, 60: 20212}


@njit(parallel=True, cache=True)
def _slice(t0s, es, north, a, hs, opn, high, low, close, kind, tape_left, last,
           ok, d_close, sigma, run_max, contact, away, toward, q_out):
    """Один срез возраста `a`; все окна `hs` за один проход по фильму."""
    nh = hs.size
    for i in prange(t0s.size):
        t0 = t0s[i]; e = es[i]; up = north[i]
        q = t0 + a
        if q > last:
            ok[i] = False
            continue
        # --- пригодность: только бары t0 … q ---
        alive = True
        s = 0.0; rmax = -np.inf
        dc = (close[t0] - e) if up else (e - close[t0])
        if dc > rmax:
            rmax = dc
        s += high[t0] - low[t0]
        for j in range(t0 + 1, q + 1):
            if kind[j] != 0:
                alive = False; break
            if low[j] <= e and e <= high[j]:
                alive = False; break
            s += high[j] - low[j]
            dc = (close[j] - e) if up else (e - close[j])
            if dc > rmax:
                rmax = dc
        if not alive:
            ok[i] = False
            continue
        ok[i] = True
        q_out[i] = q
        sigma[i] = s / (q - t0 + 1)
        run_max[i] = rmax
        c0 = close[q]
        d_close[i] = (c0 - e) if up else (e - c0)

        # --- окно q+1 … q+h. Доступность — свойство ленты, не будущего контакта ---
        tl = tape_left[q]
        aw = -np.inf; tw = -np.inf
        hit = -1
        j = q
        for k in range(nh):
            h = hs[k]
            if tl < h or q + h > last:
                contact[i, k] = -1          # окно не наблюдается целиком
                away[i, k] = np.nan
                toward[i, k] = np.nan
                continue
            while j < q + h and hit < 0:
                j += 1
                hi_ = high[j]; lo_ = low[j]
                if up:
                    if hi_ - c0 > aw:
                        aw = hi_ - c0
                    if c0 - lo_ > tw:
                        tw = c0 - lo_
                else:
                    if c0 - lo_ > aw:
                        aw = c0 - lo_
                    if hi_ - c0 > tw:
                        tw = hi_ - c0
                if lo_ <= e and e <= hi_:
                    hit = j                 # Film-1 кончился, дальше не смотрим
            contact[i, k] = 1 if hit >= 0 else 0
            away[i, k] = aw
            toward[i, k] = tw


def build(inst='NQ', terr='discovery'):
    m = ROOT / 'data/market' / inst
    opn = np.load(m / 'open.npy'); high = np.load(m / 'high.npy')
    low = np.load(m / 'low.npy'); close = np.load(m / 'close.npy')
    tsb = np.load(m / 'close_ts_utc_ns.npy')
    sess = np.load(m / 'session_id.npy')
    last = opn.size - 1

    grid, lo_, hi_ = presence_grid()
    kind, _ = gap_kinds(inst, grid, lo_, hi_)
    # tape_left[q] — сколько подряд идущих баров после q не отделены промежутком
    nxt = next_of_kind_fast(kind, (1, 2, 3))
    idx = np.arange(kind.size)
    nxt_after = np.empty(kind.size, dtype=np.int64)
    nxt_after[:-1] = nxt[1:]
    nxt_after[-1] = kind.size
    tape_left = np.minimum(nxt_after - idx - 1, last - idx).astype(np.int64)
    sess_start = np.searchsorted(sess, sess, 'left')

    f = pd.read_parquet(P081 / f'paths/films_{inst}_{terr}.parquet')
    t0 = f.t0_spine_pos.to_numpy().astype(np.int64)
    e = f.exit_boundary.to_numpy().astype(np.float64)
    north = (f.side.to_numpy() == 'north')
    hs = np.array(HS, dtype=np.int64)
    n = len(f)

    frames = {}
    for a in AGES:
        ok = np.zeros(n, np.bool_)
        d_close = np.full(n, np.nan, np.float64)
        sigma = np.full(n, np.nan, np.float64)
        run_max = np.full(n, np.nan, np.float64)
        q_out = np.zeros(n, np.int64)
        contact = np.full((n, len(HS)), -1, np.int8)
        away = np.full((n, len(HS)), np.nan, np.float64)
        toward = np.full((n, len(HS)), np.nan, np.float64)
        t = time.time()
        _slice(t0, e, north, a, hs, opn, high, low, close, kind, tape_left, last,
               ok, d_close, sigma, run_max, contact, away, toward, q_out)
        qq = q_out[ok]
        d = pd.DataFrame({
            'film': np.flatnonzero(ok),
            'riz_id': f.riz_id.to_numpy()[ok],
            'q': qq,
            'side': f.side.to_numpy()[ok],
            'tf_minutes': f.tf_minutes.to_numpy()[ok],
            'd_close': d_close[ok],
            'prefix_sigma': sigma[ok],
            'run_max': run_max[ok],
            'tape_left': tape_left[qq],
            'hour_utc': ((tsb[qq] // NS_HOUR) % 24).astype(np.int8),
            'sess_pos': (qq - sess_start[qq]).astype(np.int32),
            't0_day': f.t0_day.to_numpy()[ok],
            'via_F': (qq <= f.certified_fresh_until_pos_strict.to_numpy()[ok]),
        })
        for k, h in enumerate(HS):
            d[f'contact_{h}'] = contact[ok, k]
            d[f'away_{h}'] = away[ok, k]
            d[f'toward_{h}'] = toward[ok, k]
        #: чтения состояния определены только при положительном расстоянии
        d['state_readable'] = d.d_close > 0
        d['retraced_fraction'] = np.where(d.state_readable,
                                          1.0 - d.d_close / d.run_max, np.nan)
        d['n_bars'] = np.where(d.state_readable, d.d_close / d.prefix_sigma, np.nan)
        d.attrs['seconds'] = round(time.time() - t, 2)

        if inst == 'NQ' and terr == 'discovery':
            exp = EXPECT_NQ_DISCOVERY[a]
            assert len(d) == exp, f'популяция {len(d)} != независимого просмотра {exp}'
        frames[a] = d
        print(f'age {a}: {len(d)} eligible ({int((~d.state_readable).sum())} '
              f'с d_close<=0, {int((~d.via_F).sum())} вне q<=F), {d.attrs["seconds"]}s | '
              + '  '.join(f'h{h} at_risk {(d[f"contact_{h}"] >= 0).mean():.4f}'
                          for h in HS), flush=True)
    return frames


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    inst = sys.argv[1] if len(sys.argv) > 1 else 'NQ'
    terr = sys.argv[2] if len(sys.argv) > 2 else 'discovery'
    for a, d in build(inst, terr).items():
        p = OUT / f'win_{inst}_{terr}_a{a}.parquet'
        d.to_parquet(p, index=False, compression='zstd')
        print(p, f'{p.stat().st_size/1e6:.1f} MB')
