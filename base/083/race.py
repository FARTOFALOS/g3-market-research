#!/usr/bin/env python3
"""Слой 083: гонка собственной границы Film-1 против равноудалённого зеркала.

ВОПРОС
======
Приход живого Film-1 к собственной `exit_boundary` быстрее — это направленная
избирательность или общее ускорение цены? Ускорение приближает оба уровня
одинаково, поэтому равноудалённое зеркало делит эти объяснения по построению.

ЗЕРКАЛО
=======
    d  = d_close > 0                 расстояние от close[q] до границы
    m  = close[q] + d  (north)       граница у north лежит СНИЗУ
    m  = close[q] − d  (south)       граница у south лежит СВЕРХУ

`m = 2·close[q] − e` лежит на тиковой сетке, раз на ней лежат оба слагаемых.
Фиксируется один раз в `q`; не двигается, не переякоривается, не оптимизируется.

ОДИН ПРЕДИКАТ НА ОБА УРОВНЯ
===========================
    touch(j, L) ⇔ low[j] <= L <= high[j]          FREEZE_080A §2, заморожен

Гэп через уровень касанием не является ни для `e`, ни для `m`, и разрешением
гонки не считается. Он остаётся отдельным наблюдением (`gap_e`, `gap_m`).

ПОПУЛЯЦИЯ — ТОЛЬКО ПРЕФИКС, ОКНО ФИЛЬТРОМ НЕ РАБОТАЕТ
=====================================================
    eligible(q) ⇔ q <= last
                  ∧ kind[j] == 0                для всех j ∈ (t0, q]
                  ∧ ¬(low[j] <= e <= high[j])   для всех j ∈ (t0, q]

Неполное окно Film НЕ исключает. Если лента обрывается до разрешения гонки и
до конца бюджета, это исход `lost_observability` внутри той же префиксной
популяции. Единственное префиксное исключение — `d_close <= 0` (гэп через
уровень в `q`, DECLARE_083 §2): там не определён знак направления к границе,
значит не определено и «противоположная сторона». Численность публикуется.

ПЯТЬ ИСХОДОВ
============
    0  unresolved          лента цела все h баров, ни один уровень не накрыт
    1  boundary_first      первый накрывший бар накрыл только e
    2  mirror_first        первый накрывший бар накрыл только m
    3  same_bar_ambiguous  один бар накрыл оба, порядок внутри минуты не виден
    4  lost_observability  лента оборвалась раньше h и раньше разрешения

Сумма по пяти категориям равна всей популяции для каждого h.

ОДИН ПРОХОД
===========
Окна 5 / 15 / 60 вложены, поэтому проход делается один — до min(60, длина
непрерывной ленты). Исход для каждого h выводится из позиции разрешения и
длины доступной ленты, а не из отдельного прохода.
"""
from __future__ import annotations
import os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from numba import njit, prange

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = Path(os.environ.get('G3_083_OUT', str(ROOT / 'work/083')))
P081 = Path(os.environ.get('G3_081A_OUT', str(ROOT / 'work/081a')))
sys.path.insert(0, str(ROOT / 'base/080'))
from tape import presence_grid, gap_kinds, next_of_kind_fast            # noqa: E402

AGES = (5, 15, 60)
HS = (5, 15, 60)
HMAX = max(HS)
NS_HOUR = 3_600_000_000_000
#: контрольные численности prefix-only пригодности, NQ discovery (082 semantics.py)
EXPECT_NQ_DISCOVERY = {5: 65269, 15: 39122, 60: 20212}


@njit(parallel=True, cache=True)
def _race(t0s, es, north, a, high, low, close, kind, tape_left, last,
          ok, d_close, sigma, run_max, q_out, mirror, pre_m_touch,
          res_code, res_bars, walkable, gap_e, gap_m):
    for i in prange(t0s.size):
        t0 = t0s[i]; e = es[i]; up = north[i]
        q = t0 + a
        if q > last:
            ok[i] = False
            continue
        # --- пригодность: прямой просмотр баров t0 … q, ничего из будущего ---
        alive = True
        s = high[t0] - low[t0]
        rmax = (close[t0] - e) if up else (e - close[t0])
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
        d = (c0 - e) if up else (e - c0)
        d_close[i] = d

        if d <= 0.0:                      # гэп через уровень в q: зеркало не определено
            mirror[i] = np.nan
            res_code[i] = -1
            continue
        m = (c0 + d) if up else (c0 - d)
        mirror[i] = m

        # --- префиксная диагностика: накрыт ли зеркальный уровень уже в t0 … q ---
        pm = False
        for j in range(t0, q + 1):
            if low[j] <= m and m <= high[j]:
                pm = True; break
        pre_m_touch[i] = pm

        # --- гонка. Длина прохода — свойство ленты, не будущего исхода ---
        w = tape_left[q]
        if w > HMAX:
            w = HMAX
        if q + w > last:
            w = last - q
        walkable[i] = w

        code = 0; nb = 0
        ge = False; gm = False
        for k in range(1, w + 1):
            j = q + k
            hi_ = high[j]; lo_ = low[j]
            tb = lo_ <= e and e <= hi_
            tm = lo_ <= m and m <= hi_
            if tb and tm:
                code = 3; nb = k; break
            if tb:
                code = 1; nb = k; break
            if tm:
                code = 2; nb = k; break
            # гэп через уровень без накрывающего бара — отдельное наблюдение
            if up:
                if hi_ < e:
                    ge = True
                if lo_ > m:
                    gm = True
            else:
                if lo_ > e:
                    ge = True
                if hi_ < m:
                    gm = True
        res_code[i] = code
        res_bars[i] = nb
        gap_e[i] = ge
        gap_m[i] = gm


def build(inst='NQ', terr='discovery'):
    m = ROOT / 'data/market' / inst
    high = np.load(m / 'high.npy'); low = np.load(m / 'low.npy')
    close = np.load(m / 'close.npy'); tsb = np.load(m / 'close_ts_utc_ns.npy')
    sess = np.load(m / 'session_id.npy')
    last = close.size - 1

    grid, lo_, hi_ = presence_grid()
    kind, _ = gap_kinds(inst, grid, lo_, hi_)
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
    n = len(f)

    frames = {}
    for a in AGES:
        ok = np.zeros(n, np.bool_)
        d_close = np.full(n, np.nan, np.float64)
        sigma = np.full(n, np.nan, np.float64)
        run_max = np.full(n, np.nan, np.float64)
        mirror = np.full(n, np.nan, np.float64)
        q_out = np.zeros(n, np.int64)
        pre_m = np.zeros(n, np.bool_)
        res_code = np.full(n, -9, np.int8)
        res_bars = np.zeros(n, np.int32)
        walkable = np.zeros(n, np.int32)
        gap_e = np.zeros(n, np.bool_); gap_m = np.zeros(n, np.bool_)
        t = time.time()
        _race(t0, e, north, a, high, low, close, kind, tape_left, last,
              ok, d_close, sigma, run_max, q_out, mirror, pre_m,
              res_code, res_bars, walkable, gap_e, gap_m)
        qq = q_out[ok]
        d = pd.DataFrame({
            'film': np.flatnonzero(ok),
            'riz_id': f.riz_id.to_numpy()[ok],
            'q': qq,
            'side': f.side.to_numpy()[ok],
            'tf_minutes': f.tf_minutes.to_numpy()[ok],
            'exit_boundary': e[ok],
            'mirror': mirror[ok],
            'd_close': d_close[ok],
            'prefix_sigma': sigma[ok],
            'run_max': run_max[ok],
            'prefix_mirror_touched': pre_m[ok],
            'res_code': res_code[ok],
            'res_bars': res_bars[ok],
            'walkable': walkable[ok],
            'gap_e': gap_e[ok],
            'gap_m': gap_m[ok],
            'hour_utc': ((tsb[qq] // NS_HOUR) % 24).astype(np.int8),
            'sess_pos': (qq - sess_start[qq]).astype(np.int32),
            't0_day': f.t0_day.to_numpy()[ok],
        })
        d['state_readable'] = d.d_close > 0
        d['retraced_fraction'] = np.where(d.state_readable,
                                          1.0 - d.d_close / d.run_max, np.nan)
        d['n_bars'] = np.where(d.state_readable, d.d_close / d.prefix_sigma, np.nan)
        # --- исход на каждый бюджет: окно фильтром НЕ работает ---
        for h in HS:
            o = np.full(len(d), -1, np.int8)                 # -1 = вне контраста (d<=0)
            sr = d.state_readable.to_numpy()
            rc = d.res_code.to_numpy(); rb = d.res_bars.to_numpy()
            wk = d.walkable.to_numpy()
            resolved_in = sr & (rc > 0) & (rb <= h)
            o[resolved_in] = rc[resolved_in]                 # 1 / 2 / 3
            rest = sr & ~resolved_in
            o[rest & (wk >= h)] = 0                          # unresolved
            o[rest & (wk < h)] = 4                           # lost_observability
            d[f'outcome_{h}'] = o
        d.attrs['seconds'] = round(time.time() - t, 2)
        if inst == 'NQ' and terr == 'discovery':
            assert len(d) == EXPECT_NQ_DISCOVERY[a], \
                f'популяция {len(d)} != независимого просмотра 082'
        frames[a] = d
        N = int(d.state_readable.sum())
        print(f'age {a}: eligible {len(d)}, вне контраста d<=0 {len(d)-N}, N={N}, '
              f'{d.attrs["seconds"]}s', flush=True)
        for h in HS:
            o = d[f'outcome_{h}'].to_numpy()
            c = [int((o == k).sum()) for k in (1, 2, 3, 0, 4)]
            print(f'   h{h}: boundary {c[0]}  mirror {c[1]}  ambiguous {c[2]}  '
                  f'unresolved {c[3]}  lost {c[4]}  сумма {sum(c)}', flush=True)
    return frames


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    inst = sys.argv[1] if len(sys.argv) > 1 else 'NQ'
    terr = sys.argv[2] if len(sys.argv) > 2 else 'discovery'
    for a, d in build(inst, terr).items():
        p = OUT / f'race_{inst}_{terr}_a{a}.parquet'
        d.to_parquet(p, index=False, compression='zstd')
        print(p, f'{p.stat().st_size/1e6:.1f} MB')
