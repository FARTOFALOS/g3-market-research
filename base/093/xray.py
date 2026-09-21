#!/usr/bin/env python3
"""093 X-RAY — наложить развороты трафаретом и прочитать, где стоп, из истории.

Метод (слово трейдера): не выдумывать цену стопа и проверять, а наложить все
сцены друг на друга и посмотреть, докуда доходит цена у дошедших до `b` и где
не доходит. Стоп читается из наложения, а не назначается.

Здесь: та же recognition, что в swing.py (подтверждённый разворот от нового
дальнего экстремума, w=3), но БЕЗ стопа. Каждую сцену ведём от входа open[c+1]
до касания собственной `b` либо до закрытия NY. Пишем:
  reached            — дошла ли до `b` до закрытия;
  adv_pre_reach      — макс. ход ПРОТИВ входа СТРОГО до бара касания (сколько
                       heat держит победитель; для не дошедших — весь adverse);
  adv_full           — макс. ход против за всю сцену до резолюции;
  mfe                — макс. ход в пользу (к `b`);
  target             — путь входа до `b`;
  ext_stop           — расстояние до свинг-экстремума (+тик): стоп «за
                       экстремумом», с которым сравниваем историю.
Ничего не оптимизируем по P&L — сначала распределения.
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
from tape import presence_grid, gap_kinds                              # noqa: E402
from paths import load_films                                           # noqa: E402
from swing import _is_pivot, _count                                    # noqa: E402

TICK = {'NQ': 0.25, 'ES': 0.25, 'YM': 1.0}
REACHED, TIME_EXIT, UNKNOWN, NO_EXEC, MISSED = 0, 1, 2, 5, 6


@njit(parallel=True, cache=True)
def _xray(t0s, Fs, ends, es, north, off, w,
          opn, high, low, close, sess, last_bar, kind_gap, tick,
          o_kind, o_g, o_extstop, o_advpre, o_advfull, o_mfe, o_ttc, o_dur,
          o_first, o_age, o_texit, close_ns, ts):
    for i in prange(t0s.size):
        t0 = t0s[i]; F = Fs[i]; end_f = ends[i]; e = es[i]; up = north[i]
        base = off[i]; ext = -np.inf if up else np.inf
        idx = 0; first = 1
        for p in range(t0, F + 1):
            if p >= t0 + w and p + w <= F and _is_pivot(high, low, p, w, up, ext):
                pos = base + idx; idx += 1
                c = p + w; j = c + 1; q = c
                o_first[pos] = first; first = 0
                o_age[pos] = q - t0
                sq = sess[q]
                if sq >= 0:
                    o_ttc[pos] = (close_ns[sq] - ts[q]) / 6.0e10
                if j > F or kind_gap[j] != 0 or sess[j] < 0:
                    o_kind[pos] = NO_EXEC
                else:
                    sj = sess[j]; o = opn[j]
                    g = (o - e) if up else (e - o)
                    extlvl = (high[p] + tick) if up else (low[p] - tick)
                    es_dist = (extlvl - o) if up else (o - extlvl)
                    if g <= 0.0:
                        o_kind[pos] = MISSED
                    else:
                        lb = last_bar[sj]; limit = lb if lb < end_f else end_f
                        o_g[pos] = g; o_extstop[pos] = es_dist
                        run_adv = 0.0; adv_pre = 0.0; run_fav = 0.0
                        res = UNKNOWN; dur = 0
                        for b in range(j, limit + 1):
                            if up:
                                adv = high[b] - o; fav = o - low[b]
                                tp = low[b] <= e
                            else:
                                adv = o - low[b]; fav = high[b] - o
                                tp = high[b] >= e
                            if adv > run_adv:
                                run_adv = adv
                            if fav > run_fav:
                                run_fav = fav
                            if tp:
                                res = REACHED; dur = b - q
                                break
                            # бар не коснулся b: его adverse входит в adv_pre
                            if adv > adv_pre:
                                adv_pre = adv
                        if res == REACHED:
                            o_kind[pos] = REACHED
                        elif limit == lb:
                            res = TIME_EXIT; o_kind[pos] = TIME_EXIT
                            dur = lb - q; adv_pre = run_adv
                            px = close[lb]
                            o_texit[pos] = (o - px) if up else (px - o)
                        else:
                            o_kind[pos] = UNKNOWN; dur = limit - q; adv_pre = run_adv
                        o_advpre[pos] = adv_pre; o_advfull[pos] = run_adv
                        o_mfe[pos] = run_fav; o_dur[pos] = dur
            v = high[p] if up else low[p]
            if up:
                if v > ext:
                    ext = v
            else:
                if v < ext:
                    ext = v


def build(inst='NQ', terr='discovery', w=3):
    m = ROOT / 'data/market' / inst
    opn = np.load(m / 'open.npy'); high = np.load(m / 'high.npy')
    low = np.load(m / 'low.npy'); close = np.load(m / 'close.npy')
    ts = np.load(m / 'close_ts_utc_ns.npy')
    grid, glo, ghi = presence_grid()
    kind_gap, _ = gap_kinds(inst, grid, glo, ghi)
    st, cl = sessions(); cl = cl.astype(np.int64)
    sess, last_bar = bar_session_map(ts, st, cl)
    f = load_films(inst, terr)
    t0 = f.t0_spine_pos.to_numpy().astype(np.int64)
    F = f.certified_fresh_until_pos_strict.to_numpy().astype(np.int64)
    c = f.first_observed_contact_pos.to_numpy().astype(np.int64)
    cert = (f.film1_status.to_numpy() == 'contact_certified')
    end = np.where(cert, c, F)
    e = f.exit_boundary.to_numpy().astype(np.float64)
    north = (f.side.to_numpy() == 'north')
    cnt = np.zeros(len(f), np.int64)
    _count(t0, F, north, w, high, low, cnt)
    off = np.concatenate(([0], np.cumsum(cnt))); total = int(off[-1])
    z = lambda dt, v=0: np.full(total, v, dtype=dt)
    o = dict(kind=z(np.int8, NO_EXEC), g=z(np.float32, np.nan), extstop=z(np.float32, np.nan),
             advpre=z(np.float32, np.nan), advfull=z(np.float32, np.nan), mfe=z(np.float32, np.nan),
             ttc=z(np.float32, np.nan), dur=z(np.int32), first=z(np.int8), age=z(np.int32),
             texit=z(np.float32, np.nan))
    t = time.time()
    _xray(t0, F, end, e, north, off[:-1], w,
          opn, high, low, close, sess, last_bar, kind_gap, TICK[inst],
          o['kind'], o['g'], o['extstop'], o['advpre'], o['advfull'], o['mfe'],
          o['ttc'], o['dur'], o['first'], o['age'], o['texit'], cl, ts)
    secs = round(time.time() - t, 2)
    fi = np.repeat(np.arange(len(f)), cnt)
    entry_bar = np.clip(t0[fi] + o['age'] + 1, 0, len(ts) - 1)
    day = np.where(o['kind'] <= UNKNOWN, ts[entry_bar] // 86400000000000, -1)
    r = pd.DataFrame({'film': fi.astype(np.int32), 'riz_id': f.riz_id.to_numpy()[fi],
        'day': day.astype(np.int64),
        'first': o['first'], 'age': o['age'], 'kind': o['kind'], 'target': o['g'],
        'ext_stop': o['extstop'], 'adv_pre': o['advpre'], 'adv_full': o['advfull'],
        'mfe': o['mfe'], 'ttc': o['ttc'], 'dur': o['dur'], 'texit': o['texit'],
        't0_ns': f.t0_ts_ns.to_numpy()[fi]})
    r.attrs['seconds'] = secs
    return r


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    terr = sys.argv[1] if len(sys.argv) > 1 else 'discovery'
    inst = sys.argv[2] if len(sys.argv) > 2 else 'NQ'
    w = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    r = build(inst, terr, w)
    r.to_parquet(OUT / f'xray_{inst}_{terr}_w{w}.parquet', index=False)
    print(f'{inst}/{terr}/w{w}: {len(r)} pivots, walk {r.attrs["seconds"]}s')
    tr = r[r.kind.isin([REACHED, TIME_EXIT, UNKNOWN])].copy()
    reached = tr[tr.kind == REACHED]; notr = tr[tr.kind != REACHED]
    print(f'entries={len(tr)}  reached_b_before_close={ (tr.kind==REACHED).mean():.3f}')
    print('\n=== ТРАФАРЕТ: ход ПРОТИВ входа до касания b (пункты NQ) ===')
    qs = [.5, .75, .9, .95]
    def perc(s, lbl):
        v = s.to_numpy(); v = v[~np.isnan(v)]
        print(f'  {lbl:>22}: n={len(v):>6} median={np.median(v):5.1f}  p75={np.quantile(v,.75):5.1f}  p90={np.quantile(v,.9):6.1f}  p95={np.quantile(v,.95):6.1f}')
    perc(reached.adv_pre, 'ДОШЕДШИЕ до b (heat)')
    perc(notr.adv_full, 'НЕ дошедшие (adverse)')
    perc(reached.target, 'target ДОШЕДШИХ')
    perc(reached.ext_stop, 'стоп-за-экстремумом')
    # какая доля дошедших пережила бы стоп за экстремумом
    ok = reached.adv_pre <= reached.ext_stop
    print(f'\n  доля дошедших, чей heat <= стопа-за-экстремумом: {ok.mean():.3f}  (значит extreme+тик выбивает {1-ok.mean():.0%} победителей)')
    # трафарет: если стоп = X пунктов, сколько победителей уцелело и сколько проигравших отсеклось
    print('\n=== если стоп = X пунктов против входа (читаем по истории) ===')
    print('   X | winners_kept | losers_cut | trades_left | реализуемо на дошедших')
    for X in [4, 6, 8, 10, 15, 20, 30, 50]:
        wk = (reached.adv_pre <= X).mean()
        lc = (notr.adv_full > X).mean()
        print(f'  {X:>3} |    {wk:.3f}     |   {lc:.3f}    |      -      |')
