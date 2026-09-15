#!/usr/bin/env python3
"""Четыре шага 082 в объявленном порядке, на полностью наблюдённом окне.

    1. воспроизводится ли различие времени вообще;
    2. сколько его объясняет масштаб префикса (и отдельно — час суток);
    3. остаётся ли различие в остаточном ценовом ходе, а не только во времени;
    4. не произведено ли различие отбором и наблюдаемостью.

СТРАТА
======
Ключ `(d_close в тиках, сторона, полоса ТФ)` внутри возраста — точное
совпадение текущего положения, как в `base/081/control.py`. Опора
`MIN_SIDE = 30` в КАЖДОЙ группе; страта без опоры выбрасывается целиком, и
потерянное население публикуется.

Cross-test добавляет к ключу квартиль второго чтения, вычисленный ВНУТРИ
страты. Это фиксирует второе чтение и оставляет первому его изменчивость.

КАК ОТЛИЧИТЬ ОБЪЯСНЕНИЕ ОТ ПОТЕРИ КОНТРАСТА
===========================================
Если Δ отклика падает внутри квартилей второго чтения, это может значить
«второе чтение объясняет первое», а может — «ячейки стали такими узкими, что
у первого чтения не осталось собственной изменчивости». Различить их по Δ и CI
невозможно, поэтому рядом с каждым Δ публикуется:

    feature_sep     разность САМОГО делящего признака между high и low
    feature_sep_sd  та же разность в единицах его стандартного отклонения
    residual_rho    остаточная ранговая связь retraced ↔ σ внутри ячеек
    rows_used / support_retained / strata_kept

Объяснение выглядит как сохранившийся `feature_sep` при упавшем Δ отклика.
Потеря контраста выглядит как упавший вместе с Δ `feature_sep`.

ЗАВИСИМОСТЬ СТРОК
=================
Одна минута T0 зажигает кластер зон разных ТФ, поэтому строки не независимы.
Основной CI — блочный бутстрэп ПО ДНЯМ `t0_day`. Он снимает зависимость ВНУТРИ
дня и НЕ снимает междневную: соседние дни делят режим волатильности. Поэтому
рядом считается sensitivity скользящими блоками подряд идущих дней длины 5 и
20, и в выводе берётся самый широкий интервал. Утверждения без этой оговорки
не делаются.

Веса страт фиксируются по полным данным, чтобы оцениваемая величина не менялась
от реплики к реплике.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'work/082'

AGES = (5, 15, 60)
HS = (5, 15, 60)
TICK = {'NQ': 0.25, 'ES': 0.25, 'YM': 1.0}
TF_EDGES = [0, 2, 5, 15, 60, 240, 1441]
TF_LAB = ['1-2', '3-5', '6-15', '16-60', '61-240', '241-1440']
MIN_SIDE = 30
NBOOT = 400
BLOCKS = (1, 5, 20)          # длина блока подряд идущих дней
RNG = np.random.default_rng(20820915)


def load(inst, terr, a):
    d = pd.read_parquet(OUT / f'win_{inst}_{terr}_a{a}.parquet')
    d['tfb'] = pd.cut(d.tf_minutes, TF_EDGES, labels=TF_LAB).astype(str)
    d['dct'] = np.round(d.d_close / TICK[inst]).astype(int)
    d['base_key'] = d.dct.astype(str) + '|' + d.side + '|' + d.tfb
    return d


def _day_multipliers(uday_n, block, nboot, rng):
    """Матрица кратностей дней (n_day × nboot). block=1 — обычный по дням."""
    if block == 1:
        idx = rng.integers(0, uday_n, (uday_n, nboot))
    else:
        nb = int(np.ceil(uday_n / block))
        starts = rng.integers(0, max(uday_n - block + 1, 1), (nb, nboot))
        idx = (starts[:, None, :] + np.arange(block)[None, :, None]
               ).reshape(-1, nboot) % uday_n
    M = np.zeros((uday_n, nboot))
    for b in range(nboot):
        M[:, b] = np.bincount(idx[:, b], minlength=uday_n)
    return M


def contrast(d, feat, resp, key='base_key', refine=None, nq=4, other=None):
    """Взвешенная разность `high − low` по стратам + диагностика контраста."""
    d = d[d.state_readable & d[resp].notna() & d[feat].notna()].copy()
    if refine is not None:
        d['_qq'] = (d.groupby(key, observed=True)[refine]
                    .transform(lambda s: pd.qcut(s, nq, labels=False, duplicates='drop')))
        d = d[d._qq.notna()]
        d['_key'] = d[key] + '#' + d._qq.astype(int).astype(str)
    else:
        d['_key'] = d[key]
    if len(d) == 0:
        return {'strata_kept': 0, 'note': 'пусто'}

    sid, keys = pd.factorize(d._key)
    nk = len(keys)
    y = d[resp].to_numpy(np.float64)
    x = d[feat].to_numpy(np.float64)
    # низ/верх по медиане ВНУТРИ ячейки; равные медиане отброшены
    med = pd.Series(x).groupby(sid).transform('median').to_numpy()
    lo_m, hi_m = x < med, x > med

    n_lo = np.bincount(sid[lo_m], minlength=nk)
    n_hi = np.bincount(sid[hi_m], minlength=nk)
    keep = (n_lo >= MIN_SIDE) & (n_hi >= MIN_SIDE)
    if not keep.any():
        return {'strata_kept': 0, 'strata_total': int(nk),
                'rows_total': int(len(d)), 'note': 'опоры нет'}
    w = np.minimum(n_lo, n_hi).astype(float) * keep
    w = w / w.sum()

    def wmean(mask, v):
        s = np.bincount(sid[mask], v[mask], nk)
        c = np.bincount(sid[mask], minlength=nk).astype(float)
        m = np.divide(s, c, out=np.zeros(nk), where=c > 0)
        return float(w @ m)

    ml, mh = wmean(lo_m, y), wmean(hi_m, y)
    fl, fh = wmean(lo_m, x), wmean(hi_m, x)
    # разделение самого признака в единицах его sd внутри ячейки
    sd_in = pd.Series(x).groupby(sid).transform('std').to_numpy()
    good = np.isfinite(sd_in) & (sd_in > 0)
    sep_sd = float('nan')
    if good.any():
        z = np.zeros_like(x)
        z[good] = x[good] / sd_in[good]
        sep_sd = wmean(hi_m & good, z) - wmean(lo_m & good, z)

    # --- блочный бутстрэп по дням; все реплики одной матрицей ---
    days = d.t0_day.to_numpy()
    uday, dcode = np.unique(days, return_inverse=True)
    rows = np.arange(len(d))
    cis = {}
    for blk in BLOCKS:
        M = _day_multipliers(uday.size, blk, NBOOT, RNG)
        out = {}
        for tag, mask in (('lo', lo_m), ('hi', hi_m)):
            r = rows[mask]
            S = sparse.csr_matrix((y[mask], (sid[mask], dcode[mask])),
                                  shape=(nk, uday.size))
            C = sparse.csr_matrix((np.ones(r.size), (sid[mask], dcode[mask])),
                                  shape=(nk, uday.size))
            out[tag] = (S @ M, C @ M)
        (slo, clo), (shi, chi) = out['lo'], out['hi']
        ok = keep[:, None] & (clo > 0) & (chi > 0)
        mlo = np.divide(slo, clo, out=np.zeros_like(slo), where=clo > 0)
        mhi = np.divide(shi, chi, out=np.zeros_like(shi), where=chi > 0)
        ww = w[:, None] * ok
        tot = ww.sum(0)
        val = np.where(tot > 0, (ww * (mhi - mlo)).sum(0) / np.maximum(tot, 1e-12), np.nan)
        val = val[np.isfinite(val)]
        cis[f'block{blk}d'] = [round(float(np.quantile(val, .025)), 5),
                              round(float(np.quantile(val, .975)), 5)]
    widest = max(cis.values(), key=lambda c: c[1] - c[0])

    res = {
        'strata_kept': int(keep.sum()), 'strata_total': int(nk),
        'rows_used': int((lo_m | hi_m)[keep[sid]].sum()), 'rows_total': int(len(d)),
        'support_retained': round(float((lo_m | hi_m)[keep[sid]].sum() / len(d)), 4),
        'days': int(uday.size),
        'mean_low': round(ml, 5), 'mean_high': round(mh, 5), 'delta': round(mh - ml, 5),
        'ci95_widest': widest, 'ci95_by_block': cis,
        'feature_sep': round(fh - fl, 5), 'feature_sep_sd': round(sep_sd, 4)}
    if other is not None:
        o = d[other].to_numpy(np.float64)
        s = pd.DataFrame({'k': sid, 'a': x, 'b': o})
        rr = (s.groupby('k').apply(
            lambda g: g.a.corr(g.b, method='spearman') if len(g) > 30 else np.nan,
            include_groups=False))
        rr = rr[keep[rr.index.to_numpy()]] if len(rr) else rr
        res['residual_rho'] = (round(float(np.nanmedian(rr)), 4)
                               if len(rr) and np.isfinite(rr).any() else None)
    return res


def main(inst='NQ', terr='discovery'):
    res = {'instrument': inst, 'territory': terr,
           'note': 'возрасты — вложенные популяции доживших, не объединяются; '
                   'CI day-clustered, блоки 1/5/20 дней, публикуется самый широкий'}
    for a in AGES:
        d = load(inst, terr, a)
        A = {'eligible': int(len(d)),
             'state_unreadable_d_close_le_0': int((~d.state_readable).sum()),
             'outside_q_le_F': int((~d.via_F).sum())}

        # ---------- ШАГ 4: наблюдаемость обусловливает всё остальное ----------
        obs = {}
        for h in HS:
            ar = (d[f'contact_{h}'] >= 0)
            by_h = ar.groupby(d.hour_utc).mean()
            obs[f'h{h}'] = {
                'at_risk_share': round(float(ar.mean()), 4),
                'at_risk_by_hour_min': round(float(by_h.min()), 4),
                'at_risk_by_hour_max': round(float(by_h.max()), 4),
                'sess_pos_p50_all': float(d.sess_pos.median()),
                'sess_pos_p50_at_risk': float(d.sess_pos[ar].median()),
                'retraced_p50_all': round(float(d.retraced_fraction.median()), 4),
                'retraced_p50_at_risk': round(float(d.retraced_fraction[ar].median()), 4),
                'sigma_p50_all': round(float(d.prefix_sigma.median()), 4),
                'sigma_p50_at_risk': round(float(d.prefix_sigma[ar].median()), 4),
                'd_close_p50_all': round(float(d.d_close.median()), 4),
                'd_close_p50_at_risk': round(float(d.d_close[ar].median()), 4)}
        A['step4_observability'] = obs

        # ---------- ШАГИ 1-3 ----------
        for h in HS:
            sub = d[d[f'contact_{h}'] >= 0].copy()
            sub['rho'] = sub[f'away_{h}'] / sub.d_close
            sub['away_sig'] = sub[f'away_{h}'] / sub.prefix_sigma
            sub['appr'] = sub[f'toward_{h}'] / sub.d_close
            cell = {}
            for resp, tag in ((f'contact_{h}', 'time__contact'),
                              ('rho', 'price__away_over_dclose'),
                              ('away_sig', 'price__away_over_sigma'),
                              ('appr', 'price__toward_over_dclose')):
                hk = sub.assign(hk=sub.base_key + '|' + sub.hour_utc.astype(str))
                cell[tag] = {
                    'by_retraced': contrast(sub, 'retraced_fraction', resp,
                                            other='prefix_sigma'),
                    'by_sigma': contrast(sub, 'prefix_sigma', resp,
                                         other='retraced_fraction'),
                    'by_retraced__sigma_fixed': contrast(
                        sub, 'retraced_fraction', resp, refine='prefix_sigma',
                        other='prefix_sigma'),
                    'by_sigma__retraced_fixed': contrast(
                        sub, 'prefix_sigma', resp, refine='retraced_fraction',
                        other='retraced_fraction'),
                    'by_retraced__hour_fixed': contrast(
                        hk, 'retraced_fraction', resp, key='hk'),
                    'by_sigma__hour_fixed': contrast(
                        hk, 'prefix_sigma', resp, key='hk')}
            A[f'h{h}'] = cell
            print(f'  age {a} h {h} готов', flush=True)
        res[f'age_{a}'] = A

    p = OUT / f'contrast_{inst}_{terr}.json'
    p.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print(p, 'written')
    return res


if __name__ == '__main__':
    main(*(sys.argv[1:] or ['NQ', 'discovery']))
