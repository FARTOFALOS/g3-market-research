#!/usr/bin/env python3
"""Размер границы различимости: сколько остаточного пути лежит ВНУТРИ бара контакта.

УЗКОЕ УТВЕРЖДЕНИЕ, КОТОРОЕ ЗДЕСЬ МЕРЯЕТСЯ
=========================================
При замороженной семантике Film-1 и минутном OHLC нельзя без обусловливания на
исход отделить организацию пути ВНУТРИ бара контакта от самого прихода:
`order_within_touch_candle` объявлен неизвестным (FREEZE_080A §8), поэтому
high, low и close этого бара могут содержать движение уже ПОСЛЕ контакта.

Это ограничивает более сильный claim про организацию пути. Это НЕ утверждение,
что приход и остаточный путь неразделимы для Film-1 вообще, и НЕ закрывает
существование другой prefix-observable информации о пути ДО контакта.

ЧТО ИМЕННО ДЕЛАЕТ ГРАНИЦУ ЖЁСТКОЙ
=================================
У фильма, коснувшегося границы уже в баре `q+1`, чистых баров остаточного пути
нет ни одного: доступен только факт «контакт произошёл в этом баре», а порядок
внутри минуты неизвестен. Любой функционал, определённый на ВСЕХ живых фильмах
и не использующий движение после контакта, обязан присвоить этим фильмам
значение, зависящее только от прихода. Значит на этом подмножестве он от
прихода не независим по построению.

Граница жёсткая ровно настолько, насколько велико это подмножество и насколько
сильно оно различается между сравниваемыми группами. Это и меряется.

`first_observed_contact_pos` используется здесь как РЕТРОСПЕКТИВНОЕ описание
продолжения, а не как признак более раннего среза.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'work/082'
P081 = ROOT / 'work/081a'
sys.path.insert(0, str(HERE))
from contrast import load, AGES, HS, MIN_SIDE                          # noqa: E402


def attach_offset(d, inst, terr):
    f = pd.read_parquet(P081 / f'paths/films_{inst}_{terr}.parquet',
                        columns=['first_observed_contact_pos'])
    c = f.first_observed_contact_pos.to_numpy()[d.film.to_numpy()]
    off = np.where(c >= 0, c - d.q.to_numpy(), -1)
    return d.assign(contact_offset=off)


def group_split(d, feat, refine=None, nq=4, key='base_key'):
    """Те же низ/верх, что в cross(), чтобы доли считались на тех же группах."""
    d = d[d.state_readable & d[feat].notna()].copy()
    if refine is not None:
        d['_qq'] = (d.groupby(key, observed=True)[refine]
                    .transform(lambda s: pd.qcut(s, nq, labels=False, duplicates='drop')))
        d = d[d._qq.notna()]
        d['_key'] = d[key] + '#' + d._qq.astype(int).astype(str)
    else:
        d['_key'] = d[key]
    sid, keys = pd.factorize(d._key)
    x = d[feat].to_numpy(np.float64)
    med = pd.Series(x).groupby(sid).transform('median').to_numpy()
    lo, hi = x < med, x > med
    n_lo = np.bincount(sid[lo], minlength=len(keys))
    n_hi = np.bincount(sid[hi], minlength=len(keys))
    keep = (n_lo >= MIN_SIDE) & (n_hi >= MIN_SIDE)
    w = np.minimum(n_lo, n_hi).astype(float) * keep
    if w.sum() == 0:
        return None
    w /= w.sum()

    def share(mask, cond):
        s = np.bincount(sid[mask], cond[mask].astype(float), len(keys))
        c = np.bincount(sid[mask], minlength=len(keys)).astype(float)
        return float(w @ np.divide(s, c, out=np.zeros(len(keys)), where=c > 0))
    return d, sid, lo, hi, keep, share


def main(inst='NQ', terr='discovery'):
    res = {}
    for a in AGES:
        d = attach_offset(load(inst, terr, a), inst, terr)
        A = {}
        for h in HS:
            s = d[d[f'contact_{h}'] >= 0].copy()
            arrived = s[f'contact_{h}'] == 1
            # чистых баров остаточного пути до бара контакта
            clean = np.where(arrived, s.contact_offset - 1, h)
            A[f'h{h}'] = {
                'n_at_risk': int(len(s)),
                'arrived_share': round(float(arrived.mean()), 4),
                'arrived_in_bar_q1__zero_clean_bars': int((s.contact_offset == 1).sum()),
                'share_of_at_risk_with_zero_clean_bars':
                    round(float((s.contact_offset == 1).mean()), 4),
                'share_of_arrived_with_zero_clean_bars':
                    round(float((s.contact_offset[arrived] == 1).mean()), 4),
                'clean_bars_p50_all': float(np.median(clean)),
                'clean_bars_p50_arrived': float(np.median(clean[arrived.to_numpy()]))
                    if arrived.any() else None}
            # различается ли доля «нуля чистых баров» между группами retrace
            for tag, refine in (('no_control', None), ('sigma_fixed_nq4', 'prefix_sigma')):
                g = group_split(s, 'retraced_fraction', refine=refine, nq=4)
                if g is None:
                    A[f'h{h}'][f'zero_clean_by_retrace__{tag}'] = None
                    continue
                dd, sid, lo, hi, keep, share = g
                z = (dd.contact_offset == 1).to_numpy()
                A[f'h{h}'][f'zero_clean_by_retrace__{tag}'] = {
                    'low': round(share(lo, z), 4), 'high': round(share(hi, z), 4),
                    'delta': round(share(hi, z) - share(lo, z), 4),
                    'strata_kept': int(keep.sum())}
        res[f'age_{a}'] = A
        print(f"\n########## возраст {a}")
        for h in HS:
            x = A[f'h{h}']
            print(f"  h{h}: at_risk {x['n_at_risk']:>6}  дошли {x['arrived_share']:.4f}  "
                  f"из них контакт в баре q+1: {x['share_of_arrived_with_zero_clean_bars']:.4f}  "
                  f"(= {x['share_of_at_risk_with_zero_clean_bars']:.4f} всей популяции, "
                  f"{x['arrived_in_bar_q1__zero_clean_bars']} фильмов)")
            print(f"        чистых баров до контакта: медиана всех {x['clean_bars_p50_all']:.0f}, "
                  f"у дошедших {x['clean_bars_p50_arrived']}")
            for tag in ('no_control', 'sigma_fixed_nq4'):
                v = x[f'zero_clean_by_retrace__{tag}']
                if v:
                    print(f"        доля «ноль чистых баров», retrace {tag:>15}: "
                          f"low {v['low']:.4f} high {v['high']:.4f} Δ {v['delta']:+.4f} "
                          f"(страт {v['strata_kept']})")
    p = OUT / f'identify_{inst}_{terr}.json'
    p.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print('\n', p, 'written')


if __name__ == '__main__':
    main(*(sys.argv[1:] or ['NQ', 'discovery']))
