#!/usr/bin/env python3
"""Хронологический поток сделок с правилом конфликта (FREEZE_081A §14).

Одновременно одна позиция на инструмент. На совпавшей минуте входа разные
стороны пропускаются; для одинаковой стороны берётся самый ранний T0, затем
меньший ТФ, затем riz_id. Выбор не зависит от будущего и от прибыли.
Несколько riz_id с одной стороной, одним баром входа и одной границей — одно
исполнение; все riz_id сохраняются.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from risk_family import score_stop, COST, WIN, LOSS, AMBIG, UNKNOWN     # noqa: E402
from family2 import load                                               # noqa: E402
from evaluate import FROZEN                                            # noqa: E402

OUT = ROOT / 'work/081a'
PV = {'NQ': 20.0, 'ES': 50.0, 'YM': 5.0}      # $ за пункт, стандартные множители


def stream(inst, terr, name):
    p = FROZEN[name]
    q, f = load(inst, terr)
    c = COST[inst]
    cond = (q.age.to_numpy() >= p['age_min']) & (q.d_close.to_numpy() > p['d_mult'] * c)
    if p['theta'] is not None:
        cond &= (q.retraced_fraction.to_numpy() >= p['theta'])
    trig = q[cond].groupby('film', sort=False).head(1)
    trig = trig[trig.usable]
    fi = trig.film.to_numpy()
    d = trig.assign(riz_id=f.riz_id.to_numpy()[fi], side=f.side.to_numpy()[fi],
                    tf=f.tf_minutes.to_numpy()[fi], t0=f.t0_spine_pos.to_numpy()[fi],
                    e=f.exit_boundary.to_numpy()[fi],
                    entry_bar=None)
    d['entry_bar'] = f.t0_spine_pos.to_numpy()[fi] + d.age.to_numpy() + 1
    d['end_bar'] = f.end_pos.to_numpy()[fi]
    kind, lo, hi = score_stop(d, p['k'], inst)
    g = d.gross.to_numpy(np.float64); ma = d.mae_bound.to_numpy(np.float64)
    d = d.assign(kind=kind, lower=np.where(np.isnan(lo), -ma - c, lo),
                 upper=np.where(np.isnan(hi), g - c, hi))

    # агрегирование одинаковых экономических сделок
    d = d.sort_values(['entry_bar', 't0', 'tf', 'riz_id'])
    grp = d.groupby(['entry_bar', 'side', 'e'], sort=False)
    agg = grp.head(1).copy()
    agg['n_riz_aggregated'] = grp.size().reindex(
        pd.MultiIndex.from_arrays([agg.entry_bar, agg.side, agg.e])).to_numpy()

    # правило конфликта: одна позиция на инструмент
    agg = agg.sort_values(['entry_bar', 't0', 'tf', 'riz_id'])
    eb = agg.entry_bar.to_numpy(); xb = agg.end_bar.to_numpy()
    side = agg.side.to_numpy()
    take = np.zeros(len(agg), bool); busy_until = -1; busy_side = None
    skipped_conflict_side = 0
    for i in range(len(agg)):
        if eb[i] > busy_until:
            take[i] = True; busy_until = xb[i]; busy_side = side[i]
        elif side[i] != busy_side:
            skipped_conflict_side += 1
    sel = agg[take]
    return d, agg, sel, skipped_conflict_side


def report(inst, terr, name):
    d, agg, sel, sk = stream(inst, terr, name)
    pv = PV[inst]
    out = {'instrument': inst, 'territory': terr, 'policy': name,
           'signals_before_aggregation': int(len(d)),
           'after_aggregation': int(len(agg)),
           'after_conflict_rule': int(len(sel)),
           'skipped_conflict_opposite_side': int(sk),
           'share_taken_of_signals': round(len(sel) / max(len(d), 1), 4)}
    for tag, col in (('lower', 'lower'), ('upper', 'upper')):
        v = sel[col].to_numpy() * pv
        eq = np.cumsum(v)
        dd = np.maximum.accumulate(eq) - eq
        out[tag] = {'trades': int(len(v)), 'mean_usd': round(float(v.mean()), 2),
                    'total_usd': round(float(v.sum()), 0),
                    'max_drawdown_usd': round(float(dd.max()), 0),
                    'worst_trade_usd': round(float(v.min()), 0),
                    'best_trade_usd': round(float(v.max()), 0)}
    k = sel.kind.to_numpy()
    out['outcomes'] = {n: int((k == vv).sum()) for n, vv in
                       (('win', WIN), ('loss', LOSS), ('ambiguous', AMBIG), ('unknown', UNKNOWN))}
    # концентрация: доля результата на 1 % лучших сделок
    v = np.sort(sel.upper.to_numpy() * pv)[::-1]
    out['concentration_top1pct_share_of_positive'] = round(
        float(v[:max(1, len(v)//100)].sum() / max(v[v > 0].sum(), 1e-9)), 4)
    return out


if __name__ == '__main__':
    res = {}
    for inst in ('NQ',):
        for terr in ('discovery', 'evaluation'):
            for name in ('A_T0_early', 'B_simple', 'C_prefix'):
                r = report(inst, terr, name)
                res[f'{inst}_{terr}_{name}'] = r
                print(f"{inst}/{terr}/{name:>11}: сигналов {r['signals_before_aggregation']:>7} "
                      f"→ агрег {r['after_aggregation']:>6} → взято {r['after_conflict_rule']:>6} "
                      f"| нижняя ${r['lower']['total_usd']:>12,.0f} ({r['lower']['mean_usd']:+.1f}/сделку) "
                      f"| верхняя ${r['upper']['total_usd']:>12,.0f} ({r['upper']['mean_usd']:+.1f}) "
                      f"| просадка(верх) ${r['upper']['max_drawdown_usd']:>10,.0f}")
    (OUT / 'portfolio.json').write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                        encoding='utf-8')
