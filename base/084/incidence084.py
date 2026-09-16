#!/usr/bin/env python3
"""FREEZE_084 §8 step: incidence of the frozen close_break (bars T0 ... recognition only).

Builds the X-ray scan for the evaluation territories with the frozen operator (xray.build, window 50)
and reports close_break only. Other candidate events are computed by the same scan but not reported.
"""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'work/084'
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'base/080'))
import xray                                                                  # noqa: E402
from tape import presence_grid                                               # noqa: E402

TERR = [('NQ', 'discovery'), ('NQ', 'evaluation'), ('ES', 'evaluation'), ('YM', 'evaluation')]
Q5_EXPECT = {('NQ', 'discovery'): 64849, ('NQ', 'evaluation'): 43955,
             ('ES', 'evaluation'): 35018, ('YM', 'evaluation'): 44829}
ENDS = xray.ENDS


def freeze_hash():
    return hashlib.sha256((HERE / 'FREEZE_084.md').read_bytes()).hexdigest()


def main():
    grid = None
    res = {'freeze_sha256': freeze_hash(), 'event': 'close_break', 'window': xray.W, 'territories': {}}
    for inst, terr in TERR:
        p = OUT / f'xray_{inst}_{terr}.parquet'
        if not p.exists():
            if grid is None:
                grid = presence_grid()
            d, hist = xray.build(inst, terr, *grid, window=xray.W)
            d.to_parquet(p, index=False, compression='zstd')
            (OUT / f'xray_hist_{inst}_{terr}.json').write_text(json.dumps(hist), encoding='utf-8')
        d = pd.read_parquet(p, columns=['t0_day', 'epoch', 'end_code', 'end_k', 'r_close_break', 'side'])
        q5 = int((d.end_k > 5).sum())
        assert q5 == Q5_EXPECT[(inst, terr)], f'{inst} {terr}: q=5 population {q5} != prefix check'
        r = d.r_close_break.to_numpy(); rec = r >= 0
        rr = r[rec]
        labs = xray.EPOCHS[terr]
        T = {'T0_census': int(len(d)), 'recognized': int(rec.sum()), 'share_of_T0': round(float(rec.mean()), 4),
             'q5_population': q5,
             'recognition_bar': {f'p{int(q*100)}': float(np.quantile(rr, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
             'share_at_bar_1': round(float((rr == 1).mean()), 4),
             'day_coverage': round(float(d.loc[rec, 't0_day'].nunique() / d.t0_day.nunique()), 4),
             'not_recognized': {('window_censored_50' if c == 4 else f'ended_{ENDS[c]}'):
                                int(((~rec) & (d.end_code == c)).sum()) for c in range(len(ENDS))},
             'by_side': {s: round(float(rec[(d.side == s).to_numpy()].mean()), 4) for s in ('north', 'south')},
             'by_epoch': {}}
        for g, (lab, _) in enumerate(labs):
            m = (d.epoch == g).to_numpy()
            T['by_epoch'][lab] = {'T0': int(m.sum()), 'share_of_T0': round(float(rec[m].mean()), 4),
                                  'p50': float(np.median(r[m & rec])), 'p90': float(np.quantile(r[m & rec], 0.9))}
        res['territories'][f'{inst} {terr}'] = T
        q = T['recognition_bar']
        print(f"{inst} {terr}: T0 {T['T0_census']}  close_break {T['recognized']} ({T['share_of_T0']:.3f})  "
              f"q5 {q5}  bars p10/p25/p50/p75/p90 {q['p10']:.0f}/{q['p25']:.0f}/{q['p50']:.0f}/{q['p75']:.0f}/{q['p90']:.0f}  "
              f"at +1 {T['share_at_bar_1']:.3f}  days {T['day_coverage']:.3f}  side N/S {T['by_side']}")
        print(f"   not recognized {T['not_recognized']}")
        print('   by epoch ' + '  '.join(f"{k}: {v['share_of_T0']:.3f} p50 {v['p50']:.0f} p90 {v['p90']:.0f} (T0 {v['T0']})"
                                         for k, v in T['by_epoch'].items()))
    (OUT / 'incidence_close_break.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print('freeze', res['freeze_sha256'])


if __name__ == '__main__':
    main()
