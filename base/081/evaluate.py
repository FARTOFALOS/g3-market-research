#!/usr/bin/env python3
"""Один объявленный проход замороженной конструкции. Параметры не меняются."""
from __future__ import annotations
import json, sys
from pathlib import Path
import pandas as pd
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from candidate import with_gross                                        # noqa: E402
from family2 import load                                                # noqa: E402

OUT = ROOT / 'work/081a'
FROZEN = {                                    # FREEZE_081A §3, не изменяется
    'A_T0_early':      dict(age_min=0, d_mult=0.0, k=1.0, theta=None),
    'B_simple':        dict(age_min=0, d_mult=4.0, k=1.0, theta=None),
    "B'_matched":      dict(age_min=1, d_mult=4.0, k=1.0, theta=None),
    'C_prefix':        dict(age_min=1, d_mult=4.0, k=1.0, theta=0.75)}


def run(inst, terr):
    q, f = load(inst, terr)
    out = {}
    for name, p in FROZEN.items():
        out[name] = with_gross(q, f, inst, **p)
    return out


if __name__ == '__main__':
    res = {}
    for inst, terr in [('NQ', 'evaluation'), ('ES', 'discovery'), ('ES', 'evaluation'),
                       ('YM', 'discovery'), ('YM', 'evaluation')]:
        res[f'{inst}_{terr}'] = run(inst, terr)
        print(f'=== {inst} / {terr} ===')
        print(f"{'политика':>12}{'испол':>8}{'доля':>8}{'unk':>7}{'win':>7}{'loss':>7}"
              f"{'валовое':>9}{'после':>8}{'нижн':>8}{'верхн':>8}")
        for k, v in res[f'{inst}_{terr}'].items():
            if not v['stream']['executed']:
                print(f'{k:>12}  нет исполнений'); continue
            print(f"{k:>12}{v['stream']['executed']:>8}{v['executed_share']:>8.4f}"
                  f"{v['unknown_share']:>7.3f}{v['win']:>7.3f}{v['loss']:>7.3f}"
                  f"{v['gross_determined']:>9.4f}{v['determined']:>8.4f}"
                  f"{v['lower']:>8.3f}{v['upper']:>8.3f}")
        print()
    (OUT / 'evaluation.json').write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                         encoding='utf-8')
