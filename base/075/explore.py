"""075: exploratory overlay of the node population. Even years only.

Reports structure, not money: P(MFE >= 2R before the border is touched again)
and the plain R-expectancy of a 2R/1R bracket, sliced by each observable column.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common

pd.set_option('display.width', 200)


def slice_table(f, col, bins=None, q=8):
    if bins is None:
        v = f[col]
        if v.nunique() <= 10:
            g = v
        else:
            g = pd.qcut(v, q, duplicates='drop')
    else:
        g = pd.cut(f[col], bins)
    t = f.groupby(g, observed=True).agg(n=('R2', 'size'), R1=('R1', 'mean'), R2=('R2', 'mean'),
                                        R3=('R3', 'mean'), risk=('risk', 'median'),
                                        stop=('stopped', 'mean'))
    t['E2_R'] = 3 * t.R2 - 1
    t['E2_net$'] = (3 * t.R2 - 1) * t.risk * 20 - 15
    return t.round(4)


def main(kind='ZE'):
    f = pd.read_parquet(common.CACHE / f'feat_{kind}.parquet')
    dev = f[f.year % 2 == 0]
    print(f'=== {kind}  even years n = {len(dev)}   base R1 {dev.R1.mean():.4f}  '
          f'R2 {dev.R2.mean():.4f}  R3 {dev.R3.mean():.4f}  E2 {3*dev.R2.mean()-1:+.4f} R')
    for col in ['risk', 'risk_w', 'risk_atr', 'close_beyond', 'body', 'close_in_range',
                'range_w', 'zlen', 'zdepth', 'n_prior_Z', 'been_far', 'been_exit',
                'pushes_prev', 'engulf_prev', 'inside_prev', 'prev_pushes',
                'k', 'w_atr', 'rth', 'long', 'first']:
        print(f'\n-- {col}')
        print(slice_table(dev, col).to_string())


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'ZE')
