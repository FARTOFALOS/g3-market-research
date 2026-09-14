"""075: the trader's own example, run both ways round.

forward  : of every occurrence of a sign, what share completes
backward : of every completed case, what share carried the sign

A sign that is present in almost every completed case and also in almost every
failed one recognises nothing. The two tables have to be read together.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
from predicates import build

pd.set_option('display.width', 220)


def main():
    f = pd.read_parquet(common.CACHE / 'feat_all.parquet')
    f = f.reset_index(drop=True)
    P = build(f)
    y = f.R2.to_numpy().astype(bool)          # 2R reached before the border came back
    base = y.mean()
    rows = []
    for name, m in P.items():
        n = int(m.sum())
        if n < 2000:
            continue
        rows.append(dict(sign=name, n=n, share_of_population=round(m.mean(), 3),
                         forward=round(float(y[m].mean()), 4),
                         lift=round(float(y[m].mean()) - base, 4),
                         backward=round(float(m[y].mean()), 3),
                         among_failed=round(float(m[~y].mean()), 3)))
    t = pd.DataFrame(rows).sort_values('backward', ascending=False)
    print(f'completion = MFE reaches 2R before the cleared border is touched again')
    print(f'population = {len(f)} node events, base completion rate {base:.4f}\n')
    print(t.to_string(index=False))


if __name__ == '__main__':
    main()
