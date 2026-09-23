"""Требования контракта для веток, прошедших объявленное правило (net > 0 в 2020–25, struct того же знака в 2013–19)."""
import sys, pandas as pd
from trade import Tape
from policy import run_policy, requirements

pd.set_option('display.width', 260); pd.set_option('display.max_columns', 40)
T = Tape()
out = []
for v in sys.argv[1:]:
    tr = pd.read_csv(f'{v}_trades.csv')
    tk = run_policy(T, tr)
    tk.to_csv(f'{v}_policy.csv', index=False)
    r, day = requirements(T, tk, v)
    r26, _ = requirements(T, tk, v + ' 2026', 20260101, 20261231)
    out += [r, r26]
print(pd.DataFrame(out).to_string(index=False))
