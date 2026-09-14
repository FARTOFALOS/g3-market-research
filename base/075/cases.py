"""075: real cases of the family, read minute by minute on closes.

Four films where the sign fired -- two that reached the target, two that were
stopped -- and one near miss, the minute that closed clear of the border while
its range still touched the zone. Everything on the lines up to the node is what
a reader has; everything after it is what was still ahead.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
from candidate import load, mask
from revision import win_vector
from read_film import render

pd.set_option('display.width', 200)


def main():
    f, run, st = load()
    m = np.asarray(mask(f))
    win = win_vector(run, st)
    d = common.scene_table()
    rng = np.random.default_rng(75)
    for tag, sel in (('REACHED THE TARGET', m & win), ('STOPPED AT THE BORDER', m & ~win)):
        idx = np.flatnonzero(sel)
        for i in rng.choice(idx, 2, replace=False):
            r = f.iloc[i]
            s = d.iloc[int(r.scene)]   # the scene's own zone, not just any zone of that minute
            k = int(r.k)
            top, bot = float(s.zone_top), float(s.zone_bottom)
            w = max(top - bot, .25)
            border = float(r.risk) and None
            side = s.t0_exit_side
            print('\n' + '=' * 100)
            print(f'{tag}   kind={r.kind}  node minute k={k}  risk={r.risk:.2f} pts  '
                  f'target={2*r.risk:.2f} pts  stopped at minute {int(r.tstop) if r.tstop>0 else "-"}')
            mark = {k: '<- the push-out closes here: the sign is complete, entry at the next open'}
            render(int(r.row), top, bot, side, max(0, k - 8), min(common.HOR, k + 22), mark)
    # one near miss
    nm = pd.read_parquet(common.CACHE / 'nearmiss.parquet')
    nm = nm[(nm.risk > 2) & (nm.k > 6)].reset_index(drop=True)
    i = int(rng.integers(0, len(nm)))
    r = nm.iloc[i]
    s = d.iloc[int(r.scene)]
    print('\n' + '=' * 100)
    print(f'NEAR MISS: the minute closes clear of the border but its range still touches the zone')
    render(int(r.row), float(s.zone_top), float(s.zone_bottom), s.t0_exit_side,
           max(0, int(r.k) - 8), min(common.HOR, int(r.k) + 18),
           {int(r.k): '<- closes clear, but the range met the zone: not a push-out'})


if __name__ == '__main__':
    main()
