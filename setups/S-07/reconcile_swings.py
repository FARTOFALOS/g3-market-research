"""Сверка детектора свингов из stencil.py с линзой поля lenses/swing.py.

AGENTS.md требует: свой массовый оператор сверить с существующей линзой,
если она меряет то же самое. Результат 2026-09-08 на 300 случайных днях:
все 6 684 пивота линзы найдены и здесь, плюс 18% своих — здесь допущено
равенство справа, в линзе строгое неравенство с обеих сторон.
"""
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'setups/S-07'))
sys.path.insert(0, str(ROOT / 'src'))
import stencil as S
from g3riz.lenses.swing import _fractal, WIDTH_V1

f = S.films('NQ')
mine_low = lens_low = both = n_days = 0
mine_all, lens_all = [], []
rng = np.random.default_rng(0)
for i in rng.choice(len(f['dates']), 300, replace=False):
    m = int(f['live'][i].sum())
    if m < 10:
        continue
    w, b = f['worst'][i][:m], f['best'][i][:m]
    lows, highs = S.swings(w, b, f['live'][i], 1)
    is_hi, is_lo = _fractal(b, w, WIDTH_V1)
    mine = {j for j, _, _ in lows}
    lens = set(np.where(is_lo)[0].tolist())
    mine_low += len(mine); lens_low += len(lens); both += len(mine & lens); n_days += 1
    mine_all.append(len(mine) + len(highs))
    lens_all.append(int(is_lo.sum() + is_hi.sum()))
print(f'дней {n_days}')
print(f'моих свинг-низов {mine_low}, у линзы поля {lens_low}, совпало {both} '
      f'({both/lens_low:.1%} от линзы, {both/mine_low:.1%} от моих)')
print(f'пивотов за фильм, медиана: мои {np.median(mine_all):.0f}, линза {np.median(lens_all):.0f}')
