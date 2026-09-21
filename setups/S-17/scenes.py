"""S-17: пять реальных сессий NQ для карточки. Выбор объявлен правилом, но сделан ПОСЛЕ результата:
в 2025 году по одной сессии на группу пути — ближайшая к медиане результата D30 своей группы; пятая — худший день
первичной территории. Сцены иллюстрируют правило и дыру защиты, а не доказывают результат.

    python -B setups/S-17/scenes.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'setups' / 'S-07'))
import run as base  # noqa: E402

d = pd.read_csv(HERE / 'sessions_NQ.csv')
p = d[(d.year >= 2021) & (d.year <= 2025)].copy()
p['usd'] = p.d30_pts * 20 - 15
p['mae_c'] = -p.mae_wick_pts / p.candle
med = p.mae_c.median()
big = p.usd <= p.usd.quantile(.10)
groups = {
    'продолжение, малый ход против': (p.usd > 0) & (p.mae_c < med),
    'продолжение после значительного хода против': (p.usd > 0) & (p.mae_c >= med),
    'продолжения нет, обычная потеря': (p.usd <= 0) & ~big,
    'крупная потеря (худшие 10 %)': big,
}
picks = []
for name, mask in groups.items():
    g = p[mask & (p.year == 2025)]
    target = p.usd[mask].median()
    row = g.iloc[(g.usd - target).abs().argmin()]
    picks.append((name, row))
picks.append(('худший день 2021–2025: гэп по стороне сделки не родился — защиты C нет', p.loc[p.usd.idxmin()]))

m = base.market('NQ')
cal = pd.read_parquet(ROOT / 'setups/S-04/calendar.parquet').set_index('date')
ts = m['close_ts_utc_ns']
fig, axes = plt.subplots(len(picks), 1, figsize=(11, 3.1 * len(picks)))
meta = []
for ax, (name, r) in zip(axes, picks):
    pp = int(np.searchsorted(ts, int(cal.loc[r.date, 'open_ns']))) + 3
    b = pp + 1
    lo, hi = pp - 29, b + 30
    x = np.arange(lo, hi + 1) - b
    for k, xi in zip(range(lo, hi + 1), x):
        o, h, l, c = m['open'][k], m['high'][k], m['low'][k], m['close'][k]
        col = '#2a9d8f' if c >= o else '#e76f51'
        ax.plot([xi, xi], [l, h], color=col, lw=0.8)
        ax.plot([xi, xi], [o, c], color=col, lw=3.2, solid_capstyle='butt')
    ax.axvline(-0.5, color='k', lw=1, ls='--')
    ax.axhline(r.entry, color='#e9c46a', lw=1)
    ax.axvspan(-0.5, 29.5, color='#eeeeee', zorder=0)
    if not np.isnan(r.gap_far_below_entry_pts):
        far = r.entry - r.side * r.gap_far_below_entry_pts
        ax.hlines(far, r.gap_born_min - 1, 29.5, color='#264653', lw=1.2, ls=':')
    if not np.isnan(r.c_min):
        ax.plot([r.c_min], [m['open'][b + int(r.c_min)]], marker='v' if r.side > 0 else '^', color='#264653', ms=9)
    ax.plot([30], [m['open'][b + 30]], marker='s', color='k', ms=6)
    side = 'ЛОНГ' if r.side > 0 else 'ШОРТ'
    ax.set_title(f"{r.date}  {side}  — {name}\nD30 {r.usd:+,.0f} USD   с защитой C {r.prot_pts * 20 - 15:+,.0f} USD   ход против {r.mae_c:.1f} свечи",
                 fontsize=9, loc='left')
    ax.set_xlim(-30.5, 31)
    ax.tick_params(labelsize=8)
    meta.append(dict(group=name, date=str(r.date), side=int(r.side), d30_usd=float(r.usd), prot_usd=float(r.prot_pts * 20 - 15)))
axes[-1].set_xlabel('минуты от входа (0 = бар входа, open 09:33:00 ET); серое — 30 минут позиции; жёлтая линия — цена входа\n'
                    'точки — дальний край первого гэпа по стороне сделки; треугольник — выход по событию C; квадрат — выход по времени', fontsize=8)
fig.tight_layout()
fig.savefig(HERE / 'scenes.png', dpi=110)
(HERE / 'scenes.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding='utf-8')
print('saved', HERE / 'scenes.png')
