"""S-09 — картинка: кривая капитала обеих дверей и профиль часов, из которого
выросла дневная дверь. Считается из trades.csv и clock.csv, ничего не подбирает.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
d = pd.read_csv(HERE / 'trades.csv')
d['day'] = pd.to_datetime(d['day'], utc=True).dt.tz_localize(None)
clock = pd.read_csv(HERE / 'clock.csv')
S1 = pd.Timestamp('2025-11-01')

fig, ax = plt.subplots(1, 2, figsize=(15.5, 5.6))
fig.suptitle('S-09: два решения в день на NQ. Поиск 2020-01…2025-10, '
             'кусок после 2025-11 при поиске не открывался', fontsize=11)

names = {'utro_0933': 'утро 09:33', 'den_1338': 'день 13:38'}
for k, lbl in names.items():
    s = d[d.door == k].sort_values('day')
    ax[0].plot(s.day, s.net.cumsum(), lw=1.3, label=lbl)
tot = d.groupby('day').net.sum().sort_index()
ax[0].plot(tot.index, tot.cumsum(), lw=2, color='k', label='обе двери')
ax[0].axvline(S1, color='r', ls='--', lw=1)
ax[0].text(S1, ax[0].get_ylim()[0], ' граница поиска', color='r', fontsize=8, va='bottom')
ax[0].set_title('Накопленный чистый итог, один контракт, расход $15')
ax[0].set_ylabel('$'); ax[0].grid(alpha=.3); ax[0].legend(fontsize=9)

c = clock.set_index('минута')
ax[1].plot(c.index, c['средняя'], lw=.9, color='0.6', label='каждая минута решения')
ax[1].plot(c.index, c['средняя'].rolling(11, center=True).mean(), lw=2, label='сглажено по 11 минутам')
ax[1].axhline(0, color='k', lw=.8)
for m, lbl in ((573, '09:33'), (818, '13:38')):
    ax[1].axvline(m, color='crimson', lw=1.2)
    ax[1].text(m + 4, ax[1].get_ylim()[1] * .85, lbl, color='crimson', fontsize=9)
ticks = list(range(570, 961, 60))
ax[1].set_xticks(ticks)
ax[1].set_xticklabels([f'{t//60:02d}:{t%60:02d}' for t in ticks])
ax[1].set_title('Все 389 минут решения при одном и том же действии:\nсредняя чистая сделка, $')
ax[1].grid(alpha=.3); ax[1].legend(fontsize=9)
fig.tight_layout(rect=[0, 0, 1, .93])
fig.savefig(HERE / 'scenes.png', dpi=130)
print(HERE / 'scenes.png')
