"""S-07: две реальные сцены — решение с закрытым будущим и продолжение."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import run as base
import manage as M

HERE, ROOT, OUT, MIN = base.HERE, base.ROOT, base.OUT, base.MIN
PRE = 30          # минут предыстории на картинке
POST = 120        # минут продолжения


def draw(ax, o, h, l, c, x0, title, mid=None, entry=None, side=None, span=None):
    for i, (oo, hh, ll, cc) in enumerate(zip(o, h, l, c)):
        x = x0 + i
        up = cc >= oo
        ax.plot([x, x], [ll, hh], color='#444', linewidth=.6, zorder=2)
        ax.add_patch(plt.Rectangle((x - .35, min(oo, cc)), .7, max(abs(cc - oo), 1e-9),
                                   facecolor='#2e7d32' if up else '#c62828',
                                   edgecolor='#2e7d32' if up else '#c62828', zorder=3))
    if mid is not None:
        ax.axhline(mid, color='#1565c0', linewidth=1.1, zorder=1)
    if span is not None:
        ax.axhspan(span[0], span[1], color='#1565c0', alpha=.07, zorder=0)
    if entry is not None:
        ax.axhline(entry, color='#f9a825', linewidth=1.0, linestyle='--', zorder=1)
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=.15, linewidth=.5)
    ax.tick_params(labelsize=7)


def main(ins='NQ'):
    t = M.trajectories(ins)
    year = pd.Series(t['dates']).str.slice(0, 4).to_numpy()
    candles = M.manage(t, {})
    usd = candles * t['unit'] * t['point'] - t['cost']
    recent = year >= '2023'
    win = np.where(recent & (usd > 0))[0]
    loss = np.where(recent & (usd < 0))[0]
    # ближайшие к медианному исходу своей группы — иллюстрация, выбранная после результата
    pick = [int(win[np.argsort(np.abs(usd[win] - np.median(usd[win])))[0]]),
            int(loss[np.argsort(np.abs(usd[loss] - np.median(usd[loss])))[0]])]

    m = base.market(ins)
    u = base.unit(m)
    cal = pd.read_parquet(ROOT / 'setups/S-04/calendar.parquet')
    ts = m['close_ts_utc_ns']
    fig, axes = plt.subplots(2, 2, figsize=(13, 7))
    info = []
    for row, i in enumerate(pick):
        date = t['dates'][i]
        day = cal.loc[cal.date == date].iloc[0]
        p = int(np.searchsorted(ts, day.open_ns + base.MINUTE_OFFSET * MIN)) if hasattr(base, 'MINUTE_OFFSET') else int(np.searchsorted(ts, day.open_ns + 3 * MIN))
        b = p + 1
        a = p - PRE + 1
        mid = float(t['mid'][i])
        hi = float(max(m['high'][a:p + 1]))
        lo = float(min(m['low'][a:p + 1]))
        left = slice(a, p + 1)
        draw(axes[row, 0], m['open'][left], m['high'][left], m['low'][left], m['close'][left], 0,
             f'{ins} {date} — решение на 3-й минуте сессии, будущее закрыто', mid=mid, span=(lo, hi))
        axes[row, 0].set_ylabel('цена')
        right = slice(a, b + POST)
        draw(axes[row, 1], m['open'][right], m['high'][right], m['low'][right], m['close'][right], 0,
             f'то же + 120 минут: {"лонг" if t["side"][i] > 0 else "шорт"}, '
             f'{candles[i]:+.1f} свечи, {usd[i]:+.0f} $', mid=mid, entry=float(t['entry'][i]))
        axes[row, 1].axvline(PRE - .5, color='#f9a825', linewidth=1.2)
        info.append(dict(date=date, side=int(t['side'][i]), entry=float(t['entry'][i]),
                         mid=mid, unit_points=float(t['unit'][i]),
                         candles=float(candles[i]), usd=float(usd[i]),
                         decide_pos=p, entry_pos=b))
    for ax in axes.ravel():
        ax.set_xlabel('минуты')
    fig.suptitle('S-07: слева — всё, что известно в минуту решения; справа — продолжение. '
                 'Синяя линия — середина предрыночного диапазона, жёлтая — цена входа.', fontsize=9)
    fig.tight_layout()
    fig.savefig(HERE / 'scenes.png', dpi=130)
    (HERE / 'scenes.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(info, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
