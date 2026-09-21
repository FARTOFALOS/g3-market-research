"""S-18 — сцены: первые две сессии 2024 года со сделкой (правило отбора названо до взгляда на их исход).

Слева от пунктира первой отметки входа — всё, что известно к решению; справа — продолжение.
Серые линии — верх и низ обычной области на каждую минуту, чёрная — закрытия минут,
треугольники — входы, кресты — выходы.
"""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import run as band
from run import HERE, KMAX


def main():
    ins = 'NQ'
    cal, K, O, C = band.sessions(ins)
    T = pd.read_csv(HERE / 'band_trades_NQ.csv', parse_dates=['date'])
    days = list(T[T.date >= '2024-01-01'].date.drop_duplicates().iloc[:2])
    dates = pd.to_datetime(cal.date)
    op = O[:, 1]
    cl = np.array([C[s, min(K[s], KMAX)] for s in range(len(K))])
    move = np.abs(C / op[:, None] - 1.0)
    fig, axes = plt.subplots(len(days), 1, figsize=(11, 4.2 * len(days)))
    out = []
    for ax, d in zip(np.atleast_1d(axes), days):
        s = int(np.where(dates == d)[0][0])
        norm = np.nanmean(move[s - 14:s], axis=0)
        up = max(op[s], cl[s - 1]) * (1 + norm)
        dn = min(op[s], cl[s - 1]) * (1 - norm)
        k = np.arange(1, K[s] + 1)
        ax.plot(k, C[s, 1:K[s] + 1], color='black', lw=1)
        ax.plot(k, up[1:K[s] + 1], color='grey', lw=0.8)
        ax.plot(k, dn[1:K[s] + 1], color='grey', lw=0.8)
        tt = T[T.date == d]
        for r in tt.itertuples():
            ax.scatter([r.k_in], [O[s, r.k_in]], marker='^' if r.side > 0 else 'v', s=70,
                       color='tab:green' if r.side > 0 else 'tab:red', zorder=3)
            xo = C[s, K[s]] if r.k_out == K[s] else O[s, r.k_out]
            ax.scatter([r.k_out], [xo], marker='x', s=60, color='black', zorder=3)
            out.append({'date': str(d.date()), 'side': int(r.side), 'k_in': int(r.k_in), 'k_out': int(r.k_out),
                        'entry': float(O[s, r.k_in]), 'exit': float(xo), 'net_usd': float(r.net)})
        ax.axvline(tt.k_in.min(), color='tab:blue', ls=':', lw=1)
        ax.set_xticks(np.arange(30, K[s] + 1, 30))
        ax.set_xticklabels([f'{9 + (30 + m) // 60:02d}:{(30 + m) % 60:02d}' for m in np.arange(30, K[s] + 1, 30)], fontsize=8)
        ax.set_title(f'NQ {d.date()}: сделок {len(tt)}, итог дня ${tt.net.sum():,.0f}'.replace(',', ' '))
    fig.tight_layout()
    fig.savefig(HERE / 'scenes.png', dpi=110)
    (HERE / 'scenes.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
