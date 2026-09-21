#!/usr/bin/env python3
"""097 packet: 24 real first-contact scenes, chosen by anchor time only.

Selection uses nothing after the contact bar k. 24 equal-count time slices of
the 2021-2025 moments; slice i wants side N/S alternating and approach class
(wait==1 / wait>=2) alternating in pairs; the first matching moment at or after
the slice midpoint is taken, one scene per session.

Usage:  packet.py select | prefix | full
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[2] / 'work' / '097'
MK = ROOT / 'data/market/NQ'
O = np.load(MK / 'open.npy', mmap_mode='r')
H = np.load(MK / 'high.npy', mmap_mode='r')
L = np.load(MK / 'low.npy', mmap_mode='r')
C = np.load(MK / 'close.npy', mmap_mode='r')
TS = np.load(MK / 'close_ts_utc_ns.npy', mmap_mode='r')
SID = np.load(MK / 'session_id.npy', mmap_mode='r')
PRE, POST = 45, 60


def select():
    m = pd.read_parquet(OUT / 'moments_NQ.parquet')
    m = m[m.year <= 2025].sort_values('k').reset_index(drop=True)
    n = len(m); used = set(); rows = []
    for i in range(24):
        side = 'north' if i % 2 == 0 else 'south'
        far = (i // 2) % 2 == 1                      # wait>=2 on slices 2,3,6,7,...
        lo, hi = i * n // 24, (i + 1) * n // 24
        mid = (lo + hi) // 2
        order = list(range(mid, hi)) + list(range(lo, mid))
        for j in order:
            r = m.iloc[j]
            if r.side != side or (r.wait >= 2) != far or r.sid in used:
                continue
            used.add(r.sid); rows.append(r); break
    p = pd.DataFrame(rows).reset_index(drop=True)
    p.to_parquet(OUT / 'packet24.parquet')
    for i, r in p.iterrows():
        print('%2d %s %s tf%-4d wait %-3d w %-6.2f U %-6.2f close_out %+6.2f depth %5.2f'
              % (i, str(r.k_et)[:16], r.side[0].upper(), r.tf_minutes, r.wait, r.w, r.U,
                 r.k_close_out, r.k_depth_in))


def candles(ax, a, b, s, B, alpha=1.0):
    """Oriented candles relative to b: outward is up, zero is the line."""
    for x, j in enumerate(range(a, b)):
        o, h, l, c = (float(v[j]) for v in (O, H, L, C))
        if s < 0:
            o, h, l, c = -o, -l, -h, -c
        o, h, l, c = o - s * B, h - s * B, l - s * B, c - s * B
        col = '#1a9850' if c >= o else '#d73027'
        ax.plot([x, x], [l, h], color=col, lw=0.8, alpha=alpha)
        ax.add_patch(plt.Rectangle((x - 0.3, min(o, c)), 0.6, max(abs(c - o), 0.05),
                                   color=col, alpha=alpha))


def sheet(mode):
    p = pd.read_parquet(OUT / 'packet24.parquet')
    for page in range(4):
        fig, axes = plt.subplots(2, 3, figsize=(15.6, 10.4), dpi=100)
        for ax, (i, r) in zip(axes.ravel(), p.iloc[page * 6:(page + 1) * 6].iterrows()):
            k, t0 = int(r.k), int(r.t0_spine_pos)
            s = 1.0 if r.side == 'north' else -1.0
            a = max(min(t0 - 8, k - PRE), k - 110)
            while SID[a] != SID[k]:
                a += 1
            b = k + 1
            if mode == 'full':
                b = k + 1 + POST
                while SID[b - 1] != SID[k]:
                    b -= 1
            candles(ax, a, b, s, r.exit_boundary)
            ax.axhline(0, color='k', lw=1.0)
            ax.axhline(-r.w, color='k', lw=0.8, ls='--')
            ax.axhline(r.U, color='#2166ac', lw=0.8, ls=':')
            if t0 >= a:
                ax.axvline(t0 - a, color='#984ea3', lw=0.7, alpha=0.6)
            ax.axvline(k - a, color='#ff7f00', lw=0.7, alpha=0.8)
            ax.set_title('#%d %s %s tf%d wait%d w%.2f U%.2f' % (
                i, str(r.k_et)[:16], r.side[0].upper(), r.tf_minutes, r.wait, r.w, r.U),
                fontsize=9)
            ax.grid(alpha=0.25)
            ax.set_xlim(-1, (k - a) + POST + 2)
        fig.tight_layout()
        fig.savefig(OUT / f'sheet_{mode}_{page}.png')
        plt.close(fig)
    print('ok', mode)


if __name__ == '__main__':
    cmd = sys.argv[1]
    select() if cmd == 'select' else sheet(cmd)
