"""Address-selected x-ray of first return to the completed NY opening hour.

Exploratory scene reader only. Selection and later continuations are recorded
separately; no outcome is used in the first-event rule.
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "setups/S-20"))
import run as s20
MINUTE = s20.MINUTE

DATES = ("2021-01-04", "2021-10-01", "2022-04-01", "2023-04-03", "2024-04-02")


def candles(ax, a, positions):
    for x, p in enumerate(positions):
        o, h, l, c = (float(a[k][p]) for k in ("open", "high", "low", "close"))
        color = "#1b8f7e" if c >= o else "#c44e52"
        ax.plot([x, x], [l, h], color=color, lw=.6)
        ax.add_patch(Rectangle((x-.3, min(o, c)), .6, max(abs(c-o), .03),
                               facecolor=color, edgecolor=color))
    ix = np.linspace(0, len(positions)-1, min(6, len(positions))).astype(int)
    times = pd.to_datetime(a["ts"][positions[ix]], utc=True).tz_convert("America/New_York")
    ax.set_xticks(ix, times.strftime("%H:%M"), rotation=30)
    ax.grid(alpha=.15)


def main():
    a, _ = s20.load_market("NQ")
    cal = pd.read_csv(ROOT / "base/103/calendar.csv").set_index("date")
    out = ROOT / "work/104/scenes"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for date in DATES:
        day = cal.loc[date]
        start = pd.Timestamp(day["open"]).value
        end = int(day["close_ns"])
        p = np.flatnonzero((a["ts"] > start) & (a["ts"] <= end))
        assert len(p) >= 61 and np.all(np.diff(a["ts"][p[:60]]) == MINUTE)
        H = float(a["high"][p[:60]].max())
        L = float(a["low"][p[:60]].min())
        events = []
        for side, level in (("H", H), ("L", L)):
            away = False
            last = int(p[59])
            for q in p[60:]:
                assert a["ts"][q] - a["ts"][last] == MINUTE
                last = int(q)
                if not away:
                    away = a["high"][q] < H if side == "H" else a["low"][q] > L
                elif a["high"][q] >= H if side == "H" else a["low"][q] <= L:
                    events.append((int(q), side, level))
                    break
        q, side, level = min(events)
        ix = int(np.flatnonzero(p == q)[0])
        pre = p[max(0, ix-70):ix+1]
        post = p[ix+1:min(len(p), ix+91)]
        fig, axes = plt.subplots(1, 3, figsize=(17, 4.5), gridspec_kw={"width_ratios": [1, 1, 1.2]})
        candles(axes[0], a, pre)
        candles(axes[1], a, post)
        for ax in axes[:2]:
            ax.axhline(H, color="#315ea3", lw=1)
            ax.axhline(L, color="#a27a31", lw=1)
        axes[0].set_title("Known prefix through first retest close")
        axes[1].set_title("Next 90 closed minutes")
        remaining = p[ix:]
        et = pd.to_datetime(a["ts"][remaining], utc=True).tz_convert("America/New_York")
        axes[2].plot(et, a["close"][remaining], color="#2d3e50", lw=1)
        axes[2].axhline(level, color="#315ea3", lw=1)
        axes[2].set_title("Full later close path (x-ray only)")
        axes[2].tick_params(axis="x", rotation=30)
        axes[2].grid(alpha=.15)
        fig.suptitle(f"{date} | first retest of {side}={level:.2f} | event close={a['close'][q]:.2f}")
        fig.tight_layout()
        fig.savefig(out / f"{date}.png", dpi=140)
        plt.close(fig)
        rows.append({"date": date, "side": side, "level": level, "event_close_utc": str(pd.Timestamp(int(a['ts'][q]), tz="UTC")),
                     "event_ohlc": {k: float(a[k][q]) for k in ("open", "high", "low", "close")},
                     "next_open": float(a["open"][q+1]), "last_close": float(a["close"][p[-1]])})
    pd.DataFrame(rows).to_json(out / "index.json", orient="records", indent=2)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
