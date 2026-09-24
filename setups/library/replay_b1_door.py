"""One-trader replay: B1 (S-07 v1r1, then S-18 v0) + S-09 day door (13:38), one position, first come holds.

Rules: S-07 exactly as in B1. The day door (setups/S-09/trades.csv, door den_1338, net $ as saved) is taken only if
S-18 holds no position at the door's entry bar; while the door is open, S-18 marks whose execution bar falls inside
the door's life are skipped, and S-18 resumes flat at the next mark. With the door disabled the replay must equal
joint_daily_NQ.csv column B1 to the cent (hard check).

    python -B setups/library/replay_b1_door.py
Output: setups/library/replay_b1_door.json
"""
import json, math, sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import joint as J  # noqa: E402

S17, S18 = J.S17, J.S18
PT, COST = 20.0, 15.0
MIN = 60_000_000_000
LO, HI, SPLIT = pd.Timestamp("2020-01-02"), pd.Timestamp("2026-05-04"), pd.Timestamp("2025-11-01")


def s18_run(cal, root, block_k, windows):
    """joint.s18_blocked line by line; marks k with k < block_k[s] or k in windows[s] (set) are skipped."""
    look, mult, step, first = 14, 1.0, 30, 30
    cal, K, O, C = S18.sessions("NQ", cal, root)
    S = len(cal); op = O[:, 1]
    cl = np.array([C[s, min(K[s], S18.KMAX)] for s in range(S)])
    move = np.abs(C / op[:, None] - 1.0)
    usd_out, legs_out = {}, {}
    for s in range(look, S):
        prev = cl[s - 1]
        if not np.isfinite(op[s]) or not np.isfinite(prev) or not np.isfinite(cl[s]): continue
        hist = move[s - look:s]; cnt = np.isfinite(hist).sum(axis=0)
        with np.errstate(invalid="ignore"):
            norm = np.where(cnt >= int(np.ceil(0.7 * look)), np.nanmean(hist, axis=0), np.nan)
        up = max(op[s], prev) * (1 + mult * norm); dn = min(op[s], prev) * (1 - mult * norm)
        marks = [k for k in range(first, S18.KMAX, step) if k + 1 <= K[s] - 1]
        pos, entry, ek, usd, bad, legs = 0, np.nan, 0, 0.0, False, []
        skip = windows.get(s, set())
        for k in marks:
            if k < block_k[s] or k in skip: continue
            p = C[s, k]
            if not np.isfinite(p) or not np.isfinite(up[k]): bad = True; break
            want = 1 if p > up[k] else (-1 if p < dn[k] else 0)
            if want != pos:
                px = O[s, k + 1]
                if not np.isfinite(px): bad = True; break
                if pos != 0:
                    usd += pos * (px - entry) * PT - COST; legs.append((ek, k + 1))
                pos, entry, ek = want, px, k + 1
        if bad: continue
        if pos != 0:
            usd += pos * (cl[s] - entry) * PT - COST; legs.append((ek, min(K[s], S18.KMAX) + 1))
        usd_out[s] = usd; legs_out[s] = legs
    return cal, K, usd_out, legs_out


def curve(x):
    x = np.asarray(x, float); k = max(1, math.ceil(0.05 * (x != 0).sum())); tot = x.sum()
    cum = np.cumsum(x)
    return dict(sessions=len(x), usd_per_session=round(float(x.mean()), 1),
                t=round(float(x.mean() / x.std(ddof=1) * math.sqrt(len(x))), 2),
                share_days_ge_0=round(float((x >= 0).mean()), 3),
                top5_share_of_total=round(float(np.sort(x[x != 0])[::-1][:k].sum() / tot), 3) if tot > 0 else None,
                max_drawdown=round(float((cum - np.maximum.accumulate(cum)).min()), 0), worst_day=round(float(x.min()), 0))


def main():
    cal = S17.history_calendar(); root = ROOT / "data/market/NQ"
    e = J.s07_s17(root, cal)
    block = J.block_from(e, "s07_exit_min", cal)
    ref = pd.read_csv(HERE / "joint_daily_NQ.csv", parse_dates=["date"]).set_index("date")
    cal2, K, u18, legs18 = s18_run(cal, root.parent, block, {})
    dates = pd.to_datetime(cal2.date); pos_of = {d: s for s, d in enumerate(dates)}
    # regression: no door -> B1
    em = e.set_index("date"); no = em.status.reindex(dates).fillna("").str.startswith("no_entry").to_numpy()
    s07 = np.where(no, 0.0, em.s07_usd.reindex(dates).to_numpy())
    b1 = pd.Series([s07[s] + u18.get(s, np.nan) for s in range(len(dates))], index=dates).loc[LO:HI]
    r = ref.B1.reindex(b1.index)
    if not np.allclose(b1.dropna(), r[b1.notna()], atol=0.01) or (b1.isna() != r.isna()).any():
        raise SystemExit("replay without the door does not reproduce B1")
    # the door
    tr = pd.read_csv(ROOT / "setups/S-09/trades.csv"); tr = tr[tr.door == "den_1338"].copy()
    tr["date"] = pd.to_datetime(tr.day.str[:10])
    windows, door = {}, {}
    taken = skipped = 0
    for row in tr.itertuples():
        s = pos_of.get(row.date)
        if s is None or not (LO <= row.date <= HI): continue
        ke = int((row.ts - cal2.open_ns.iloc[s]) // MIN) + 1        # bar whose open is the entry
        kx = ke + int(row.minutes) - 1
        busy = any(a <= ke < b for a, b in legs18.get(s, []))
        if busy: skipped += 1; continue
        taken += 1; door[s] = row.net
        windows[s] = set(range(ke - 1, kx))                         # marks whose execution bar falls in the door's life
    _, _, u18d, _ = s18_run(cal, root.parent, block, windows)
    lib = pd.Series([s07[s] + u18d.get(s, np.nan) + door.get(s, 0.0) for s in range(len(dates))], index=dates).loc[LO:HI]
    door_s = pd.Series([door.get(s, 0.0) for s in range(len(dates))], index=dates).loc[LO:HI]
    s18_change = pd.Series([u18d.get(s, np.nan) - u18.get(s, np.nan) for s in range(len(dates))], index=dates).loc[LO:HI]
    ok = lib.notna() & b1.notna()
    out = dict(rule="B1 + S-09 den_1338, one position, first come holds", door_taken=taken, door_skipped_s18_busy=skipped,
               regression="no-door replay equals joint B1 on every session", segments={})
    for name, m in (("2020-01-02..2025-10-31 (S-09 search)", ok & (lib.index < SPLIT)), ("2025-11-01..2026-05-04 (S-09 holdout)", ok & (lib.index >= SPLIT)),
                    ("all", ok)):
        out["segments"][name] = dict(B1=curve(b1[m]), B1_plus_door=curve(lib[m]),
                                     door_usd_per_session=round(float(door_s[m].mean()), 1),
                                     s18_change_usd_per_session=round(float(s18_change[m].fillna(0).mean()), 1))
    (HERE / "replay_b1_door.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
