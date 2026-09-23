"""S-20 v1: first return to a completed NY opening-hour extreme.

Declared rule: setups/S-20-pervyy-retest-utrennego-ekstremuma.md.
Reads frozen NQ minute tape; no RIZ, materializer, or Volume access.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
try:
    import exchange_calendars as xc
except ModuleNotFoundError:
    sys.path.insert(0, str(ROOT / "work/103/deps"))  # local dependency cache, optional
    import exchange_calendars as xc

MINUTE = 60_000_000_000

COST = 1.0
BEGIN = "2020-01-01"
END = "2025-12-31"


def load_market(instrument):
    market = ROOT / "data/market" / instrument
    manifest = json.loads((market / "manifest.json").read_text(encoding="utf-8"))
    source = json.loads((ROOT / "SOURCE_DATA.json").read_text(encoding="utf-8"))
    assert manifest["corpus_id"] == source["instruments"][instrument]["corpus_id"]
    stop = pd.Timestamp("2026-01-01", tz="UTC").value
    ts = np.load(market / "close_ts_utc_ns.npy", mmap_mode="r")
    n = int(np.searchsorted(ts, stop))
    a = {k: np.load(market / f"{k}.npy", mmap_mode="r")[:n]
         for k in ("open", "high", "low", "close")}
    a["ts"] = ts[:n]
    return a, manifest


def one_day(a, date, open_ns, close_ns):
    ts = a["ts"]
    first = int(np.searchsorted(ts, open_ns + MINUTE))
    stop = int(np.searchsorted(ts, close_ns, side="right"))
    expected = open_ns + MINUTE * np.arange(1, 61, dtype=np.int64)
    base = {"date": date, "status": "known", "reason": "", "side": "", "level": None,
            "event_pos": None, "entry_pos": None, "exit_pos": None, "gross": 0.0,
            "net": 0.0, "trades": 0}
    if first + 60 > stop or not np.array_equal(ts[first:first+60], expected):
        return dict(base, status="unknown", reason="opening_hour_incomplete", net=None, gross=None)
    H = float(a["high"][first:first+60].max())
    L = float(a["low"][first:first+60].min())
    base.update(morning_high=H, morning_low=L)
    away_h = away_l = False
    event = None
    last = first + 59
    for q in range(first+60, stop):
        if ts[q] - ts[last] != MINUTE:
            return dict(base, status="unknown", reason="gap_before_first_retest", net=None, gross=None)
        last = q
        h = float(a["high"][q]); l = float(a["low"][q])
        hit_h = away_h and h >= H
        hit_l = away_l and l <= L
        if hit_h or hit_l:
            if hit_h and hit_l:
                return dict(base, status="unknown", reason="both_extremes_same_minute", net=None, gross=None)
            side, level, d = ("H", H, 1) if hit_h else ("L", L, -1)
            event = (q, side, level, d)
            break
        if not away_h and h < H:
            away_h = True
        if not away_l and l > L:
            away_l = True
    if event is None:
        if stop == 0 or ts[stop-1] != close_ns:
            return dict(base, status="unknown", reason="session_tail_missing", net=None, gross=None)
        return dict(base, reason="no_first_retest")
    q, side, level, d = event
    o = float(a["open"][q]); c = float(a["close"][q])
    base.update(side=side, level=level, direction=d, event_pos=q,
                event_open=o, event_close=c)
    if not (d * (o - level) < 0 and d * (c - level) > 0):
        return dict(base, reason="first_retest_no_body_cross")
    entry = q + 1
    if entry >= stop or entry >= len(ts) or ts[entry] - ts[q] != MINUTE:
        if q + 1 >= stop and ts[q] == close_ns:
            return dict(base, reason="signal_at_session_close")
        return dict(base, status="unknown", reason="next_open_unknown", net=None, gross=None)
    entry_price = float(a["open"][entry])
    if d * (entry_price - level) <= 0:
        return dict(base, reason="next_open_back_inside")
    base.update(entry_pos=entry, entry_price=entry_price, trades=1)
    close_pos = int(np.searchsorted(ts, close_ns))
    close_known = close_pos < len(ts) and ts[close_pos] == close_ns
    for k in range(entry, stop):
        if k > entry and ts[k] - ts[k-1] != MINUTE:
            return dict(base, status="unknown", reason="gap_during_position", gross=None, net=None)
        if close_known and k == close_pos:
            exit_pos = k; exit_price = float(a["close"][k]); why = "session_close"
        elif d * (float(a["close"][k]) - level) <= 0:
            exit_pos = k + 1
            if exit_pos >= len(ts) or ts[exit_pos] - ts[k] != MINUTE or ts[exit_pos] > close_ns:
                return dict(base, status="unknown", reason="exit_open_unknown", gross=None, net=None)
            exit_price = float(a["open"][exit_pos]); why = "closed_back_inside"
        else:
            continue
        gross = d * (exit_price - entry_price)
        end = exit_pos + 1 if why == "session_close" else exit_pos
        lows = a["low"][entry:end]; highs = a["high"][entry:end]
        mae = entry_price - float(lows.min()) if d > 0 else float(highs.max()) - entry_price
        mfe = float(highs.max()) - entry_price if d > 0 else entry_price - float(lows.min())
        held = float((ts[exit_pos]-ts[entry])/MINUTE + (why == "session_close"))
        return dict(base, reason=why, exit_pos=exit_pos, exit_price=exit_price,
                    gross=gross, net=gross-COST, held_minutes=held,
                    mae=mae, mfe=mfe)
    return dict(base, status="unknown", reason="session_close_unknown", gross=None, net=None)


def block_ci(values, known, block=20, draws=10000, seed=10420):
    rng = np.random.default_rng(seed)
    n = len(values)
    out = np.empty(draws)
    for i in range(draws):
        starts = rng.integers(0, n-block+1, size=int(np.ceil(n/block)))
        ix = (starts[:, None] + np.arange(block)).ravel()[:n]
        den = known[ix].sum()
        out[i] = values[ix].sum()/den if den else np.nan
    return [float(v) for v in np.nanquantile(out, [.025, .975])]


def main():
    a, manifest = load_market("NQ")
    cal = xc.get_calendar("XNYS", start=BEGIN, end=END)
    rows = []
    for r in cal.schedule.itertuples():
        date = r.Index.strftime("%Y-%m-%d")
        rows.append(one_day(a, date, pd.Timestamp(r.open).value, pd.Timestamp(r.close).value))
    days = pd.DataFrame(rows)
    out = ROOT / "setups/S-20"
    days.to_csv(out / "daily.csv", index=False)
    known = days.status.eq("known")
    trades = days[days.trades.eq(1)]
    complete_trades = trades[trades.status.eq("known")]
    net = days.net.fillna(0).to_numpy()
    mask = known.to_numpy(float)
    mean = float(days.loc[known, "net"].mean())
    by_year = days.loc[known].assign(year=lambda z: z.date.str[:4]).groupby("year").agg(
        sessions=("net", "size"), mean=("net", "mean"), trades=("trades", "sum")
    ).reset_index().to_dict("records")
    result = {
        "rule": "S-20-v1-first-retest-opening-hour-body-cross",
        "corpus_id": manifest["corpus_id"], "period": [BEGIN, END],
        "runtime_versions": {"numpy": np.__version__, "pandas": pd.__version__,
                             "exchange_calendars": xc.__version__},
        "exposure": "20 calendar-addressed quarter-first sessions and five x-ray scenes, 2021-2025; prior G3 archive exposure",
        "sessions": len(days), "known_sessions": int(known.sum()),
        "unknown_sessions": int((~known).sum()),
        "reasons": days.reason.value_counts().to_dict(),
        "recognized_body_cross": int(((days.direction.isin([-1, 1]))
                                      & ((days.direction*(days.event_open-days.level)) < 0)
                                      & ((days.direction*(days.event_close-days.level)) > 0)).sum()),
        "known_entries": len(complete_trades), "unknown_trade_outcomes": int((trades.status != "known").sum()),
        "known_net_sum_points": float(days.loc[known, "net"].sum()),
        "known_net_per_session_points": mean,
        "ci95_block20_conditional": block_ci(net, mask),
        "known_zero_cost_mean": float((days.loc[known, "net"]+days.loc[known, "trades"]).mean()),
        "known_cost_0_75_mean": float((days.loc[known, "net"]+.25*days.loc[known, "trades"]).mean()),
        "known_cost_2_mean": float((days.loc[known, "net"]-days.loc[known, "trades"]).mean()),
        "known_trade_mean_net": float(complete_trades.net.mean()) if len(complete_trades) else None,
        "known_trade_median_net": float(complete_trades.net.median()) if len(complete_trades) else None,
        "known_trade_worst_net": float(complete_trades.net.min()) if len(complete_trades) else None,
        "known_trade_win_rate": float((complete_trades.net>0).mean()) if len(complete_trades) else None,
        "by_year_conditional": by_year,
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "interpretation_boundary": "Known-session economics only; unknown sessions are not zeros, and selected v1 has no independent validation."
    }
    (out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
