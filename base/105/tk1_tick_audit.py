"""TK1 — representation audit: does the hidden order inside a minute (high first vs low first) carry information
about the future beyond the minute's OHLC and prefix? Declared in base/105/LOG.md before any tick download.

    python -B base/105/tk1_tick_audit.py --self-test        # synthetic ticks: no dependence -> nothing; injected -> found
    python -B base/105/tk1_tick_audit.py --years 2024 2025  # needs data/forward/_incoming/edgearbiter_nq_tick_rithmic_<Y>.parquet

Only the columns 'timestamp' (UTC) and 'price' are read; volume and aggressor are never loaded.
"""
import argparse, json, math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SRC = ROOT / "data/forward/_incoming"
OUT = HERE / "tk1_out"
HS = (1, 5, 15, 60)
NPERM, SEED = 200, 20260924


def minutes_from_ticks(batches):
    """Stream (ts_ms int64, price float64) batches sorted by time -> per-minute o,h,l,c and order (+1 high first)."""
    parts, carry = [], None
    for ts, px in batches:
        df = pd.DataFrame(dict(ts=ts, px=px))
        if carry is not None: df = pd.concat([carry, df], ignore_index=True)
        df["m"] = df.ts // 60000
        last_m = df.m.iloc[-1]
        carry = df[df.m == last_m][["ts", "px"]]; df = df[df.m != last_m]
        if len(df): parts.append(agg(df))
    if carry is not None and len(carry):
        carry = carry.assign(m=carry.ts // 60000); parts.append(agg(carry))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def agg(df):
    g = df.groupby("m", sort=True).px
    out = pd.DataFrame(dict(o=g.first(), h=g.max(), l=g.min(), c=g.last()))
    ih, il = g.idxmax(), g.idxmin()                      # first occurrence of the extreme
    out["order"] = np.where(ih < il, 1, np.where(il < ih, -1, 0))
    return out.reset_index()


def label(M):
    """Add ET clock, day, scale u, outcomes and control cells to a minute table (m = minute index, UTC)."""
    t = pd.to_datetime(M.m * 60, unit="s", utc=True).dt.tz_convert("America/New_York")
    M["day"] = (t.dt.year * 10000 + t.dt.month * 100 + t.dt.day).to_numpy()
    M["mod"] = (t.dt.hour * 60 + t.dt.minute).to_numpy()
    M = M[(M["mod"] >= 120) & (M["mod"] < 960)].reset_index(drop=True)
    rng = (M.h - M.l).to_numpy()
    M["u"] = pd.Series(rng).rolling(30, min_periods=30).median().shift(1).to_numpy()
    c = M.c.to_numpy(); m = M.m.to_numpy(); day = M.day.to_numpy()
    jump = np.r_[0.0, np.abs(np.diff(c))]
    for h in HS:
        idx = np.arange(len(M)) + h
        ok = idx < len(M)
        j = np.where(ok, idx, 0)
        contig = ok & (m[j] - m == h) & (day[j] == day)
        # exclude windows containing a roll/gap jump
        cj = np.r_[0.0, np.cumsum(jump > np.maximum(50.0, 20 * np.nan_to_num(M.u.to_numpy(), nan=np.inf)))]
        clean = contig & (cj[np.minimum(j + 1, len(cj) - 1)] - cj[np.arange(len(M)) + 1] == 0)
        M[f"f{h}"] = np.where(clean, c[j] - c, np.nan)
    prev5 = np.r_[np.full(5, np.nan), c[5:] - c[:-5]]
    same5 = np.r_[np.zeros(5, bool), (m[5:] - m[:-5] == 5) & (day[5:] == day[:-5])]
    M["prev5"] = np.where(same5, np.sign(prev5), np.nan)
    M["dir"] = (M.c >= M.o).astype(int)
    M["cpos"] = np.where(rng > 0, (M.c - M.l) / np.where(rng > 0, rng, 1), np.nan)
    M["rrel"] = rng / M.u
    M = M[(M.order != 0) & M.u.notna() & (M.u > 0) & M.prev5.notna()].reset_index(drop=True)
    M["cq"] = pd.qcut(M.cpos, 3, labels=False, duplicates="drop")
    M["rq"] = pd.qcut(M.rrel, 3, labels=False, duplicates="drop")
    M["cell"] = ((M["mod"] - 120) // 30).astype(str) + "|" + M.dir.astype(str) + "|" + M.cq.astype(str) + "|" + M.rq.astype(str) + "|" + M.prev5.astype(str)
    return M


def delta(M, col, order):
    """Within-cell HF - LF difference of mean future, weighted by cell size; day-clustered t."""
    y = M[col].to_numpy(float); ok = np.isfinite(y)
    cells = M.cell.to_numpy()[ok]; y = y[ok]; o = order[ok]; day = M.day.to_numpy()[ok]
    cm = pd.Series(y).groupby(cells).transform("mean").to_numpy()
    e = y - cm
    hf, lf = o == 1, o == -1
    ph = pd.Series(hf.astype(float)).groupby(cells).transform("mean").to_numpy()
    # regression of within-cell residual on the within-cell centred order indicator
    z = hf.astype(float) - ph
    denom = (z * z).sum()
    if denom == 0: return float("nan"), float("nan")
    b = (z * e).sum() / denom
    s = pd.Series(z * (e - b * z)).groupby(day).sum().to_numpy()
    se = math.sqrt((s * s).sum()) / denom
    return float(b), float(b / se) if se > 0 else float("nan")


def audit(M, rng):
    res = {}
    for layer, sub in (("02:00-09:30", M[M["mod"] < 570]), ("09:30-16:00", M[M["mod"] >= 570])):
        sub = sub.reset_index(drop=True)
        o = sub.order.to_numpy()
        codes = pd.factorize(sub.cell)[0]; base = np.argsort(codes, kind="stable")
        perms = []
        for _ in range(NPERM):                              # random permutation of order inside every cell
            p = np.lexsort((rng.random(len(o)), codes)); perm = np.empty_like(o); perm[p] = o[base]; perms.append(perm)
        for h in HS:
            b, t = delta(sub, f"f{h}", o)
            bu, _ = delta(sub.assign(**{f"f{h}u": sub[f"f{h}"] / sub.u}), f"f{h}u", o)
            null = [delta(sub, f"f{h}", perm)[0] for perm in perms]
            res[f"{layer}|{h}"] = dict(n=int(np.isfinite(sub[f"f{h}"]).sum()), delta_pts=b, delta_u=bu, t=t,
                                       null_q005=float(np.quantile(null, .005)), null_q995=float(np.quantile(null, .995)))
    return res


def ticks_batches(path, batch_rows=5_000_000):
    f = pq.ParquetFile(path)
    for b in f.iter_batches(batch_size=batch_rows, columns=["timestamp", "price"]):
        ts = b.column("timestamp").cast("int64").to_numpy(); px = b.column("price").to_numpy()
        yield ts, px


def synthetic(days, inject, rng):
    """Random-walk ticks 02:00-16:00 ET; with inject, the next minutes drift by +0.5 pt after a high-first minute."""
    out = []
    for d in pd.bdate_range("2023-01-03", periods=days):
        start = pd.Timestamp(d.date()).tz_localize("America/New_York") + pd.Timedelta(hours=2)
        t0 = int(start.tz_convert("UTC").value // 1_000_000); p = 15000.0; drift = 0.0
        for k in range(840):
            n = rng.integers(20, 60); ts = t0 + k * 60000 + np.sort(rng.integers(0, 60000, n))
            steps = rng.choice([-0.25, 0.25], n) + drift / n
            px = p + np.cumsum(steps); p = px[-1]
            drift = 0.0
            if inject and px.argmax() < px.argmin(): drift = 0.5
            out.append((ts, np.round(px * 4) / 4))
    ts = np.concatenate([a for a, _ in out]); px = np.concatenate([b for _, b in out])
    yield ts, px


def main():
    a = argparse.ArgumentParser(); g = a.add_mutually_exclusive_group(required=True)
    g.add_argument("--self-test", action="store_true"); g.add_argument("--years", nargs="+", type=int)
    x = a.parse_args(); rng = np.random.default_rng(SEED)
    global NPERM
    if x.self_test:
        NPERM = 20
        for inject in (False, True):
            M = label(minutes_from_ticks(synthetic(40, inject, np.random.default_rng(1))))
            r = audit(M, rng)
            k = "09:30-16:00|1"
            print(f"inject={inject}: n {r[k]['n']}, delta {r[k]['delta_pts']:+.3f} pt, t {r[k]['t']:+.1f}, null99 [{r[k]['null_q005']:+.3f}, {r[k]['null_q995']:+.3f}]")
        return
    OUT.mkdir(exist_ok=True); res = {}
    for y in x.years:
        path = SRC / f"edgearbiter_nq_tick_rithmic_{y}.parquet"
        M = label(minutes_from_ticks(ticks_batches(path)))
        res[str(y)] = dict(minutes=len(M), share_hf=float((M.order == 1).mean()), audit=audit(M, rng))
    verdict = []
    for layer in ("02:00-09:30", "09:30-16:00"):
        for h in (1, 5, 15):
            rs = [res[str(y)]["audit"][f"{layer}|{h}"] for y in x.years]
            if len(rs) == 2 and np.sign(rs[0]["delta_pts"]) == np.sign(rs[1]["delta_pts"]) and all(
                    abs(r["t"]) >= 3 and not (r["null_q005"] <= r["delta_pts"] <= r["null_q995"]) for r in rs):
                verdict.append(f"{layer}|{h}")
    res["hidden_order_carries_information"] = verdict
    (OUT / "summary.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
