"""Build a forward continuation of the local minute tape from contract-level minute bars.

Writes ONLY under data/forward/. The canonical spine (data/market), the raw attachment (data/raw)
and the frozen RIZ field (data/field) are read-only inputs and are never modified.

Steps
  1. extension CSV in the exact raw source format (America/Chicago END-labelled minute),
     rows strictly after the last local label, front contract chosen by the roll rule that the
     local tape itself follows (established on two rolls: switch at the Sunday session open of
     expiration week);
  2. the same canonical transform as the original ingest (chicago-end-label-to-eastern-open-v1),
     self-tested by re-deriving the tail of the ORIGINAL raw file and comparing with the stored spine;
  3. forward spine = old arrays + extension rows, loadable by g3riz.market.MarketSpine.open_store.

No operator, signal or outcome is computed here: mechanical integrity only.
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[3]
FWD = REPO / "data" / "forward"
LYNX = FWD / "_incoming" / "lynx1231_equity_index_minute.parquet"
MARKET_SCHEMA = "g3-market-spine/1"
PARSER_VERSION = "chicago-end-label-to-eastern-open-v1"
ARRAYS = ["source_ts_local_ns", "close_ts_utc_ns", "open", "high", "low", "close", "volume", "session_id", "source_row_pos"]
DTYPES = {"source_ts_local_ns": "int64", "close_ts_utc_ns": "int64", "open": "float64", "high": "float64",
          "low": "float64", "close": "float64", "volume": "int64", "session_id": "int32", "source_row_pos": "int64"}

# (contract, first CT open label inclusive, last CT open label exclusive); None = unbounded.
# Rule read from the local tape on overlap: Z25->H26 at Sun 2025-12-14, H26->M26 at Sun 2026-03-15.
ROLL = {
    "NQ": [("NQM26", None, "2026-06-13"), ("NQU26", "2026-06-14", None)],
    "ES": [("ESM26", None, "2026-06-13"), ("ESU26", "2026-06-14", None)],
}


def sha256_file(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            d.update(block)
    return d.hexdigest()


def fmt_price(x: float) -> str:
    return ("%.2f" % x).rstrip("0").rstrip(".")


def transform(label: pd.Series):
    """Verbatim canonical transform of the original ingest (git dd278b6, src/g3riz/market.py)."""
    localized = label.dt.tz_localize("America/Chicago", ambiguous="raise", nonexistent="raise")
    open_et = localized.dt.tz_convert("US/Eastern") - pd.Timedelta(minutes=1)
    close_ns = localized.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy(dtype="datetime64[ns]").astype(np.int64)
    source_ns = label.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    normalized = open_et.dt.tz_localize(None).dt.normalize()
    session_dates = normalized + pd.to_timedelta((open_et.dt.hour >= 18).astype("int8"), unit="D")
    days = (session_dates.to_numpy(dtype="datetime64[ns]").astype(np.int64) // 86_400_000_000_000).astype(np.int32)
    return source_ns, close_ns, days


def read_tail_csv(path: Path, nbytes: int = 600_000) -> pd.DataFrame:
    with path.open("rb") as fh:
        fh.seek(0, 2)
        size = fh.tell()
        fh.seek(max(0, size - nbytes))
        raw = fh.read().decode("utf-8", errors="strict")
    lines = raw.split("\n")[1:]  # drop the partial first line
    rows = [ln.strip().split(",") for ln in lines if ln.strip()]
    df = pd.DataFrame(rows, columns=["Date", "Time", "Open", "High", "Low", "Close", "Volume"])
    for c in ("Open", "High", "Low", "Close"):
        df[c] = df[c].astype("float64")
    df["Volume"] = df["Volume"].astype("int64")
    return df


def main(inst: str) -> None:
    inst = inst.upper()
    old_root = REPO / "data" / "market" / inst
    old_manifest = json.loads((old_root / "manifest.json").read_text(encoding="utf-8"))
    n_old = int(old_manifest["rows"])
    old = {name: np.load(old_root / f"{name}.npy", mmap_mode="r") for name in ARRAYS}
    with np.load(old_root / "sessions.npz") as z:
        sess = {k: z[k].copy() for k in z.files}
    last_label = pd.Timestamp(int(old["source_ts_local_ns"][-1]))
    print(inst, "old rows", n_old, "last source label (CT, end)", last_label)

    # ---- self-test of the transform on the original raw tail
    raw_path = (REPO / "data" / "raw" / f"{inst}_1min.csv").resolve()
    tail = read_tail_csv(raw_path)
    k = len(tail)
    lab = pd.to_datetime(tail["Date"] + " " + tail["Time"], format="%Y-%m-%d %H:%M:%S", errors="raise")
    s_ns, c_ns, days = transform(lab)
    assert np.array_equal(s_ns, np.asarray(old["source_ts_local_ns"][-k:])), "source_ts mismatch on raw tail"
    assert np.array_equal(c_ns, np.asarray(old["close_ts_utc_ns"][-k:])), "close_ts mismatch on raw tail"
    for col, name in (("Open", "open"), ("High", "high"), ("Low", "low"), ("Close", "close")):
        assert np.array_equal(tail[col].to_numpy(), np.asarray(old[name][-k:])), f"{name} mismatch on raw tail"
    day_of_sid = sess["session_date_local_days"]
    assert np.array_equal(days, day_of_sid[np.asarray(old["session_id"][-k:])]), "session mapping mismatch on raw tail"
    print("self-test: transform reproduces stored spine on the last", k, "raw rows (ts, OHLC, sessions)")

    # ---- extension rows from contract bars
    lx = pq.read_table(LYNX).to_pandas()
    parts = []
    for code, start, stop in ROLL[inst]:
        g = lx[lx.contract_code == code].copy()
        g["n_open"] = pd.to_datetime(g.timestamp, unit="ms")  # Chicago wall clock, bar-open label
        if start is not None:
            g = g[g.n_open >= pd.Timestamp(start)]
        if stop is not None:
            g = g[g.n_open < pd.Timestamp(stop)]
        g["label"] = g.n_open + pd.Timedelta(minutes=1)  # END label, as in the raw source
        g = g[g.label > last_label]
        g["contract"] = code
        parts.append(g)
    ext = pd.concat(parts).sort_values("label").reset_index(drop=True)
    assert ext.label.is_unique and ext.label.is_monotonic_increasing
    o, h, l, c = (ext[x].to_numpy() for x in ("open", "high", "low", "close"))
    assert not (np.any(l > np.minimum(o, c)) or np.any(h < np.maximum(o, c)) or np.any(h < l)), "OHLC invariant"
    assert np.isfinite(np.column_stack((o, h, l, c))).all()
    tick = 0.25
    assert np.allclose(np.column_stack((o, h, l, c)) / tick, np.round(np.column_stack((o, h, l, c)) / tick)), "tick grid"

    # junction: the last old bar must equal the contract bar carrying the same label
    front = lx[lx.contract_code == ROLL[inst][0][0]].copy()
    front["label"] = pd.to_datetime(front.timestamp, unit="ms") + pd.Timedelta(minutes=1)
    jrow = front[front.label == last_label]
    junction = None
    if len(jrow) == 1:
        r = jrow.iloc[0]
        junction = {"label": str(last_label), "old_hlc": [float(old["high"][-1]), float(old["low"][-1]), float(old["close"][-1])],
                    "contract_hlc": [float(r.high), float(r.low), float(r.close)]}
        print("junction bar:", junction)

    raw_dir = FWD / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    first_d, last_d = ext.label.iloc[0].strftime("%Y%m%d"), ext.label.iloc[-1].strftime("%Y%m%d")
    ext_csv = raw_dir / f"{inst}_1min_ext_{first_d}_{last_d}.csv"
    out = pd.DataFrame({
        "Date": ext.label.dt.strftime("%Y-%m-%d"), "Time": ext.label.dt.strftime("%H:%M:%S"),
        "Open": ext.open.map(fmt_price), "High": ext.high.map(fmt_price), "Low": ext.low.map(fmt_price),
        "Close": ext.close.map(fmt_price), "Volume": ext.volume.astype("int64"),
    })
    out.to_csv(ext_csv, index=False, lineterminator="\n")
    ext_sha = sha256_file(ext_csv)
    print("extension csv", ext_csv.name, "rows", len(out), ext.label.iloc[0], "..", ext.label.iloc[-1])

    # ---- canonical transform of the extension, sessions continued
    s_ns, c_ns, days = transform(ext.label)
    assert c_ns[0] > int(old["close_ts_utc_ns"][-1]) and np.all(np.diff(c_ns) > 0)
    lookup = {int(d): i for i, d in enumerate(sess["session_date_local_days"])}
    s_days = list(map(int, sess["session_date_local_days"]))
    s_open = list(map(int, sess["session_open_utc_ns"]))
    s_first = list(map(int, sess["first_minute_pos"]))
    s_stop = list(map(int, sess["stop_minute_pos"]))
    ids = np.empty(len(ext), dtype=np.int32)
    for day in np.unique(days):
        pos = np.flatnonzero(days == day)
        sid = lookup.get(int(day))
        if sid is None:
            sid = len(s_days)
            lookup[int(day)] = sid
            s_days.append(int(day))
            anchor = pd.Timestamp(int(day) * 86_400_000_000_000).tz_localize("US/Eastern") - pd.Timedelta(hours=6)
            s_open.append(int(anchor.tz_convert("UTC").value))
            s_first.append(n_old + int(pos[0]))
            s_stop.append(n_old + int(pos[-1]) + 1)
        else:
            s_stop[sid] = n_old + int(pos[-1]) + 1
        ids[pos] = sid

    new = {
        "source_ts_local_ns": s_ns, "close_ts_utc_ns": c_ns, "open": o, "high": h, "low": l, "close": c,
        "volume": ext.volume.to_numpy(dtype="int64"), "session_id": ids,
        "source_row_pos": np.arange(n_old, n_old + len(ext), dtype=np.int64),
    }
    dst = FWD / "market" / inst
    dst.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name in ARRAYS:
        arr = np.lib.format.open_memmap(dst / f"{name}.npy", mode="w+", dtype=np.dtype(DTYPES[name]), shape=(n_old + len(ext),))
        arr[:n_old] = old[name]
        arr[n_old:] = new[name].astype(DTYPES[name])
        arr.flush()
        del arr
    np.savez(dst / "sessions.npz",
             session_id=np.arange(len(s_days), dtype=np.int32),
             session_date_local_days=np.asarray(s_days, dtype=np.int32),
             session_open_utc_ns=np.asarray(s_open, dtype=np.int64),
             first_minute_pos=np.asarray(s_first, dtype=np.int64),
             stop_minute_pos=np.asarray(s_stop, dtype=np.int64))
    # prefix identity: the first n_old rows must be byte-equal to the canonical spine
    for name in ARRAYS:
        a = np.load(dst / f"{name}.npy", mmap_mode="r")
        assert np.array_equal(np.asarray(a[:n_old]), np.asarray(old[name])), f"prefix differs: {name}"
        hashes[f"{name}.npy"] = sha256_file(dst / f"{name}.npy")
        del a
    hashes["sessions.npz"] = sha256_file(dst / "sessions.npz")
    print("prefix identity: first", n_old, "rows equal the canonical spine in all", len(ARRAYS), "arrays")

    per_sess = pd.Series(days).map(lambda d: str(pd.Timestamp(int(d) * 86_400_000_000_000).date())).value_counts().sort_index()
    roll_rows = []
    for (code_a, _, _), (code_b, _, _) in zip(ROLL[inst][:-1], ROLL[inst][1:]):
        a_last = ext[ext.contract == code_a].iloc[-1]
        b_first = ext[ext.contract == code_b].iloc[0]
        roll_rows.append({"from": code_a, "to": code_b, "last_label_from": str(a_last.label), "first_label_to": str(b_first.label),
                          "close_from": float(a_last.close), "open_to": float(b_first.open), "gap_points": float(b_first.open - a_last.close)})
    manifest = {
        "schema": MARKET_SCHEMA,
        "instrument": inst,
        "rows": n_old + len(ext),
        "sessions": len(s_days),
        "kind": "FORWARD CONTINUATION - not the canonical spine; canonical prefix is byte-identical",
        "canonical_prefix": {"rows": n_old, "corpus_id": old_manifest["corpus_id"], "source_sha256": old_manifest["source"]["sha256"]},
        "forward_extension": {
            "rows": len(ext),
            "first_label_ct_end": str(ext.label.iloc[0]), "last_label_ct_end": str(ext.label.iloc[-1]),
            "extension_csv": str(ext_csv.relative_to(REPO)).replace("\\", "/"), "extension_csv_sha256": ext_sha,
            "source_dataset": "huggingface:lynx1231/historical-equity-index-futures-data-sample",
            "source_commit": "1e10ad1f78044d494fe00e0b252fad6cbc097e98",
            "source_file": "data/minute/part-00000.parquet", "source_file_sha256": sha256_file(LYNX),
            "source_clock": "integer ms = America/Chicago wall clock, bar-OPEN label (resolved on overlap, not documented by the source)",
            "contracts": [{"contract": c_, "rows": int((ext.contract == c_).sum())} for c_, _, _ in ROLL[inst]],
            "roll_rule": "switch to next quarterly at the Sunday session open of expiration week; read from the local tape on 2025-12-14 and 2026-03-15",
            "rolls": roll_rows, "junction": junction,
            "rows_per_session": {k_: int(v) for k_, v in per_sess.items()},
        },
        "transform": old_manifest["transform"],
        "first_close_utc_ns": int(old_manifest["first_close_utc_ns"]),
        "last_close_utc_ns": int(c_ns[-1]),
        "array_sha256": hashes,
    }
    manifest["corpus_id"] = hashlib.sha256(
        f"{MARKET_SCHEMA}|{inst}|{old_manifest['source']['sha256']}+{ext_sha}|{PARSER_VERSION}".encode()).hexdigest()
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("forward spine ->", dst, "rows", manifest["rows"], "sessions", manifest["sessions"])
    print("rolls:", roll_rows)
    print("rows per session (extension):")
    print(per_sess.to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "NQ")
