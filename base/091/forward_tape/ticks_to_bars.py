"""Aggregate Rithmic tick parquet (EdgeArbiter/rithmic-tick-data) into 1-minute bars, row group by row group.

Bar label = floor(timestamp, 1 min) in the file's own naive clock (time zone resolved later on overlap).
"""
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[3]


def build(src: Path, dst: Path) -> None:
    pf = pq.ParquetFile(src)
    parts = []
    for rg in range(pf.metadata.num_row_groups):
        t = pf.read_row_group(rg, columns=["timestamp", "price", "volume"]).to_pandas()
        t["m"] = t["timestamp"].dt.floor("min")
        g = t.groupby("m", sort=True).agg(
            open=("price", "first"), high=("price", "max"), low=("price", "min"), close=("price", "last"),
            volume=("volume", "sum"), ticks=("price", "size"),
        )
        parts.append(g)
        if rg % 10 == 0:
            print("row group", rg, "/", pf.metadata.num_row_groups, g.index[0], g.index[-1], flush=True)
    allp = pd.concat(parts)
    bars = allp.groupby(level=0, sort=True).agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"),
        volume=("volume", "sum"), ticks=("ticks", "sum"),
    )
    bars.to_parquet(dst)
    print("bars", len(bars), bars.index[0], "..", bars.index[-1], "->", dst)


if __name__ == "__main__":
    inst = sys.argv[1].lower()
    build(
        REPO / f"data/forward/_incoming/edgearbiter_{inst}_tick_rithmic_2026.parquet",
        REPO / f"data/forward/_incoming/rithmic_{inst}_1m_naive.parquet",
    )
