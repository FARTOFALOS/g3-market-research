"""Шаг 1 входной проверки 091-TEMPORAL: линза против поля на старой ленте.

Линза (lens_riz.py) запускается на канонической ленте NQ с пустого состояния с первой
сессии 2024 года, все ТФ 1..1440. Сравниваются RIZ с T0 на отрезке 2025-11-01 ... конец
канонической ленты — отложенном куске 086. Совпадение по riz_id (хеш корпуса, ТФ, времени
рождения, границ, стороны), затем по минуте T0, закрытию T0 и стороне выхода.
Исходов не читает.

    python -B base/091/temporal/gate_lens_vs_field.py
Результат: base/091/temporal/gate_lens_vs_field.json, lens RIZ -> work/091t/lens_canonical_NQ.parquet
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
import lens_riz as LR  # noqa: E402

WARM = "2024-01-01"
FROM = "2025-11-01"
OUT = REPO / "work" / "091t"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    store = LR.load_store(REPO / "data" / "market" / "NQ")
    ts = store["close_ts_utc_ns"]
    warm_pos = int(np.searchsorted(ts, pd.Timestamp(WARM, tz="UTC").value))
    from_pos = int(np.searchsorted(ts, pd.Timestamp(FROM, tz="UTC").value))
    t = time.time()
    parts = [LR.lens_riz(store, "NQ", tf, warm_pos, from_pos) for tf in range(1, 1441)]
    lens = pd.concat(parts, ignore_index=True)
    secs = round(time.time() - t, 1)
    lens.to_parquet(OUT / "lens_canonical_NQ.parquet", index=False)

    cols = ["riz_id", "tf_minutes", "zone_top", "zone_bottom", "t0_spine_pos", "t0_close",
            "t0_exit_side", "t0_kind"]
    field = pd.concat([pq.read_table(REPO / "data/field/NQ/cells" / f"tf_{tf:04d}" / "passports.parquet",
                                     columns=cols).to_pandas() for tf in range(1, 1441)], ignore_index=True)
    field = field[field.t0_spine_pos >= from_pos].reset_index(drop=True)

    m = field.merge(lens, on="riz_id", how="outer", suffixes=("_f", "_l"), indicator=True)
    both = m[m._merge == "both"]
    same_t0 = (both.t0_spine_pos_f == both.t0_spine_pos_l)
    same_all = same_t0 & (both.t0_close_f == both.t0_close_l) & (both.t0_exit_side_f == both.t0_exit_side_l)
    res = {
        "territory": [FROM, "end of canonical NQ tape"], "warm_start": WARM, "tf": "1..1440",
        "lens_seconds": secs,
        "field_riz": int(len(field)), "lens_riz": int(len(lens)),
        "matched_by_riz_id": int(len(both)),
        "field_only": int((m._merge == "left_only").sum()), "lens_only": int((m._merge == "right_only").sum()),
        "matched_same_t0_minute": int(same_t0.sum()),
        "matched_same_t0_close_and_side": int(same_all.sum()),
        "recall_of_field": round(float(same_all.sum() / max(1, len(field))), 5),
        "precision_of_lens": round(float(same_all.sum() / max(1, len(lens))), 5),
    }
    fo = m[m._merge == "left_only"]
    lo = m[m._merge == "right_only"]
    res["field_only_by_tf_band"] = fo.tf_minutes_f.pipe(lambda s: pd.cut(s, [0, 5, 30, 120, 480, 1440]).value_counts().sort_index().astype(int).to_dict()) if len(fo) else {}
    res["lens_only_by_tf_band"] = lo.tf_minutes_l.pipe(lambda s: pd.cut(s, [0, 5, 30, 120, 480, 1440]).value_counts().sort_index().astype(int).to_dict()) if len(lo) else {}
    res = json.loads(json.dumps(res, default=str))
    (HERE / "gate_lens_vs_field.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))
    if len(fo):
        fo[["riz_id", "tf_minutes_f", "t0_spine_pos_f"]].head(20).to_csv(OUT / "field_only_sample.csv", index=False)
    if len(lo):
        lo[["riz_id", "tf_minutes_l", "t0_spine_pos_l"]].head(20).to_csv(OUT / "lens_only_sample.csv", index=False)


if __name__ == "__main__":
    main()
