"""Consolidation and machine-readable materialization status."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import pyarrow.parquet as pq

from .field import build_identity
from .market import MarketSpine
from .market import sha256_file
from .schema import EVENT_SCHEMA, PASSPORT_SCHEMA, SEMANTIC_VERSION


def read_status(field_root: Path, instruments: tuple[str, ...] = ("ES", "NQ", "YM")) -> dict:
    result = {"schema": "g3-materialization-status/1", "expected_cells": len(instruments) * 1440,
              "complete_cells": 0, "instruments": {}}
    for instrument in instruments:
        cells = field_root / instrument / "cells"
        complete: list[int] = []
        stale: list[int] = []
        invalid: list[dict] = []
        market_path = field_root.parent / "market" / instrument
        market = MarketSpine.open_store(market_path) if (market_path / "manifest.json").exists() else None
        temporary = sorted(
            p.name for pattern in (".tf_*", ".tf=*") for p in cells.glob(pattern) if p.is_dir()
        ) if cells.exists() else []
        for tf in range(1, 1441):
            path = cells / f"tf_{tf:04d}" / "manifest.json"
            if not path.exists():
                continue
            try:
                manifest = json.loads(path.read_text(encoding="utf-8"))
                if manifest.get("status") == "complete" and manifest.get("tf_minutes") == tf:
                    if market is not None and manifest.get("build_identity") != build_identity(market, instrument, tf):
                        stale.append(tf)
                    else:
                        complete.append(tf)
                else:
                    invalid.append({"tf": tf, "reason": "manifest not complete or wrong timeframe"})
            except (OSError, json.JSONDecodeError) as exc:
                invalid.append({"tf": tf, "reason": str(exc)})
        missing = sorted(set(range(1, 1441)) - set(complete))
        consolidated = field_root / instrument / "consolidated" / "manifest.json"
        consolidated_manifest = None
        if consolidated.exists():
            try:
                consolidated_manifest = json.loads(consolidated.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                consolidated_manifest = {"status": "invalid"}
        result["instruments"][instrument] = {
            "complete_count": len(complete), "complete_tfs": complete,
            "missing_count": len(missing), "missing_tfs": missing,
            "stale_count": len(stale), "stale_tfs": stale,
            "invalid": invalid, "temporary_directories": temporary,
            "consolidated": consolidated_manifest,
        }
        result["complete_cells"] += len(complete)
    result["missing_cells"] = result["expected_cells"] - result["complete_cells"]
    result["complete"] = result["missing_cells"] == 0 and not any(
        item["invalid"] for item in result["instruments"].values()
    )
    return result


def consolidate(field_root: Path, instrument: str) -> dict:
    cells_root = field_root / instrument / "cells"
    market = MarketSpine.open_store(field_root.parent / "market" / instrument)
    manifests: list[dict] = []
    missing: list[int] = []
    for tf in range(1, 1441):
        path = cells_root / f"tf_{tf:04d}" / "manifest.json"
        if not path.exists():
            missing.append(tf)
            continue
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if (manifest.get("status") != "complete" or manifest.get("tf_minutes") != tf
                or manifest.get("build_identity") != build_identity(market, instrument, tf)):
            raise ValueError(f"invalid cell manifest for {instrument} TF {tf}")
        manifests.append(manifest)
    if missing:
        raise ValueError(f"cannot consolidate {instrument}: {len(missing)} cells missing; first={missing[:10]}")

    target = field_root / instrument / "consolidated"
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".consolidated.", dir=target.parent))
    passport_writer = pq.ParquetWriter(tmp / "passports.parquet", PASSPORT_SCHEMA,
                                       compression="zstd", use_dictionary=True)
    event_writer = pq.ParquetWriter(tmp / "events.parquet", EVENT_SCHEMA,
                                    compression="zstd", use_dictionary=True)
    passport_rows = event_rows = 0
    try:
        for tf in range(1, 1441):
            cell = cells_root / f"tf_{tf:04d}"
            ptable = pq.read_table(cell / "passports.parquet", schema=PASSPORT_SCHEMA)
            etable = pq.read_table(cell / "events.parquet", schema=EVENT_SCHEMA)
            passport_writer.write_table(ptable)
            event_writer.write_table(etable)
            passport_rows += ptable.num_rows
            event_rows += etable.num_rows
        passport_writer.close()
        event_writer.close()
        digest_payload = "\n".join(m["build_identity"] for m in manifests)
        manifest = {
            "schema": "g3-consolidated-field/1", "status": "complete",
            "semantic_version": SEMANTIC_VERSION, "instrument": instrument,
            "cell_count": 1440, "tf_min": 1, "tf_max": 1440,
            "passports": passport_rows, "events": event_rows,
            "cell_set_digest": hashlib.sha256(digest_payload.encode()).hexdigest(),
            "output_sha256": {
                "passports.parquet": sha256_file(tmp / "passports.parquet"),
                "events.parquet": sha256_file(tmp / "events.parquet"),
            },
        }
        (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                            encoding="utf-8")
        if target.exists():
            old = target.with_name(target.name + ".previous")
            if old.exists():
                shutil.rmtree(old)
            os.replace(target, old)
            os.replace(tmp, target)
            shutil.rmtree(old)
        else:
            os.replace(tmp, target)
        return manifest
    except BaseException:
        passport_writer.close()
        event_writer.close()
        shutil.rmtree(tmp, ignore_errors=True)
        raise
