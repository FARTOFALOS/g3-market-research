"""Идентичность объектов поля: как считается riz_id и отпечаток сборщика.

Менять здесь что-либо значит менять сами объекты, а не их чтение.
"""
from __future__ import annotations

import hashlib

from .schema import CELL_SCHEMA_VERSION, SEMANTIC_VERSION


FROZEN_MATERIALIZER_CODE_IDENTITY = (
    "8032f281849a7d4c723357b34db37935401cf9dad6514466e137cfd1c1398adb"
)


def _digest(*parts: object) -> str:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def riz_id(corpus_id: str, instrument: str, tf_minutes: int, precursor_formed_ts_ns: int,
           top: float, bottom: float, direction: int) -> str:
    return "riz_" + _digest("g3-riz-id/1", corpus_id, instrument, tf_minutes,
                             precursor_formed_ts_ns, float(top).hex(), float(bottom).hex(),
                             direction)[:32]


def event_id(zone_id: str, event_family: str, event_kind: str, event_ts_ns: int,
             native_bar_index: int, event_seq: int) -> str:
    return "evt_" + _digest("g3-riz-event/1", zone_id, event_family, event_kind,
                             event_ts_ns, native_bar_index, event_seq)[:32]


def build_identity(market, instrument: str, tf_minutes: int) -> str:
    value = (f"{CELL_SCHEMA_VERSION}|{SEMANTIC_VERSION}|{FROZEN_MATERIALIZER_CODE_IDENTITY}|"
             f"{market.manifest['corpus_id']}|{instrument}|{tf_minutes}")
    return hashlib.sha256(value.encode()).hexdigest()
