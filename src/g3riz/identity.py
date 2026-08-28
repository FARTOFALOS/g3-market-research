from __future__ import annotations

import hashlib


def _digest(*parts: object) -> str:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def riz_id(corpus_id: str, instrument: str, tf_minutes: int, birth_known_ts_ns: int,
           top: float, bottom: float, direction: int) -> str:
    return "riz_" + _digest("g3-riz-id/1", corpus_id, instrument, tf_minutes,
                             birth_known_ts_ns, float(top).hex(), float(bottom).hex(),
                             direction)[:32]


def event_id(zone_id: str, event_family: str, event_kind: str, event_ts_ns: int,
             native_bar_index: int, event_seq: int) -> str:
    return "evt_" + _digest("g3-riz-event/1", zone_id, event_family, event_kind,
                             event_ts_ns, native_bar_index, event_seq)[:32]
