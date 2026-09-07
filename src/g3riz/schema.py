"""Схемы паспорта, событий и ячейки плюс версии семантики поля.

Версии здесь — адрес поколения поля; они меняются только вместе с новым
поколением, по отдельному решению трейдера.
"""
from __future__ import annotations

import pyarrow as pa

PASSPORT_SCHEMA_VERSION = "g3-riz-passport/2"
EVENT_SCHEMA_VERSION = "g3-riz-lifecycle-event/2"
CELL_SCHEMA_VERSION = "g3-riz-cell/2"
SEMANTIC_VERSION = "g3-t0-primary-x3-anchors/2"

PASSPORT_SCHEMA = pa.schema([
    ("riz_id", pa.string()), ("corpus_id", pa.string()), ("instrument", pa.string()),
    ("tf_minutes", pa.int16()), ("zone_top", pa.float64()), ("zone_bottom", pa.float64()),
    ("zone_width", pa.float64()), ("direction", pa.int8()), ("bullish", pa.bool_()),
    ("precursor_formed_ts_ns", pa.int64()), ("precursor_formed_spine_pos", pa.int64()),
    ("precursor_native_bar_index", pa.int64()), ("t0_ts_ns", pa.int64()),
    ("t0_spine_pos", pa.int64()), ("t0_kind", pa.string()), ("t0_exit_side", pa.string()),
    ("t0_close", pa.float64()), ("t0_span_count", pa.int32()),
    ("t0_native_bar_index", pa.int64()),
    ("native_blue_confirmation_ts_ns", pa.int64()),
    ("native_blue_confirmation_spine_pos", pa.int64()),
    ("native_blue_confirmation_native_bar_index", pa.int64()),
    ("x3_t0_ts_ns", pa.int64()), ("x3_t0_spine_pos", pa.int64()),
    ("x3_t0_native_bar_index", pa.int64()), ("x3_t0_close", pa.float64()),
    ("x3_t0_exit_side", pa.string()), ("x3_t0_confirmed", pa.bool_()),
    ("blue_eligibility_end_ts_ns", pa.int64()), ("blue_eligibility_end_spine_pos", pa.int64()),
    ("c1_deletion_ts_ns", pa.int64()), ("c1_deletion_spine_pos", pa.int64()),
    ("c1_deletion_native_bar_index", pa.int64()), ("censored", pa.bool_()),
    ("still_alive_at_archive_end", pa.bool_()), ("last_observed_ts_ns", pa.int64()),
    ("last_observed_spine_pos", pa.int64()), ("final_tier", pa.int8()),
    ("final_activated", pa.bool_()), ("final_span_count", pa.int32()),
    ("final_north_alive", pa.bool_()), ("final_south_alive", pa.bool_()),
    ("final_last_span_index", pa.int64()), ("final_broke_south", pa.bool_()),
])

EVENT_SCHEMA = pa.schema([
    ("event_id", pa.string()), ("riz_id", pa.string()), ("instrument", pa.string()),
    ("tf_minutes", pa.int16()), ("event_family", pa.string()), ("event_kind", pa.string()),
    ("event_ts_ns", pa.int64()), ("market_spine_pos", pa.int64()),
    ("native_bar_index", pa.int64()), ("event_seq", pa.int32()),
    ("terminal", pa.bool_()), ("deletion_cause", pa.string()),
    ("breaker_direction", pa.string()),
    ("tier", pa.int8()), ("activated", pa.bool_()), ("span_count", pa.int32()),
    ("north_alive", pa.bool_()), ("south_alive", pa.bool_()),
    ("last_span_index", pa.int64()), ("broke_south", pa.bool_()),
])
