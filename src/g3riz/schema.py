from __future__ import annotations

import pyarrow as pa

PASSPORT_SCHEMA_VERSION = "g3-riz-passport/1"
EVENT_SCHEMA_VERSION = "g3-riz-lifecycle-event/1"
CELL_SCHEMA_VERSION = "g3-riz-cell/1"
SEMANTIC_VERSION = "g2-c1-late-b394cd8-plus-complete-sparse-observer/1"

PASSPORT_SCHEMA = pa.schema([
    ("riz_id", pa.string()), ("corpus_id", pa.string()), ("instrument", pa.string()),
    ("tf_minutes", pa.int16()), ("zone_top", pa.float64()), ("zone_bottom", pa.float64()),
    ("zone_width", pa.float64()), ("direction", pa.int8()), ("bullish", pa.bool_()),
    ("birth_label_ts_ns", pa.int64()), ("birth_known_ts_ns", pa.int64()),
    ("birth_known_spine_pos", pa.int64()), ("birth_native_bar_index", pa.int64()),
    ("first_activation_ts_ns", pa.int64()), ("first_activation_spine_pos", pa.int64()),
    ("first_activation_native_bar_index", pa.int64()), ("t0_ts_ns", pa.int64()),
    ("t0_spine_pos", pa.int64()), ("t0_kind", pa.string()), ("t0_exit_side", pa.string()),
    ("t0_close", pa.float64()), ("t0_span_count", pa.int32()),
    ("t0_native_bar_index", pa.int64()), ("t0_preview_outcome", pa.string()),
    ("t0_preview_outcome_known_at_ns", pa.int64()),
    ("t0_preview_outcome_known_at_spine_pos", pa.int64()),
    ("first_confirmation_ts_ns", pa.int64()), ("first_confirmation_spine_pos", pa.int64()),
    ("first_confirmation_native_bar_index", pa.int64()),
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
    ("event_ts_ns", pa.int64()), ("known_at_ns", pa.int64()), ("market_spine_pos", pa.int64()),
    ("native_bar_index", pa.int64()), ("event_seq", pa.int32()),
    ("occurrence_index", pa.int32()), ("terminal", pa.bool_()),
    ("cancelled_stage", pa.string()), ("preview_outcome", pa.string()),
    ("preview_outcome_known_at_ns", pa.int64()),
    ("preview_outcome_known_at_spine_pos", pa.int64()),
    ("tier", pa.int8()), ("activated", pa.bool_()), ("span_count", pa.int32()),
    ("north_alive", pa.bool_()), ("south_alive", pa.bool_()),
    ("last_span_index", pa.int64()), ("broke_south", pa.bool_()),
])
