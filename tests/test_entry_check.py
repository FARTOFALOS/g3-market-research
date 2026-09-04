"""The entry check must measure the reader, not leak the answer.

Two failures would silently turn this instrument into theatre: a native grid
that does not reproduce the bars the field was built on, and an anonymisation
that changes the very relations the questions are about.
"""

from __future__ import annotations

import numpy as np

from g3riz import entry_check as ec
from conftest import needs_field


@needs_field
def test_native_grid_reproduces_the_bars_the_field_recorded(nq_field):
    starts, stops = ec.native_grid(nq_field.market, 54)
    rows = nq_field.passports(tf=54).to_pylist()[:200]
    for row in rows:
        j = int(row["t0_native_bar_index"])
        assert starts[j] <= row["t0_spine_pos"] < stops[j]
        k = int(row["precursor_native_bar_index"])
        assert stops[k] - 1 == row["precursor_formed_spine_pos"]


@needs_field
def test_the_price_shift_leaves_every_span_relation_intact(nq_field):
    corpus = ec.Corpus.load(nq_field, 54)
    row, pos = ec._pool_minute_spans_native_not(corpus)[0]
    scene = ec._scene(corpus, row, pos, 1234.5)
    last = scene["minutes"][-1]
    native_open = scene["current_native_bar"]["open"]
    south, north = scene["zone_south"], scene["zone_north"]
    # The minute body still traverses the zone and the developing native body
    # still does not: this pair is the whole point of the item.
    assert min(last["open"], last["close"]) < south < north < max(last["open"], last["close"])
    assert not (min(native_open, last["close"]) < south
                and max(native_open, last["close"]) > north)


@needs_field
def test_the_paper_carries_no_address_back_into_the_field(nq_field):
    import json

    paper, key = ec.build_paper(nq_field, 54, seed="0" * 32)
    text = json.dumps(paper, ensure_ascii=False)
    assert "riz_" not in text and "ts_ns" not in text and "spine_pos" not in text
    assert key["_meta"]["paper_sha256"] == paper["paper_sha256"]
    for pid, spec in key.items():
        if not pid.startswith("_"):
            assert spec["anchor"]["same_meaning"] in ("разные", "одно и то же")


def test_scoring_reads_the_answer_the_reader_actually_gave():
    key = {"P1": {"anchor": {"a_t0_now": "да", "same_meaning": "разные"}},
           "_meta": {}}
    got = ec.score_anchors(key, {"P1": {"a_t0_now": "Yes", "same_meaning": "одинаковые"}})
    assert got["items"]["P1"]["a_t0_now"]["ok"] is True
    assert got["items"]["P1"]["same_meaning"]["ok"] is False
    assert (got["correct"], got["total"]) == (1, 2)
