"""The entry check must measure the reader, not leak the answer.

Two failures would silently turn this instrument into theatre: a native grid
that does not reproduce the bars the field was built on, and an anonymisation
that changes the very relations the questions are about.
"""

from __future__ import annotations

import numpy as np
import pytest

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


@needs_field
def test_decision_paper_removes_ambiguous_identity_score_and_preserves_legacy(nq_field):
    import copy
    readings = copy.deepcopy(ec.READINGS)
    paper, key = ec.build_decision_paper(nq_field, 54, seed="0" * 32)
    assert ec.READINGS == readings
    assert len(paper["items"]) == 6
    assert all("same_meaning" not in it["ask"] for it in paper["items"])
    assert "sources" in key["_meta"]
    assert "riz_" not in ec.paper_markdown(paper)
    for item in paper["items"][:2]:
        assert item["scene_a"]["eligible_tier"] is True
        assert item["scene_b"]["eligible_tier"] is True
    assert "ПОСЛЕ T0" in paper["items"][4]["ask"]["b_within_shown"]
    # Known absence inside a complete window must not be graded as unknown.
    assert key["P5"]["anchor"]["b_within_shown"] == "нет"
    assert key["P5"]["anchor"]["b_later"] == "нет"


def test_a_foreign_answer_sheet_is_not_a_pass():
    key = {"_meta": {"version": 2, "paper_sha256": "one"}}
    with pytest.raises(ValueError, match="identify"):
        ec.score_anchors(key, {"_meta": {"paper_fingerprint": "another"}})


def test_the_printed_digest_identifies_the_paper_under_either_field_name():
    """A reader who hashed the .md file instead of copying the printed digest
    lost a whole cold run on 2026-09-07. The paper now says which one it wants;
    the older field name still scores the runs already recorded."""
    key = {"_meta": {"version": 2, "paper_sha256": "printed"},
           "P1": {"anchor": {"a_t0_now": "да"}}}
    answered = {"P1": {"a_t0_now": "да"}}
    for meta in ({"paper_fingerprint": "printed"}, {"paper_sha256": "printed"}):
        assert ec.score_anchors(key, dict(answered, _meta=meta))["correct"] == 1
    with pytest.raises(ValueError, match="identify"):
        ec.score_anchors(key, dict(answered, _meta={"paper_sha256": "sha256-of-the-file"}))


def test_correct_words_cannot_hide_wrong_denominator_or_execution():
    key = {"_meta": {"version": 2, "paper_sha256": "same"},
           "P6": {"anchor": {"denominator": "12", "unknown": "3", "entry_bar": "5"}}}
    wrong = {"_meta": {"paper_sha256": "same"},
             "P6": {"denominator": 5, "unknown": 0, "entry_bar": 0,
                    "question": "Объект сохраняем, смотрим по префиксу."}}
    result = ec.score_anchors(key, wrong)
    assert result["correct"] == 0 and result["total"] == 3
    assert result["anchors_pass"] is False
    assert result["free_text_review_required"] is True


def test_calendar_horizon_distinguishes_absence_from_a_gap():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("contract_example", Path(__file__).resolve().parents[1] / "reference/contract_example.py")
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    ts = np.arange(13, dtype=np.int64) * example.MINUTE
    high, low = np.full(13, 9.0), np.full(13, 8.0)
    assert example.label(high, low, ts, 10.0) == ("no", None)
    assert example.label(high[:-1], low[:-1], ts[:-1], 10.0) == ("unknown", None)
    mask = np.arange(13) != 7
    assert example.label(high[mask], low[mask], ts[mask], 10.0) == ("unknown", None)
