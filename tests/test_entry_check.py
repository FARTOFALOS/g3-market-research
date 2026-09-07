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


def _boundary(obj: dict) -> float:
    return obj["zone_north"] if obj["t0_exit_side"] == "north" else obj["zone_south"]


def _first_contact_from_tape(rows: list, boundary: float):
    for r in rows:
        if r["t"] > 0 and r["low"] <= boundary <= r["high"]:
            return r["t"]
    return None


def _has_departure(rows: list, boundary: float, side: str, before: int) -> bool:
    """Two consecutive fully outside candles after T0 and before the contact."""
    outside = [r["t"] for r in rows
               if 0 < r["t"] < before
               and (r["low"] > boundary if side == "north" else r["high"] < boundary)]
    return any(t + 1 in outside for t in outside)


@needs_field
def test_the_key_answers_the_tape_the_reader_is_shown(nq_field):
    """A generator bug could print one tape and key another. Then a correct
    reader fails on the examiner's arithmetic, which is what happened on
    2026-09-07 for the tier line and the P5 wording. Recompute every contact
    answer from the printed candles alone."""
    for seed in ("0" * 32, "1" * 32, "a1b2" * 8):
        paper, key = ec.build_decision_paper(nq_field, 54, seed=seed)
        items = {it["id"]: it for it in paper["items"]}

        p3 = items["P3"]
        for label in ("a", "b"):
            got = _first_contact_from_tape(p3["minutes_from_t0"], _boundary(p3[f"object_{label}"]))
            assert str(got) == key["P3"]["anchor"][f"{label}_first_contact"], (seed, "P3", label)

        for pid in ("P4", "P5"):
            for label in ("a", "b"):
                scene = items[pid][f"scene_{label}"]
                boundary = _boundary(scene)
                got = _first_contact_from_tape(scene["minutes_from_t0"], boundary)
                expected = key[pid]["anchor"][f"{label}_first_contact"]
                assert (str(got) if got is not None else "не наблюдалось") == expected, (seed, pid, label)
                if pid == "P4":
                    retest = got is not None and _has_departure(
                        scene["minutes_from_t0"], boundary, scene["t0_exit_side"], got)
                    assert ("да" if retest else "нет") == key["P4"]["anchor"][f"{label}_retest"], (seed, label)
        # Unobserved is never a known absence: B's later outcome stays "нет".
        assert key["P5"]["anchor"]["b_later"] == "нет"


@needs_field
def test_p1_p2_scenes_print_every_condition_the_answer_depends_on(nq_field):
    """The 2026-09-07 reader answered «нельзя определить» and was right: the
    tier condition was not in the scene, and the contract forbids assuming a
    missing condition. Every admission fact the key relies on must be visible."""
    paper, key = ec.build_decision_paper(nq_field, 54, seed="7" * 32)
    md = ec.paper_markdown(paper)
    assert md.count("Текущий tier допустим для Blue и не является breaker: да") == 4
    assert "Blue ещё не наблюдался" in md
    for item in paper["items"][:2]:
        for label in ("a", "b"):
            scene = item[f"scene_{label}"]
            for fact in ("accepted_spans_so_far", "last_accepted_span_native_bars_ago",
                         "boundaries_alive", "current_native_bar", "eligible_tier"):
                assert fact in scene, (item["id"], label, fact)
            assert f"**{scene['current_native_bar']['open']}**" in md
