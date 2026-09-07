"""Cold-entry semantic check: does the entry path transfer the object?

This is a measurement of the ENTRY DOCUMENTS, not a grade for an agent. A
fresh reader gets pairs of anonymised prefixes that look alike on the candles
and differ in meaning. Answering needs the scene, not the column names.

Every scene is affine-shifted in price and stripped of dates, instrument and
`riz_id`, so no item can be answered by finding the row in the field. Nothing
here writes to the field or instantiates the machine.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import secrets

import numpy as np

RU_SIDE = {"north": "север", "south": "юг"}
MINUTE_NS = 60_000_000_000
TICK = 0.25


def native_grid(market, tf: int) -> tuple[np.ndarray, np.ndarray]:
    """First and stop minute positions of every native bar, session by session.

    Read-only aggregation over the stored spine, on the same session grid the
    field was built with. It reproduces bar boundaries; it does not re-run the
    RIZ machine or change any stored byte.
    """
    starts: list[int] = []
    stops: list[int] = []
    sessions = market.sessions
    for sid in range(len(sessions["session_id"])):
        a = int(sessions["first_minute_pos"][sid])
        b = int(sessions["stop_minute_pos"][sid])
        anchor = int(sessions["session_open_utc_ns"][sid])
        keys = ((np.asarray(market.close_ts_utc_ns[a:b]) - anchor) // MINUTE_NS - 1) // tf
        first = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1]) + 1]
        last = np.r_[first[1:], b - a]
        starts.extend((first + a).tolist())
        stops.extend((last + a).tolist())
    return np.asarray(starts), np.asarray(stops)


@dataclass
class Corpus:
    field: object
    tf: int
    starts: np.ndarray
    stops: np.ndarray
    passports: list
    events: dict

    @classmethod
    def load(cls, field, tf: int) -> "Corpus":
        starts, stops = native_grid(field.market, tf)
        rows = field.passports(tf=tf).to_pylist()
        events: dict[str, list] = {}
        for e in field.events(tf=tf).to_pylist():
            events.setdefault(e["riz_id"], []).append(e)
        for lst in events.values():
            lst.sort(key=lambda e: (e["market_spine_pos"], e["event_seq"]))
        return cls(field, tf, starts, stops, rows, events)

    def bar_of(self, pos: int) -> int:
        return int(np.searchsorted(self.stops, pos, side="right"))

    def spans(self, riz_id: str) -> list:
        return [e for e in self.events.get(riz_id, []) if e["event_kind"] == "accepted_span"]


def _round_tick(x: float) -> float:
    return round(round(x / TICK) * TICK, 2)


def _scene(corpus: Corpus, row: dict, at_pos: int, offset: float, tape: int = 8) -> dict:
    """The prefix a trader would have at `at_pos`: no future field columns."""
    m = corpus.field.market
    j = corpus.bar_of(at_pos)
    bar_start = int(corpus.starts[j])
    accepted = [e for e in corpus.spans(row["riz_id"]) if e["market_spine_pos"] <= at_pos]
    last_bar = max((int(e["native_bar_index"]) for e in accepted), default=None)
    minutes = []
    for p in range(max(at_pos - tape + 1, bar_start - 40), at_pos + 1):
        minutes.append({
            "t": int(p - at_pos),
            "open": _round_tick(float(m.open[p]) + offset),
            "high": _round_tick(float(m.high[p]) + offset),
            "low": _round_tick(float(m.low[p]) + offset),
            "close": _round_tick(float(m.close[p]) + offset),
        })
    alive = {"north": True, "south": True}
    for e in corpus.events.get(row["riz_id"], []):
        if e["market_spine_pos"] <= at_pos:
            alive = {"north": bool(e["north_alive"]), "south": bool(e["south_alive"])}
    return {
        "native_tf_minutes": corpus.tf,
        "tick": TICK,
        "zone_south": _round_tick(float(row["zone_bottom"]) + offset),
        "zone_north": _round_tick(float(row["zone_top"]) + offset),
        "accepted_spans_so_far": len(accepted),
        "last_accepted_span_native_bars_ago": None if last_bar is None else j - last_bar,
        "boundaries_alive": alive,
        "current_native_bar": {
            "open": _round_tick(float(m.open[bar_start]) + offset),
            "minutes_since_bar_open": int(at_pos - bar_start),
        },
        "minutes": minutes[-tape:],
    }


def _film(corpus: Corpus, row: dict, offset: float, window: int) -> tuple[dict, object]:
    """Minutes after T0 plus the honest end of the shown window."""
    m = corpus.field.market
    t0 = int(row["t0_spine_pos"])
    side = row["t0_exit_side"]
    boundary = float(row["zone_top"] if side == "north" else row["zone_bottom"])
    end_pos = int(row["last_observed_spine_pos"])
    stop = min(t0 + window, end_pos)
    minutes = []
    contact = None
    for p in range(t0, stop + 1):
        k = p - t0
        minutes.append({
            "t": int(k),
            "open": _round_tick(float(m.open[p]) + offset),
            "high": _round_tick(float(m.high[p]) + offset),
            "low": _round_tick(float(m.low[p]) + offset),
            "close": _round_tick(float(m.close[p]) + offset),
        })
        if k >= 1 and contact is None and float(m.low[p]) <= boundary <= float(m.high[p]):
            contact = k
    shown = {
        "native_tf_minutes": corpus.tf,
        "tick": TICK,
        "zone_south": _round_tick(float(row["zone_bottom"]) + offset),
        "zone_north": _round_tick(float(row["zone_top"]) + offset),
        "t0_exit_side": side,
        "minutes_from_t0": minutes,
        "tape_ends_after": len(minutes) - 1,
        "tape_end_reason": "archive_edge" if stop == end_pos else "observation_budget",
    }
    return shown, contact


def _offset(rng, level: float) -> float:
    return _round_tick(rng.uniform(5000.0, 25000.0) - level)


def _minute_body_spans(m, pos: int, bottom: float, top: float) -> bool:
    o, c = float(m.open[pos]), float(m.close[pos])
    return min(o, c) < bottom and max(o, c) > top


def _native_body_spans(corpus: Corpus, pos: int, bottom: float, top: float) -> bool:
    m = corpus.field.market
    j = corpus.bar_of(pos)
    o, c = float(m.open[int(corpus.starts[j])]), float(m.close[pos])
    return min(o, c) < bottom and max(o, c) > top


def _pool_t0_minute_ignition(corpus: Corpus) -> list:
    out = []
    for r in corpus.passports:
        if r["t0_kind"] != "minute_ignition" or r["t0_span_count"] != 1:
            continue
        spans = corpus.spans(r["riz_id"])
        if not spans:
            continue
        j1 = min(int(e["native_bar_index"]) for e in spans)
        if int(r["t0_native_bar_index"]) < j1 + 2:
            continue
        pos = int(r["t0_spine_pos"])
        if _minute_body_spans(corpus.field.market, pos, r["zone_bottom"], r["zone_top"]):
            continue
        out.append(r)
    return out


def _pool_minute_spans_native_not(corpus: Corpus) -> list:
    """Pre-T0 minutes whose MINUTE body traverses the zone while the developing
    native body does not. A detector written on the minute body says T0 here."""
    m = corpus.field.market
    out = []
    for r in corpus.passports:
        spans = corpus.spans(r["riz_id"])
        if not spans:
            continue
        j1 = min(int(e["native_bar_index"]) for e in spans)
        t0 = int(r["t0_spine_pos"])
        lo = int(corpus.stops[j1 + 1]) if j1 + 1 < len(corpus.stops) else t0
        if t0 - lo < 3 or t0 - lo > 4000:
            continue
        bottom, top = r["zone_bottom"], r["zone_top"]
        o = np.asarray(m.open[lo:t0])
        c = np.asarray(m.close[lo:t0])
        hit = np.flatnonzero((np.minimum(o, c) < bottom) & (np.maximum(o, c) > top))
        for i in hit:
            pos = lo + int(i)
            if corpus.bar_of(pos) < j1 + 2:
                continue
            if _native_body_spans(corpus, pos, bottom, top):
                continue
            out.append((r, pos))
            break
    return out


def _first_contact(corpus: Corpus, row: dict, window: int) -> tuple[object, int, str]:
    m = corpus.field.market
    t0 = int(row["t0_spine_pos"])
    end = int(row["last_observed_spine_pos"])
    side = row["t0_exit_side"]
    boundary = float(row["zone_top"] if side == "north" else row["zone_bottom"])
    stop = min(t0 + window, end)
    for p in range(t0 + 1, stop + 1):
        if float(m.low[p]) <= boundary <= float(m.high[p]):
            return p - t0, stop - t0, ("archive_edge" if stop == end else "observation_budget")
    return None, stop - t0, ("archive_edge" if stop == end else "observation_budget")


AXES = {
    "P1": "чьё тело читает машина в T0",
    "P2": "будущее не дотягивается назад",
    "P3": "тождество принадлежит объекту, а не минуте",
    "P4": "контакт и уход — разные события",
    "P5": "ненаблюдённое не равно небывшему",
}

# Pre-registered before any answer is seen. A reader of columns lands on
# `naive`; a reader of the scene lands on `object`. Classifying an answer into
# one of the two is what this check measures; the anchors below only catch a
# gross misread.
READINGS = {
    "P1": {
        "same_object": False,
        "naive": "B — это T0, потому что минутное тело прошило область целиком; "
                 "в A такого прошива нет, значит T0 нет.",
        "object": "T0 задаёт развивающееся тело НАТИВНОЙ свечи, а не минутной. "
                  "A — уже объект популяции Blue/2X, B — ещё предыстория с одним "
                  "принятым спаном, сколько бы минутных тел ни прошило область.",
    },
    "P2": {
        "same_object": True,
        "naive": "Неподтверждённая сцена — не RIZ или ошибочная запись; "
                 "по префиксу видно, какая из двух подтвердится.",
        "object": "На префиксе они не различаются, и различить их там нельзя. "
                  "Обе — наблюдавшийся RIZ; подтверждение или его отсутствие "
                  "наступает позже и прошлого наблюдения не отменяет.",
    },
    "P3": {
        "same_object": False,
        "naive": "Одна минута T0 — одна сцена, и исход у неё один.",
        "object": "Два объекта со своими границами и своими фильмами. Общая "
                  "минута говорит о зависимости наблюдений, а не о тождестве; "
                  "схлопывание по минуте стирает один из двух путей.",
    },
    "P4": {
        "same_object": True,
        "naive": "Обе сцены — ретест выходной границы, A просто быстрее.",
        "object": "Обе — первый контакт, ни одна не установлена как ретест. "
                  "Ретест требует отдельно определённого и наблюдённого ухода; "
                  "продолжение контакта на +1 уходом не является.",
    },
    "P5": {
        "same_object": False,
        "naive": "В B контакта не было; это отрицательный исход.",
        "object": "В B контакт не наблюдался: лента кончилась раньше. Это "
                  "цензура наблюдения. Случай остаётся в знаменателе вопроса "
                  "о частоте и нулём исхода не становится.",
    },
}

FIRST_CONTACT_ASK = {
    "a_first_contact": "На какой минуте после T0 в A первый контакт с выходной "
                       "границей? Ответь числом либо словами «не наблюдалось».",
    "b_first_contact": "То же для B: число либо «не наблюдалось».",
}

ASK = {
    "same_meaning": "Разные ли это по смыслу вещи? Ответь одним словом из двух: "
                    "«разные» либо «одно и то же».",
    "difference": "Чем эти две сцены различаются КАК ОБЪЕКТЫ, или почему не "
                  "различаются. Не перечисляй числа: скажи, что это за вещи.",
    "question": "Один осмысленный рыночный вопрос об этой паре и наблюдение, "
                "которое его разрешит.",
    "unknowable": "Чего в эти минуты знать нельзя.",
}


def build_paper(field, tf: int = 54, seed: str | None = None, *,
                record_sources: bool = False) -> tuple[dict, dict]:
    """Five pairs, one misreading axis each.

    Two pairs look different and are the same object question; three look
    similar and are different. A reader that always answers "different" fails
    as surely as one that always answers "same".
    """
    import random

    seed = seed or secrets.token_hex(16)
    rng = random.Random(int(seed, 16))
    corpus = Corpus.load(field, tf)
    items: list[dict] = []
    key: dict[str, dict] = {}
    ru = {"north": "север", "south": "юг"}

    # P1 -- the native body decides T0, the minute body does not -------------
    a_row = rng.choice(_pool_t0_minute_ignition(corpus))
    b_row, b_pos = rng.choice(_pool_minute_spans_native_not(corpus))
    items.append({
        "id": "P1",
        "title": "Две минуты у границы области",
        "scene_a": _scene(corpus, a_row, int(a_row["t0_spine_pos"]),
                          _offset(rng, float(a_row["t0_close"]))),
        "scene_b": _scene(corpus, b_row, b_pos,
                          _offset(rng, float(field.market.close[b_pos]))),
        "ask": dict(ASK,
                    a_t0_now="Есть ли T0 в оцениваемую минуту сцены A? «да» либо «нет».",
                    b_t0_now="Есть ли T0 в оцениваемую минуту сцены B? «да» либо «нет»."),
    })
    key["P1"] = {"anchor": {"a_t0_now": "да", "b_t0_now": "нет"},
                 "a_exit_side": ru[a_row["t0_exit_side"]]}

    # P2 -- the same prefix, different futures -------------------------------
    pool = _pool_t0_minute_ignition(corpus)
    c_row = rng.choice([r for r in pool if r["native_blue_confirmation_ts_ns"] is not None])
    d_row = rng.choice([r for r in pool if r["native_blue_confirmation_ts_ns"] is None])
    pair = [(c_row, "confirmed"), (d_row, "flicker")]
    rng.shuffle(pair)
    items.append({
        "id": "P2",
        "title": "Два наблюдения Blue",
        "note": "У одной из этих сцен нативный бар позже подтвердил Blue, у другой нет.",
        "scene_a": _scene(corpus, pair[0][0], int(pair[0][0]["t0_spine_pos"]),
                          _offset(rng, float(pair[0][0]["t0_close"]))),
        "scene_b": _scene(corpus, pair[1][0], int(pair[1][0]["t0_spine_pos"]),
                          _offset(rng, float(pair[1][0]["t0_close"]))),
        "ask": dict(ASK, which_is_confirmed="Какая из сцен позже подтвердится "
                    "нативным баром? «A», «B» либо «нельзя определить»."),
    })
    key["P2"] = {"anchor": {"which_is_confirmed": "нельзя определить"},
                 "hidden_truth": pair[0][1]}

    # P3 -- one T0 minute carrying two objects -------------------------------
    by_pos: dict[int, list] = {}
    for r in corpus.passports:
        by_pos.setdefault(int(r["t0_spine_pos"]), []).append(r)
    shared = []
    for rows in by_pos.values():
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                x, y = rows[i], rows[j]
                bx = x["zone_top"] if x["t0_exit_side"] == "north" else x["zone_bottom"]
                by = y["zone_top"] if y["t0_exit_side"] == "north" else y["zone_bottom"]
                kx = _first_contact(corpus, x, 30)[0]
                ky = _first_contact(corpus, y, 30)[0]
                if bx != by and kx != ky:
                    shared.append((x, y, kx, ky))
    x_row, y_row, kx, ky = rng.choice(shared)
    off = _offset(rng, float(x_row["t0_close"]))
    film_x, _ = _film(corpus, x_row, off, 30)
    film_y, _ = _film(corpus, y_row, off, 30)
    items.append({
        "id": "P3",
        "title": "Одна минута T0, две области",
        "note": "У обоих объектов T0 в одну и ту же минуту одного инструмента. "
                "Лента минут общая, области и выходные границы разные.",
        "object_a": {k: v for k, v in film_x.items() if k != "minutes_from_t0"},
        "object_b": {k: v for k, v in film_y.items() if k != "minutes_from_t0"},
        "minutes_from_t0": film_x["minutes_from_t0"],
        "ask": dict(ASK, **FIRST_CONTACT_ASK),
    })
    key["P3"] = {"anchor": {
        "a_first_contact": "не наблюдалось" if kx is None else str(kx),
        "b_first_contact": "не наблюдалось" if ky is None else str(ky)}}

    # P4 -- contact at +1 against contact after an absence -------------------
    imm, late = [], []
    for r in corpus.passports:
        k, shown, _ = _first_contact(corpus, r, 40)
        if k == 1 and shown >= 20:
            imm.append(r)
        elif k is not None and 8 <= k <= 25:
            late.append(r)
    e_row, f_row = rng.choice(imm), rng.choice(late)
    film_e, ke = _film(corpus, e_row, _offset(rng, float(e_row["t0_close"])), 30)
    film_f, kf = _film(corpus, f_row, _offset(rng, float(f_row["t0_close"])), 30)
    items.append({
        "id": "P4",
        "title": "Два первых касания выходной границы",
        "scene_a": film_e,
        "scene_b": film_f,
        "ask": dict(ASK, **FIRST_CONTACT_ASK),
    })
    key["P4"] = {"anchor": {"a_first_contact": str(ke), "b_first_contact": str(kf)}}

    # P5 -- an observed absence against an unobserved one --------------------
    inside, outside = [], []
    for r in corpus.passports:
        k20 = _first_contact(corpus, r, 20)[0]
        k40 = _first_contact(corpus, r, 40)[0]
        if k20 is not None and 3 <= k20 <= 18:
            inside.append(r)
        elif k20 is None and k40 is not None:
            outside.append(r)
    h_row = rng.choice(inside)
    g_row = rng.choice(outside)
    film_h, kh = _film(corpus, h_row, _offset(rng, float(h_row["t0_close"])), 20)
    film_g, kg = _film(corpus, g_row, _offset(rng, float(g_row["t0_close"])), 20)
    assert kg is None
    items.append({
        "id": "P5",
        "title": "Две ленты без продолжения",
        "scene_a": film_h,
        "scene_b": film_g,
        "ask": dict(ASK, **FIRST_CONTACT_ASK),
    })
    key["P5"] = {"anchor": {"a_first_contact": str(kh),
                            "b_first_contact": "не наблюдалось"}}

    paper = {
        "what_this_is": "Проверка входа, а не аттестация агента. Пять пар сцен: "
                        "в каждой две сцены похожи на свечах и могут различаться "
                        "как объекты, а могут и не различаться. Отвечай по префиксу.",
        "rules": [
            "Не открывай src/g3riz/entry_check.py и не запускай генератор.",
            "Цены сдвинуты, даты и инструмент убраны: искать эти сцены в поле бессмысленно.",
            "Где на префиксе ответа нет — так и напиши: это правильный ответ.",
            "Отвечай тем, что это за объекты, а не тем, как называются колонки.",
            "Пиши сам ответ, а не подсказку формата из списка «Ответь».",
        ],
        "native_tf_minutes": tf,
        "items": items,
    }
    paper["paper_sha256"] = hashlib.sha256(
        json.dumps(paper, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    for pid in list(key):
        key[pid]["axis"] = AXES[pid]
        key[pid]["readings"] = READINGS[pid]
        key[pid]["anchor"]["same_meaning"] = (
            "одно и то же" if READINGS[pid]["same_object"] else "разные")
    key["_meta"] = {"seed": seed, "tf": tf, "paper_sha256": paper["paper_sha256"]}
    if record_sources:
        key["_meta"]["sources"] = {
            "P1": [(a_row["riz_id"], int(a_row["t0_spine_pos"])), (b_row["riz_id"], b_pos)],
            "P2": [(r["riz_id"], int(r["t0_spine_pos"])) for r, _ in pair],
            "P3": [(r["riz_id"], int(r["t0_spine_pos"])) for r in (x_row, y_row)],
            "P4": [(r["riz_id"], int(r["t0_spine_pos"])) for r in (e_row, f_row)],
            "P5": [(r["riz_id"], int(r["t0_spine_pos"])) for r in (h_row, g_row)],
        }
        for pid in ("P1", "P2"):
            item = next(it for it in items if it["id"] == pid)
            for label, (rid, pos) in zip(("scene_a", "scene_b"), key["_meta"]["sources"][pid]):
                item[label]["current_native_bar"]["closed"] = bool(pos == corpus.stops[corpus.bar_of(pos)] - 1)
                seen = [e for e in corpus.events[rid] if e["market_spine_pos"] <= pos]
                item[label]["eligible_tier"] = int(seen[-1]["tier"]) in (1, 2)
    return paper, key


def build_decision_paper(field, tf: int = 54, seed: str | None = None) -> tuple[dict, dict]:
    """V2 asks for consequences, without v1's ambiguous 'same meaning' score.

    V1 remains reproducible. Prices/relations come from its field scenes;
    addresses stay with the examiner. Free reasoning still requires review.
    """
    import copy

    paper, key = build_paper(field, tf, seed, record_sources=True)
    key = copy.deepcopy(key)  # v2 must not mutate the shared legacy READINGS
    paper["version"] = 2
    paper["what_this_is"] = "Прочитай сцены и выбери исследовательское действие. " \
        "Это учебная выборка, подобранная по различиям, включая исходы; по ней нельзя оценивать частоты."
    paper["rules"] += [
        "Скопируй строку «Отпечаток задания» ниже в _meta.paper_fingerprint своего ответа. "
        "Это отпечаток самого задания, а не хеш файла: считать sha256 файла не нужно.",
        "Номера свечей — порядковые позиции ленты; календарное время через разрывы из них не выводится.",
        "В P1/P2 до оцениваемой свечи Blue ещё не наблюдался. Прочие условия допуска бери из сцены.",
        "Для каждой пары объясни: какой вопрос к данным меняется, какой тест отпадает, какой нужен взамен.",
    ]
    for item in paper["items"]:
        pid = item["id"]
        item["ask"].pop("same_meaning")
        item["ask"]["difference"] = "Какие свойства совпадают и какие различаются? Приведи основание из сцены."
        item["ask"]["question"] = "Вопрос к данным, неподходящий тест и следующий проверяемый тест."
        key[pid]["anchor"].pop("same_meaning")
    p2 = paper["items"][1]
    p2["ask"]["keep_both"] = "Включить ли обе сцены в выборку по уже наблюдаемому T0? «да» / «нет»."
    key["P2"]["anchor"]["keep_both"] = "да"
    key["P2"]["readings"]["object"] = (
        "Разные префиксы имеют одинаковый статус наблюдаемого Blue. Будущее не известно; "
        "гипотеза о вероятности подтверждения допустима, исключение будущего flicker из раннего отбора — нет.")
    p3 = paper["items"][2]
    # Both answers must be recoverable from the SAME displayed tape.
    for label in ("a", "b"):
        obj = p3[f"object_{label}"]
        boundary = obj["zone_north"] if obj["t0_exit_side"] == "north" else obj["zone_south"]
        found = next((r["t"] for r in p3["minutes_from_t0"]
                      if 0 < r["t"] <= obj["tape_ends_after"] and r["low"] <= boundary <= r["high"]), None)
        key["P3"]["anchor"][f"{label}_first_contact"] = "не наблюдалось" if found is None else str(found)
    p3["ask"].update(objects="Сколько RIZ нужно сохранить? Число.",
                    independent="Доказаны ли две независимые возможности? «да» / «нет».")
    key["P3"]["anchor"].update(objects="2", independent="нет")
    p4 = paper["items"][3]
    p4["note"] = ("Только для этой исследовательской линзы уход означает две подряд полностью "
                  "внешние свечи после T0 ДО первого контакта. Это не условие принадлежности к RIZ.")
    for label in ("a", "b"):
        scene = p4[f"scene_{label}"]
        k = int(key["P4"]["anchor"][f"{label}_first_contact"])
        up = scene["t0_exit_side"] == "north"
        boundary = scene["zone_north"] if up else scene["zone_south"]
        before = [r for r in scene["minutes_from_t0"] if 0 < r["t"] < k]
        outside = [r["low"] > boundary if up else r["high"] < boundary for r in before]
        departure = any(a and b for a, b in zip(outside, outside[1:]))
        p4["ask"][f"{label}_retest"] = f"Первый контакт {label.upper()} — ретест по заданной линзе? «да» / «нет»."
        key["P4"]["anchor"][f"{label}_retest"] = "да" if departure else "нет"
    key["P4"]["readings"]["object"] = "Контакт определяет лента; ретест требует отдельно заданного ухода. Применить предикат, не угадывать по задержке."
    p5 = paper["items"][4]
    p5["ask"].update(b_within_shown="Был ли контакт B ПОСЛЕ T0 внутри показанного окна? «да» / «нет».",
                    b_later="Известно ли, что контакт B не наступит позже показанного окна? «да» / «нет».")
    key["P5"]["anchor"].update(b_within_shown="нет", b_later="нет")
    key["P5"]["readings"]["object"] = "Отсутствие в полностью показанном окне известно; исход за окном неизвестен. Для более длинного горизонта это недосмотренный случай."
    paper["items"].append({
        "id": "P6", "title": "До будущего паттерна", "task":
        "Учебная ситуация: T0 уже наблюдён. На закрытии +4 узнаётся состояние S. "
        "Событие Y — первый контакт после S до +14. Правило A берёт все S, включая будущие "
        "неуспехи; B берёт только S с будущим Y и исключает будущие flicker. "
        "Из 12 отобранных по A случаев: 5 с Y, 4 полностью наблюдались до +14 без Y, "
        "у 3 лента оборвалась раньше +14 без Y. Успешный Y в одном примере начинается на +8. "
        "Решение принимается после закрытия +4; модель исполнения — open следующей свечи, "
        "если она есть без разрыва. Оба правила придуманы после просмотра этого архива.",
        "ask": {"rule": "Какое правило отбирает по доступному состоянию? «A» / «B».",
                "denominator": "Сколько всего возможностей в знаменателе? Число.",
                "known_no": "Сколько известных неуспехов до +14? Число.",
                "unknown": "Сколько неизвестных исходов до +14? Число.",
                "entry_bar": "На open какой свечи первое допустимое исполнение? Число.",
                "independent_confirmation": "Повтор на этом же архиве независимо подтверждает идею? «да» / «нет».",
                "question": "Предложи один эксперимент о раннем состоянии: узнавание, исход, опережение, отрицательный результат и решение по нему.",
                "rejected_test": "Какой соблазнительный тест здесь отвечает на другой вопрос и почему?"}})
    key["P6"] = {"axis": "префикс → будущий ответ → действие",
                 "anchor": {"rule": "A", "denominator": "12", "known_no": "4", "unknown": "3",
                            "entry_bar": "5", "independent_confirmation": "нет"},
                 "readings": {"naive": "Искать предвестник только среди успешных Y, прибыль считать от T0.",
                              "object": "Отбор по S, Y отдельно, весь знаменатель, неизвестное отдельно; действие после узнавания. Нужен проверяемый следующий эксперимент."}}
    paper.pop("paper_sha256")
    paper["paper_sha256"] = hashlib.sha256(json.dumps(paper, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    key["_meta"].update(version=2, instrument=field.instrument, paper_sha256=paper["paper_sha256"])
    return paper, key


def score_anchors(key: dict, answers: dict) -> dict:
    """Machine part only. The readings are classified by the operator."""
    if key.get("_meta", {}).get("version") == 2:
        meta = answers.get("_meta", {})
        # `paper_fingerprint` is what the paper asks for; `paper_sha256` is the
        # older field name, kept so recorded runs stay scorable.
        shown = meta.get("paper_fingerprint", meta.get("paper_sha256"))
        if shown != key["_meta"]["paper_sha256"]:
            raise ValueError("answers do not identify this decision paper: "
                             "_meta.paper_fingerprint must repeat the printed digest")
    def norm(v) -> str:
        s = str(v).strip().lower().replace("ё", "е")
        table = {"yes": "да", "no": "нет", "true": "да", "false": "нет",
                 "north": "север", "south": "юг",
                 "не наблюдалось": "не наблюдалось", "censored": "не наблюдалось",
                 "цензура": "не наблюдалось", "нет данных": "не наблюдалось",
                 "нельзя определить": "нельзя определить",
                 "cannot be determined": "нельзя определить"}
        return table.get(s, s)

    out = {"items": {}, "correct": 0, "total": 0}
    for pid, spec in key.items():
        if pid.startswith("_"):
            continue
        got = answers.get(pid, {})
        detail = {}
        for field, expected in spec["anchor"].items():
            actual = got.get(field, "")
            ok = norm(actual) == norm(expected)
            detail[field] = {"expected": expected, "answered": actual, "ok": ok}
            out["total"] += 1
            out["correct"] += int(ok)
        out["items"][pid] = detail
    if key.get("_meta", {}).get("version") == 2:
        out["anchors_pass"] = out["total"] > 0 and out["correct"] == out["total"]
        out["free_text_review_required"] = True
    return out


def _candles(rows: list, first_col: str) -> list[str]:
    out = [f"| {first_col} | open | high | low | close |", "|---|---|---|---|---|"]
    for r in rows:
        t = r.get("t")
        out.append(f"| {t:+d} | {r['open']} | {r['high']} | {r['low']} | {r['close']} |")
    return out


def _scene_md(name: str, s: dict) -> list[str]:
    out = [f"### {name}", ""]
    if "minutes" in s:
        last = s["last_accepted_span_native_bars_ago"]
        out += [
            f"Область: юг **{s['zone_south']}**, север **{s['zone_north']}**. "
            f"Тик {s['tick']}, нативный ТФ {s['native_tf_minutes']} минут.",
            f"Принятых спанов на эту минуту: **{s['accepted_spans_so_far']}**"
            + (f"; последний закрылся назад нативных баров: {last}." if last is not None else "."),
            f"Обе границы живы: север {s['boundaries_alive']['north']}, юг {s['boundaries_alive']['south']}.",
            f"Текущий нативный бар открылся {s['current_native_bar']['minutes_since_bar_open']} "
            f"минут назад на **{s['current_native_bar']['open']}**"
            + (" и закрылся на оцениваемой свече." if s['current_native_bar'].get('closed') else " и ещё не закрылся."),
            "",
            "Минуты, 0 — оцениваемая:",
            "",
        ]
        out += _candles(s["minutes"], "мин")
        if "eligible_tier" in s:
            out += ["", "Текущий tier допустим для Blue и не является breaker: "
                    + ("да." if s["eligible_tier"] else "нет.")]
    else:
        out += [
            f"Область: юг **{s['zone_south']}**, север **{s['zone_north']}**. "
            f"Сторона ухода в T0: **{RU_SIDE[s['t0_exit_side']]}**.",
            f"Лента показана до +{s['tape_ends_after']}; она обрывается по причине "
            f"**{s['tape_end_reason']}**.",
            "",
            "Минуты от T0, 0 — сама минута T0:",
            "",
        ]
        out += _candles(s["minutes_from_t0"], "от T0")
    out.append("")
    return out


def paper_markdown(paper: dict) -> str:
    title = "шесть исследовательских решений" if paper.get("version") == 2 else "пять пар сцен"
    L = [f"# Проверка входа: {title}", "", paper["what_this_is"], ""]
    L += ["Правила:", ""] + [f"- {r}" for r in paper["rules"]] + [""]
    digest = paper['paper_sha256'] if paper.get("version") == 2 else paper['paper_sha256'][:16]
    L += [f"Отпечаток задания: `{digest}`", ""]
    for it in paper["items"]:
        L += [f"## {it['id']} — {it['title']}", ""]
        if it.get("note"):
            L += [it["note"], ""]
        if it.get("task"):
            L += [it["task"], ""]
        elif it["id"] == "P3":
            a, b = it["object_a"], it["object_b"]
            L += [
                f"Объект A: область юг **{a['zone_south']}**, север **{a['zone_north']}**, "
                f"сторона ухода **{RU_SIDE[a['t0_exit_side']]}**.",
                f"Объект B: область юг **{b['zone_south']}**, север **{b['zone_north']}**, "
                f"сторона ухода **{RU_SIDE[b['t0_exit_side']]}**.",
                f"Лента показана до +{a['tape_ends_after']}, обрыв по причине **{a['tape_end_reason']}**.",
                "", "Общая минутная лента от T0:", "",
            ]
            if paper.get("version") == 2:
                L += [f"Для B наблюдение допустимо до +{b['tape_ends_after']}; за его пределами исход неизвестен.", ""]
            L += _candles(it["minutes_from_t0"], "от T0")
            L.append("")
        else:
            L += _scene_md("Сцена A", it["scene_a"])
            L += _scene_md("Сцена B", it["scene_b"])
        L += ["**Ответь:**", ""]
        for field, prompt in it["ask"].items():
            L.append(f"- `{field}` — {prompt}")
        L.append("")
    L += [
        "## Как сдать",
        "",
        "Один файл JSON: ключ верхнего уровня — идентификатор пары, внутри — поля "
        "из списка «Ответь» этой пары. Короткие поля одним словом, свободные — текстом.",
        "",
    ]
    return "\n".join(L)
