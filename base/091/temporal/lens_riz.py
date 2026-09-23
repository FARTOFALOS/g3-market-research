"""091-TEMPORAL: RIZ (Blue/2X, минутный T0) на ленте, где поля нет.

Это исследовательская линза рядом с замороженным полем, а не сборщик поля: ничего не
пишется в data/field, идентичности поля не меняются. Правила дословно повторяют
замороженный сборщик (git dd278b6, src/g3riz/fast.py `_kernel`, src/g3riz/native.py
`build_native_arrays`, src/g3riz/identity.py `riz_id`):

- нативные свечи: сетка от открытия сессии, key = (offset_minutes - 1) // tf внутри сессии;
- превью на минутных часах: только минуты ДО закрытия нативной свечи, на состоянии до
  обновления; условие tier 1/2, обе стороны живы, i >= last_span + 2; развивающееся тело
  (open нативной свечи, close минуты) прошивает зону -> T0 = эта минута (minute_ignition);
- закрытие нативной свечи: буквальный порядок ветвей (брейкер-возврат, принятый спан,
  брейкер, снятие сторон, удаление); после него подтверждение синего -> T0 на последней
  минуте свечи (native_confirmation), если T0 ещё не было;
- рождение зоны после прохода обновления; riz_id = sha256(g3-riz-id/1, corpus, inst, tf,
  precursor_formed_ts, top.hex, bottom.hex, direction)[:32].

Линза стартует с пустого состояния с даты прогрева: зоны, родившиеся раньше, ей не видны.
Насколько это важно, меряет сверка с полем на общем отрезке (gate091.py).
Volume не читается.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numba as nb
import numpy as np
import pandas as pd

MIN = 60_000_000_000


def load_store(root: Path) -> dict:
    a = {n: np.load(root / f"{n}.npy") for n in ("open", "high", "low", "close", "close_ts_utc_ns")}
    with np.load(root / "sessions.npz") as z:
        s = {k: z[k].copy() for k in z.files}
    a["sessions"] = s
    a["corpus_id"] = json.loads((root / "manifest.json").read_text(encoding="utf-8"))["corpus_id"]
    return a


def native_arrays(store: dict, tf: int, first_pos: int):
    """build_native_arrays, векторно; только сессии, начинающиеся не раньше first_pos."""
    s = store["sessions"]
    keep = s["first_minute_pos"] >= first_pos
    firsts = s["first_minute_pos"][keep].astype(np.int64)
    stops = s["stop_minute_pos"][keep].astype(np.int64)
    anchors = s["session_open_utc_ns"][keep].astype(np.int64)
    lens = stops - firsts
    pos = np.concatenate([np.arange(f, t, dtype=np.int64) for f, t in zip(firsts, stops)])
    sess = np.repeat(np.arange(len(firsts)), lens)
    ts = store["close_ts_utc_ns"][pos]
    off = (ts - anchors[sess]) // MIN
    key = (off - 1) // tf
    brk = np.r_[True, (sess[1:] != sess[:-1]) | (key[1:] != key[:-1])]
    starts = np.flatnonzero(brk)
    stops_ = np.r_[starts[1:], len(pos)]
    mstart = pos[starts]
    mstop = pos[stops_ - 1] + 1
    o = store["open"][mstart]
    h = np.maximum.reduceat(store["high"][pos], starts)
    l = np.minimum.reduceat(store["low"][pos], starts)
    c = store["close"][mstop - 1]
    return o, h, l, c, mstart, mstop


@nb.njit(cache=True)
def kernel(bo, bh, bl, bc, mstart, mstop, mo, mc, cap):
    top = np.empty(cap); bottom = np.empty(cap); bull = np.zeros(cap, np.uint8)
    pf_pos = np.empty(cap, np.int64)
    tier = np.zeros(cap, np.int8); act = np.zeros(cap, np.uint8); sc = np.zeros(cap, np.int32)
    north = np.ones(cap, np.uint8); south = np.ones(cap, np.uint8); ls = np.full(cap, -1, np.int64)
    broke = np.zeros(cap, np.uint8); was_blue = np.zeros(cap, np.uint8)
    t0_pos = np.full(cap, -1, np.int64); t0_kind = np.zeros(cap, np.int8)
    t0_exit = np.zeros(cap, np.int8); t0_close = np.full(cap, np.nan)
    del_pos = np.full(cap, -1, np.int64)
    live = np.empty(cap, np.int64); nl = 0; nz = 0
    for i in range(len(bo)):
        pos = mstop[i] - 1
        new_nl = 0
        for li in range(nl):
            z = live[li]
            # превью на минутных часах, состояние до обновления
            if t0_pos[z] < 0 and (tier[z] == 1 or tier[z] == 2) and north[z] == 1 and south[z] == 1 and i >= ls[z] + 2:
                a = mstart[i]; sbc = mstop[i] - 1
                if sbc > a:
                    dev = mo[a]
                    if dev < bottom[z]:
                        for p in range(a, sbc):
                            if mc[p] > top[z]:
                                t0_pos[z] = p; t0_kind[z] = 1; t0_exit[z] = 1; t0_close[z] = mc[p]
                                break
                    elif dev > top[z]:
                        for p in range(a, sbc):
                            if mc[p] < bottom[z]:
                                t0_pos[z] = p; t0_kind[z] = 1; t0_exit[z] = -1; t0_close[z] = mc[p]
                                break
            # закрытие нативной свечи, буквальный порядок ветвей
            survived = True
            bmin = bo[i] if bo[i] < bc[i] else bc[i]
            bmax = bo[i] if bo[i] > bc[i] else bc[i]
            spanned = bmin < bottom[z] and bmax > top[z]
            if tier[z] == 3:
                returned = bh[i] >= bottom[z] if broke[z] == 1 else bl[i] <= top[z]
                survived = not returned
            elif spanned and (act[z] == 0 or i >= ls[z] + 2):
                if act[z] == 0:
                    act[z] = 1
                    tier[z] = 1 if (north[z] == 1 and south[z] == 1) else 2
                    sc[z] = 1
                else:
                    sc[z] += 1
                ls[z] = i
            else:
                kind = 0
                if not spanned:
                    if south[z] == 1 and north[z] == 0 and bo[i] > bottom[z] and bc[i] < bottom[z]:
                        kind = -1
                    elif north[z] == 1 and south[z] == 0 and bo[i] < top[z] and bc[i] > top[z]:
                        kind = 1
                    elif north[z] == 1 and south[z] == 1 and bo[i] > bottom[z] and bc[i] < bottom[z] and bh[i] >= top[z] and bc[i] <= top[z]:
                        kind = -1
                    elif north[z] == 1 and south[z] == 1 and bo[i] < top[z] and bc[i] > top[z] and bl[i] <= bottom[z] and bc[i] >= bottom[z]:
                        kind = 1
                if kind != 0:
                    broke[z] = 1 if kind == -1 else 0
                    north[z] = 0; south[z] = 0; tier[z] = 3; act[z] = 1; ls[z] = i
                else:
                    if bl[i] <= top[z] and top[z] <= bh[i]:
                        north[z] = 0
                    if bl[i] <= bottom[z] and bottom[z] <= bh[i]:
                        south[z] = 0
                    survived = north[z] == 1 or south[z] == 1
            if not survived:
                del_pos[z] = pos; was_blue[z] = 0
            else:
                live[new_nl] = z; new_nl += 1
            if survived and (tier[z] == 1 or tier[z] == 2) and sc[z] >= 2 and north[z] == 1 and south[z] == 1:
                if was_blue[z] == 0 and t0_pos[z] < 0:
                    t0_pos[z] = pos; t0_kind[z] = 2
                    t0_exit[z] = 1 if bc[i] > top[z] else -1
                    t0_close[z] = bc[i]
                was_blue[z] = 1
            else:
                was_blue[z] = 0
        nl = new_nl
        # рождение после прохода обновления
        if i >= 2 and nz < cap - 2:
            c1t = bo[i - 2] if bo[i - 2] > bc[i - 2] else bc[i - 2]
            c2t = bo[i - 1] if bo[i - 1] > bc[i - 1] else bc[i - 1]
            c2b = bo[i - 1] if bo[i - 1] < bc[i - 1] else bc[i - 1]
            c1b = bo[i - 2] if bo[i - 2] < bc[i - 2] else bc[i - 2]
            c3t = bo[i] if bo[i] > bc[i] else bc[i]
            c3b = bo[i] if bo[i] < bc[i] else bc[i]
            if bl[i] > bh[i - 2] and c2t > bh[i - 2] and c2b < bl[i]:
                bot = c1t if c2b > c1t else bh[i - 2]
                tp = c3b if c3b > c2t else bl[i]
                if tp > bot:
                    z = nz; nz += 1; top[z] = tp; bottom[z] = bot; bull[z] = 1; pf_pos[z] = pos
                    live[nl] = z; nl += 1
            if bh[i] < bl[i - 2] and c2b < bl[i - 2] and c2t > bh[i]:
                tp = c1b if c2t < c1b else bl[i - 2]
                bot = c3t if c3t < c2b else bh[i]
                if tp > bot:
                    z = nz; nz += 1; top[z] = tp; bottom[z] = bot; bull[z] = 0; pf_pos[z] = pos
                    live[nl] = z; nl += 1
    return (top[:nz], bottom[:nz], bull[:nz], pf_pos[:nz], t0_pos[:nz], t0_kind[:nz],
            t0_exit[:nz], t0_close[:nz], del_pos[:nz])


def riz_id(corpus_id, inst, tf, pf_ts, top, bottom, direction):
    payload = "\x1f".join(str(p) for p in ("g3-riz-id/1", corpus_id, inst, tf, pf_ts,
                                           float(top).hex(), float(bottom).hex(), direction)).encode("utf-8")
    return "riz_" + hashlib.sha256(payload).hexdigest()[:32]


def lens_riz(store: dict, inst: str, tf: int, warm_pos: int, t0_from_pos: int, corpus_id: str | None = None):
    """RIZ одного ТФ с T0 не раньше t0_from_pos; прогрев с сессии, начинающейся с warm_pos."""
    o, h, l, c, ms, me = native_arrays(store, tf, warm_pos)
    cap = 2 * len(o) + 16
    top, bot, bull, pf, t0, kind, ex, t0c, dele = kernel(o, h, l, c, ms, me, store["open"], store["close"], cap)
    sel = t0 >= t0_from_pos
    ts = store["close_ts_utc_ns"]
    cid = corpus_id or store["corpus_id"]
    rows = pd.DataFrame({
        "tf_minutes": tf, "zone_top": top[sel], "zone_bottom": bot[sel],
        "direction": np.where(bull[sel] == 1, 1, -1), "precursor_formed_spine_pos": pf[sel],
        "precursor_formed_ts_ns": ts[pf[sel]], "t0_spine_pos": t0[sel], "t0_ts_ns": ts[t0[sel]],
        "t0_kind": np.where(kind[sel] == 1, "minute_ignition", "native_confirmation"),
        "t0_exit_side": np.where(ex[sel] == 1, "north", "south"), "t0_close": t0c[sel],
        "c1_deletion_spine_pos": dele[sel]})
    rows.insert(0, "riz_id", [riz_id(cid, inst, tf, int(a), b, d, int(e)) for a, b, d, e in
                              zip(rows.precursor_formed_ts_ns, rows.zone_top, rows.zone_bottom, rows.direction)])
    return rows
