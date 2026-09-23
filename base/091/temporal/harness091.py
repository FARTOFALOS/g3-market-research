"""091-TEMPORAL: замороженный оператор 084 -> 086 -> 091 на произвольной ленте и списке фильмов.

Используются замороженные функции без изменений:
  base/084/xray.py      _scan              — первый close_break живого Film-1 в окне +1..+50;
  base/086/fork086.py   _extreme, _fork, cell_ids, TICK — вилка b против «первый тик за M», клетка;
  base/086/cells086.json медианы клетки;   base/091/exec091.py — классы исполнения (повторены ниже
  строка в строку, потому что exec091 читает ленту по жёсткому пути).
Обвязка только подаёт массивы ленты, дыры ленты, календарь закрытия XNYS и фильмы.

Одиночный трейдер — FREEZE_091_TEMPORAL §1, §4, §7.2, §7.3 (см. single_unit).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "base/084"))
sys.path.insert(0, str(REPO / "base/086"))
import xray as X  # noqa: E402
import fork086 as F  # noqa: E402

MED = json.loads((REPO / "base/086/cells086.json").read_text(encoding="utf-8"))["medians"]
SEL_CELL = 1          # u+ k- ttc- r-, знак -1: сторона продолжения
COST = 1.00
WIN, CANCEL, SAME, GAP_B, GAP_A, LOST, EDGE = 0, 1, 2, 3, 4, 5, 6


def gap_kind_from_ts(ts: np.ndarray) -> np.ndarray:
    """kind[j] != 0, если перед баром j пропущена хотя бы одна минута (род дыры для исхода не важен)."""
    k = np.zeros(ts.size, np.int8)
    k[1:] = (np.diff(ts) > 60_000_000_000).astype(np.int8)
    return k


def close_break(films: pd.DataFrame, high, low, close, kind) -> pd.DataFrame:
    """084: первый close_break в окне +50, как X.build, без записи файлов."""
    n = len(films)
    t0 = films.t0_spine_pos.to_numpy().astype(np.int64)
    e = films.exit_boundary.to_numpy().astype(np.float64)
    north = films.side.to_numpy() == "north"
    grp = np.zeros(n, np.int8)
    ne = len(X.EVENTS)
    end_code = np.full(n, -1, np.int8); end_k = np.zeros(n, np.int32)
    r_ev = np.full((n, ne), -1, np.int32)
    flat_ev = np.zeros((n, ne), np.bool_); tie_ev = np.zeros((n, ne), np.bool_)
    dc_ev = np.full((n, ne), np.nan, np.float32)
    kh = X.KH
    risk = np.zeros((1, ne, kh), np.int64); hits = np.zeros((1, ne, kh), np.int64)
    alive = np.zeros((1, kh), np.int64); ended = np.zeros((1, 2, kh), np.int64)
    X._scan(t0, e, north, grp, high, low, close, kind, close.size - 1, kh, X.W,
            end_code, end_k, r_ev, flat_ev, tie_ev, dc_ev, risk, hits, alive, ended)
    r = r_ev[:, X.EVENTS.index("close_break")]
    out = films.assign(end_code=end_code, end_k=end_k, r_close_break=r)
    return out


def geometry(p: pd.DataFrame, high, low, close, ts, cal_close_ns) -> pd.DataFrame:
    """fork086.geometry строка в строку, на поданных массивах (NQ)."""
    t0 = p.t0_spine_pos.to_numpy().astype(np.int64); q = p.q.to_numpy().astype(np.int64)
    north = (p.side == "north").to_numpy()
    M = np.empty(len(p)); F._extreme(t0, q, north, high, low, M)
    c = close[q]; b = p.exit_boundary.to_numpy()
    d = np.abs(c - b)
    a = np.where(north, M + F.TICK["NQ"], M - F.TICK["NQ"])
    s = np.abs(a - c)
    assert (d > 0).all() and (s > F.TICK["NQ"] - 1e-9).all()
    assert (np.where(north, b < c, b > c) & np.where(north, c < a, c > a)).all(), "b, c, a out of order"
    cs = np.concatenate(([0.0], np.cumsum(high - low)))
    sigma = (cs[q + 1] - cs[t0]) / (q - t0 + 1)
    idx = np.searchsorted(cal_close_ns, ts[q], side="left")
    ttc = np.where(idx < cal_close_ns.size, (cal_close_ns[np.minimum(idx, cal_close_ns.size - 1)] - ts[q]) / 60e9, np.nan)
    g = pd.DataFrame({"riz_id": p.riz_id.to_numpy(), "side": p.side.to_numpy(), "tf_minutes": p.tf_minutes.to_numpy(),
                      "t0_pos": t0, "q": q, "t0_day": p.t0_day.to_numpy(), "north": north, "b": b, "c": c, "M": M,
                      "a": a, "d": d, "s": s, "p0": s / (d + s), "sigma": sigma, "u": d / sigma,
                      "k": (q - t0).astype(np.int64), "ttc": ttc, "r": s / d})
    g.attrs["instrument"] = "NQ"
    return g


def cell(g: pd.DataFrame) -> pd.DataFrame:
    g = g[np.isfinite(g.ttc.to_numpy())].reset_index(drop=True)
    cid = F.cell_ids(g, MED)
    out = g[cid == SEL_CELL].reset_index(drop=True)
    out.attrs["instrument"] = "NQ"
    return out


def execute(g: pd.DataFrame, op, high, low, close, kind) -> pd.DataFrame:
    """exec091.run_cell: вилка от q и классы исполнения по open(q+1)."""
    last = close.size - 1
    code = np.full(len(g), -1, np.int8); kbar = np.zeros(len(g), np.int64)
    F._fork(g.q.to_numpy(), g.b.to_numpy(), g.a.to_numpy(), g.north.to_numpy(), high, low, kind, last, code, kbar)
    assert (code >= 0).all()
    q = g.q.to_numpy(); north = g.north.to_numpy()
    a = g.a.to_numpy(); b = g.b.to_numpy()
    dirn = np.where(north, 1.0, -1.0)
    qn = q + 1
    avail = (qn <= last) & (kind[np.clip(qn, 0, last)] == 0)
    E = np.where(avail, op[np.clip(qn, 0, last)], np.nan)
    tgt = np.isin(code, (CANCEL, GAP_A)); stp = np.isin(code, (WIN, GAP_B)); unk = np.isin(code, (SAME, LOST, EDGE))
    exit_lvl = np.full(len(g), np.nan); exit_lvl[tgt] = a[tgt]; exit_lvl[stp] = b[stp]
    cls = np.array(["?"] * len(g), dtype=object)
    missed = avail & (dirn * (E - a) >= 0)
    adverse_gap = avail & (dirn * (E - b) <= 0)
    cls[~avail] = "exec_unavailable"
    cls[avail & unk] = "unknown"
    cls[avail & tgt & ~missed & ~adverse_gap] = "target_first"
    cls[avail & stp & ~missed & ~adverse_gap] = "boundary_first"
    cls[avail & missed] = "missed"
    cls[avail & adverse_gap & ~missed] = "adverse_gap"
    out = g.assign(code=code, kbar=kbar, exit_bar=q + kbar, cls=cls, E=E, dirn=dirn,
                   gross_E=dirn * (exit_lvl - E),
                   gross_fav=dirn * (a - E), gross_adv=dirn * (b - E))
    return out


def run(films, high, low, close, op, ts, kind, cal_close_ns):
    x = close_break(films, high, low, close, kind)
    ev = x[x.r_close_break >= 0].copy()
    ev["q"] = ev.t0_spine_pos + ev.r_close_break
    g = geometry(ev.reset_index(drop=True), high, low, close, ts, cal_close_ns)
    c = cell(g)
    return x, g, execute(c, op, high, low, close, kind)


# --------------------------------------------------------------------- одиночный трейдер
def single_unit(sig: pd.DataFrame) -> pd.DataFrame:
    """FREEZE_091_TEMPORAL §1 и §7.2.

    Сигналы по порядку (q_event, riz_id). Пока флэт — берётся самый ранний. Сигнал годен,
    только если q_event не раньше бара выхода прошлой позиции (open(q+1) ещё впереди).
    Пока позиция открыта — skipped_overlap. §7.2: open уже за a или за b -> позиция не
    создаётся, P&L 0. Нет бара входа -> exec_unavailable, позиция не создаётся.
    Выход — первое касание a/b с бара q+1; неизвестный порядок (same_bar / потеря ленты /
    край архива) — позиция закрыта на этом баре, исход в границах: благоприятная = a, неблагоприятная = b.
    """
    s = sig.sort_values(["q", "riz_id"]).reset_index(drop=True)
    busy_until = -1                       # бар выхода открытой позиции
    status = []
    for q, cls, xb in zip(s.q.to_numpy(), s.cls.to_numpy(), s.exit_bar.to_numpy()):
        if q < busy_until:
            status.append("skipped_overlap"); continue
        if cls == "exec_unavailable":
            status.append("exec_unavailable"); continue
        if cls == "missed":
            status.append("preentry_missed_target"); continue
        if cls == "adverse_gap":
            status.append("preentry_invalidated_stop"); continue
        status.append("taken_" + cls)
        busy_until = int(xb)
    s["status"] = status
    taken = s.status.str.startswith("taken_")
    unk = s.status == "taken_unknown"
    s["net_identified"] = np.where(taken & ~unk, s.gross_E - COST, np.nan)
    s["net_favorable"] = np.where(taken, np.where(unk, s.gross_fav, s.gross_E) - COST, np.nan)
    s["net_adverse"] = np.where(taken, np.where(unk, s.gross_adv, s.gross_E) - COST, np.nan)
    return s


def day_block_ci(values_by_day: np.ndarray, counts_by_day: np.ndarray, seed=20260916):
    """Самый широкий 95% интервал отношения сумм по блокам 1/5/20 календарных дней (протокол 086)."""
    D = values_by_day.size
    widest = None
    for L in (1, 5, 20):
        rng = np.random.default_rng(seed + L)
        nb = int(np.ceil(D / L))
        starts = rng.integers(0, max(D - L + 1, 1), size=(2000, nb))
        idx = (starts[:, :, None] + np.arange(L)[None, None, :]).reshape(2000, -1)[:, :D]
        den = counts_by_day[idx].sum(1)
        ok = den > 0
        rep = values_by_day[idx].sum(1)[ok] / den[ok]
        ci = [float(np.quantile(rep, 0.025)), float(np.quantile(rep, 0.975))]
        if widest is None or ci[1] - ci[0] > widest[1] - widest[0]:
            widest = ci
    return [round(widest[0], 4), round(widest[1], 4)]
