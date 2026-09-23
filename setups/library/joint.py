"""Общий расчёт S-07 v1r1 + S-17 D30 + S-18 v0 на NQ. Объявление — JOINT_S07_S17_S18.md.

Движки берутся из карточек без изменений: setups/S-17/forward.py::engine (S-07 v1r1 и D30)
и setups/S-18/run.py::run (S-18 v0). Для режима «один трейдер» S-18 повторён строка в
строку с одним добавлением — отметки, на которых позиция S-07/S-17 ещё открыта, пропускаются;
без блокировки повтор обязан совпасть с сохранённым band_daily_NQ.csv.

    python -B setups/library/joint.py
Результат: setups/library/joint_result.json, joint_daily_NQ.csv
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
H_FROM, H_TO = "2020-01-02", "2026-05-04"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


S17 = _load("s17forward", ROOT / "setups/S-17/forward.py")
S18 = _load("s18run", ROOT / "setups/S-18/run.py")
S18F_HOL = {"2026-05-25", "2026-06-19", "2026-07-03"}


def s18_blocked(ins, cal, root, block_k):
    """S-18 run() строка в строку (look 14, mult 1, step 30, first 30, gap True), плюс блок отметок k < block_k[s]."""
    look, mult, step, first = 14, 1.0, 30, 30
    cal, K, O, C = S18.sessions(ins, cal, root)
    S = len(cal)
    op = O[:, 1]
    cl = np.array([C[s, min(K[s], S18.KMAX)] for s in range(S)])
    move = np.abs(C / op[:, None] - 1.0)
    rows = []
    for s in range(look, S):
        rec = {"date": cal.date.iloc[s], "status": "ok"}
        prev = cl[s - 1]
        if not np.isfinite(op[s]) or not np.isfinite(prev) or not np.isfinite(cl[s]):
            rec["status"] = "unknown:нет open, вчерашнего close или close сессии"; rows.append(rec); continue
        hist = move[s - look:s]
        cnt = np.isfinite(hist).sum(axis=0)
        with np.errstate(invalid="ignore"):
            norm = np.where(cnt >= int(np.ceil(0.7 * look)), np.nanmean(hist, axis=0), np.nan)
        up = max(op[s], prev) * (1 + mult * norm)
        dn = min(op[s], prev) * (1 - mult * norm)
        marks = [k for k in range(first, S18.KMAX, step) if k + 1 <= K[s] - 1]
        pos, entry, ek = 0, np.nan, 0
        usd, n_tr, bad = 0.0, 0, False
        blk = block_k[s]
        for k in marks:
            if k < blk:
                continue
            p = C[s, k]
            if not np.isfinite(p) or not np.isfinite(up[k]):
                bad = True; break
            want = 1 if p > up[k] else (-1 if p < dn[k] else 0)
            if want != pos:
                px = O[s, k + 1]
                if not np.isfinite(px):
                    bad = True; break
                if pos != 0:
                    usd += pos * (px - entry) * S18.POINT[ins] - S18.COST[ins]; n_tr += 1
                pos, entry, ek = want, px, k + 1
        if bad:
            rec["status"] = "unknown:нет цены на отметке, нормы или цены исполнения"; rows.append(rec); continue
        if pos != 0:
            usd += pos * (cl[s] - entry) * S18.POINT[ins] - S18.COST[ins]; n_tr += 1
        rec.update(usd=usd, n_tr=n_tr)
        rows.append(rec)
    D = pd.DataFrame(rows)
    D["date"] = pd.to_datetime(D.date)
    return D


def s07_s17(root, cal):
    e = S17.engine(S17.load(root), cal)
    e["date"] = pd.to_datetime(e.date)
    return e


def block_from(e, col_exit, cal):
    """Минута (от 09:30), с которой S-18 может действовать: 3 + минута выхода; нет входа -> 0; выход неизвестен -> NaN."""
    m = e.set_index("date")
    d = pd.to_datetime(cal.date)
    st = m.status.reindex(d).fillna("no_calendar")
    ex = m[col_exit].reindex(d)
    blk = np.where(st.str.startswith("no_entry").to_numpy(), 0.0, 3.0 + ex.to_numpy())
    return blk


def stats(x: pd.Series, dates: pd.Series) -> dict:
    x = x.astype(float); n = len(x)
    if n == 0:
        return {"sessions": 0}
    cum = x.cumsum(); dd = (cum - cum.cummax()).min()
    rng = np.random.default_rng(20260923); L = 20; nb = int(np.ceil(n / L))
    starts = rng.integers(0, max(n - L + 1, 1), size=(4000, nb))
    idx = (starts[:, :, None] + np.arange(L)[None, None, :]).reshape(4000, -1)[:, :n]
    reps = x.to_numpy()[idx].mean(1)
    k = max(1, int(round(0.05 * n)))
    yrs = pd.Series(x.to_numpy(), index=pd.to_datetime(dates).dt.year).groupby(level=0).sum().round(0)
    return {"sessions": int(n), "usd_per_session": round(float(x.mean()), 2),
            "t": round(float(x.mean() / x.std(ddof=1) * np.sqrt(n)), 2) if n > 2 else None,
            "ci95_block20": [round(float(np.quantile(reps, .025)), 2), round(float(np.quantile(reps, .975)), 2)],
            "total_usd": round(float(x.sum()), 0), "max_drawdown_usd": round(float(dd), 0),
            "worst_day_usd": round(float(x.min()), 0), "share_positive": round(float((x > 0).mean()), 3),
            "total_without_top5pct_days": round(float(np.sort(x.to_numpy())[:-k].sum()), 0),
            "by_year": {str(k_): float(v) for k_, v in yrs.items()}}


def build(root, cal, lo, hi):
    e = s07_s17(root, cal)                      # root — каталог ленты NQ
    r18 = root.parent                           # S-18 читает <root>/<инструмент>
    s18 = s18_blocked("NQ", cal, r18, np.zeros(len(cal)))
    b1 = s18_blocked("NQ", cal, r18, block_from(e, "s07_exit_min", cal))
    b2 = s18_blocked("NQ", cal, r18, block_from(e, "d30_exit_min", cal))
    d = pd.DataFrame({"date": pd.to_datetime(cal.date)})
    m = e.set_index("date")
    no = m.status.reindex(d.date).fillna("").str.startswith("no_entry").to_numpy()
    d["s07"] = np.where(no, 0.0, m.s07_usd.reindex(d.date).to_numpy())
    d["s17"] = np.where(no, 0.0, m.d30_usd.reindex(d.date).to_numpy())
    for name, D in (("s18", s18), ("s18_after_s07", b1), ("s18_after_s17", b2)):
        ok = D[D.status == "ok"].set_index("date").usd
        d[name] = ok.reindex(d.date).to_numpy()
    d = d[(d.date >= lo) & (d.date <= hi)].reset_index(drop=True)
    d["A3"] = d.s07 + d.s17 + d.s18
    d["A2"] = d.s07 + d.s18
    d["B1"] = d.s07 + d.s18_after_s07
    d["B2"] = d.s17 + d.s18_after_s17
    return d, e, s18


def summarize(d):
    out = {}
    for col in ("s07", "s17", "s18", "A3", "A2", "B1", "B2"):
        k = d[np.isfinite(d[col])]
        out[col] = stats(k[col], k.date)
    k = d.dropna(subset=["s07", "s17", "s18"])
    out["correlation_daily"] = {"s07_s17": round(float(k.s07.corr(k.s17)), 3),
                                "s07_s18": round(float(k.s07.corr(k.s18)), 3),
                                "s17_s18": round(float(k.s17.corr(k.s18)), 3)}
    k = d.dropna(subset=["s18", "s18_after_s07"])
    out["s18_blocked_by_s07"] = {"sessions_changed": int((k.s18 != k.s18_after_s07).sum()),
                                 "usd_lost_vs_standalone": round(float((k.s18_after_s07 - k.s18).sum()), 0)}
    k = d.dropna(subset=["s18", "s18_after_s17"])
    out["s18_blocked_by_s17"] = {"sessions_changed": int((k.s18 != k.s18_after_s17).sum()),
                                 "usd_lost_vs_standalone": round(float((k.s18_after_s17 - k.s18).sum()), 0)}
    return out


def validate(e_hist, s18_hist, e_fwd, s18_fwd):
    res = {}
    ref = pd.read_csv(ROOT / "setups/S-17/engine_history_NQ.csv", parse_dates=["date"])
    j = ref.merge(e_hist, on="date", suffixes=("_ref", ""))
    for c in ("s07_usd", "d30_usd", "s07_exit_min", "d30_exit_min"):
        a, b = j[c + "_ref"].to_numpy(float), j[c].to_numpy(float)
        res[f"history_{c}_equal"] = int((np.isclose(a, b) | (np.isnan(a) & np.isnan(b))).sum())
    res["history_sessions"] = int(len(j))
    ref = pd.read_csv(ROOT / "setups/S-18/band_daily_NQ.csv", parse_dates=["date"])
    j = ref.merge(s18_hist, on="date", suffixes=("_ref", ""))
    ok = j[(j.status_ref == "ok") & (j.status == "ok")]
    res["s18_history_status_equal"] = int((j.status_ref == j.status).sum())
    res["s18_history_sessions"] = int(len(j))
    res["s18_history_usd_equal_to_cent"] = int(np.isclose(ok.usd_ref, ok.usd).sum())
    res["s18_history_ok"] = int(len(ok))
    ref = pd.read_csv(ROOT / "setups/S-17/forward_sessions_NQ.csv", parse_dates=["date"])
    j = ref.merge(e_fwd, on="date", suffixes=("_ref", ""))
    a, b = j.s07_usd_ref.to_numpy(float), j.s07_usd.to_numpy(float)
    res["forward_s07_equal"] = int((np.isclose(a, b) | (np.isnan(a) & np.isnan(b))).sum())
    a, b = j.d30_usd_ref.to_numpy(float), j.d30_usd.to_numpy(float)
    res["forward_d30_equal"] = int((np.isclose(a, b) | (np.isnan(a) & np.isnan(b))).sum())
    res["forward_sessions"] = int(len(j))
    ref = pd.read_csv(ROOT / "setups/S-18/forward_ledger_NQ.csv", parse_dates=["date"])
    j = ref.merge(s18_fwd, on="date", suffixes=("_ref", ""))
    ok = j[(j.status_ref == "ok") & (j.status == "ok")]
    res["s18_forward_usd_equal_to_cent"] = int(np.isclose(ok.usd_ref, ok.usd).sum())
    res["s18_forward_ok"] = int(len(ok))
    return res


def main():
    hist_cal = S17.history_calendar()
    dh, eh, sh = build(ROOT / "data/market/NQ", hist_cal, pd.Timestamp(H_FROM), pd.Timestamp(H_TO))
    h = hist_cal[hist_cal.date < S17.F_FROM]
    fwd_cal = pd.concat([h, S17.forward_calendar()], ignore_index=True)
    df, ef, sf = build(ROOT / "data/forward/market/NQ", fwd_cal, pd.Timestamp(S17.F_FROM), pd.Timestamp(S17.F_TO))
    s18_fwd_dir = ROOT / "data/forward/market"
    res = {"validation": validate(eh, S18.run("NQ", cal=hist_cal)[0], S17.engine(S17.load(ROOT / "data/forward/market/NQ"),
                                  S17.forward_calendar()).assign(date=lambda x: pd.to_datetime(x.date)),
                                  S18.run("NQ", cal=fwd_cal, root=s18_fwd_dir)[0])}
    res["history_2020_2026_05_04"] = summarize(dh)
    res["history_2021_2025"] = summarize(dh[(dh.date >= "2021-01-01") & (dh.date <= "2025-12-31")].reset_index(drop=True))
    res["forward_2026_05_04_07_10_seen"] = summarize(df)
    dh.to_csv(HERE / "joint_daily_NQ.csv", index=False, float_format="%.2f")
    df.to_csv(HERE / "joint_daily_forward_NQ.csv", index=False, float_format="%.2f")
    (HERE / "joint_result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
