"""Consistency checks of a saved F0 A/B replay; writes validation.json.

Run from repository root after the probe:
    python -B base/105/f0_ab_validate.py [--out base/105/f0_ab_probe_out]

Every check is recomputed from the saved outputs, the current input arrays and
the current source files; nothing is copied from the probe's own verdict except
`stage1_passed`, which is checked against the falsifier record it summarizes.
The legacy-journal comparison needs the local, untracked
work/nq-manual/F0_policy.csv; when that file is absent the check is reported
as unavailable instead of passed.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LEGACY_JOURNAL = ROOT / "work/nq-manual/F0_policy.csv"
TOL = 1e-9


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def headroom(ra, rb):
    return float(np.mean(np.maximum(ra, rb)) - max(np.mean(ra), np.mean(rb)))


def opportunity_oids(events, policy):
    g = events[(events.event == "opportunity") & (events.policy == policy)]
    return {int(d): list(x.oid) for d, x in g.groupby("date")}


def validate(out):
    days = pd.read_csv(out / "f0_ab_days.csv")
    ops = pd.read_csv(out / "f0_ab_opportunities.csv")
    events = pd.read_csv(out / "f0_ab_events.csv", low_memory=False)
    rec = pd.read_csv(out / "f0_ab_reconciliation.csv")
    summary = json.loads((out / "f0_ab_summary.json").read_text(encoding="utf-8"))
    passport = json.loads((out / "passport.json").read_text(encoding="utf-8"))
    c = {}

    subst = summary["continuation_substitution"]
    c["stage1_passed"] = bool(summary["stage1_passed"] and subst["passed"]
                              and not (subst["s07_failures"] or subst["pv2_failures"]
                                       or subst["phase_access_violations"] or subst["policy_state_failures"]))

    c["all_dates_unique"] = bool(days.date.is_unique)

    ok = True
    for p in ("A", "B"):
        resolved = days[f"{p}_status"] == "resolved"
        ok &= bool(days.loc[resolved, f"R{p}"].notna().all() and days.loc[~resolved, f"R{p}"].isna().all())
    c["unknown_is_nan"] = ok

    p0 = days.p0_status == "yes"
    pair = p0 & (days.A_status == "resolved") & (days.B_status == "resolved")
    n = summary["counts"]
    c["pair_count"] = bool(int(pair.sum()) == n["P_AB"]
                           and int((p0 & (days.A_status == "resolved")).sum()) == n["P_A"]
                           and int((p0 & (days.B_status == "resolved")).sum()) == n["P_B"]
                           and int(p0.sum()) == n["P0_known"]
                           and int((days.p0_status == "unknown").sum()) == n["P0_membership_unknown"]
                           and len(days) == n["calendar_dates"])

    pp = days[pair]
    decomp = pp.morning_A + pp.downstream_A - pp.downstream_B
    c["D_and_decomposition"] = bool(np.allclose(pp.D, pp.RA - pp.RB, rtol=0, atol=TOL)
                                    and np.allclose(pp.D, pp.D_decomp, rtol=0, atol=TOL)
                                    and np.allclose(pp.D, decomp, rtol=0, atol=TOL)
                                    and days.loc[~pair, ["D", "D_decomp"]].isna().all().all())

    ok = True
    groups = [("all", pp, summary["headroom_resolved"])]
    groups += [(ep, pp[pp.epoch == ep], v["headroom"]) for ep, v in summary["by_epoch"].items()]
    for _, g, h in groups:
        if len(g) < 2:
            continue
        ra, rb = g.RA.to_numpy(float), g.RB.to_numpy(float)
        d = ra - rb
        H = headroom(ra, rb)
        ident = min(np.mean(np.maximum(d, 0)), np.mean(np.maximum(-d, 0)))
        ok &= bool(abs(H - ident) <= TOL and abs(H - h["H"]) <= TOL and abs(h["H_identity_from_D"] - ident) <= TOL)
    c["headroom_identity"] = ok

    stream = {int(d): list(x.oid) for d, x in ops.groupby("date")}
    oa, ob = opportunity_oids(events, "A"), opportunity_oids(events, "B")
    # Each policy sees an ordered prefix of the one saved stream (it may stop
    # early at an unknown); paired days must see the whole stream.
    prefix = lambda seen, full: seen == full[:len(seen)]
    c["one_shared_opportunity_stream"] = bool(
        all(prefix(oa.get(int(d), []), stream.get(int(d), [])) and prefix(ob.get(int(d), []), stream.get(int(d), []))
            for d in days.loc[p0, "date"])
        and all(oa.get(int(d), []) == ob.get(int(d), []) == stream.get(int(d), []) for d in pp.date)
        and set(stream) <= set(days.loc[p0, "date"].astype(int)))

    o = events[events.event == "opportunity"]
    a_auth = o[o.policy == "A"].authorized.astype(bool)
    b = o[o.policy == "B"]
    b_skip = ~b.authorized.astype(bool)
    c["single_intervention"] = bool(a_auth.all()
                                    and (b_skip == (b.branch == "S07")).all()
                                    and (b.loc[b_skip, "reason"] == "B_intervention").all())

    s = events[events.event == "settlement"]
    ok = True
    for p in ("A", "B"):
        g = s[s.policy == p]
        total = g.groupby("date").net.sum()
        morning = g[g.branch == "S07"].groupby("date").net.sum()
        down = g[g.branch != "S07"].groupby("date").net.sum()
        r = days[p0 & (days[f"{p}_status"] == "resolved")].set_index("date")
        ok &= bool(np.allclose(total.reindex(r.index, fill_value=0.0), r[f"R{p}"], rtol=0, atol=TOL))
        ok &= bool(np.allclose(down.reindex(r.index, fill_value=0.0), r[f"downstream_{p}"], rtol=0, atol=TOL))
        if p == "A":
            ok &= bool(np.allclose(morning.reindex(r.index, fill_value=0.0), r.morning_A, rtol=0, atol=TOL))
        else:
            ok &= bool(morning.reindex(r.index).isna().all())
    c["ledger_matches_resolved_payoffs"] = ok

    ok = True
    for p, seen in (("A", oa), ("B", ob)):
        for d in days.loc[p0 & (days[f"{p}_status"] == "resolved"), "date"]:
            ok &= seen.get(int(d), []) == stream.get(int(d), [])
    c["resolved_policies_receive_all_opportunities"] = bool(ok)

    c["reconciliation_all_allowed"] = bool(rec.allowed.all()
                                           and summary["reconciliation"]["forbidden_differences"] == 0
                                           and rec.category.value_counts().to_dict() == summary["reconciliation"]["categories"])

    data = ROOT / "data/forward/market/NQ"
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    ident = passport["data_identity"]
    ok = ident["corpus_id"] == manifest["corpus_id"]
    for name, h in ident["array_sha256"].items():
        ok &= h == manifest["array_sha256"][name] == sha256(data / name)
    ok &= ident["source_gate_sha256"] == sha256(ROOT / "base/091/source_gate_forward_2026-09-21.json")
    for name, h in passport["file_sha256"].items():
        ok &= h == sha256(HERE / name)
    ok &= summary["passport"] == passport
    c["input_source_fingerprints"] = bool(ok)

    rebuilt = sha256(out / "legacy_f0_trades_rebuilt.csv")
    original = sha256(LEGACY_JOURNAL) if LEGACY_JOURNAL.exists() else None
    c["legacy_original_journal_identical"] = None if original is None else rebuilt == original

    unavailable = [k for k, v in c.items() if v is None]
    return dict(checks=c, all_passed=all(v for v in c.values() if v is not None),
                unavailable=unavailable, legacy_original_sha256=original,
                generator="base/105/f0_ab_validate.py")


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--out", default=str(HERE / "f0_ab_probe_out"))
    out = Path(a.parse_args().out).resolve()
    v = validate(out)
    (out / "validation.json").write_text(json.dumps(v, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(v, ensure_ascii=False, indent=2))
    if not v["all_passed"]:
        raise SystemExit("validation failed; see validation.json")


if __name__ == "__main__":
    main()
