"""Describe NQ minute gaps and their reach into the frozen S-19 replay.

No missing OHLC is invented. A non-scheduled no-bar interval is an archive gap,
not proof that trades occurred while the exchange was open. The archived field
and both trade policies are read only.
"""
import json

import numpy as np
import pandas as pd

from census import ROOT, MINUTE, load

NQ_USD_PER_POINT = 20.0


def main():
    a, _, manifest = load()
    ts = a["ts"]
    sid = a["session_id"]
    begin = pd.Timestamp("2020-01-01", tz="UTC").value
    end = pd.Timestamp("2026-01-01", tz="UTC").value
    observed = (ts >= begin) & (ts < end)

    ix = np.flatnonzero((np.diff(ts) != MINUTE) & (np.diff(sid) == 0)) + 1
    ix = ix[(ts[ix] >= begin) & (ts[ix] < end)]
    length = ((ts[ix] - ts[ix - 1]) // MINUTE - 1).astype(int)
    assert np.all(length > 0)
    eastern = pd.to_datetime(ts[ix], utc=True).tz_convert("America/New_York")
    # Same scheduled halt correction used by prepare.py, supported by CME SER-8788.
    scheduled = (
        (length == 15)
        & (eastern.strftime("%Y-%m-%d") < "2021-06-28")
        & (eastern.hour == 16)
        & (eastern.minute == 31)
    )
    other_ix, other_length = ix[~scheduled], length[~scheduled]
    other_first = ts[other_ix - 1] + MINUTE
    other_last = ts[other_ix] - MINUTE
    observed_count = int(observed.sum())
    absent_count = int(other_length.sum())
    expected_count = observed_count + absent_count
    one_minute_range = a["high"][observed] - a["low"][observed]
    selected_pos = np.flatnonzero(observed)
    selected_sid = sid[selected_pos]
    session_starts = np.r_[0, np.flatnonzero(np.diff(selected_sid) != 0) + 1]
    session_stops = np.r_[session_starts[1:], len(selected_pos)]
    nominal_grid = int(sum((ts[selected_pos[j - 1]] - ts[selected_pos[i]]) // MINUTE + 1
                           for i, j in zip(session_starts, session_stops)))
    assert nominal_grid == observed_count + int(length.sum())

    cal = pd.read_csv(ROOT / "base/103/calendar.csv")
    parent = pd.read_parquet(ROOT / "work/103/parent.parquet")
    policy_dates = cal.date.astype(str).tolist()
    direct = {}
    for row in cal.itertuples():
        prior = (pd.Timestamp(row.date) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        market_open = pd.Timestamp(prior + " 18:00", tz="America/New_York").value
        close = int(row.close_ns)
        overlap = (other_first <= close) & (other_last >= market_open)
        direct[row.date] = {
            "gap_intervals": int(overlap.sum()),
            "missing_minute_slots": int(
                np.maximum(0, np.minimum(other_last[overlap], close)
                           - np.maximum(other_first[overlap], market_open))
                .sum() // MINUTE + int(overlap.sum())
            ),
        }
    direct_days = {d for d, v in direct.items() if v["gap_intervals"]}

    relevant = parent[(parent.clock_status == "entry_window")
                      & (parent.structural_status == "prefix_unknown")]
    any_prefix_gap = parent[parent.prefix_gap_count > 0]
    indexed_parent = parent.set_index("riz_id")
    assert indexed_parent.index.is_unique

    policy = {}
    first_uncertainty_addresses = []
    for version in ("v1", "v2"):
        base = ROOT / "work/103/replay"
        day = pd.read_parquet(base / f"{version}_daily.parquet")
        decisions = pd.read_parquet(base / f"{version}_decisions.parquet")
        trades = pd.read_parquet(base / f"{version}_trades.parquet")
        assert set(day.date) == set(policy_dates)
        buckets = {}
        for has_direct_gap in (False, True):
            for known in (False, True):
                dates = set(day.loc[(day.date.isin(direct_days) == has_direct_gap)
                                    & ((day.status == "known") == known), "date"])
                subset = day[day.date.isin(dates)]
                t = trades[trades.date.isin(dates)]
                label = ("direct_gap_" if has_direct_gap else "no_direct_gap_") \
                        + ("policy_known" if known else "policy_unknown")
                buckets[label] = {
                    "sessions": len(dates),
                    "known_trades": int((t.status == "known").sum()),
                    "unknown_trade_outcomes": int((t.status != "known").sum()),
                    "net_sum_if_fully_known": float(subset.net.sum()) if known else None,
                    "net_usd_if_fully_known": float(subset.net.sum() * NQ_USD_PER_POINT) if known else None,
                    "certified_prefix_net_sum_only": float(subset.certified_prefix_net.sum()),
                    "certified_prefix_net_usd_only": float(subset.certified_prefix_net.sum() * NQ_USD_PER_POINT),
                }
        causes = decisions[decisions.reason.isin(
            ["prefix_unknown_changes_selection", "entry_unknown", "entered_unknown"]
        )].groupby("reason").date.nunique().to_dict()
        stopped_by_prefix = decisions[decisions.reason == "prefix_unknown_changes_selection"]
        first_gap_context = []
        shared_gap_positions = set()
        for row in stopped_by_prefix.itertuples():
            z = indexed_parent.loc[row.riz_id]
            lo = np.searchsorted(other_ix, int(z.c3_start), side="right")
            hi = np.searchsorted(other_ix, int(z.t0_spine_pos), side="right")
            gap_pos = other_ix[lo:hi]
            gap_lengths = other_length[lo:hi]
            assert len(gap_pos) > 0
            shared_gap_positions.update(gap_pos.tolist())
            birth = int(z.precursor_formed_spine_pos)
            first_span = int(z.first_span_pos)
            before_span = bool(np.any(gap_pos <= first_span))
            after_span = bool(np.any(gap_pos > first_span))
            if int(z.direction) > 0:
                all_edges = np.maximum(a["high"][gap_pos - 1], a["high"][gap_pos])
                nearest_edge_to_E = float(max(0, np.min(float(z.target) - all_edges)))
            else:
                all_edges = np.minimum(a["low"][gap_pos - 1], a["low"][gap_pos])
                nearest_edge_to_E = float(max(0, np.min(all_edges - float(z.target))))
            min_edge_to_target = None
            if after_span and not before_span:
                later = gap_pos[gap_pos > first_span]
                if int(z.direction) > 0:
                    edge = np.maximum(a["high"][later - 1], a["high"][later])
                    min_edge_to_target = float(np.min(float(z.target) - edge))
                else:
                    edge = np.minimum(a["low"][later - 1], a["low"][later])
                    min_edge_to_target = float(np.min(edge - float(z.target)))
            first_gap_context.append({
                "tf": int(z.tf_minutes),
                "gap_intervals": len(gap_pos),
                "missing_minutes": int(gap_lengths.sum()),
                "max_gap_minutes": int(gap_lengths.max()),
                "within_formation_c3": bool(np.any(gap_pos <= birth)),
                "between_birth_and_first_span": bool(np.any((gap_pos > birth)
                                                       & (gap_pos <= first_span))),
                "before_or_at_first_span": before_span,
                "after_first_span": after_span,
                "only_after_first_span": after_span and not before_span,
                "nearest_observed_gap_edge_to_E_points": nearest_edge_to_E,
                "min_observed_edge_to_target_points": min_edge_to_target,
            })
            first_uncertainty_addresses.append({
                "version": version,
                "kind": "prefix_selection",
                "date": row.date,
                "riz_id": row.riz_id,
                "tf_minutes": int(z.tf_minutes),
                "t0_utc": str(pd.Timestamp(int(z.t0_ts_ns), tz="UTC")),
                "zone_bottom": float(z.zone_bottom),
                "zone_top": float(z.zone_top),
                "target_E": float(z.target),
                "gap_intervals": len(gap_pos),
                "missing_minutes": int(gap_lengths.sum()),
                "max_gap_minutes": int(gap_lengths.max()),
                "gap_in_forming_C3": bool(np.any(gap_pos <= birth)),
                "gap_before_or_at_first_span": before_span,
                "gap_after_first_span": after_span,
                "nearest_observed_gap_edge_to_E_points": nearest_edge_to_E,
            })
        ctx = pd.DataFrame(first_gap_context)
        only_after = ctx[ctx.only_after_first_span]
        single_one_minute = ctx[(ctx.gap_intervals == 1) & (ctx.missing_minutes == 1)]
        entered_unknown = trades[trades.status != "known"]
        position_gap_lengths = []
        for row in entered_unknown.itertuples():
            if row.reason != "gap_while_open":
                continue
            k = int(row.unknown_pos)
            position_gap_lengths.append(int((ts[k] - ts[k - 1]) // MINUTE - 1))
            first_uncertainty_addresses.append({
                "version": version,
                "kind": "open_position_gap",
                "date": row.date,
                "riz_id": row.riz_id,
                "tf_minutes": int(row.tf),
                "t0_utc": str(pd.Timestamp(int(ts[int(row.q)]), tz="UTC")),
                "zone_bottom": None,
                "zone_top": None,
                "target_E": float(row.target),
                "gap_intervals": 1,
                "missing_minutes": int((ts[k] - ts[k - 1]) // MINUTE - 1),
                "max_gap_minutes": int((ts[k] - ts[k - 1]) // MINUTE - 1),
                "gap_in_forming_C3": None,
                "gap_before_or_at_first_span": None,
                "gap_after_first_span": None,
                "nearest_observed_gap_edge_to_E_points": None,
            })
        policy[version] = {
            "known_sessions": int((day.status == "known").sum()),
            "unknown_sessions": int((day.status != "known").sum()),
            "first_uncertainty_reason_sessions": {k: int(v) for k, v in causes.items()},
            "first_blocking_prefix_gap_context": {
                "sessions": len(ctx),
                "distinct_market_gap_intervals_across_these_prefixes": len(shared_gap_positions),
                "exactly_one_missing_minute": int(((ctx.gap_intervals == 1)
                                                   & (ctx.missing_minutes == 1)).sum()),
                "only_gaps_of_at_most_five_minutes": int((ctx.max_gap_minutes <= 5).sum()),
                "any_gap_over_sixty_minutes": int((ctx.max_gap_minutes > 60).sum()),
                "gap_before_or_at_first_span": int(ctx.before_or_at_first_span.sum()),
                "gap_after_first_span": int(ctx.after_first_span.sum()),
                "gap_in_forming_c3_bar": int(ctx.within_formation_c3.sum()),
                "gap_between_birth_and_first_span": int(ctx.between_birth_and_first_span.sum()),
                "only_gap_after_first_span": len(only_after),
                "native_tf_at_least_60_minutes": int((ctx.tf >= 60).sum()),
                "single_one_minute_gap_on_tf_at_least_60": int(
                    ((ctx.gap_intervals == 1) & (ctx.missing_minutes == 1)
                     & (ctx.tf >= 60)).sum()),
                "single_one_minute_gap_nearest_edge_to_E_points": {
                    "count": len(single_one_minute),
                    "median": float(single_one_minute.nearest_observed_gap_edge_to_E_points.median()),
                    "p10": float(single_one_minute.nearest_observed_gap_edge_to_E_points.quantile(.1)),
                    "p90": float(single_one_minute.nearest_observed_gap_edge_to_E_points.quantile(.9)),
                },
                "missing_minutes_median": float(ctx.missing_minutes.median()),
                "missing_minutes_p90": float(ctx.missing_minutes.quantile(.9)),
                "gap_intervals_median": float(ctx.gap_intervals.median()),
                "only_after_span_min_observed_edge_to_E_points": {
                    "median": float(only_after.min_observed_edge_to_target_points.median()),
                    "p10": float(only_after.min_observed_edge_to_target_points.quantile(.1)),
                    "p90": float(only_after.min_observed_edge_to_target_points.quantile(.9)),
                    "min": float(only_after.min_observed_edge_to_target_points.min()),
                },
            },
            "entered_unknown_position_gap_lengths": position_gap_lengths,
            "buckets": buckets,
        }
        assert sum(v["sessions"] for v in buckets.values()) == len(cal)

    bins = {
        "1": int((other_length == 1).sum()),
        "2_to_5": int(((other_length >= 2) & (other_length <= 5)).sum()),
        "6_to_15": int(((other_length >= 6) & (other_length <= 15)).sum()),
        "16_to_60": int(((other_length >= 16) & (other_length <= 60)).sum()),
        "over_60": int((other_length > 60).sum()),
    }
    result = {
        "scope": "NQ, UTC-close-labelled minutes 2020-01-01 through 2025-12-31; same session_id intervals",
        "corpus_id": manifest["corpus_id"],
        "nq_usd_per_point": NQ_USD_PER_POINT,
        "nq_multiplier_source": "https://www.cmegroup.com/markets/equities/nasdaq/e-mini-nasdaq-100.contractSpecs.html",
        "observed_minutes": observed_count,
        "nominal_session_span_minute_slots_including_halts": nominal_grid,
        "scheduled_15_minute_halts": int(scheduled.sum()),
        "scheduled_halt_minute_slots": int(length[scheduled].sum()),
        "other_gap_intervals": len(other_ix),
        "other_missing_minute_slots": absent_count,
        "expected_minute_slots_excluding_documented_halts": expected_count,
        "other_missing_fraction": absent_count / expected_count,
        "observed_one_minute_range_points": {
            "median": float(np.quantile(one_minute_range, .5)),
            "p99": float(np.quantile(one_minute_range, .99)),
            "p999": float(np.quantile(one_minute_range, .999)),
        },
        "other_gap_length_bins_intervals": bins,
        "other_gap_max_missing_minutes": int(other_length.max()),
        "observed_market_sessions": int(np.unique(sid[observed]).size),
        "market_sessions_with_other_gap": int(np.unique(sid[other_ix]).size),
        "xnys_policy_sessions_2021_2025": len(cal),
        "xnys_policy_sessions_with_direct_market_gap": len(direct_days),
        "riz_with_any_prefix_gap": len(any_prefix_gap),
        "entry_window_riz_with_unknown_prefix": len(relevant),
        "entry_window_t0_minutes_with_unknown_prefix": int(relevant.t0_spine_pos.nunique()),
        "entry_window_sessions_with_unknown_prefix": int(relevant.session_date.nunique()),
        "policy": policy,
        "qualification": (
            "Other gaps are absent archive bars after removing the documented regular halt; "
            "individual exceptional exchange closures are not adjudicated. Policy-unknown "
            "sessions have only certified-prefix money, not full-session P&L. A gap in a "
            "RIZ prefix can predate the affected policy session. First-uncertainty addresses "
            "are conservative replay stops, not proven changed trades. Neighbor-to-E distances "
            "describe required excursions and are not hard price bounds."
        ),
    }
    assert sum(bins.values()) == len(other_ix)
    assert result["xnys_policy_sessions_2021_2025"] == 1255
    out = ROOT / "base/103/gap_impact.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(first_uncertainty_addresses).sort_values(
        ["version", "date", "kind", "riz_id"]
    ).to_csv(ROOT / "base/103/first_uncertainty_addresses.csv", index=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
