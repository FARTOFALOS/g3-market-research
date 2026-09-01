"""What this directory is for, and the fixtures it shares.

Tests here insure shared assumptions with a large blast radius: field read
identity, T0 and end/censor/budget semantics, point-in-time properties, and the
difference between reusable predicates that look alike. A defect that already
happened once earns a test so it cannot happen quietly again.

NOT tested here: the format of a card, a one-off formula, the order hypotheses
are picked in, or whether a question is worth asking at all. A green test proves
the predicate it encodes — never that the right thing was asked of the market.
No test is a permit to start researching, and none is owed before a question.

Fixtures: hand-built films, and the real field when it is present.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from g3riz.film import Film

REPO = Path(__file__).resolve().parents[1]


def make_film(high, low, *, open_=None, close=None, top=10.0, bottom=0.0,
              exit_up=True, pre_roll=0, stop="deletion",
              end_reason="none") -> Film:
    """A film from bare high/low. Bodies default to the middle of the range."""
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    mid = (high + low) / 2.0
    open_ = mid if open_ is None else np.asarray(open_, dtype=np.float64)
    close = mid if close is None else np.asarray(close, dtype=np.float64)
    n = high.size
    return Film(
        riz_id="riz_test", instrument="NQ", tf_minutes=10,
        zone_top=top, zone_bottom=bottom, exit_up=exit_up,
        stop=stop, end_reason=end_reason, n_pre_roll=pre_roll,
        spine_pos=np.arange(n, dtype=np.int64),
        close_ts_utc_ns=np.arange(n, dtype=np.int64) * 60_000_000_000,
        open=open_, high=high, low=low, close=close,
        volume=np.ones(n, dtype=np.int64),
        session_id=np.zeros(n, dtype=np.int32),
        bar_ord=np.arange(n, dtype=np.int64) - pre_roll)


def field_available() -> bool:
    return (REPO / "data" / "market" / "NQ" / "manifest.json").exists()


needs_field = pytest.mark.skipif(
    not field_available(),
    reason="local data/market and data/field are absent; a clone carries no bytes")


@pytest.fixture(scope="session")
def nq_field():
    from g3riz.query import Field

    return Field(REPO, "NQ")
