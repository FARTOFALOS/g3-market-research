"""078A — the three frozen generic representations and the frozen distance.

Everything here is fixed by base/078/FREEZE_078A.md and must not be edited to
change a result. No RIZ-derived quantity enters any coordinate. Y is never
defined, computed or approached. Volume is never loaded.

One eligibility set serves every declared variant: the padding is the strictest
one any declared sensitivity needs (longest window, shifted by R1, plus the
longest ruler). All comparisons therefore run on an identical population.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research"))
from calendar_utils import eastern  # noqa: E402

MINUTE = 60_000_000_000
RULER = 60                 # 076/077 prior ruler
RULERS = (30, 60, 120)     # LOOKBACKS of base/076/prior_scale.py -> M3
L_G1 = 5                   # 065 / 061 / 075 prefix language
L_G2 = 30                  # S-09 / 061 / 008 / 013 / 015 / 016 prefix language
L_MAX = L_G2
PAD = L_MAX + 1 + max(RULERS)   # 151 contiguous minutes ending at p
EPOCH0, EPOCH_SPAN = 2006, 20


@dataclass(frozen=True)
class Spec:
    """One frozen representation plus one frozen metric variant."""
    name: str
    lookback: int | None      # None -> G0, no shape block
    include_p: bool = True    # False -> sensitivity R1
    w_clock: float = 1.0
    w_epoch: float = 1.0
    w_scale: float = 1.0
    w_shape: float = 1.0
    ruler: int = RULER

    @property
    def shift(self) -> int:
        return 0 if self.include_p else 1

    @property
    def dims(self) -> int:
        return 4 + (0 if self.lookback is None else 4 * self.lookback)


G0 = Spec("G0", None)
G1 = Spec("G1", L_G1)
G2 = Spec("G2", L_G2)
BASE_SPECS = (G0, G1, G2)

SENSITIVITIES = (
    Spec("G1 R1 no-p", L_G1, include_p=False),
    Spec("G2 R1 no-p", L_G2, include_p=False),
    Spec("G0 M1 no-epoch", None, w_epoch=0.0),
    Spec("G1 M1 no-epoch", L_G1, w_epoch=0.0),
    Spec("G2 M1 no-epoch", L_G2, w_epoch=0.0),
    Spec("G1 M2 shape-only", L_G1, w_clock=0.0, w_epoch=0.0, w_scale=0.0),
    Spec("G2 M2 shape-only", L_G2, w_clock=0.0, w_epoch=0.0, w_scale=0.0),
    Spec("G0 M3 ruler30", None, ruler=30),
    Spec("G1 M3 ruler30", L_G1, ruler=30),
    Spec("G2 M3 ruler30", L_G2, ruler=30),
    Spec("G0 M3 ruler120", None, ruler=120),
    Spec("G1 M3 ruler120", L_G1, ruler=120),
    Spec("G2 M3 ruler120", L_G2, ruler=120),
)


class Tape:
    """The frozen one-minute spine plus everything both classes share."""

    def __init__(self, instrument: str = "NQ"):
        market = ROOT / "data" / "market" / instrument
        self.instrument = instrument
        self.open = np.load(market / "open.npy").astype(np.float64)
        self.high = np.load(market / "high.npy").astype(np.float64)
        self.low = np.load(market / "low.npy").astype(np.float64)
        self.close = np.load(market / "close.npy").astype(np.float64)
        self.ts = np.load(market / "close_ts_utc_ns.npy")
        self.session_id = np.load(market / "session_id.npy")
        self.n = len(self.close)
        self._sigma: dict[tuple[int, int], np.ndarray] = {}
        self._rng = None
        self._clock = None
        self._eligible = None

    # -- clock / epoch, America/New_York (fix of 068) ---------------------
    def clock(self):
        if self._clock is None:
            e = eastern(self.ts)
            self._clock = (
                (e.hour.to_numpy() * 60 + e.minute.to_numpy()).astype(np.int32),
                e.year.to_numpy().astype(np.int32),
            )
        return self._clock

    # -- prior ruler ------------------------------------------------------
    @property
    def rng_hl(self) -> np.ndarray:
        if self._rng is None:
            self._rng = (self.high - self.low).astype(np.float32)
        return self._rng

    def sigma_at(self, pos: np.ndarray, ruler: int = RULER,
                 shift: int = 0) -> np.ndarray:
        """median(high-low) over `ruler` minutes ending at p - L_MAX - shift.

        Always strictly before the start of the longest declared window, so the
        ruler never overlaps any representation window (FREEZE section 3).
        Computed for the requested positions only; nothing is cached per tape.
        """
        pos = np.asarray(pos, dtype=np.int64)
        end = L_MAX + shift                      # last ruler index is p - end
        out = np.full(pos.size, np.nan)
        rng = self.rng_hl
        for s in range(0, pos.size, 200_000):
            chunk = pos[s:s + 200_000]
            beg = chunk - end - ruler + 1
            idx = beg[:, None] + np.arange(ruler)[None, :]
            out[s:s + len(chunk)] = np.median(rng[idx], axis=1)
        return out

    def sigma_full(self) -> np.ndarray:
        """sigma for the base variant over the whole tape. Cached once."""
        if (RULER, 0) not in self._sigma:
            out = np.full(self.n, np.nan)
            first = L_MAX + RULER - 1
            allp = np.arange(first, self.n, dtype=np.int64)
            out[first:] = self.sigma_at(allp, RULER, 0)
            self._sigma[(RULER, 0)] = out
        return self._sigma[(RULER, 0)]

    # -- eligibility, one set for every declared variant -------------------
    def eligible(self) -> np.ndarray:
        """151 contiguous finite minutes ending at p, and a positive base ruler.

        PAD covers the longest declared ruler (120) plus the longest declared
        window shifted by R1, so every declared variant is computable on exactly
        this population. A variant ruler that comes out zero on a flat stretch
        is counted and reported, not silently dropped.
        """
        if self._eligible is not None:
            return self._eligible
        finite = (np.isfinite(self.open) & np.isfinite(self.high)
                  & np.isfinite(self.low) & np.isfinite(self.close))
        step = np.zeros(self.n, dtype=bool)
        step[1:] = (np.diff(self.ts) == MINUTE) & finite[1:] & finite[:-1]
        reset = np.where(~step, np.arange(self.n), 0)
        np.maximum.accumulate(reset, out=reset)
        run = np.arange(self.n) - reset          # consecutive valid steps ending at i
        s = self.sigma_full()
        ok = (run >= (PAD - 1)) & np.isfinite(s) & (s > 0)
        self._eligible = ok
        return ok


def sigma_sd(tape: Tape, spec: Spec, pool: np.ndarray) -> float:
    """SD of log sigma over the eligible comparison pool. Frozen constant."""
    s = tape.sigma_at(pool, spec.ruler, spec.shift)
    s = s[np.isfinite(s) & (s > 0)]
    return float(np.std(np.log(s)))


def coords(tape: Tape, pos: np.ndarray, spec: Spec, s_sd: float) -> np.ndarray:
    """Frozen weighted coordinates; plain Euclidean on them IS the frozen metric.

    d^2 = w_clock*(1-cos dTheta)/2 + w_epoch*(dy)^2
        + w_scale*((dlog sigma)/s)^2 + w_shape*mean_j (dx_j)^2

    is reproduced exactly by scaling coordinate j by sqrt(c_j).
    """
    pos = np.asarray(pos, dtype=np.int64)
    mod, yr = tape.clock()
    sig = tape.sigma_at(pos, spec.ruler, spec.shift)

    theta = 2.0 * np.pi * mod[pos].astype(np.float64) / 1440.0
    cols = [
        np.sqrt(spec.w_clock * 0.25) * np.cos(theta),
        np.sqrt(spec.w_clock * 0.25) * np.sin(theta),
        np.sqrt(spec.w_epoch) * (yr[pos].astype(np.float64) - EPOCH0) / EPOCH_SPAN,
        np.sqrt(spec.w_scale) * np.log(sig) / s_sd,
    ]
    out = [c[:, None] for c in cols]

    if spec.lookback is not None:
        L = spec.lookback
        end = pos - spec.shift
        idx = end[:, None] + np.arange(-L + 1, 1, dtype=np.int64)[None, :]
        anchor = tape.close[end - L]
        vals = np.empty((len(pos), L, 4), dtype=np.float64)
        for j, a in enumerate((tape.open, tape.high, tape.low, tape.close)):
            vals[:, :, j] = a[idx]
        vals -= anchor[:, None, None]
        vals /= sig[:, None, None]
        out.append(vals.reshape(len(pos), 4 * L) * np.sqrt(spec.w_shape / (4.0 * L)))

    return np.ascontiguousarray(np.hstack(out), dtype=np.float32)
