"""Canonical C1 RIZ state machine.

This is a small independent port of the operator-supplied Pine reference. Its
executable semantics are kept literal: native state changes only at native-bar
close, a refused adjacent re-span falls through to boundary retirement, and
Blue confirmation requires 2X+ with both boundaries alive.
"""

from __future__ import annotations

from dataclasses import dataclass

TIER_LATENT = 0
TIER_ACTIVE_BOTH = 1
TIER_ACTIVE_ONE_SIDED = 2
TIER_BREAKER = 3


@dataclass(slots=True)
class NativeBar:
    index: int
    open_ts_ns: int
    close_ts_ns: int
    open: float
    high: float
    low: float
    close: float
    minute_start: int
    minute_stop: int


@dataclass(slots=True)
class Zone:
    top: float
    bottom: float
    birth_label_ts_ns: int
    bullish: bool
    north_alive: bool = True
    south_alive: bool = True
    activated: bool = False
    last_span_index: int = -1
    tier: int = TIER_LATENT
    broke_south: bool = False
    span_count: int = 0

    @property
    def direction(self) -> int:
        return 1 if self.bullish else -1


def body_bounds(bar: NativeBar) -> tuple[float, float]:
    return min(bar.open, bar.close), max(bar.open, bar.close)


def spans(zone: Zone, bmin: float, bmax: float) -> bool:
    return bmin < zone.bottom and bmax > zone.top


def zone_key(zone: Zone) -> tuple[int, float, float, bool]:
    return (zone.birth_label_ts_ns, zone.top, zone.bottom, zone.bullish)


def _breaker_kind(zone: Zone, bar: NativeBar, spanned: bool) -> str | None:
    if spanned:
        return None
    o, c, h, low = bar.open, bar.close, bar.high, bar.low
    n, s = zone.north_alive, zone.south_alive
    sbk = s and not n and o > zone.bottom and c < zone.bottom
    nbk = n and not s and o < zone.top and c > zone.top
    sbk2 = n and s and o > zone.bottom and c < zone.bottom and h >= zone.top and c <= zone.top
    nbk2 = n and s and o < zone.top and c > zone.top and low <= zone.bottom and c >= zone.bottom
    if sbk or sbk2:
        return "south"
    if nbk or nbk2:
        return "north"
    return None


def advance_one(zone: Zone, bar: NativeBar) -> bool:
    """Advance one pre-existing zone; return whether it survives."""
    if zone.tier == TIER_BREAKER:
        returned = bar.high >= zone.bottom if zone.broke_south else bar.low <= zone.top
        return not returned

    bmin, bmax = body_bounds(bar)
    spanned = spans(zone, bmin, bmax)
    if spanned and (not zone.activated or bar.index >= zone.last_span_index + 2):
        if not zone.activated:
            zone.activated = True
            zone.tier = TIER_ACTIVE_BOTH if zone.north_alive and zone.south_alive else TIER_ACTIVE_ONE_SIDED
            zone.span_count = 1
        else:
            zone.span_count += 1
        zone.last_span_index = bar.index
        return True

    kind = _breaker_kind(zone, bar, spanned)
    if kind is not None:
        zone.broke_south = kind == "south"
        zone.north_alive = False
        zone.south_alive = False
        zone.tier = TIER_BREAKER
        zone.activated = True
        zone.last_span_index = bar.index
        return True

    if bar.low <= zone.top <= bar.high:
        zone.north_alive = False
    if bar.low <= zone.bottom <= bar.high:
        zone.south_alive = False
    return zone.north_alive or zone.south_alive


def detect_births(c1: NativeBar, c2: NativeBar, c3: NativeBar) -> list[Zone]:
    born: list[Zone] = []
    c1_top, c1_bottom = max(c1.open, c1.close), min(c1.open, c1.close)
    c2_top, c2_bottom = max(c2.open, c2.close), min(c2.open, c2.close)
    c3_top, c3_bottom = max(c3.open, c3.close), min(c3.open, c3.close)
    if c3.low > c1.high and c2_top > c1.high and c2_bottom < c3.low:
        bottom = c1_top if c2_bottom > c1_top else c1.high
        top = c3_bottom if c3_bottom > c2_top else c3.low
        if top > bottom:
            born.append(Zone(top, bottom, c1.open_ts_ns, True))
    if c3.high < c1.low and c2_bottom < c1.low and c2_top > c3.high:
        top = c1_bottom if c2_top < c1_bottom else c1.low
        bottom = c3_top if c3_top < c2_bottom else c3.high
        if top > bottom:
            born.append(Zone(top, bottom, c1.open_ts_ns, False))
    return born


def blue_confirmed(zone: Zone) -> bool:
    return (
        zone.tier in (TIER_ACTIVE_BOTH, TIER_ACTIVE_ONE_SIDED)
        and zone.span_count >= 2
        and zone.north_alive
        and zone.south_alive
    )
