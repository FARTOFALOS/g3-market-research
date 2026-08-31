# Glossary

Every term below is either a **field fact** — something the materialized field
records, with the column that holds it — or a **derived read**, which is ours to
define and is not in the field. Do not blur the two. `RIZ.md` explains why any
of this matters.

## The object

**RIZ** — a price range the market left untraded and has since run through with
a candle body at least twice. The research object of this program. One RIZ =
one row in `passports`, identified by `riz_id`.

**BISI / SIBI** — direction of the imbalance. BISI is bullish (gap up), SIBI is
bearish (gap down). Fields: `bullish`, `direction` (+1 / -1).

**North / south** — the top and bottom price of the range. Fixed at birth,
never move. Fields: `zone_top`, `zone_bottom`. These are the lines price
interacts with.

**Width** — north minus south. Field: `zone_width`. Known to dominate almost
any question asked about a zone, so condition on it before asking anything.

**Timeframe** — which native bar size the zone belongs to, 1 to 1440 minutes.
Field: `tf_minutes`. The same price area carries zones from many timeframes at
once.

## Birth and the road to significance

**Precursor** — the three-candle shape that creates the latent range: a gap
between the first and third candle, with the middle candle's body covering that
gap. The middle body is what makes it a real impulse rather than a wick poke.
The zone appears at the close of the third candle. Fields:
`precursor_formed_ts_ns`, `precursor_formed_spine_pos`,
`precursor_native_bar_index`.

**Span** — a candle body runs the entire range, edge to edge. Not a touch, not
a partial entry: the body's open-to-close must cover the whole zone. This is
the unit of significance.

**Activation** — the first span. The range stops being latent.

**Spacing rule** — a second span only counts if at least one full bar has closed
since the previous one. A span that comes too soon does not count and usually
kills the zone instead.

**2X / Blue** — two accepted spans with both boundaries still alive. The zone is
now significant and enters the field. Zones that never reach 2X are not stored
at all.

**3X** — a third accepted span. Same object, same `riz_id`, higher count. Rarer
and stronger. Fields: `x3_t0_ts_ns`, `x3_t0_spine_pos`, `x3_t0_close`,
`x3_t0_exit_side`, `x3_t0_confirmed`, `final_span_count`.

## The four times

These are four different clocks. Confusing them is the most common error.

**T0** — the minute the zone first qualified as 2X, caught on the one-minute
tape while the native bar was still forming. This is the birth of the research
object and the anchor for everything else. Fields: `t0_ts_ns`,
`t0_spine_pos`, `t0_close`, `t0_exit_side`, `t0_kind`, `t0_span_count`.

**Native confirmation** — the same qualification, registered at the close of the
zone's own bar. Always at or after T0. Fields:
`native_blue_confirmation_ts_ns`, `native_blue_confirmation_spine_pos`.

**Flicker** — T0 fired but the native bar closed without confirming. The object
existed on the minute tape and never appeared on a closed chart. Not an error.
Detect it as a passport with a T0 and no native confirmation.

**x3 ignition** — the same minute-level catch, for the third span.
`x3_t0_confirmed` says whether the native bar accepted it.

## Death

**Boundary retirement** — a wick touches a boundary and that side is dead.
Boundaries never come back. Fields: `final_north_alive`, `final_south_alive`.

**Breaker** — the body did not run the range but pierced one side: opened inside,
closed outside. The zone stops being a zone and becomes a one-sided magnet
pulling toward the pierced side. Fields: `final_tier` = 3, `final_broke_south`.

**Annihilation** — both boundaries retired. The zone is deleted and no longer
exists. Fields: `c1_deletion_ts_ns`, `c1_deletion_spine_pos`.

**Blue eligibility end** — the moment the zone could no longer be blue, which
can precede deletion. Field: `blue_eligibility_end_spine_pos`.

**Censored** — the archive ended while the zone was still alive. Its death was
never observed. Fields: `censored`, `still_alive_at_archive_end`. Never treat a
censored zone as long-lived.

## Not in the machine — ours to define

None of these exist in the field. Any definition is a research decision, and it
belongs in `FINDINGS.md` with the observation that used it.

**Retest** — price returning to a line after leaving it. The machine has no such
concept; it knows only span, wick touch and pierce.

**Confluence / stack** — several zones from different timeframes occupying
roughly the same price area. "Roughly" is undefined and must be stated wherever
it is used.

**Magnet** — the pull toward an unfinished range. Motivation, not a field fact,
except in the specific breaker sense above.

**Setup** — a repeatable situation stated precisely enough to act on. The output
of this program; none exist yet.

## Reading the field

Anchor everything on `t0_spine_pos` (or `x3_t0_spine_pos`) and slice the minute
tape around it:

```python
from pathlib import Path
from g3riz.query import Field

field = Field(Path.cwd(), "NQ")
z = field.passports(tf=10)
w = field.minute_windows(z["t0_spine_pos"].to_numpy(), before=30, after=120)
```

`Field.objects_at(minute_pos, state="blue")` gives every zone alive at a given
minute across all timeframes — the entry point for the second research surface.
