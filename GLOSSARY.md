# G3 glossary

The bridge between spoken market language — the trader's or the agent's own —
canonical English terms, and the stored field. Load meanings from here; do not
reconstruct them from column names or generic trading literature.

Each entry is one of:

- **field fact** — recorded by the ready field;
- **machine provenance** — needed to explain a stored RIZ, but not the ordinary
  research population;
- **research term** — defined by a study and never smuggled in as a field fact;
- **project architecture** — how meaning is preserved between agents.

## Object and coordinates

### RIZ — `РИЗ`, `риз`

**Field fact.** A zone that reached Blue/2X and entered this repository's
research population. One stored RIZ has one `riz_id`. In ordinary G3 research,
the object begins at T0; a pre-T0 candidate is not counted as a RIZ population
member.

### North / south — `север` / `юг`

**Field fact.** The upper and lower fixed prices of the zone:
`zone_top`, `zone_bottom`. A side may later retire, but its price does not move.

### Exit boundary — `выходная граница`, `сторона ухода`

**Field fact, composed of two.** The boundary price the RIZ left through at T0:
`zone_top` when `t0_exit_side` is `north`, `zone_bottom` when it is `south`.
Exposed as `Film.exit_boundary`; the other one is `Film.far_boundary`.

This is the line the trader's post-T0 language is about, and it belongs to the
RIZ for the whole life of the film — it never moves and never switches sides,
whatever price later does. The T0 minute closes strictly beyond it in 100% of
the field and strictly beyond the other boundary in 0% (checked on NQ TF 5, 54,
240), so a cold agent never needs to re-detect the span to find it.

It is a per-RIZ fact, not a per-minute one. See `Episode vs row` for the
measured consequence.

### Width — `ширина риза`

**Field fact.** `zone_top - zone_bottom`, stored as `zone_width`. Width is an
available conditioner, not an established explanation and not presumed to
dominate a result until measured.

### Native timeframe — `родной ТФ`, `таймфрейм риза`

**Field fact.** The bar duration on which the zone's machine runs, stored as
`tf_minutes`. G3 contains every integer value from 1 to 1440 minutes.

### Minute tape — `минутная лента`

**Field fact.** The canonical one-minute market sequence used to observe T0,
slice context and eventually make decisions. A native RIZ timeframe and the
minute observation clock are different things.

### Origin direction — BISI / SIBI, `бычий` / `медвежий риз`

**Field fact.** The direction of the imbalance from which the zone originated:
`bullish` and `direction` (`+1` / `-1`). It is not the direction of a future
trade.

## Lifecycle before and into T0

These terms explain ancestry of an already stored RIZ. They do not change the
rule that the research population begins at T0.

### Pre-T0 ancestry — `предыстория риза`

**Machine provenance.** The latent zone and events that led to Blue/2X. The
field preserves direct addresses such as `precursor_formed_ts_ns`, but a cold
agent does not start by constructing or studying every latent gap.

### Span — `спан`, `прошив телом`

**Machine provenance / field event.** A candle body traverses the full zone,
strictly beyond both boundaries. A wick touch or partial body entry is not a
span.

### Activation — `активация`, `первый спан`

**Machine provenance.** The first accepted full-body span. It activates the
latent zone but does not yet make it a stored Blue/2X research object.

### Re-span — `повторный спан`, `повторный прошив`

**Machine provenance / field event.** A later accepted span of the same zone.
The spacing rule requires at least one complete native bar between accepted
spans; an adjacent traversal does not increment the counter.

### Blue / 2X — `синий риз`, `2X`

**Field qualification.** The zone has two accepted spans while both boundaries
remain alive. This is the filter that admits the RIZ into the field. It says the
place is research-worthy, not what price will do next.

### T0 — `T0`, `момент отсчёта`

**Field fact and primary research anchor.** The close of the one-minute candle
on which the Blue/2X condition first became observable on the minute tape:
`t0_ts_ns`, `t0_spine_pos`, `t0_close`, `t0_exit_side`, `t0_kind`,
`t0_span_count`.

T0 gives a place and moment, not a trading hypothesis. Minutes before T0 may be
used as information already known at T0. A proposed decision before T0 is a
separate precursor question and may not use future knowledge that T0 will occur.

**T0 is not "a minute candle whose body spanned the zone".** The 2X box turns
blue intrabar, and the Pine reads the body of the FORMING NATIVE bar
(`is2xPreview`, line 261 of the pinned source), so the span can be half made of
the native bar's open. On NQ, the T0 minute's own body spans the zone in 68.7%
of cases at TF 5, 39.8% at TF 54 and 28.1% at TF 240. A hand-written
minute-body detector therefore measures a different market event on most
timeframes; this is the exact failure mode AGENTS.md calls
«тем же способом, каким это записано в поле».

What always holds on the minute tape is the close: strictly beyond
`t0_exit_side`, never beyond the other side. Take the exit boundary from the
passport, do not re-derive the span.

### T0 kind — `t0_kind`, `минутное зажигание`, `нативное зажигание`

**Field fact.** How T0 was reached: `minute_ignition` when the intrabar preview
was what first made Blue/2X visible on the tape, `native_confirmation` when a
native bar close did. `minute_ignition` dominates (1,029 of 1,046 on NQ TF 54)
and carries `t0_span_count = 1`, because the second span has not been accepted
at native close yet — which is not a contradiction of Blue/2X but the reason
`native_blue_confirmation_*` and `Flicker` exist. See
[`011`](base/011-nativnoe-podtverzhdenie-pereskazyvaet-put.md) and
[`012`](base/012-odinokoe-zazhiganie-podglyadyvalo-vperyod.md).

### Native confirmation — `нативное подтверждение`, `подтверждение на родном ТФ`

**Field fact.** The later close of the RIZ's native bar that confirms the Blue
state: `native_blue_confirmation_ts_ns` and
`native_blue_confirmation_spine_pos`. It is at or after T0 and must not be
back-ported into earlier minutes.

### Flicker — `фликер`, `мигающий риз`

**Field fact derived from recorded clocks.** T0 occurred on the minute tape but
the native bar later failed to confirm Blue. This is a valid research object,
not a data error.

### 3X / x3 ignition — `3X`, `третий спан`

**Field fact.** The same RIZ reaches a third accepted span. It remains the same
`riz_id`; it is not a new zone. `x3_t0_*` records minute observation and
`x3_t0_confirmed` records native acceptance. G3 does not presume 3X is stronger
until a study measures that proposition.

## Later life and end states

### Life of a RIZ — `жизнь риза`, `цикл риза`

**Research frame over field facts.** Everything observable from T0 until the
relevant boundary retirement, breaker transition, deletion or archive end.
Every study must name which endpoint it uses; these endpoints are not synonyms.

### Boundary retirement — `снятие границы`, `север/юг снят`

**Field fact, and a per-scale one.** The rule is evaluated at NATIVE BAR close,
on that bar's own high and low, and it requires the bar's range to *contain* the
boundary (`bar.low <= boundary <= bar.high`) — not merely to reach it. It is
also skipped entirely on a bar that produces an accepted span or a breaker,
because those paths return before the test. A RIZ may continue to exist with one
side alive. Fields include `final_north_alive` and `final_south_alive`;
lifecycle events give the event time.

The machine has always worked this way; it is this entry that described
something else. Nothing was learned about retirement — a wrong description was
corrected. Two consequences follow, and any study touching lifecycle order needs
both:

- **Retirement is not a minute-tape wick touch.** Measured against the tape, the
  recorded minute lands a median 70 minutes after the first minute the tape
  reached the boundary price, even at TF 1.
- **A shared price is not a shared event.** Layers holding the *same exact*
  boundary retire on one identical minute in only ~10% of cases, and a larger
  timeframe frequently retires *earlier* than a smaller one, because a longer
  bar covers the price sooner in wall-clock terms and every timeframe sits at a
  different phase of the bar grid. Reading "TF 240 retired before TF 15" as
  market structure is reading bar-grid phase. Strip the word and what remains is
  price returning to a level, which needs no machine semantics to observe — see
  [`009`](base/009-okruzhenie-rizov-ne-delit-sleduyushchiy-chas.md) and
  [`010`](base/010-pamyat-kontaktov-s-urovnem-ploskaya.md).

### Blue eligibility end — `конец синего состояния`

**Field fact.** The minute after which the object can no longer satisfy the
strict Blue condition. `blue_eligibility_end_spine_pos`. This can precede final
deletion.

### Breaker — `брейкер`

**Field fact.** A non-span body crosses the relevant live boundary and the
object enters the machine's one-sided breaker tier. Fields include
`final_tier = 3` and `final_broke_south`. Breaker is a lifecycle state, not by
itself a proven magnet or setup.

### Deletion / annihilation — `удаление`, `аннигиляция`, `риз умер`

**Field fact.** The canonical machine no longer carries the object, recorded by
`c1_deletion_ts_ns` and `c1_deletion_spine_pos`. Informal `риз умер` must be
mapped to this exact endpoint or replaced by the intended earlier endpoint.

### Censored — `цензурирован`, `история закончилась раньше риза`

**Field fact.** The historical corpus ended while the object or relevant side
was still alive: `censored`, `still_alive_at_archive_end`. Censoring is unknown
future life, not long survival and not deletion.

## Research language not supplied by the machine

The following are **research terms**. A study card must operationalize them if
the exact meaning affects selection or outcome.

### History — `история`

Careful: in Russian this word carries two meanings — **the loaded tape** and **a
causal story**. In G3 it means only the tape: twenty years of one-minute market
history. For a causal account say `механика`.

### Mechanism — `механика`, `почему это работает`

A causal account of why price behaves this way. **Optional, and never required to
accept a finding.** Repeatability on an honestly counted sample is sufficient
ground. If a mechanism shows up later, good; demanding one up front restricts the
search to what is already understood.

### Episode vs row — `эпизод` / `строка`

**Research term, and the main way to be fooled here.** The same tape is replayed
1,440 times per instrument, once per native timeframe, so a RIZ on TF 10 and a
RIZ on TF 15 at the same place and time are two rows that **may** be one market
event. Overlapping RIZ in time inflate the count further. Rows are not
independent observations by default, and a study must say how many independent
units it actually has.

There is no repository-wide episode predicate, and none is owed. Sharing a T0
minute is evidence that two RIZ are related, not proof they are one object:
they may be one event seen at several scales, separate interactions, or an outer
zone containing an inner one. A study that needs an independent unit defines and
names its own, and says what that definition throws away. Deduplicating by T0
minute alone is a diagnostic statistic, never an episode count, which is why
`Field.t0_minute_groups()` is named after the grouping and not after an event.

**The collapse is measured, not feared.** On NQ, 44.6% of T0 minutes carry more
than one RIZ, and 56.6% of those minutes carry two or more DIFFERENT exit
boundaries — up to 170 on a single minute. Among those, the first post-T0
contact with the exit boundary falls on a different minute for 46.1% of the
minutes (p90 spread 22 minutes, worst case 599). One pair: 2006-06-12 07:13 UTC,
TF 41 leaves north through 1576.00 and TF 44 through 1575.75; first contact is
+1 for one and +205 for the other. Films are per `riz_id`, always.

### Film — `фильм`

**Research frame, not a stored field fact.** The minute tape around ONE
`riz_id`, anchored at ITS T0, carrying its own pre-roll and a named end —
`deletion`, `blue_end`, `archive_edge`, an observation budget, or a position the
study closed itself. The end is always stated, never guessed. The canonical
field does not store films: `Field.films()` builds them on demand from recorded
facts. A study may materialise its own corpus of films locally, and then that
corpus — not the word — carries provenance: its own identity, the semantic
generation it was built from, and its selection rule.

**Careful with older cards: the word there usually names a fixed window.**
`H15-фильм` (006, 007), `120-минутный фильм` (008, 016) and `165-минутный
фильм` (017) are observation budgets over T0-minute groups, not per-RIZ films
with a market end. A budget is a legitimate named end. It is not the object
below, and a card measuring one is not evidence about the other.

### Film-1 — `первый фильм`, `Film-1`

**Research frame with an unambiguous end, and the minimal object of per-RIZ
work.** The minutes from a RIZ's T0 through the FIRST minute after T0 whose
range meets that RIZ's exit boundary, inclusive. A wick counts.
`first_exit_contact_v1` returns that ordinal; `Field.film(end_position=...)`
cuts the film there.

The T0 minute itself never qualifies: its range straddles the line it closed
beyond, by construction.

**One film per RIZ is the unit of the OBJECT, never the unit of INDEPENDENCE.**
Two RIZ must have two films — that is what the reset is about — but two films on
the same tape, close in price and time, can have almost the same outcome for the
same reason. `riz_id` fixes what is being watched; it says nothing about how
many independent things were observed. Counting 100,000 films and speaking as if
100,000 independent observations were made is the same substitution as before,
wearing the corrected object's clothes. A study reports how many films it
counted AND how few independent occasions stand behind them — distinct T0
minutes, distinct sessions, distinct price neighbourhoods — and lets the
weakest count carry the claim. See `Episode vs row`.

**No departure is required, and the word `retest` is avoided for that reason** —
`retest` smuggles in "price first went away", and here it usually did not. On
40,000 sampled NQ RIZ the first contact is at +1 minute in 57.7% of cases,
within 3 minutes in 73.0%, within 15 in 88.4%, and absent inside 600 minutes in
4.0%. The median Film-1 is one minute long. A study that wants a departure adds
it to its own selection and reports what that drops.

Nothing canonical follows Film-1. After the first contact the market branches —
continued contact, departure, return, full traversal, breaker — and no
repository-wide Film-2 / Film-3 cut exists or is owed. What is owed is that the
later life stays recoverable without loss: a film with an observation budget
plus `exit_boundary_touch_v1` over every minute gives that.

Worked out on one real scene in [`reference/SCENE.md`](reference/SCENE.md).

### Context — `контекст`

Information known at the evaluated minute: earlier one-minute bars, RIZ state,
session, other already-existing RIZ and any other point-in-time feature named by
the study.

### Retest — `ретест`, `повторный тест`

Price returns to a specified boundary or zone after a specified departure.
The machine does not provide a universal retest flag. A study must define the
line, direction of approach, qualifying touch/close, and whether repeated
contacts count.

The word carries a condition — that price first left — and the field does not
supply it. Where the trader says "first retest" about the end of the first film,
the repository says `first exit contact`; see `Film-1`. Use `retest` only when
departure is actually in your selection.

### Return / rejection / continuation / reversal

`возврат` / `отбой` / `продолжение` / `разворот`.

Different possible price behaviors, never implied by RIZ origin direction.
Each needs an exact outcome and horizon in the study that uses it.

### Stack / cluster / confluence

`стек ризов` / `скопление ризов` / `конфлюэнс`.

Several RIZ related in price and time. The field exposes the objects; the
distance, overlap, age and timeframe rules are research definitions.

### RIZ interaction / ecology — `взаимодействие ризов`, `экология ризов`

Relations among multiple RIZ: nesting, overlap, opposed origin, shared
boundary, sequence, clustering, or inheritance of a level. No single relation
is assumed useful before measurement.

### Conditional asymmetry — `условная асимметрия`, `закономерность`

A measured change in later price behavior conditional on a named RIZ state or
history, relative to an explicit comparison. It is bounded by the measured
territory and is not yet a setup.

### Pattern / observation — `паттерн`, `наблюдение`

A visual or numerical lead worth asking about. Until measured, it is an `IDEA`,
not knowledge.

### Edge — `эдж`, `преимущество`

A persistent positive trading asymmetry after realistic decision timing, risk,
costs and failure conditions. A descriptive effect is not automatically edge.

### Setup — `сетап`

A complete trade logic: applicability, trigger, direction, action,
invalidation, management and exit. The project is intended to grow into a
library of narrow setups, each with its own territory and failure mode.

## Reading the field

```python
from pathlib import Path
from g3riz.query import Field

field = Field(Path.cwd(), "NQ")
zones = field.passports()          # all 1,440 timeframes; tf=N is a slice, not the field
windows = field.minute_windows(
    zones["t0_spine_pos"].to_numpy(), before=30, after=120
)
```

`Field.objects_at(minute_pos, state="blue")` exposes all Blue-eligible RIZ at a
minute across timeframes and is the main entry into interaction research.
