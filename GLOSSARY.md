# G3 glossary

The bridge between spoken market language — the trader's or the agent's own —
canonical English terms, and the stored field. This is the single reference for
exact definitions. Begin with `reference/SCENE.md`; look up the terms you need
here rather than reading every entry at startup. Do not infer meanings from
column names or generic trading literature.

Each entry is one of:

- **field fact** — recorded by the ready field;
- **machine provenance** — needed to explain a stored RIZ, but not the ordinary
  research population;
- **research term** — defined by a study and never smuggled in as a field fact;
- **project architecture** — how meaning is preserved between agents.

## What this contract establishes

Definitions fix object identity, event predicates and observation clocks.
Numbers cited from studies describe their measured populations; they do not
silently become admission filters or causal facts. A mechanism may be unknown.

The source minute OHLC and timestamps are observations retained with the
provenance and uncertainties in `SOURCE_DATA.json`. Canonical time/session
coordinates and RIZ passports/events are derived from those observations under
the pinned transform and machine. A **field fact** is a persisted observation
under that contract, not direct evidence of hidden liquidity or causal force.
Film and lenses derive views from the field; claims belong in `base/`.

## Object and coordinates

### RIZ — `РИЗ`, `риз`

**Field identity and research meaning.** RIZ research concerns an observable
scene of repeated price interaction with a previously formed price region,
after the prescribed lifecycle sequence. The zone locates the scene; T0 anchors
its first Blue/2X observation; the passport addresses it. The subject is the
price history and subsequent interactions, not the rectangle or timestamp alone.

One stored RIZ still has one `riz_id`, with the existing Blue/2X admission rule
unchanged. A pre-T0 candidate is ancestry, not yet a RIZ population member.
The full worked ancestry and the boundary between observed price geometry and
a hypothesis of unfilled interest are in `reference/SCENE.md`.

### North / south — `север` / `юг`

**Field fact.** The upper and lower fixed prices of the zone:
`zone_top`, `zone_bottom`. A side may later retire, but its price does not move.

### Exit boundary — `выходная граница`, `сторона ухода`

**Field fact, composed of two.** The boundary price the RIZ crossed at T0:
`zone_top` when `t0_exit_side` is `north`, `zone_bottom` when it is `south`.

The name `сторона ухода` identifies a side, not a developed departure.
The proximity measurements and their population are in
[`046`](base/046-t0-eto-vozvratnaya-hodka-a-ne-uhod.md); they do not define RIZ.
Exposed as `Film.exit_boundary`; the other one is `Film.far_boundary`.

This is the line the trader's post-T0 language is about, and it belongs to the
RIZ for the whole life of the film — it never moves and never switches sides,
whatever price later does. The T0 minute closes on its exit side; the
strict-close relation is checked in the scene calibration. Take the recorded
side and fixed prices directly; do not replace them with a new minute-body
span detector.

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

### Cluster — `кластер`

**Research term over field facts, and the unit the field does not store.** All
RIZ whose T0 falls on the same minute of the tape, reduced to distinct zones by
boundary price. One minute of tape ignites them together, and in every measured
case with five or more members they share a single `t0_exit_side`: one candle
body closes beyond several zones at once.

The field has no row, id or column for this object — it stores passports one per
`riz_id`. A cold agent therefore counts RIZ one by one and measures a different
population than the trader sees on the chart. Two consequences, measured on NQ
2006–2025 in [`060`](base/060-vizity-k-linii-klastera.md):

- **Per-RIZ counting inflates density 2.9×.** Neighbouring timeframes produce
  near-identical bars and hence copies of one zone: mean 5.29 RIZ per ignition
  minute against 1.83 distinct zones. NQ 2024-10-17 09:00 records sixteen RIZ on
  TF 455…1066 which are three zones. Reduce by boundary price, tolerance about a
  quarter of the minute candle, before any population claim.
- **Only part of a cluster is visible to anyone.** The pinned Pine pair draws
  every minute up to 59 plus 60, 120, 240, 420 and D. Zones native to TF 61–1439
  outside those values appear on no chart in the world, and 18.8% of ignition
  minutes contain no visible timeframe at all. Whether an effect needs an
  observer is therefore testable on neighbours — TF 240 against TF 239 and 241 —
  and any population claim should say which of the two sets it used.

The cluster's **line** is the first boundary price met on the way back: the
highest `zone_top` for a north cluster, the lowest `zone_bottom` for a south
one. It stays live while at least one of its zones has not retired that side.

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

### Precursor formation — `рождение кандидата`, `precursor`

**Machine provenance.** A latent FVG-derived candidate born at C3's native
close under the pinned machine's gap, displacement and boundary rules.
`precursor_formed_ts_ns` addresses formation; the source's C1 origin coordinate
is not the time at which a completed three-candle pattern was known.
An arbitrary visual gap does not substitute for this machine predicate.

### Span — `спан`, `прошив телом`

**Machine provenance / field event.** The NATIVE candle body traverses the
full zone: `min(open, close) < zone_bottom` and
`max(open, close) > zone_top`. Equality, a wick touch or partial body entry is
not a machine span. A minute observation of a developing native body is not
the minute body's own traversal. This OHLC body predicate does not establish
executed volume at every intermediate price or any resting-order inventory.
Study lens `full_traversal_v1` has inclusive
boundary comparisons; it is a different named research predicate.

### Activation — `активация`, `первый спан`

**Machine provenance.** The first accepted full-body span. It activates the
latent zone but does not yet make it a stored Blue/2X research object.

### Re-span — `повторный спан`, `повторный прошив`

**Machine provenance / field event.** A later accepted span of the same zone.
The spacing rule requires at least one complete native bar between accepted
spans; an adjacent traversal does not increment the counter.

**An `accepted_span` event is reported at native close, which may be later
than the market crossing.** Its `market_spine_pos` is the report address.
Reconstruct the minute path when the question concerns the crossing itself;
the worked example is in `reference/SCENE.md`.

### Blue / 2X — `синий риз`, `2X`

**Field qualification with two observation routes.** At native close, Blue
requires at least two accepted spans, both boundaries alive, and the machine's
eligible non-breaker tier. Before native close, the second qualifying span can
be visible on the developing native body under the same spacing/live-boundary
rules. This minute preview can admit the object at T0 while the persisted
accepted count is still one. Do not require later native confirmation to keep
that object, or call the first accepted span (activation) Blue/2X.

The clock split is:
`latent -> first accepted span (activation) -> earliest Blue observation (T0)`;
then the native bar can confirm that observation or fail to confirm it.
T0 can also first occur at native confirmation. These are two recorded routes,
not a requirement to wait for future acceptance. Neither predicts direction.

A frequent shape described in [046](base/046-t0-eto-vozvratnaya-hodka-a-ne-uhod.md)
is a round trip with a counter-directional second crossing. This is an empirical
description, not a filter: a different qualifying shape remains a RIZ.

### T0 — `T0`, `момент отсчёта`

**Field fact and primary research anchor.** The close of the one-minute candle
on which the Blue/2X condition first became observable on the minute tape:
`t0_ts_ns`, `t0_spine_pos`, `t0_close`, `t0_exit_side`, `t0_kind`,
`t0_span_count`.

T0 records the first observable Blue state. It does not require a developed
departure or a specified distance from the exit boundary. Subsequent price
movement and population distance measurements are research questions.

T0 gives a place and moment, not a trading hypothesis. Minutes before T0 may be
used as information already known at T0. A proposed decision before T0 is a
separate precursor question and may not use future knowledge that T0 will occur.

**T0 is not "a minute candle whose body spanned the zone".** The 2X box turns
blue intrabar, and the Pine reads the body of the FORMING NATIVE bar
(`is2xPreview`, line 261 of the pinned source), so the span can be half made of
the native bar's open. On NQ, the T0 minute's own body spans the zone in 68.7%
of cases at TF 5, 39.8% at TF 54 and 28.1% at TF 240. A hand-written
minute-body detector therefore measures a different market event on most
timeframes. This is the failure `AGENTS.md` names when it says to read the
event the field already stores instead of writing a candle detector for it.

What always holds on the minute tape is the close: strictly beyond
`t0_exit_side`, never beyond the other side. Take the exit boundary from the
passport, do not re-derive the span.

**T0 does not by itself establish a departure.** The measured close sits
beyond the boundary by a median 0.30 of that minute's own range — about five
ticks on NQ ([046](base/046-t0-eto-vozvratnaya-hodka-a-ne-uhod.md)) — and the
first contact with that boundary follows a median one minute later, within two
minutes for 71.7% of clusters in the original 060 operator, whose touch/visit
semantics now require review (see [audit](setups/S-08/AUDIT.md)). An entry at T0
aiming at its own boundary may have too little room to cover costs; this must
be checked at its actual fill price. Prior studies do not all measure this entry:
056 and 057 explicitly require a departure and later decision. Their negatives
cannot all be attributed to T0 proximity. The worked picture is in
[`reference/scenes/`](reference/scenes/README.md).

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

**Field fact when present.** The native close at or after T0 that confirms
Blue: `native_blue_confirmation_ts_ns` and
`native_blue_confirmation_spine_pos`. Confirmation can be absent. Its presence,
absence or eventual timing must not be back-ported into earlier decisions.

### Flicker — `фликер`, `мигающий риз`

**Field fact derived from recorded clocks.** T0 occurred on the minute tape but
the native bar later failed to confirm Blue. This is a valid research object,
not a data error. The eventual failure is not information available at T0.

### 3X / x3 ignition — `3X`, `третий спан`

**Field fact.** The same RIZ reaches a third-span observation. It remains the same
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

### End of observation — `archive_edge`, `observation_budget`

**Fact about the film, not about the object.** A film always names why it
stopped. `archive_edge`: the loaded corpus ended there. `observation_budget`: a
named cap (`max_bars`, counted in post-T0 minutes) bit first. Neither is a death
of the RIZ, and neither turns a missing event into an absent one. Up to the
named horizon "it did not happen" is a known fact; past it the outcome is
unknown and stays in the denominator of the question. `deletion` and `blue_end`
end the object; these two end the looking.

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

For a concrete distinction, NQ 2006-06-12 07:13 UTC has two RIZ: TF 41
exits north through 1576.00 and TF 44 through 1575.75. Their first contacts are
+1 and +205 minutes. A shared T0 does not make their films interchangeable.

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

**The first research segment of one RIZ scene.** The minutes from that RIZ's
T0 through the FIRST minute after T0 whose range meets its exit boundary,
inclusive. A wick counts. `first_exit_contact_v1` returns that ordinal;
`Field.film(end_position=...)` cuts the film there. The object retains its
ancestry and later life beyond this first segment.

The T0 minute is excluded from contact search by definition. Its own body
need not span the zone. A contact at +1 means one elapsed minute after T0
and TWO candles inclusive. If observation ends before contact, report that
limit and censoring; do not invent a contact or a negative outcome.

**One film per RIZ preserves object identity; it does not establish
independence.** Two RIZ retain two films even when they share a T0. Report
object counts, distinct decision moments, levels and sessions, then state
how dependence is treated. None of these counts automatically supplies the
number of independent observations. See `Episode vs row`.

**No departure is required, so first contact is not automatically a retest.**
A study using `retest` must define and observe a prior departure and report
the resulting narrower population. Short contact latency alone cannot prove
that no departure occurred, especially within a one-minute bar. Population
frequencies and the proximity interpretation belong to
[`046`](base/046-t0-eto-vozvratnaya-hodka-a-ne-uhod.md); they are not admission rules.

After Film-1 the same RIZ's life continues. There is no canonical Film-2 /
Film-3 cut. Preserve OHLC and the chosen observation limits when studying
later contacts, departures, returns or traversal; an exit-contact mask alone
does not describe the whole path or recover unobserved intraminute order.

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

A candidate action policy around a RIZ scene. It may begin as an incomplete
idea. Before testing execution, it specifies applicability, observable trigger,
action or abstention, invalidation, management and exit. It can use several
RIZ and successive decisions; no fixed duration or single entry/target form is
required. Each candidate has its own stated territory and failure conditions.
Discovery evidence, new-information evidence and permission to trade are
different properties; a file in `setups/` establishes none of them by itself.

### X-ray — `рентген`

Discovery-only use of the whole finished film to see **where continuations of
comparable scenes actually diverge**. The future is a locator for a question, never
a state of the past. A divergence found by X-ray must be rewound to a prefix fact
before it can inform recognition; without one it remains a description of the future.
See [`reference/SETUP_DISCOVERY_METHOD.md`](reference/SETUP_DISCOVERY_METHOD.md).

### Recognition — `распознавание`, `узнавание`

The earliest closed-prefix minute at which a candidate condition is honestly
observable. Every component of the decision must be available at or before it. If
information a rule needs appears after the initial cursor, recognition has moved, and
population, context, remaining future and executable opportunity are redefined
accordingly. Movement between an earlier cursor and honest recognition is not credited.

### Structural candidate — `структурный кандидат`

A prefix-recognizable scene condition with a measured conditional outcome (e.g. a
fork's probability), stated **from a structural reference price** and not yet shown to
be executable. It is not a trade: an executable-entry test (recognition → earliest
honest fill → costs → missed/gap handling) is a separate, required step.

### Executable historical candidate — `исполнимый исторический кандидат`

A structural candidate whose advantage survives honest recognition and a real
available fill (e.g. next open) on its **own historical territory**, after costs. It
carries neither temporal persistence nor cross-instrument transport nor forward
validation — those are separate levels of proof. Worked example: 086→091.

### Action policy — `торговое действие`, `action policy`

The complete deterministic trader that turns per-scene candidates into one stream of
real actions: concurrency limit, deterministic selection among simultaneous signals,
sizing, skipped overlaps, re-entry rule, target/stop ownership, cost accounting and
treatment of unknowns. Positive per-scene expectancy does not define it; a forward or
temporal test requires it frozen. See
[`base/091/FREEZE_091_TEMPORAL.md`](base/091/FREEZE_091_TEMPORAL.md).

### Idea error vs protective limit — `ошибка идеи` / `защитное ограничение`

Two different things a stop can mean, and they need not coincide.

An **idea error** is an observation that weakens the guess the trade was built
on: after it, the reason to hold the position is gone. Naming one requires
saying which observation, and it is falsifiable.

A **protective limit** caps the loss and nothing else. A catastrophic limit
chosen because a wider one costs too much money is a protective limit even when
it sits on a round level; it carries no claim about the market being wrong. It
is legitimate and often necessary, but structural meaning is not to be invented
for it. A card states which of the two a level is, and how it was chosen — by
money or by a named observation. Both may exist in one rule at different
distances.

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
