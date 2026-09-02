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

**Field fact.** A wick touches a live boundary and that side retires. A RIZ may
continue to exist with one side alive. Fields include `final_north_alive` and
`final_south_alive`; lifecycle events give the event time.

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
1,440 times per instrument, once per native timeframe. A RIZ on TF 10 and a RIZ
on TF 15 at the same place and time are two rows and one market event. Overlapping
RIZ in time inflate the count further. A study must state how many independent
episodes it has, not how many rows.

There is no repository-wide episode predicate, and none is owed. Sharing a T0
minute is evidence that two RIZ are related, not proof they are one object:
they may be one event seen at several scales, separate interactions, or an outer
zone containing an inner one. A study that needs an independent unit defines and
names its own, and says what that definition throws away. Deduplicating by T0
minute alone is a diagnostic statistic, never an episode count.

### Film — `фильм`

**Research frame, not a stored field fact.** The minute tape around one T0,
carrying its own pre-roll and a named end — `deletion`, `blue_end`,
`archive_edge`, an observation budget, or a position the study closed itself.
The end is always stated, never guessed. The canonical field does not store
films: `Field.films()` builds them on demand from recorded facts. A study may
materialise its own corpus of films locally, and then that corpus — not the word
— carries provenance: its own identity, the semantic generation it was built
from, and its selection rule.

### Context — `контекст`

Information known at the evaluated minute: earlier one-minute bars, RIZ state,
session, other already-existing RIZ and any other point-in-time feature named by
the study.

### Retest — `ретест`, `повторный тест`

Price returns to a specified boundary or zone after a specified departure.
The machine does not provide a universal retest flag. A study must define the
line, direction of approach, qualifying touch/close, and whether repeated
contacts count.

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
