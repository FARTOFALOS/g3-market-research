# Read-only semantic reference

`reference/` contains a pinned copy of the Pine source supplied directly by the
operator for this project, a worked scene and contrasting calibration cases.
Together they let a cold agent see the exact RIZ lifecycle semantics, and the
market those semantics describe, without relying on another repository or
conversation.

The Pine file is not a runtime dependency and is never an ordinary edit target.

## `SCENE.md`

Begin with the market history and the calibration exercise. The examples
distinguish minute Blue observation, native confirmation and a precursor that
has only its first accepted span. Detailed source candles, the trader's original
example and earlier semantic corrections are optional sections. Exact definitions
live in `GLOSSARY.md`; the scene teaches their application.

The numerical examples were checked against the frozen field. This does not
claim that an independent new agent has already passed the transfer exercise.

## Reading a scene from the field

With local data attached, run `python -B -m g3riz.cli status` from the repo root.

```python
from pathlib import Path
from g3riz.query import Field
from g3riz.lenses.interaction import first_exit_contact_v1

field = Field(Path.cwd(), "NQ")
zones = field.passports(tf=54)
riz_id = zones["riz_id"][0].as_py()
film = field.film(riz_id, tf=54, stop="archive_edge", max_bars=600, pre_roll=2)
contact = first_exit_contact_v1(film)  # ordinal after T0, or None
```

The 600-bar window is an observation budget, not Film-1. If contact is absent,
report that limit. Reconstruct longer ancestry from `events` and minute OHLC
when the question needs it; an automatic two-bar pre-roll cannot provide it.
For recognition at a particular minute, expose only the prefix and known events.
Do not use future fields from a full passport as earlier information.

## `pine/RIZ_BLUE_v1.0.pine`

| Property | Value |
|---|---|
| Operator-supplied source | `RIZ BLUE v1.0` |
| SHA-256 | `ebad537958902fc3524c95f7b0cd182764ba63335b93e2e15425152bd1e3ef88` |
| Size | 27,945 bytes; 425 LF-terminated lines |

Verify from the repository root:

```powershell
(Get-FileHash -Algorithm SHA256 reference/pine/RIZ_BLUE_v1.0.pine).Hash
```

The `f_machine()` body is the semantic source relevant to the field: zone
birth, first accepted span, spacing rule, subsequent spans, live boundaries,
Blue/2X qualification, breaker transition and deletion. Box drawing, labels,
multi-timeframe request fan-out and anti-flicker rendering are presentation
logic and do not define stored research facts.

The Python field implementation is an independent executable port. The pinned
Pine bytes remain the human-readable reference for semantic review; the ready
field and its manifests remain the source of persisted facts.

Changing these bytes would imply a new semantic generation and is outside
ordinary research work.
