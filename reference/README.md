# Read-only semantic reference

`reference/` contains a pinned copy of the Pine source supplied directly by the
operator for this project, and one worked scene. Together they let a cold agent
see the exact RIZ lifecycle semantics, and the market those semantics describe,
without relying on another repository or conversation.

The Pine file is not a runtime dependency and is never an ordinary edit target.

## `SCENE.md`

One RIZ, one T0, one exit boundary, one minute film, read candle by candle: the
trader's own calibration screenshot explained, plus the same object located in
the corpus with its real OHLC. It is not a `base/` card — it establishes what
object we are counting, not what the market does. Read it before building a
first film.

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
