# G3 RIZ research

G3 is a trader-agent repository for finding repeatable price asymmetries around
RIZ and developing the survivors into trading setups. Its substrate already
exists: roughly twenty years of one-minute ES, NQ and YM history, with Blue/2X
RIZ materialized on every integer native timeframe from 1 to 1440 minutes.

The field is the map, not the edge. T0 tells us where and when to look; it does
not tell us whether to buy, sell, fade, continue or do nothing.

## Start

Agents enter through [`AGENTS.md`](AGENTS.md). Meaning lives in
[`RIZ.md`](RIZ.md), language in [`GLOSSARY.md`](GLOSSARY.md), and accumulated
research in [`research/README.md`](research/README.md).

```powershell
pip install -e .
g3-riz status
```

The primary workstation should report 4,320 complete cells, 1,440 per
instrument, zero missing, and all three market spines present. The live census
is authoritative; [`PROJECT_STATE.json`](PROJECT_STATE.json) is only its compact
snapshot.

## Read the field

```powershell
python -m g3riz.cli query --instrument NQ --tf 10 --limit 5
```

```python
from pathlib import Path
from g3riz.query import Field

field = Field(Path.cwd(), "NQ")
zones = field.passports(tf=10)
windows = field.minute_windows(
    zones["t0_spine_pos"].to_numpy(), before=30, after=120
)
stack = field.objects_at(minute_pos, state="blue")
```

`passports`, `events`, `minute_windows` and `objects_at` read persisted facts;
they do not replay the machine.

## Research memory

GitHub is the semantic layer. Every measured question gets one short `R###`
card containing its territory, measurement, answer, standing, local pointer,
closed road and next branches. Before new work, agents search all cards —
including negative results — and continue rather than repeat them.

One-off scripts and heavy outputs remain local under the same ID. Code enters
`src/` only when it becomes a reused project capability.

## Local bytes

Large data and working artifacts are not committed:

```text
data/raw/                 source attachment
data/market/{ES,NQ,YM}/   canonical one-minute spines
data/field/<instrument>/  ready RIZ field
data/research/R###/       heavy study output
work/R###/                one-off study code and logs
```

A fresh clone can understand the project and inspect stable code, study cards,
source hashes and the pinned Pine reference. Full research requires attaching
the existing `data/market` and `data/field` stores, then running `g3-riz status`.
Missing local data is never interpreted as an empty market or a reason to
silently rebuild.

[`SOURCE_DATA.json`](SOURCE_DATA.json) records source hashes, transforms, row
counts and unresolved provenance limits.

## Map

| Path | Purpose |
|---|---|
| `AGENTS.md` | Cold-agent contract |
| `RIZ.md` | Object, mission and destination |
| `GLOSSARY.md` | Canonical lifecycle and trader language |
| `research/` | Prior work and live frontier |
| `src/g3riz/` | Stable field access and frozen semantics |
| `reference/` | Operator-supplied Pine source, pinned by checksum |
| `tests/` | Existing load-bearing integrity checks |

No setup or execution layer is claimed yet. It appears only after research
produces something worth promoting.
