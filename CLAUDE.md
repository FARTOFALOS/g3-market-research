# G3 RIZ research field

Research program over a fixed historical ES/NQ/YM one-minute corpus. The field
is built and frozen; the work is finding trading setups in it, not rebuilding
it.

Start here — meaning first, machinery second:

@AGENTS.md

## Environment

```
pip install -e .
```

Then `g3-riz` and `pytest` work from the repository root with no `PYTHONPATH`.

## First command

```
g3-riz status
```

The census is the actual materialization state; do not infer it from chat
history. `market_spine` must read `present`.
