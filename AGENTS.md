# Cold entry

Read in this order:

1. **[`RIZ.md`](RIZ.md)** — what a RIZ is and what this program is trying to
   find. Read it first; without it the rest is just columns.
2. **[`GLOSSARY.md`](GLOSSARY.md)** — every term, and whether it is a field fact
   or a derived read you must define yourself.
3. **[`FINDINGS.md`](FINDINGS.md)** — what has been observed so far, and where
   your own observations go.
4. **[`README.md`](README.md)** — machinery: install, query, current state.

Then verify the field is actually there:

```
g3-riz status
```

That census is the real state. Do not infer it from anything else. It must read
4,320 complete cells and `market_spine: present` for ES, NQ and YM.

## Standing rules

- **The field is finished and frozen.** Rebuilding it is not research. Resume a
  build only if `status` reports verified missing cells, or the operator
  authorizes a new semantic generation.
- **The zone is an anchor, not the subject.** The subject is what price does
  around it, and how zones relate to each other.
- **Field facts and your own definitions are different things.** Anything not in
  `GLOSSARY.md`'s field-fact list — retest, confluence, setup — you are defining
  yourself. Say so, and say exactly what you mean, in the same entry.
- **Nothing here is proven yet.** There is no standard of evidence in this
  repository on purpose. Record observations honestly; do not promote them.
- **Never reuse incompatible, temporary, or `.previous` outputs.**

## Semantics that are settled

Only zones that reach T0 are stored; zones that never turn blue do not exist
here. The precursor forms at the native close of the third candle. T0 is the
minute-level birth of the research object. Native confirmation is a separate,
later event. `x3_t0` is the same zone's first minute-level third span.

The canonical machine is the `f_machine()` of
[`reference/pine/RIZ_BLUE_v1.0.pine`](reference/pine/RIZ_BLUE_v1.0.pine),
pinned from G2 with its checksum. It is a read-only reference, never an edit
target and never a runtime dependency.

## Optional tooling

[`TOOLING.md`](TOOLING.md) covers the read-only TradingView bridge: chart state,
candles, Pine drawings, screenshots. The chart is an eye, not an oracle — the
field decides RIZ facts. Never write Pine without an explicit operator
instruction naming the script.

[`design/`](design/README.md) holds visual benches for working through semantics
with the operator. Neither field nor dependency.
