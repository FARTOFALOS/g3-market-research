# G3 RIZ research field

One-time research field over the fixed historical ES/NQ/YM one-minute corpus.
The field is materialized and its semantics are frozen: normal work reads the
field, it does not rebuild it.

Cold-entry instructions, frozen semantics and standing rules:

@AGENTS.md

## Environment

Install the package editable once. After that `g3-riz` and `pytest` both work
from the repository root with no `PYTHONPATH` setting:

```
pip install -e .
```

## First command

```
g3-riz status
```

Compact per-instrument census by default; `--full` adds the per-timeframe
lists. This census is the actual materialization state — do not infer it from
chat history. `market_spine` must read `present`, otherwise cell identities
could not be checked and the cells are reported as `unverifiable`.

## Chart and Pine tooling

One user-scope MCP server is available and is **not** a runtime dependency:
`tradingview` — read-only eyes inside the running TradingView Desktop (chart
state, OHLCV, Pine boxes/lines/labels, screenshots, symbol/TF navigation). It
is deliberately the only bridge.

Read [`TOOLING.md`](TOOLING.md) before using it. It carries the connection
procedure — TradingView must be launched with `--remote-debugging-port=9222`,
which on this Windows MSIX install needs `scripts\launch_tv_debug.bat` from the
`tradingview-mcp` checkout — the session-restart CLI fallback, and the standing
rules: the chart is an eye and not an oracle, never write Pine without an
explicit instruction, and never use the server to rebuild the field.

The Blue 2X+ script running on the live chart is frozen in this repository at
[`reference/pine/RIZ_BLUE_v1.0.pine`](reference/pine/RIZ_BLUE_v1.0.pine); its
`f_machine()` is the canonical v9.4 machine verbatim. It is a copy pinned from
G2, which stays the Pine authority — never treat the G3 copy as the source of
truth. Everything under `reference/` is read-only, never an edit target, and
carries its origin and checksum in `reference/README.md`.

## Наглядные стенды

[`design/`](design/README.md) — интерактивные страницы для разбора семантики
вместе с оператором, не часть поля и не зависимость. Актуальный:
[`design/riz-sandbox/`](design/riz-sandbox/README.md) — песочница RIZ, где
свечи задаются руками, а машина v9.4 считает по ним зоны. В её README записано,
как пересобрать и проверить стенд, и что по семантике уже установлено —
перепроверять это заново не нужно.
