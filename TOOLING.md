# Agent tooling for G3

One MCP server is registered at Claude Code **user scope** (`~/.claude.json`),
so it is available in this repository without any project-local `.mcp.json`. It
is not a runtime dependency of the `g3riz` package: `pytest`, `g3-riz`, and
every research query work with the server absent.

`tradingview` is deliberately the only bridge. It is read-only chart
inspection: screenshots, candles, Pine boxes/lines/labels, symbol and timeframe
navigation.

| Server | Command | Source | What it is for |
|---|---|---|---|
| `tradingview` | `node C:/Users/Admin/Claude/tradingview-mcp/src/server.js` | [tradesdontlie/tradingview-mcp](https://github.com/tradesdontlie/tradingview-mcp) | Eyes inside the running TradingView Desktop — read chart state, OHLCV, Pine-drawn boxes/lines/labels, take screenshots |

An offline Pine v6 documentation server (PineMCP) was evaluated and
**deliberately not installed** — G3 reads the chart, it does not author Pine.

## `tradingview` — connecting

The server talks to TradingView Desktop over the Chrome DevTools Protocol on
`127.0.0.1:9222`. TradingView must have been started with the debug port open;
a normally-launched instance will not answer.

TradingView for Windows ships as an MSIX package under
`C:\Program Files\WindowsApps\`, so it cannot be started from a plain path. Use
the bundled script, which resolves the install via `Get-AppxPackage`:

```bash
cd C:/Users/Admin/Claude/tradingview-mcp && cmd //c "scripts\launch_tv_debug.bat"
```

It force-closes any running TradingView first, then waits for the port. On
builds where launching from `WindowsApps` is denied, the `tv_launch` MCP tool
falls back to a one-time ~330 MB local copy of the package. Never try to fix
this with `icacls` on `WindowsApps` — it fails and can break app servicing.

Check the connection with `tv status`; `cdp_connected: true` plus a populated
`chart_symbol` means the bridge is live.

Every MCP tool is also reachable as a CLI from the same repository, over the
same CDP channel and returning the same data:

```bash
node C:/Users/Admin/Claude/tradingview-mcp/src/cli/index.js state
```

That CLI is the fallback whenever the MCP tools are not loaded in the current
session — for example right after registering the server, before a restart.

## Standing rules

- **The chart is an eye, not an oracle.** G2 decision D-040 retired TradingView
  as a source of truth; the materialized field is authoritative for RIZ facts.
  Use the chart to see and explain, not to adjudicate a lifecycle question the
  field already answers. Where the two genuinely disagree, that is a finding to
  report, not a licence to correct the field.
- **Do not write Pine.** The `tradingview` server exposes `pine_set_source`,
  `pine_save`, and compile tools. Reading (`pine_get`), boxes, lines, labels,
  and screenshots are fine at any time. Do not create, modify, overwrite, save,
  or replace any Pine script without an explicit operator instruction naming
  that script.
- **Do not use this server to rebuild the field.** Materialization is
  complete and frozen; see `AGENTS.md`.

## Read-only semantic authorities

Наглядный стенд по этой семантике: [`design/riz-sandbox/`](design/riz-sandbox/README.md).

Frozen in this repository under [`reference/`](reference/README.md):

- `reference/pine/RIZ_BLUE_v1.0.pine` — the Blue 2X+ script running on the live
  chart; its `f_machine()` is the canonical v9.4 machine verbatim. Pinned from
  G2 `reference/RIZ BLUE v1.0.pine` at commit `7dee3619`; G2 remains the
  authority. Provenance and checksum: `reference/README.md`.

Outside this repository (never edit):

- `C:\Users\Admin\Claude\Projects\riz indicator\RIZ_история_и_архитектура.md` —
  RIZ indicator history and architecture. Canonical machine semantics are
  **v9.4**; v9.5/v9.6 were rolled back by the operator on 2026-07-02, and v9.7
  is a drawing-only extension carrying the same v9.4 machine.
- `C:\Users\Admin\Claude\g2-market-research\spec\C1 RIZ machine — L2 spec.md`
  v1.4 — ladder/observer authority.
- `C:\Users\Admin\Claude\g2-market-research\derived\v2_riz_1h_visual_positive_control\` —
  ES/NQ/YM 1H positive control, operator-visually-checked against the chart.
