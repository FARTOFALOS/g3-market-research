# Read-only semantic references

Everything under `reference/` is a **pinned copy** of an external authority,
kept here so an agent can read canonical semantics without leaving the
repository. These files are not runtime dependencies and not edit targets. G3's
code never imports or executes them.

G3 is not the authority for any of them. Each copy records the repository, the
path inside it, the commit that pinned it, and the SHA-256 of the exact bytes.
Verify before trusting a copy; if it has drifted, re-copy from the origin
rather than editing here. If the upstream authority genuinely changes, replace
the copy deliberately and update the provenance row in the same change.

## `pine/RIZ_BLUE_v1.0.pine`

| | |
|---|---|
| Origin repository | `C:\Users\Admin\Claude\g2-market-research` |
| Path in origin | `reference/RIZ BLUE v1.0.pine` |
| Pinning commit | `7dee3619948dbdf6f57d68e032e8b7e31d7c253a` (2026-07-16, *D-001 bootstrap G-2 canon; D-002 source artifacts registered — RIZ BLUE v1.0 tracked*) |
| SHA-256 | `ebad537958902fc3524c95f7b0cd182764ba63335b93e2e15425152bd1e3ef88` |
| Size | 27 945 bytes, 425 LF-terminated lines |

G2 is the authority. This copy is byte-identical to the G2 file at that commit,
which is also G2's current `HEAD` state for that path. `.gitattributes` marks
`reference/pine/*.pine` as `-text` so no checkout ever rewrites the line
endings and silently breaks the checksum.

Verify with:

```bash
sha256sum reference/pine/RIZ_BLUE_v1.0.pine
```

### Why G3 keeps it

`f_machine()` in this script is the canonical **v9.4** state machine verbatim,
and G3's frozen semantics are defined against exactly that machine: latent
BISI/SIBI birth at the native close of C3, activation on the first full body
span of the zone, re-span accepted only at `cr+2` or later, and Blue 2X at
`nx >= 2` with both boundaries still live.

The rest of the script — box drawing, duration labels, the multi-timeframe
`request.security` fan-out, the anti-flicker reconciliation — is presentation
only and has no counterpart in the field. Do not read semantics into it.

Note that v9.4 is the canonical machine even though the file names v9.7 as its
parent: v9.5 and v9.6 were rolled back by the operator on 2026-07-02, and v9.7
is a drawing-only extension that still carries the v9.4 machine.

Authorities that are *not* copied into this repository — RIZ history and
architecture, the G2 L2 spec, the V2 1H positive control — are listed with
their paths in [`../TOOLING.md`](../TOOLING.md).
