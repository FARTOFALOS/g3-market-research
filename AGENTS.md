# G3 cold-entry instruction

Read `README.md`, the final semantic-freeze amendment at the top of
`G3_RIZ_FIELD_IMPLEMENTATION_BRIEF.md`, `PROJECT_STATE.json`, and
`SOURCE_DATA.json`. Then run `python -m g3riz.cli status` with `PYTHONPATH=src`.
The command's cell-manifest census is the actual materialization state; do not
infer it from chat history.

The frozen population contains only zones that reach T0. Precursor formation is
C3 native close, T0 begins RIZ research-object existence, native confirmation
is separate, accepted spans are authoritative X-state, and `x3_t0` is the same
RIZ's first minute-level third-span ignition. Research interpretations remain
derived from the factual lifecycle and shared one-minute tape.

Production materialization is complete: 4,320/4,320 current-generation cells,
1,440 per instrument. Use `status` to verify the manifests and the documented
query entry points for research. Do not rebuild by default; resume only if
`status` identifies verified missing cells or the operator authorizes a new
semantic generation. Never reuse incompatible, temporary, or `.previous`
outputs. G2 and Pine/RIZ sources are read-only semantic references, not runtime
dependencies or edit targets.

Read-only chart inspection is available through one user-scope MCP server,
`tradingview`; `TOOLING.md` has the connection procedure and the standing
rules. It is an optional tool, never a runtime dependency: the chart is an eye
and not an oracle, Pine is never written without an explicit operator
instruction, and the server is not a route to rebuilding the field. Pinned Pine
copies under `reference/` carry their G2 origin and checksum; G2 stays the Pine
authority. `design/` holds visual benches for working through semantics with the
operator — start from `design/README.md`; they are neither field nor dependency.
