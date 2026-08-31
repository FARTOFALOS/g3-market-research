# Research desk

This is the live map of what G3 knows, what failed in scope, and what remains
worth asking. Study cards are the durable record; this page is only the compact
frontier.

## Now

- Field: ready — ES/NQ/YM, native TF 1..1440, one-minute tape.
- Population: stored Blue/2X RIZ from T0.
- Measured effects: none in this clean research line.
- Approved setups: none.
- Frontier: choose the first bounded question with the operator.

## Catalog

| ID | Question | Standing | Territory | Answer | Card |
|---|---|---|---|---|---|
| — | No studies yet | — | — | — | — |

## Open queue

Ideas are not claims. Keep each to one line:

`Q### — question; why it may matter; related R### or none.`

No old hypotheses are inherited into this clean line.

## Research loop

1. Map the trader's phrase through [`../GLOSSARY.md`](../GLOSSARY.md).
2. Search previous cards by canonical terms and aliases:

   ```powershell
   rg -n -i "retest|повторный тест|north|север" research
   ```

3. If the same question and territory already exist, continue or replicate the
   named card; do not disguise a repeat with new wording.
4. Copy [`studies/_TEMPLATE.md`](studies/_TEMPLATE.md) to
   `studies/R###-short-slug.md`. Use the same ID for local `work/R###/` and
   `data/research/R###/` when needed.
5. Run the smallest measurement that can change the decision.
6. Record the numerical answer, what it does and does not establish, the road
   now closed, and one to three useful next questions.
7. Update this page.

One measured question has one card. The card receives an ID before substantial
work and a disposition before the desk moves on.

## Standings

- `ACTIVE` — being measured; no conclusion.
- `EFFECT` — conditional asymmetry measured on the stated territory; not a trade.
- `NO_SUPPORT` — not supported by this measurement on this territory.
- `INCONCLUSIVE` — data or method could not decide; reason named.
- `SETUP_CANDIDATE` — full trade logic defined; deeper validation justified.
- `APPROVED_SETUP` — explicitly approved by the operator after validation.
- `SUPERSEDED` — replaced by a linked card, retained for search.

## What deserves rigor

Preserve point-in-time inputs, the true denominator, measured territory,
inspected variants, effect size, uncertainty, and an appropriate comparison
when the claim needs one. A broken instrument is not a negative market result.
Rigor grows with the claim; it is not a global ceremony imposed on every idea.

New questions should come from operator observations, field anomalies, the
boundary where an effect breaks, or the next branch named by a prior card — not
from a giant speculative matrix.

## Storage rule

GitHub keeps semantic memory: question, territory, answer, standing, local
pointer and next branches. One-off code, logs and heavy output stay ignored in
`work/` and `data/research/`. Promote code into `src/` only after it becomes a
reused project capability. Git history is the archive.

No constitutions, obligation ledgers, decision logs, exposure registries,
compiled frontiers or tests for market claims.
