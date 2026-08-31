# G3 RIZ FIELD — IMPLEMENTATION BRIEF

**Status:** operator-approved implementation handoff  
**Repository to create:** `FARTOFALOS/g3-market-research` (private, matching the current G2 repository visibility)  
**Local working folder:** `C:\Users\Admin\Claude\g3-market-research`  
**Terminal owner:** one strong coding agent  
**Terminal result:** a complete, fast, reproducible RIZ research field over ES/NQ/YM, not a prototype and not a new research institute.

---

## Final semantic-freeze amendment (2026-08-28)

This amendment supersedes conflicting terminology or field-shape requirements
below. The persisted population contains only underlying zones that actually
reach a valid Blue-2X T0; never-Blue latent zones are not research objects.

- `precursor_formed_ts_ns` is the factual native close of C3, when the
  canonical machine creates the latent BISI/SIBI zone. Its direct market-spine
  and native-bar addresses, geometry, and direction are retrospective precursor
  facts. There is no precursor-knowability clock.
- Pine `time[2]` / C1 open is an internal provenance label only and is absent
  from the research-facing passport.
- T0 is the first completed one-minute observation satisfying the unchanged
  canonical Blue-2X predicate. It is the birth and lower existence boundary of
  the RIZ research object. Native confirmation is a separate later fact when
  T0 occurs intrabar.
- Every accepted native span is persisted once, in order, with its resulting
  `span_count` (`nx`). This sequence is authoritative for confirmed X-state.
- 3X is a subtype of the same `riz_id`. `x3_t0` is the first completed minute,
  after confirmed `nx=2`, satisfying the canonical developing-body span and
  `cr+2` predicates for a possible third accepted span. This remains available
  to a surviving one-sided RIZ; Blue/T0's two-live-boundary gate is not added
  to native span acceptance. Its native bar may later fail;
  confirmed 3X is independently established by accepted history reaching
  `span_count >= 3`.
- The sparse journal contains only `precursor_formed`, `accepted_span`, `t0`,
  distinct `native_confirmation`, `x3_t0`, `north_boundary_retired`,
  `south_boundary_retired`, `breaker_entry`, `deleted`, and `archive_censor`.
  Candidate episodes, cancellation/re-arming, retests, excursions, swings,
  motifs, setups, films, and statistics are derived research views over the
  journal plus shared one-minute tape.
- Normal RIZ existence/as-of reads begin at T0 and end at canonical deletion.
  Pre-T0 precursor history remains retrospectively available after selecting a
  qualifying RIZ.

The semantic generation is `g3-t0-primary-x3-anchors/2`. Outputs produced under
earlier generations are incompatible and must not enter normal reads. Current
materialization standing and the live manifest census are exposed by
`PROJECT_STATE.json` and `python -m g3riz.cli status`.

---

## 1. Goal

Build a fresh, independent research substrate over the supplied ES/NQ/YM continuous 1-minute archive so that **RIZ discovery and lifecycle reconstruction are paid once**, while future hypotheses around any RIZ event can be tested by cheap, vectorized reads over already-materialized facts.

The completed field must contain:

1. one canonical, vector-friendly 1-minute market spine for each of ES, NQ and YM over that instrument's full available history;
2. a complete RIZ population for every integer native timeframe `1..1440` minutes on each instrument;
3. one compact passport per RIZ;
4. a sparse, complete lifecycle journal for each RIZ from factual C3-close precursor formation through the first valid minute-close Blue 2X activation and all later lifecycle changes until true C1 deletion, or explicit censoring if the archive ends first;
5. direct, deterministic addresses from RIZ events into the corresponding 1-minute market spine;
6. a thin, bulk/vectorized access surface that lets later agents select RIZ populations and arbitrary minute windows without replaying the RIZ machine or doing per-film disk I/O.

The field is successful when a later agent can ask a new question such as:

> Take every NQ 10-minute RIZ in the full history; inspect five observed 1-minute bars before its first valid T0 and ten after; separate cases with wick excursions beyond the near boundary before T0 from clean first-break cases; compare what follows.

and execute that experiment from the materialized field alone, without rebuilding RIZ, without pre-materialized films, and without a new multi-hour data-construction run.

**Central invariant: expensive extraction happens once; research interpretation remains cheap and disposable.**

---

## 2. What this task is — and is not

This task builds the **field**, not the institute around the field.

### Build now

- canonical ES/NQ/YM 1m stores;
- exact RIZ semantics needed to identify and follow RIZ objects;
- all native TFs `1..1440`;
- RIZ passports;
- sparse lifecycle events;
- stable identities and 1m/native-bar addresses;
- resumable full materialization;
- fast bulk/vectorized read primitives;
- focused semantic/data/performance validation;
- complete full-history output for all three instruments.

### Do not build now

- trading setups;
- retest taxonomies;
- HH/HL/LH/LL feature families;
- swing/motif/phrase discovery;
- Atlas or a dense `minute × RIZ × TF` cube;
- stored film/path corpus unless measurement proves a narrowly necessary exception;
- cross-market ES↔NQ↔YM relationships or correlation logic;
- cross-TF “clusters”, “magnets”, setup families or causal stories;
- train/test/holdout governance;
- preregistration, exposure ledgers, publication cockpit, obligations, gates, canon machinery;
- large certification/test bureaucracy;
- scientific claims or trading conclusions;
- migration of old G2 statistics;
- renewed NQ-vs-Golden equivalence adjudication;
- price adjustment or invented roll logic to force old/new data agreement.

Future research agents must be free to derive retests, swings, approaches, films, setup branches and statistics from the field. Do not freeze today's research vocabulary into the substrate.

---

## 3. Relationship to G2 / Golden

`FARTOFALOS/g2-market-research` is a **read-only benchmark and traversed road**, not the new research population and not a runtime dependency.

Use G2 to recover:

- the canonical RIZ machine and its late corrections;
- session-anchored resampling semantics;
- the full `1..1440`-TF census pattern;
- event/lifecycle distinctions learned after early census defects;
- global 1m spine addressing;
- persistence/lazy-view lessons;
- performance lessons and failure modes.

Do **not** inherit:

- G2 scientific results;
- its Golden NQ population;
- its split/holdout state;
- its governance machinery;
- its test/certification accumulation;
- its publication lifecycle.

G2 must remain unchanged. Read it; do not modify it.

The Pine/RIZ repository is also a read-only semantic/visual reference. The already-completed V2 1H positive control in the local G2 checkout was manually checked by the operator against the chart and judged adequate. Use that as practical bootstrap evidence; do not reopen the abandoned Golden-equivalence branch.

### High-value G2 reference points

Inspect the current repository, not only these pointers, and follow later superseding decisions where they exist:

- `src/riz_machine.py`
- `src/riz_resample.py`
- `src/r7_zone_census.py`
- late observer / C4 RIZ field and skeleton specifications
- local V2 positive-control runner:
  `C:\Users\Admin\Claude\g2-market-research\tools\v2_riz_1h_visual_positive_control.py`
- local V2 positive-control artifacts:
  `C:\Users\Admin\Claude\g2-market-research\derived\v2_riz_1h_visual_positive_control\`
- local execution handoff if still present:
  `C:\Users\Admin\Claude\g2-market-research\V2_RIZ_1H_VISUAL_POSITIVE_CONTROL_EXECUTION_HANDOFF.md`

Historical decision anchors:

- **D-083 / commit `e193b22288d3841d1317c070f25422016d2868da`** — same frozen census extended to every integer TF `1..1440`.
- **D-085/D-086 / commit `124708694cf410686a81a595c6ca8688e5c23115`** — early census event semantics were too coarse; event-sourced RIZ field adopted; `terminal` was shown not to mean true death; C4 separated field facts from research views.
- **D-105 / commit `b79b20bfef748df30ad1e1188d04b2852540e306`** — rebuild-per-question was rejected at the persistence layer; sparse projections and lazy relationships were preferred.
- **D-188 / commit `88df7356d16654ccffbf8ff5761412583e569e67`** — large Atlas/film materializations became a memory wall; film-by-film reduction and caching the reduction fixed the failure.

Use these as lessons, not as authority to copy implementation byte-for-byte.

---

## 4. Source data

Primary source archive already supplied by the operator:

`C:\Users\Admin\Documents\Codex\2026-08-28\c\outputs\CME_continuous_1m_free_2006_2026`

Expected source populations from the prior intake:

- ES: `6,855,058` rows; SHA-256  
  `ca2d8aa64977801f01d8f153574fa16cf8fa0dff7fcdfe75c58c5025df441c3b`
- NQ: `6,418,541` rows; SHA-256  
  `7404eb845350c84361ce2316edfffd830b3eea7761e301a8f1e6368c9178934a`
- YM: `6,438,663` rows; SHA-256  
  `d249ab94ed8ee372b7c347ac39de76e4529ab660f42e78210d8f09fcec379ddd`

The source is the public Hugging Face dataset `danielharkin21/futuresss`. Its feed origin, licence, timezone metadata and roll rule are not fully documented by the source. Do not invent missing vendor metadata.

The practical clock interpretation used by the validated V2 positive control is the previously recovered one: source labels behave as Chicago end-labelled minutes, shifted by `-1 minute` to the bar start and mapped to the research/exchange clock. Recover the exact implemented transform from the working V2 runner and its manifest rather than re-deriving it from memory.

Preserve source `Volume` in the canonical market field because later research may use it, but **volume must not alter RIZ semantics** unless the canonical RIZ machine itself says so.

Raw source bytes are immutable. Large raw and materialized data should be repository-local but Git-ignored/regenerable; commit code, schemas, manifests, digests and compact reports, not multi-gigabyte generated corpora unless there is a measured reason to do otherwise.

---

## 5. The two fundamental entities

The field has only two fundamental data domains.

### A. Market

A canonical 1-minute spine per instrument:

`instrument + minute_position -> timestamp + OHLCV + minimal objective temporal/session coordinates`

Requirements:

- full available history per instrument independently; do not trim all three to a common date range;
- sorted, deterministic, duplicate-safe;
- missing source minutes remain missing;
- no synthetic zero bars;
- preserve enough provenance to reproduce the canonical minute store;
- make high/low/close/open/volume/timestamp available in a vector-friendly form;
- maintain a stable integer observation position over **actually observed** 1m bars so “five bars before” means five observed market bars, not five wall-clock minutes across a gap.

The exact physical format is not prescribed. Measure Parquet/DuckDB/memory-map/partition alternatives as needed and choose the simplest design that satisfies correctness, bulk-read speed, size and resumability.

### B. RIZ

A RIZ is independent per `(instrument, native_tf)`. Do not create cross-instrument counterpart relationships.

The main RIZ population includes every underlying zone that produces a valid minute-close Blue 2X activation at least once, including cases that later cancel before native-TF confirmation.

Zones that never produce a valid Blue 2X minute-close activation are not part of the main field.

For every included RIZ, preserve its precursor history from factual C3-close formation of the underlying BC/CB zone, even though the RIZ research object does not exist until T0.

One RIZ = one passport, not one row per ignition/retest.

---

## 6. RIZ semantics that must survive

Do not infer these from the early `r7_zone_census` alone. Recover the **latest canonical semantics** from the current G2 machine, observer and superseding specifications.

At minimum the field must preserve enough information to distinguish:

- factual C3-close precursor formation;
- every accepted native span and resulting `nx`;
- first valid **1-minute-close** Blue 2X activation / T0;
- native-TF confirmation separately from minute-level T0;
- first valid **1-minute-close** x3 ignition separately from accepted `nx=3`;
- boundary-state changes / boundary retirement;
- Blue-eligibility end;
- breaker entry;
- true C1 deletion;
- censored / still alive at archive end.

### Two clocks are essential

Do not collapse:

1. the **1m observation clock**, where a completed minute can make a high-TF RIZ visibly/actionably Blue before the native candle closes; and
2. the **native-TF close clock**, where the native bar confirms or cancels state.

A 4H or 1D RIZ can therefore produce a valid minute-level activation long before the 4H/1D bar closes. Preserve both timestamps/addresses and their relationship.

Each persisted event uses the timestamp/address of the completed minute or
native close at which that factual event occurs. No generic duplicate
knowledge-time column is part of this generation; the two real clocks remain
explicit through T0/x3 anchors and their separate native-close events.

### T0 clarification

For this project, `T0` is the first valid minute-close event that makes the 2X RIZ active/Blue under the canonical RIZ rule. It is **not** the birth of the precursor BC/CB zone and it is **not** synonymous with later native-bar confirmation.

The few minutes of price action immediately before or after T0 are **market microstructure**, not lifecycle ontology. Do not encode “clean ignition”, “slow approach”, “wick attempts”, etc. into the RIZ passport. Later agents obtain those patterns by slicing the common 1m tape around T0.

---

## 7. Passport and lifecycle journal

### Passport: one compact row per RIZ

The exact schema is an implementation decision, but it must provide stable, deterministic identity and the invariant coordinates needed to query the object.

Conceptually include:

- source/corpus identity;
- instrument;
- native timeframe;
- stable `riz_id`;
- factual C3-close precursor-formation/native-bar address;
- fixed zone geometry (`top`, `bottom`, width);
- direction / side semantics from the canonical machine;
- first valid T0 timestamp and 1m spine position;
- T0 kind/context needed to distinguish minute activation from native confirmation;
- native confirmation timestamp/address if it occurs;
- true deletion timestamp/address if observed;
- censoring/live-at-end status.

Do not make a transient Python object identity or file order part of `riz_id`.

### Lifecycle journal: sparse events, not a film

Keep the complete, ordered sequence of meaningful changes to the RIZ/underlying zone from precursor formation through deletion.

Each event should be addressable back to:

- the RIZ;
- the instrument;
- the native TF;
- the native bar when relevant;
- the exact 1m observation position/timestamp when observable at minute resolution;
- the factual event timestamp/address on its minute or native clock;
- the typed event/state payload needed to reproduce the canonical lifecycle.

Do not store every minute merely because the RIZ was alive. The shared 1m market spine already holds the minute tape.

---

## 8. Why films are not stored

A future “film” is a cheap view:

`market[instrument][start_position:end_position]`

where `start_position` and `end_position` are chosen relative to a RIZ event or any other address.

Examples:

- `T0 - 5 observed 1m bars -> T0 + 10`;
- precursor formation -> T0;
- T0 -> true deletion;
- native confirmation -> 300 observed bars later;
- first research-defined retest -> second research-defined retest.

The field must make these slices fast. It must **not** pre-materialize one copy of the 1m tape for every RIZ.

The old Golden performance came from exactly this shape: precomputed zones on disk plus integer offsets into a preloaded 1m array. Stored minute paths were not required for the later fast wide sweeps.

---

## 9. Physical design: intentional freedom

Do not blindly reproduce Golden's `1,440 separate parquet files` merely because it worked there.

Preserve the **properties** that made it work:

- columnar bulk reads;
- cheap filtering by instrument/TF/time/lifecycle state;
- vectorized arrays rather than Python-dict-per-film loops;
- no per-film disk I/O;
- direct integer addressing into the 1m tape;
- deterministic regeneration;
- chunked/resumable computation;
- already-completed partitions are not recomputed after interruption.

Two design choices are deliberately left to measurement:

1. exact storage engine/partition layout for market/passport/events;
2. whether complete native `2..1440m` bar tables should be persisted or whether exact native-bar coordinates plus deterministic lazy reconstruction are faster/smaller overall.

Benchmark both where necessary. Choose the lowest-complexity design that keeps future mass queries fast.

Do not build a database server, service mesh, DSL or institutional API unless measurement shows the simple local columnar design cannot satisfy the workload.

---

## 10. Minimal access surface

Leave a thin, documented programmatic interface for later agents. This is not a research engine.

It must make operations like these straightforward and vectorized:

- load one instrument's 1m spine/columns;
- select RIZ passports by instrument, one TF, TF range, time range or lifecycle state;
- load lifecycle events for a selected RIZ population;
- convert event addresses into arbitrary 1m windows in bulk;
- select which RIZ objects were alive/Blue at a given observation time without replaying the RIZ machine;
- recover source/native/1m coordinates from a RIZ/event.

Return arrays/dataframes/columnar tables, not tens of thousands of rich Python dictionaries unless a narrow caller explicitly asks for objects.

Do not encode retest, swing, setup or statistical logic in this interface.

---

## 11. Full execution scope

This agent owns the entire terminal build.

Do not stop at a pilot.

Required coverage:

- ES: all available history × TF `1..1440`
- NQ: all available history × TF `1..1440`
- YM: all available history × TF `1..1440`

The computation must be chunked/resumable so a process/session/model interruption cannot force a full restart.

A practical shape is independent work units by `(instrument, TF or TF-chunk)` with atomic completion markers/manifests, but choose the implementation that best fits the measured cost.

The final materialized field must prove that all `3 × 1440 = 4320` instrument/TF cells are complete or explicitly empty because no qualifying RIZ occurred—not silently missing.

---

## 12. Validation philosophy

Use **small, load-bearing validation**, not G2-style test accumulation.

Validation should protect only the facts that would corrupt the field if wrong.

At minimum prove:

### Source / market

- source hashes/row counts reconcile;
- timestamp normalization is deterministic;
- OHLC invariants hold;
- missing bars are not synthesized;
- market spine positions are stable;
- rerun produces the same logical content.

### RIZ semantics

- the validated 1H V2 positive-control path is preserved;
- the new implementation conforms to the canonical late RIZ machine/observer on representative bounded slices and edge cases;
- minute T0 and native confirmation are not collapsed;
- precursor history is retained for RIZ that eventually qualify;
- true deletion is not confused with Blue-eligibility end or breaker entry;
- censored live-at-end objects remain representable;
- stable RIZ identities and event ordering regenerate deterministically.

### Referential integrity

- every stored RIZ/event references a valid instrument/TF;
- every event that claims a 1m address resolves to the correct market row;
- native-bar references resolve or carry an explicit typed reason;
- no event exists after true deletion;
- one RIZ passport owns its event sequence.

### Resumability

- interrupt a chunked build;
- resume it;
- prove completed work was reused and the final logical output matches a clean build on a bounded control.

Do not create hundreds/thousands of tests just to certify the framework. Prefer a small number of high-information conformance/property/integration tests.

---

## 13. Performance acceptance

Performance is part of correctness because the field exists to support mass hypothesis search.

Historical operator-supplied Golden measurements, for orientation only and **not hard SLAs**:

- load a 3.67M-row 1m file: ~`0.23 s`;
- compute ATR/derived columns: ~`0.47 s`;
- read 60 zone files: ~`0.59 s`;
- vectorized 46k films × 240 bars (~11M bars touched): ~`0.20 s`;
- read all 1,440 zone files: ~`15.9 s`;
- full 1,440-TF / ~163k-film vectorized pass: ~`18.3 s` end-to-end.

The new field has more history, three instruments and a richer lifecycle journal, so do not cargo-cult these exact numbers. Use them to remember the architectural lesson:

> **1,440 timeframes are not, by themselves, a reason for future hypothesis runs to be slow.**

### Required workload probe 1 — T0 microstructure

Using **only the completed field**, no RIZ replay:

1. select all NQ 10m RIZ;
2. obtain each first valid minute T0;
3. bulk-slice five observed 1m bars before and ten after;
4. compute a simple temporary research split:
   - at least one pre-T0 wick excursion past the near boundary without valid close;
   - no such excursion;
5. produce counts and a simple next-10-bar comparison;
6. report elapsed time and peak memory.

The scientific result is irrelevant. The test proves the field supports the intended research style.

### Required workload probe 2 — wide corpus

On one instrument, select the full `1..1440` RIZ corpus and run a simple vectorized fixed-horizon future high/low/close reduction around T0 using the shared 1m tape.

No per-film I/O. No RIZ replay. No prebuilt film corpus.

Measure load time, compute time, peak memory and population size. If a simple pass is unexpectedly slow, profile and fix the data organization before declaring completion.

---

## 14. What future research must remain possible

Do not implement these now; ensure the field does not make them expensive or impossible.

A future agent may treat any meaningful state in a RIZ life as an observation point:

`pre-T0 approach -> T0 -> early development -> first retest -> reaction -> second retest -> reclaim/hold/traverse -> later swings -> failure/continuation`

The research method is to compare films that arrive at similar visible states but later diverge, expand backward and forward, and find the earliest live-recognizable history that separates recurring future branches. One RIZ may contain several independent tradable moments. This method is why the field must support arbitrary point-in-time slicing rather than only a fixed post-T0 feature table.

Do not bake those future observation points or setup names into the field. They are research views over the market + RIZ facts.

---

## 15. Pseudo-replication lesson — preserve the ability to correct it, do not solve it now

Golden later showed that 1,440 overlapping TFs can multiply observations without multiplying independent market information: many RIZ instances can share the same market minute/session.

The field must therefore preserve exact instrument, timestamp, observation position, session/native coordinates and stable RIZ identity so later research can deduplicate or cluster dependence correctly.

Do **not** build statistical deduplication, z-scores, bootstrap or research policy into this first task.

---

## 16. Repository bootstrap and ownership

Create a new clean local repository:

`C:\Users\Admin\Claude\g3-market-research`

Create the matching **private** GitHub repository:

`FARTOFALOS/g3-market-research`

Default branch: `main`.

Put this brief in the repository root as:

`G3_RIZ_FIELD_IMPLEMENTATION_BRIEF.md`

The new repo owns:

- implementation code;
- schemas;
- compact manifests/digests;
- focused tests;
- benchmark scripts/results;
- documentation;
- local Git-ignored raw/canonical/materialized field directories.

G2 and RIZ repos are read-only inputs/benchmarks. Do not commit changes to them.

If the remote repository already exists when execution begins, inspect it before writing and reuse it only if it is clearly this intended empty/new project.

Suggested one-time operator/bootstrap commands if repository creation has not already been performed:

```powershell
cd C:\Users\Admin\Claude
mkdir g3-market-research
cd g3-market-research
git init -b main
gh repo create FARTOFALOS/g3-market-research --private --source . --remote origin
```

The coding agent may choose a safer equivalent sequence if needed.

Large source and derived field bytes should not be pushed to ordinary Git history. Keep them reproducible and local under the repository, with manifests/checksums tracked.

---

## 17. Implementation freedom and anti-drift rule

The semantic outcome is binding; the physical implementation is replaceable.

The agent may:

- redesign the old census runner;
- use streaming/chunking/parallelism;
- choose Parquet/DuckDB/Arrow/memory-map or another measured local layout;
- consolidate per-TF files;
- persist selected native-bar materializations if measurement justifies it;
- add small indexing structures;
- replace the current positive-control wrapper with a cleaner production implementation;
- port only the minimal RIZ code necessary into G3 with provenance to the source commit.

The agent must **not** change RIZ meaning merely to simplify implementation.

When an old G2 mechanism conflicts with a later correction, prefer the later canonical semantics.

When a choice only affects storage/performance and not semantics, measure it and choose autonomously.

When a choice would alter what counts as a RIZ, T0, confirmation, lifecycle event or true deletion, stop and recover the canonical late semantics before proceeding; do not invent.

---

## 18. Terminal acceptance checklist

The task is complete only when all are true:

- new local `g3-market-research` repo exists;
- private GitHub remote exists and code/manifests/tests/brief are pushed to `main`;
- G2 and RIZ repos remain untouched;
- canonical ES/NQ/YM 1m market spines exist over each full source history;
- source identity/provenance is reproducible;
- every integer TF `1..1440` is processed independently for each instrument;
- the full qualifying RIZ population is materialized;
- one deterministic passport exists per RIZ;
- precursor history is preserved for every RIZ that eventually qualifies;
- sparse lifecycle journal covers the canonical lifecycle through true deletion/censoring;
- minute T0 and native confirmation are separately represented;
- all necessary RIZ events have direct 1m/native addresses;
- the materialization can resume after interruption without recomputing completed work;
- field-only T0 microstructure workload runs successfully with measured cost;
- field-only all-TF wide workload runs successfully with measured cost;
- neither workload replays RIZ or performs per-film disk I/O;
- no dense film/Atlas corpus was built without a measured necessity;
- no scientific setup/statistical/cross-market research was smuggled into the field build;
- validation is focused and load-bearing rather than an expanding institutional certification suite;
- a successor agent can open the repo, read the brief/README, load a market/RIZ population and run a new vectorized hypothesis without reconstructing project history.

---

## 19. Final instruction to the implementation agent

Do not return a proposal.

Recover the latest canonical RIZ semantics and the already-validated V2 clock/positive-control path from the read-only G2 checkout; create the clean G3 repository; design the smallest fast local substrate that preserves the required facts; implement it; complete the full ES/NQ/YM × `1..1440` materialization; validate it; benchmark real research-style reads; and leave the result usable by the next research agent.

Use Golden/G2 as a benchmark of what worked **and of what later failed**. Take the late lessons without importing the institution that accumulated around them.

The product is the **field**.

Not the bureaucracy around it.
