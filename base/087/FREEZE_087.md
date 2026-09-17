# WOECB′ — frozen candidate operator (predecessor + event). NOT yet a proven process split.

Discovery corpus: NQ development. Read off real films via candle-to-candle relational geometry.
Outward coordinates: for side s∈{+1 north,−1 south}, per bar
`W_out=max(sH,sL)`, `W_in=min(sH,sL)`, `B_out=max(sO,sC)`, `B_in=min(sO,sC)`.
Running fronts: `M`=max W_out so far (wick front), `MB`=max B_out so far (body front).
An "outward update" bar: it advances `M` and/or `MB`. Type: `joint` (both), `M-only`, `MB-only`.

## Definition (three structural facets, T0-independent, no magnitude threshold)
WOECB′(t) holds iff:
1. **predecessor** — the last outward-update bar before t was **joint** (`M` and `MB` advanced together);
2. **event** — bar t is **M-only**: `W_out[t] > M_prev` and `B_out[t] ≤ MB_prev` (wick front advances, body front does not);
3. **close-back** — bar t's body points inward: `s·C[t] < s·O[t]` (`Co < Oo`).

Reading: wick-front and body-front were advancing together; then, for the first time, the
wick-front advances alone AND the bar closes back inward. A change in the relation of the two
fronts, not "the Nth minute after T0".

## Semantic evidence (why each facet)
- Raw "M-only + body-front flat" fires on trivial @+1 micro-pokes (earliest-bar median = 1). The
  `joint→` predecessor removes them WITHOUT referencing T0 (they have no prior joint update);
  genuine post-leg cases have their last update = joint in ~66–69% at k≥2.
- Facet 3: among fresh joint→M-only decouplings, inward-body coincides with close-back —
  within inward-body only 0.6% are "held" (clo_pos<0.33), 84.3% are "back" (clo_pos>0.66).
  So `Co<Oo` captures close-back threshold-free; clo_pos adds nothing inside inward-body.
- Counterexamples kept: A = joint→M-only close-**held** (outward body); C = joint→joint (body
  front keeps advancing); D = M-only→M-only (non-fresh decoupling). Doji (`Co=Oo`) set aside.

## Units and scope
- Detection unit = **object-scene**: each `riz_id` from its T0 to its own terminal.
- Evidence unit = **physical episode** (physical act = candidate absolute bar + side). At first
  fresh decoupling, physical-act dedup removed 8061/22141 = 36% copies (NQ dev) → ~14,080 acts.
- RIZ TF/width kept as provenance/context; no cuts by them yet.

## Continuation tests 3→2→1 (NQ development, deduped to physical acts)
Status: **observable process transition established; executable edge not established; censoring structural.**

- **(3) persistence** — from a common joint-parent, branch = next outward update. WOECB′ vs joint
  grammars do NOT reconverge in 1–2 events (post-branch event-TV 0.29→0.19 by offset; minute-slice
  TV s3=0.54, s10=0.28). WOECB′ → resolution toward b; joint → sustained joint advance. Persists
  ~10 bars. The difference is a resolve-vs-continue propensity, not a distinct live vocabulary.
  Survives the falsifier (no fast reconvergence).
- **(2) remaining movement** strictly after close(WOECB′) (2903 acts): available return to b median
  1.95σ / 4.75 pt, reached 0.656; adverse outward 1.81σ ≈ target → no clean stop room (062/S-15 wall);
  continuation side 1.26σ. Executable edge not established.
- **(1) censoring is structural, not external:** censored 0.344 (gap 0.302, cap 0.042, cross X≈0).
  Censored films survived 186 bars and extended 5.7σ vs reached-B 13 bars / 0.4σ — censoring removes
  the long far-extended developments, via session-boundary tape gaps (UTC 20–21/00 ~0.6–0.7 vs NY
  11–13 ~0.20). Year-uniform (0.27–0.43). So WOECB′ is likely bimodal (quick b-return vs long extend);
  P(B eventually) unknown for the ~34% extenders; P(B|observed)=0.999 is censoring-biased.

Not carried to NQ evaluation / ES / YM — this is a discovery-corpus result.
