# Prefix-only support probe, 2026-09-14

This diagnostic asks whether a specified observable history varies at EXACTLY
equal specified geometry in the ready NQ TF54 field. It does not estimate any
future outcome, conditional information, profitability, or population impossibility.
Definitions are recorded before executing this probe. Earlier G3 teaching scenes,
research summaries and the three March prefixes have already informed this choice.

The observational unit is a physical scene at (instrument, T0, p). Each TF54
zone is an addressed projection of that scene. All zone IDs are retained;
multiple projections or overlapping price histories are not independent evidence.

Population: all stored NQ TF54 T0, one cursor p=T0+2 observed bars per anchor.
The +2 cursor is the existing pedagogical choice, not a discovered information
arrival time. No lifecycle survival or future completeness filter is used.
Known insufficient or gapped prefixes remain explicitly unresolved.

Observed interval: the first observed minute of the session-aligned native bar
containing T0, through p. This is a bounded history projection, not all scene
ancestry. The native bar's scheduled start is determined from the stored session
anchor and T0 close timestamp. No developing bar's future high/low/close is used.
Coordinates: x=side*(price-exit_boundary), side=+1 north, -1 south. The far
boundary is -zone_width. Prices retain exact stored precision; no rounding bins,
distance metric, matching weights or cross-instrument equivalence are introduced.

G0 = exit side, T0 kind, zone width, interval first open, interval min and max,
current-minute O/H/L/C (all prices in the above coordinates), interval duration.
The current minute is included in G0 and its known information is not claimed
as additional history. TF54 and the cursor policy are fixed for this diagnostic.
The north/south reflection is a coordinate convention, not a transfer claim;
exit side remains part of G0, so mirrored cases are not pooled by exact equality.

At each minute, H observes two inclusive range-contact flags (exit/far boundary)
and close location (far-exterior / far-boundary / inside / exit-boundary / exterior).
H_order is the sequence of these joint readings with adjacent identical readings
compressed. Simultaneous contact of both boundaries is one reading; their
intraminute order is unknown and is never inferred. H_timed retains the first
minute of each run, so it combines order and timing. Neither is asserted to be
a process identity, sufficient state, or a unique information source.

G1 additionally preserves the number of minutes contacting each boundary and
the total minutes in each of the five close locations. It asks about history
beyond geometry AND these counts/dwell totals, unlike G0's broader question.

Outputs: all prefix descriptions, exact-equality group sizes, whether a group
contains multiple H values across distinct physical scenes, coverage by anchor,
scene and session day, and four example prefixes chosen without Y. Missing
comparisons mean absent empirical exact overlap for this definition and corpus,
not no path dependence and not no overlap under any continuous estimator.

Checks: direct extraction must agree with extraction from arrays truncated at p;
artificially permuted feasible candle sequences with identical G0/G1 must be
distinguished when their observable H differs. The latter checks the reader,
not the existence of those histories in RIZ or the power of a prediction method.
The protocol and script do not define, load or evaluate Y. No tolerance sweep
will be performed after seeing the support count.
