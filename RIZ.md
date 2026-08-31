# What a RIZ is, and what this repository is for

Read this before anything else. It is the meaning layer. `GLOSSARY.md` is the
vocabulary, `README.md` is the machinery, `FINDINGS.md` is the running journal.

## The market idea

A market maker is obliged to quote both sides at every price. In a market that
works, every price level gets traded through properly — buyers and sellers meet
there, and the level is done with.

Sometimes that does not happen. Price leaves in one direction hard enough that
the opposite side never gets filled. What stays behind is a price range the
market passed through without actually trading it out. An imbalance. Retail
literature calls it a fair value gap; it is the same three-candle shape.

The market has an unfinished piece of business there. Not a law — a tendency,
and the working assumption of this program: price tends to come back to ranges
it skipped.

## Why most of them do not matter

There are far too many of these gaps. Every instrument, every timeframe, all
day long. Taken raw they are noise, and a research program built on all of them
learns nothing.

So the population is filtered by what the market does afterwards. Price comes
back to the range — and instead of quietly trading it out and moving on, a
candle body runs the whole range end to end a second time. The market returned
to the scene and left again decisively rather than settling the matter.

That second full traversal is **2X**, and it is the entry ticket. A third is
**3X** — the best and the rarest.

This is a filter of significance, not a prediction. A 2X zone is a place the
market has now visited twice without finishing its business. What price does
next is exactly what we do not know, and exactly what this repository exists to
find out.

## What a qualifying zone gives us

Two prices and one moment.

- **North and south** — the top and bottom of the range. These are the lines
  price will interact with: approach, touch, reject, break, return.
- **T0** — the minute the zone became significant. The birth of the research
  object. Before it there is only a candidate; from it the object exists.

Everything else is measured relative to those three things.

## What the field is

Twenty years of one-minute tape on ES, NQ and YM, with every qualifying zone
marked on every timeframe from 1 minute to 1440 — every integer, not a selected
handful.

**The field is a coordinate system, not a result.** It says where and when the
market showed its hand. It says nothing about what to do there.

Two consequences worth stating plainly:

- The same price at the same moment is covered by many zones from many
  timeframes at once — nested, overlapping, edge-sharing, opposed in direction.
  No chart can show this. The field can.
- The field is complete and frozen. Rebuilding it is not research and is not
  on the table.

## What we are actually researching

Two surfaces. Both already sit in the field; neither needs new computation.

**1. Price around a zone.** What price does before it reaches the line, at the
moment of contact, and after T0. Setups can live in any of the three, and they
are different setups.

**2. Zones against each other.** Nesting, level agreement, several timeframes
stacking in one price pocket, north of one meeting south of another, sequence
and inheritance of a level across zones. The machine links none of this — every
relation is ours to define and measure.

The output we want is a **setup**: a repeatable situation, stated precisely
enough that an agent can act on it on live data later.

## What this is not

- Not a claim that price must fill a gap. That is the motivation for choosing
  this event, not a finding.
- Not a rebuild of the RIZ machine. It is done, frozen, and correct by
  definition of this program.
- Not a chart-visual project. A zone can exist in the field and never appear on
  a closed chart; that is a real and known difference, not an error.
- Not yet a claim about anything. There is deliberately no standard of proof in
  this repository right now. Observations accumulate first.

## Where the research stands

No standard of evidence has been set. This is on purpose. The predecessor
program built an elaborate apparatus of gates and procedures and measured
almost no market with it. Here the order is reversed: look first, accumulate
honest observations in `FINDINGS.md`, and set the bar once it is clear what is
actually being found.

Until that bar exists, nothing in this repository is a proven effect, and no
observation licenses a trade.
