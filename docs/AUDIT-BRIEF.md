# Audit brief — repository restructure

Written for the agent auditing this restructure. The operator does not want a
rubber stamp; they want a second reading of their intent, and they suspect the
author of this restructure of being too narrow and too engineering-minded.

## Branches

| Branch | What it holds |
|---|---|
| `snapshot/pre-restructure` | Everything exactly as it stood before any of this. Nothing was deleted anywhere; this branch is the recovery point. |
| `restructure/vision` | This proposal. |

Nothing was removed on either branch. One file moved: the implementation brief,
into `archive/`, with the reason recorded there.

## The operator's intent, as stated

Reconstructed from a working session. Where their words carry the meaning
better than a paraphrase, they are kept.

- **The field is finished. Stop touching it.** «Мы сделали уже поле. Мы просто
  использовали эту машину RIZ. Не фокусируйся на саму машину.» Every question
  about how the machine works, which Pine version, how zones are computed — that
  work is over.
- **The zone is an anchor, not the subject.** T0 is «рождение нашей сверхновой»
  — the origin of coordinates. The subject is what price does around it.
- **Two research surfaces.** Price behaviour around a zone, and the relations
  between zones themselves — nesting, level agreement, several timeframes
  stacking in one price pocket. The second was raised explicitly and is not
  secondary.
- **The purpose is setups.** «Нам нужно только найти торговые реально сетапы
  прибыльные, опираясь на историю» — repeating patterns, stated well enough that
  agents trade them automatically on live data later.
- **No standard of proof yet, on purpose.** Asked directly how an agent should
  know a finding is not noise, the operator answered: not yet. Observations
  accumulate first.
- **The repository is built for agents, not for humans**, and in English.
- **The predecessor is a warning.** G2 built an elaborate apparatus of gates,
  tests and ledgers and measured almost no market with it. «Там эти тесты, хуй
  тесты, нам не позволяли реально исследовать рынок.» Any structure that
  recreates that is a failure, however rigorous it looks.

## What was done

Four documents added, two rewritten, one moved.

- **`RIZ.md`** — the meaning layer, which did not exist. Market-maker obligation,
  imbalance, why most gaps are noise, why 2X is a significance filter and not a
  prediction, what a qualifying zone gives (two lines and a moment), what the
  field is (a coordinate system), the two research surfaces, and an explicit list
  of what is not being claimed.
- **`GLOSSARY.md`** — every term split into **field facts** (with the passport
  column holding them) and **derived reads** that do not exist in the machine and
  must be defined by whoever uses them. Retest, confluence, magnet and setup are
  in the second group.
- **`FINDINGS.md`** — an append-only journal with a short entry format and the
  operator's four open hunches, marked as unmeasured.
- **`AGENTS.md`** — reading order and standing rules, replacing a procedural
  cold-entry note.
- **`README.md`** — reordered so meaning comes before machinery.
- **`CLAUDE.md`** — reduced to a pointer.

## Decisions an auditor should challenge

1. **Meaning before machinery.** The judgement is that a cold agent who reads
   column names first will produce competent nonsense. Cost: a returning agent
   who only wants the query API reads one extra hop.
2. **No standard of evidence anywhere in the repository.** This follows the
   operator's answer and the G2 lesson, and it is genuinely risky: nothing here
   stops an agent from mistaking one lucky slice for a discovery. The bet is that
   premature gates cost more. Is there a cheap floor — one sentence, not an
   apparatus — worth adding now?
3. **`FINDINGS.md` as a single flat journal.** Chosen by the operator over
   per-hypothesis files. It will not scale past some size, and the migration
   point is not defined.
4. **The field-fact / derived-read split.** The central device of the glossary.
   It is a strong claim that this line is the one worth drawing.
5. **The implementation brief moved to `archive/`.** Argument: build
   instructions at the root of a finished field invite rebuilding. Counter: it
   is the only complete statement of the frozen contract.
6. **Scientific framing left open.** Searching found no academic literature on
   fair value gaps at all; the mechanism the operator describes maps onto
   market-microstructure work (order-flow imbalance, price impact, inventory
   effects), and the method is an intraday event study. The operator wanted to
   discuss the framing further before fixing it, so `RIZ.md` states the
   motivation honestly and claims nothing. This is unfinished and known.

## Where this proposal is most likely wrong

- The author repeatedly drifted into machine mechanics and measurement details
  after being told the machine is done. Similar narrowness may be baked into the
  documents themselves.
- The second research surface — relations between zones — is described but not
  developed. The field supports it and no vocabulary exists for it. This is
  probably the largest gap.
- `RIZ.md` states the market idea in the operator's terms. It has not been
  checked against how the field actually behaves, and it should not be treated
  as validated.
