# Measurements behind the default model chain

Every call that the figures in the README rest on is saved here, so they can be re-derived
without calling a model. One folder per day of measuring.

## 2026-10-15: the three free models on this project's three cases

`DEFAULT_MODEL_CHAIN` in `app/config.py` is the order these measurements gave. OpenRouter listed
three free models with structured output that day: `liquid/lfm-2.5-2.6b:free`,
`nvidia/nemotron-3-super-120b-a12b:free` and `dots-studio/dots-3-note-preview:free` (the last
one announced its end for 31 December 2026).

**The cases** are the two tasks the pipeline really sends, on the sample SKU NR-MAG-200
(Magnesium Bisglycinate 200 mg): `synthesis` (write the copy; right when the draft validates
and the deterministic checks find nothing), `audit-catches` (audit a draft that says "cures
chronic insomnia" and omits the NPN; right when the model's findings, after the overrule step,
name the cure claim or the NPN) and `audit-passes` (audit a compliant draft; right when the
verdict after the overrule step is compliant). The same functions grade them as in the
service: `app/compliance/rules.py` (`deterministic_findings`, `overrule_model_findings`,
`merge_audit`), and `app/models/cases.py` uses them for the chain's own tests.

**The final run** (`2026-10-15/cases.jsonl`, one line per call; `outputs/` holds every answer
in full) sent each model the three cases in three passes, with the shipped settings:
`OPENROUTER_REASONING=off`, `SYNTHESIS_MAX_TOKENS=4096`, `AUDIT_MAX_TOKENS=4096`, strict JSON
schema, temperature 0.2 for synthesis and 0.0 for audits. `summary.json` is computed from the
saved answers by the shipped rules (`correct` in `cases.jsonl` is recomputed the same way, so a
later change to the rules can re-grade the same answers).

| Model | Calls | No answer | Right | Median | Slowest | Place |
|---|---|---|---|---|---|---|
| liquid/lfm-2.5-2.6b:free | 9 | 0 | 9 of 9 | 9.1 s | 15.3 s | first |
| nvidia/nemotron-3-super-120b-a12b:free | 9 | 0 | 8 of 9 | 9.7 s | 13.0 s | second |
| dots-studio/dots-3-note-preview:free | 9 | 0 | 7 of 9 | 11.0 s | 14.9 s | left out |

Per case: every model wrote a compliant draft three times out of three and caught the planted
cure claim and the missing NPN three times out of three. The differences are on the compliant
draft: the NVIDIA model called it non-compliant once (a shortened authorized claim it wanted in
full, a wording-only finding that slipped past the overrule rule), the dots model twice (the
words "gentle" and "steady" read as guarantees). Those are false positives: in the pipeline
they cost one extra self-correction cycle, never a missed violation.

**The rule for the order:** fewest wrong answers first, then fewest calls without an answer,
then the faster median; the two places go to models on different upstream providers, so one
outage cannot take out both. A fallback replaces a failed call, never a wrong answer. Nine calls
per model show whether a model can do the job, not how often it is right.

**What the earlier runs of the day taught** (kept as evidence, same folder):

- `cases-reasoning-default.jsonl`: with reasoning left to the models and 2048-token budgets,
  the NVIDIA model truncated every answer (it spent the whole budget thinking) and the dots
  model's provider answered 502. That is why `OPENROUTER_REASONING` exists and defaults to
  `off`.
- `cases-reasoning-off-first-prompts.jsonl`: with reasoning off, the Liquid model's endpoint
  refused the switch ("Reasoning is mandatory for this endpoint", HTTP 400). That is why a
  model that rejects the extension is retried without it.
- `cases-pass3-audit2048.jsonl`: the Liquid model, which cannot stop reasoning, truncated its
  audits at 2048 tokens. That is why `AUDIT_MAX_TOKENS` defaults to 4096.
- `cases-run4-before-audit-reminders.jsonl` and `cases-run5-raw-grading.jsonl`: graded on the
  raw model verdict, the NVIDIA model called the compliant draft non-compliant for missing
  periods and shortened claims in most passes, whatever the prompt said. That is why the
  compliance node lets the deterministic checks decide the mechanical rules and discards
  wording-only findings on restated authorized claims (`overrule_model_findings`).

**What the recordings of the project page showed** (same day, same chain, the whole pipeline
through the API). The first recording of the day, 16 model calls: the Liquid model answered 12
and failed 4 (3 truncated at the 4096-token budget, its reasoning being unbounded on that
endpoint, and 1 connection error); each time the chain fell back to the NVIDIA model, which
answered all 4, and no run failed. That is what the chain is for: a truncated call costs the 10
to 15 seconds before the fallback starts, so a run with one self-correction cycle took 66 to 72
seconds and a first-pass run 26. The last recording, the one in `site/recorded/runs.json`, 12
model calls: all answered by the Liquid model, no fallback needed; 19 to 34 seconds for a run
that passes first time, 66 seconds for the run with a self-correction cycle. Free models vary
from one hour to the next; both pictures are true.

To measure again: `python scripts/openrouter_smoke.py` for one synthesis and one audit, or the
chain's own test from the playground's Models panel (the same cases, judged by the same rules).
