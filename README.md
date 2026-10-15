# Autonomous Multi-Agent Operations and Marketing Pipeline

A LangGraph state machine that turns one inventory SKU into channel marketing copy a regulator
would accept, and refuses to let that copy leave the building until a person has signed it off.
The graph reads the licensed product data for a Natural Health Product (NHP) from PostgreSQL,
asks a language model to draft a product page, five marketplace bullets, a social post and an
image prompt, audits the draft against Health Canada's NHP rules with a second model call plus a
set of deterministic Python checks, sends violations back to the writer for a bounded number of
revisions, and then pauses. A reviewer approves, edits or rejects the draft through the API; only
approval or an edit resumes the graph, and the last node fires an HMAC-signed webhook. Every
node's output is checkpointed in PostgreSQL through `langgraph-checkpoint-postgres`, so a thread
can wait for its reviewer for a day and be resumed from the exact state it paused in.

It is Project 3 of an Applied AI portfolio, and it runs on the same zero-cost stack as the two
projects before it: free OpenRouter models (an ordered chain that repairs itself when a model is
withdrawn), Neon for PostgreSQL, Render for hosting, GitHub Pages for the project page. The code
is FastAPI with Pydantic v2 at every boundary: the models never return free text, a request is
validated before a model is called, and error responses carry an incident id instead of an
upstream body.

* **The project page:** <https://nickdesjardins.github.io/autonomous-operation-marketing/> is
  the address to give to people. It is a static page on GitHub Pages: it is there at once, shows
  recorded runs of the pipeline exactly as the API returned them, explains the graph, and wakes
  the live demo while it is read.
* **The live demo:** to be filled in when deployed (a free Render web service; it sleeps between
  visits and takes about a minute to wake). It runs in demo mode: runs and reviews are open but
  budgeted, revisions are capped, and the webhook can only reach the service's own receiver. See
  [Public showcase](#public-showcase-demo-mode).
* **Putting it online:** [docs/DEPLOY.md](docs/DEPLOY.md) walks through it, and says exactly
  what has and has not been verified.

---

## Architecture

```text
POST /api/v1/workflows/start                                   thread_id = uuid4, one checkpoint per node
        │
        ▼
┌──────────────┐  invalid request   ┌─────┐
│  supervisor  │───────────────────►│ END │  status invalid_request
└──────┬───────┘                    └─────┘
       │ inventory not loaded yet (a replay with inventory_data goes straight to synthesis)
       ▼
┌─────────────────────┐  unknown SKU, or stock 0   ┌─────┐
│ inventory_sql_agent │───────────────────────────►│ END │  status sku_not_found / out_of_stock
│ (lookup_sku tool,   │                            └─────┘
│  parameterized SQL) │
└──────────┬──────────┘
           │ found and in stock
           ▼
┌──────────────────────┐          ┌──────────────────────────────────┐
│ copy_synthesis_agent │─────────►│ compliance_validator             │
│ (model call, strict  │          │ (model call + deterministic      │
│  JSON schema)        │◄─────────│  checks; findings always win)    │
└──────────────────────┘ revise   └───────────────┬──────────────────┘
      violations and revision_count < max_revisions │ compliant, or violations with the
                                                    │ budget spent (requires_human_escalation)
                                                    ▼
                                   ┌─────────────────────┐
                                   │ human_approval_gate │  status awaiting_human_review
                                   └──────────┬──────────┘
                                              │ interrupt_before: the graph stops here
                              ................│.......................................
                              POST /api/v1/workflows/{thread_id}/review
                                approve / edit  -> aupdate_state, then ainvoke(None)
                                reject          -> aupdate_state(as_node=webhook_dispatch): never runs
                              ................│.......................................
                                              ▼
                                   ┌──────────────────┐   X-Signature-256 (HMAC-SHA256)
                                   │ webhook_dispatch │──►  X-Timestamp, X-Delivery-Id
                                   └──────────┬───────┘   retries on 429/5xx/timeouts,
                                              │           every attempt in webhook_audit_log
                                              ▼
                                            ┌─────┐
                                            │ END │  dispatched / dispatch_failed / dispatch_refused
                                            └─────┘
```

The same graph as a state diagram (the node names are the constants in `app/graph/builder.py`,
and `topology()` in that module exposes the nodes, edges and breakpoint as data for the
playground and the tests):

```mermaid
stateDiagram-v2
    [*] --> supervisor
    supervisor --> inventory_sql_agent: inventory not loaded yet
    supervisor --> copy_synthesis_agent: inventory already loaded (replay)
    supervisor --> [*]: invalid request
    inventory_sql_agent --> copy_synthesis_agent: SKU found and in stock
    inventory_sql_agent --> [*]: SKU unknown or out of stock
    copy_synthesis_agent --> compliance_validator
    compliance_validator --> human_approval_gate: compliant
    compliance_validator --> copy_synthesis_agent: violations, revision_count below max_revisions
    compliance_validator --> human_approval_gate: violations, budget spent (escalation)
    human_approval_gate --> webhook_dispatch: interrupt_before, resumed by approve or edit
    human_approval_gate --> [*]: reject (written as if webhook_dispatch had run)
    webhook_dispatch --> [*]
```

---

## State machine mechanics

### The state and its reducers (`app/graph/state.py`)

`PipelineState` is a `TypedDict`. Every node returns a partial update and never mutates the
state it was given. Keys without a reducer are overwritten by the latest update; three keys are
annotated with `operator.add`, so they accumulate across the self-correction cycles and nothing
the critic said is ever lost.

| Field | Reducer | Written by |
|---|---|---|
| `thread_id`, `sku_code`, `target_channels`, `campaign_angle` | overwrite | the API (`initial_state`); the supervisor upper-cases `sku_code` |
| `inventory_data` | overwrite | `inventory_sql_agent` (the `SkuRecord` as a dict) |
| `draft_copy` | overwrite | `copy_synthesis_agent`; the review endpoint on `edit` |
| `compliance_report` | overwrite | `compliance_validator` (the merged `ComplianceAuditResult`) |
| `audit_history` | `operator.add` | `compliance_validator`, one `AuditEntry` per audit |
| `revision_instructions` | `operator.add` | `compliance_validator`, the feedback text for the next draft |
| `revision_count`, `max_revisions` | overwrite | the supervisor resets the count to 0 on a fresh start; the validator increments it on a failed audit |
| `requires_human_escalation` | overwrite | `compliance_validator` |
| `human_review_status`, `human_reviewer_id`, `human_reviewer_notes` | overwrite | the gate sets `pending`; the review endpoint sets `approved`, `edited` or `rejected` |
| `webhook_delivery_result` | overwrite | `webhook_dispatch` (the `DeliveryResult` as a dict) |
| `status`, `error` | overwrite | every node (table below) |
| `trace` | `operator.add` | every node appends one event (`node`, `at`, `ms`, `detail`); the review endpoint appends a `human_review` event |
| `created_at`, `updated_at` | overwrite | the API, then every node |

### Status values

Every value is a `STATUS_` constant in `app/graph/state.py`. `TERMINAL_STATUSES` holds the ones
after which the thread does nothing more.

| `status` | Set by | When |
|---|---|---|
| `started` | `initial_state`, `supervisor` | the request is accepted and the supervisor found nothing wrong with it |
| `invalid_request` | `supervisor` | the SKU, the channels, the brief or `max_revisions` are unusable; the run ends (terminal) |
| `inventory_loaded` | `inventory_sql_agent` | the SKU exists and has stock |
| `sku_not_found` | `inventory_sql_agent` | the SKU is not in the inventory, or the code does not even pass the tool's argument model (terminal) |
| `out_of_stock` | `inventory_sql_agent` | the SKU exists but `stock_quantity` is 0 (terminal) |
| `drafted` | `copy_synthesis_agent` | a validated `ChannelCopyDraft` came back |
| `compliant` | `compliance_validator` | the merged audit lists no violation |
| `revision_requested` | `compliance_validator` | violations, and the incremented `revision_count` is still below `max_revisions` |
| `escalated_to_human` | `compliance_validator` | violations, and the incremented `revision_count` has reached `max_revisions` |
| `awaiting_human_review` | `human_approval_gate` | the graph is about to stop at the breakpoint |
| `rejected` | the review endpoint | the reviewer rejected; written as if the webhook node had run (terminal) |
| `dispatched` | `webhook_dispatch` | delivered with a 2xx, or `WEBHOOK_TARGET=dry_run` (terminal) |
| `dispatch_failed` | `webhook_dispatch` | every attempt failed, or `self` has no address to post to (terminal) |
| `dispatch_refused` | `webhook_dispatch` | the node ran while `human_review_status` was neither `approved` nor `edited` (terminal) |

Two more values pass through `status` for an instant: the review endpoint writes `approved` or
`edited` into it together with `human_review_status`, and the webhook node overwrites it with
its own outcome a moment later. A `GET` between the two shows the reviewer's word.

### Routing rules (`app/graph/nodes.py`)

The routers are pure functions of the state after the node ran.

* `route_after_supervisor`: `END` when `status` is `invalid_request`; `copy_synthesis_agent`
  when `inventory_data` is already present (a replay); else `inventory_sql_agent`.
* `route_after_inventory`: `END` when `status` is `sku_not_found` or `out_of_stock`; else
  `copy_synthesis_agent`.
* `copy_synthesis_agent` always goes to `compliance_validator`.
* `route_after_compliance`: `human_approval_gate` when the report is compliant; otherwise
  `copy_synthesis_agent` while `revision_count < max_revisions`, and `human_approval_gate` with
  `requires_human_escalation=True` once it is not. The validator increments `revision_count`
  before the router reads it, so `max_revisions` is the number of audits a draft may fail before
  a person sees it: `3` allows two redrafts, `2` allows one, and `1` or `0` escalates after the
  first failed audit.
* `human_approval_gate` always goes to `webhook_dispatch`, and `webhook_dispatch` to `END`.

A run therefore makes at most `2 * max(1, max_revisions)` model calls (a draft and an audit per
attempt). The demo budget reserves `2 * (max_revisions + 1)`, one attempt more than can happen.

### Human in the loop, exactly as implemented

The graph is compiled with `interrupt_before=[webhook_dispatch]` (`app/graph/builder.py`). The
API (`app/main.py`) drives it like this, against `langgraph==1.2.15`:

1. **Start.** `POST /api/v1/workflows/start` builds the initial state, calls
   `graph.ainvoke(state, {"configurable": {"thread_id": ...}})` and waits. The call returns when
   the graph stops: at `END` for an invalid, unknown or out-of-stock SKU, or at the breakpoint,
   where the snapshot's `next` is `("webhook_dispatch",)`. The response reports that as
   `awaiting_human_review: true` and `next_nodes: ["webhook_dispatch"]`.
2. **Review.** `POST /api/v1/workflows/{thread_id}/review` first reads the snapshot and answers
   `409` unless `next` is exactly `("webhook_dispatch",)`: a thread can only be reviewed while
   it waits. Then, by action:
   * **approve**: `graph.aupdate_state(config, {human_review_status: "approved", status:
     "approved", human_reviewer_id, human_reviewer_notes, updated_at, trace: [...]})`, then
     `graph.ainvoke(None, config)`. The update is applied as the last node that ran (the gate),
     and the resume runs `webhook_dispatch` and reaches `END`.
   * **edit**: the same update with `draft_copy` replaced by the reviewer's `override_copy`
     (a complete `ChannelCopyDraft`, validated by the request model), then `ainvoke(None)`.
     The webhook node validates the draft once more before it builds the payload, whoever wrote
     it. The override is not audited again: the reviewer is the audit.
   * **reject**: `graph.aupdate_state(config, {...human_review_status: "rejected", status:
     "rejected"...}, as_node="webhook_dispatch")`. Writing the update *as* the webhook node
     makes the graph treat that node as done: `next` becomes `()` and the node never runs.
     Nothing leaves the system, and the thread is terminal.
3. **The node's own refusal.** `webhook_dispatch` checks `human_review_status` itself and
   returns `dispatch_refused` unless it is `approved` or `edited`. Resuming the graph by any
   other route (a bare `ainvoke(None)`, a tool that replays checkpoints) cannot fire the webhook
   without a recorded human decision.

Everything above survives a restart: the thread's state lives in the checkpoint tables, and
`GET /api/v1/workflows/{thread_id}` reads it back through `graph.aget_state`.

---

## The compliance rules

`app/compliance/rules.py` holds the six rules the auditor is briefed with (they are rendered into
both system prompts) and the checks Python runs on every draft, whatever the model said. The
rules are modelled on Health Canada's framework for NHPs: a licensed product carries an
eight-digit Natural Product Number that must appear on the label, may only make the claims its
licence authorizes, must carry its risk information, and may not be advertised as a treatment,
cure or preventive for the diseases in Schedule A of the Food and Drugs Act. They encode those
principles for a demo; they are not legal advice.

| Rule | Severity | What it requires | Deterministic check |
|---|---|---|---|
| `NPN_DISPLAY` | high | The product page and the social post show the NPN exactly as licensed, and the draft reports the number it displayed | `npn_findings`: the licence number is a substring of `npn_displayed`, `web_description_html` and `social_post_copy` |
| `AUTHORIZED_CLAIMS_ONLY` | high | Every health claim is one of the licence's authorized claims, or a plain restatement of one | model finds; the code discards a wording-only finding whose quoted text is made of an authorized claim's own words (`overrule_model_findings`), and keeps every finding whose quote adds a word the licence does not contain |
| `NO_DISEASE_CLAIMS` | high | No treatment, cure, prevention, healing or reversal claims for a disease; no antibiotic, antiviral or "replaces your medication" language | `prohibited_findings`: eleven compiled patterns over the title, description, bullets, social post and image prompt (cure in any form; treat, prevent, heal, fight, reverse or eliminate within four words of a disease term; antibiotic, antiviral, antifungal; kills bacteria, viruses, germs; replaces a prescription or a doctor) |
| `DOSAGE_ACCURACY` | medium | An amount written next to an ingredient is the licensed amount per serving, in the licensed unit | `dosage_findings`: for every occurrence of an ingredient's name, every amount within 60 characters must equal the licensed amount (or the total across ingredients sharing the unit, for a multi-strain probiotic); percentages are ignored as standardization figures |
| `RISK_INFORMATION` | high | Every mandatory risk statement of the licence appears word for word in `disclaimers_included` | `risk_statement_findings`: each mandatory statement, normalized (NFKC, lower case, tags and punctuation out), is a substring of the normalized disclaimers |
| `NO_GUARANTEES` | medium | No guaranteed results, miracle claims, safety absolutes, "clinically proven", or approval and endorsement claims | `prohibited_findings`: guarantee, miracle, 100% safe or effective, no side effects, safe for everyone, clinically or scientifically proven, Health Canada or FDA approved |

What the checks prove: a draft that reaches the human gate as `compliant` shows the licence
number where it must, contains none of the prohibited wordings in any scanned field, states no
ingredient amount other than the licensed one next to that ingredient, and carries every
mandatory warning verbatim. What they do not prove is the rest: an invented benefit phrased
without any of those words is the model auditor's job, which is why both run.

`overrule_model_findings` runs first and drops two kinds of model finding: one under
`NPN_DISPLAY`, `DOSAGE_ACCURACY` or `RISK_INFORMATION` in a field where the code found nothing
(the code is the authority on those), and a wording-only `AUTHORIZED_CLAIMS_ONLY` finding whose
quoted text is made of an authorized claim's own words and whose fix is about punctuation,
wording or a shortened claim. A quote that adds a word the licence does not contain ("relieves
arthritis pain", "melts belly fat") is never dropped, whatever the fix says. The count goes to
`discarded_findings` in the audit entry, and the discarded items with their reason to the
`compliance_validator` trace event. `merge_audit` then combines what is left with the
deterministic findings: the model's upheld violations come first, the deterministic findings
that the model did not already name are appended, the risk level is raised to the worst finding,
and `is_compliant` is true only when the merged list is empty. `format_revision_instructions`
turns the merged audit into the numbered feedback the next synthesis call receives.

---

## Model calls

The models never return free text. Each node asks for exactly one schema:

* **Copy synthesis** (`SYNTHESIS_SYSTEM_PROMPT`, `synthesis_messages` in
  `app/graph/prompts.py`) returns a `ChannelCopyDraft`: `web_title`, `web_description_html`,
  exactly five `marketplace_bullets`, `social_post_copy`, `image_generation_prompt`,
  `disclaimers_included` and `npn_displayed`. The user message carries the SKU record as JSON,
  the target channels and the brief; a revision call adds the previous draft, the violations to
  fix and the latest revision notes. `SYNTHESIS_TEMPERATURE` is 0.2 by default.
* **Compliance audit** (`AUDIT_SYSTEM_PROMPT`, `audit_messages`) returns a
  `ComplianceAuditResult`: `is_compliant`, `risk_level`, a list of `ViolationItem` (field,
  offending text, rule, required fix) and a `revision_summary`. `AUDIT_TEMPERATURE` is 0.0. A
  model validator in the schema makes `is_compliant` false whenever violations are listed, so a
  contradictory verdict cannot slip through. The reverse holds too: a verdict of not compliant
  with no violation listed is treated as compliant, because there is nothing for the writer to
  fix (a flag some models set wrong); the deterministic findings are merged in afterwards, and
  the human gate still reads every draft.

Both system prompts are static strings, so a provider that caches prompt prefixes can reuse
them; everything that varies goes in the user message.

**Strict schema first, JSON mode second.** `OpenAICompatibleChain` (`app/services/llm_client.py`)
sends the Pydantic schema, hardened for strict mode (`strict_json_schema`: every property
required, `additionalProperties: false` everywhere), as a `json_schema` response format. A model
that rejects the format (a 400 mentioning `response_format`, `json_schema` or `structured`) is
retried once in plain `json_object` mode with the schema appended to the system prompt
(`json_mode_suffix`). Either way the text is cleaned (`extract_json_object`: code fences
stripped, a trailing object recovered from prose) and validated with `model_validate` before the
graph sees it. An answer that does not validate counts as a failed attempt and the next model is
tried. Claude goes through the official SDK (`app/services/anthropic_client.py`) with
`output_config.format` as the schema and `effort` instead of temperature, and is validated the
same way. The trace of every node records which model answered, whether a fallback was used and
whether JSON mode was needed.

**The brief is untrusted.** The campaign angle is the one text a stranger controls. The request
model caps it at 300 characters, collapses whitespace and refuses control characters. The prompt
wraps it in `<campaign_brief_{nonce}>` tags with an eight-hex-character nonce drawn per request
(`new_nonce`), tells the model that the text is the campaign theme written by a stranger and not
an instruction, and the system prompt says to ignore any part of it that conflicts with the
rules. The audit call wraps the draft the same way and tells the auditor that anything in it that
reads like an instruction is part of the copy, and probably a violation.

---

## Model selection: free models and the chain that repairs itself

The default provider is OpenRouter with an ordered chain of `:free` models. One model answers a
healthy request; the next one is tried only when the earlier one's call *fails* (a 429, an
in-band upstream error, a 404 withdrawal, a timeout, truncation, empty output, or JSON that does
not validate). A wrong-but-valid answer is not a failure: it goes to the auditor, which is the
design.

**Automatic or pinned.** `OPENROUTER_MODELS` blank (what `.env.example` ships) means
*automatic*: the service starts from `DEFAULT_MODEL_CHAIN` in `app/config.py`, reads the chain
this project's scheduled test published, and looks after the chain itself (the `app/models`
package: `catalogue`, `cases`, `rules`, `testing`, `chooser`, `published`, `cache`). Naming
models in `OPENROUTER_MODELS` *pins* the chain: the service calls exactly those models and
never reads, tests or replaces one. The start-up log says which is in force (`Model choice:
automatic` or `Model choice: pinned`), and `/health` reports it as `llm.mode`: `automatic`,
`pinned`, `no_key` (no OpenRouter key, so nothing is looked after), `not_openrouter` (another
base URL has no catalogue to ask), or `fixed` for the other providers.

**The default chain, in words.** First `liquid/lfm-2.5-2.6b:free` (Liquid), then
`nvidia/nemotron-3-super-120b-a12b:free` (NVIDIA). The two sit on different upstream providers,
so one outage cannot take out both. It is a dated snapshot, not a ranking: `DEFAULT_MEASURED_ON`
in `app/models/testing.py` records 15 October 2026 as the day it was measured by hand through
this service's own cases, three passes of the three cases per model, with every call saved under
`benchmarks/2026-10-15/` and the grading explained in `benchmarks/README.md`. The first place
goes to the model with the fewest wrong answers. To measure again: `scripts/openrouter_smoke.py`
(the catalogue's word on the chain, the day's allowance, then one synthesis and one audit of the
planted draft: two free requests) and the model test (the *Models* section of the playground, or
`python -m app.models.published --previous chain.json --out chain.json --force`, which writes
the same file the scheduled job publishes).

<!-- snapshot:start -->
| Model | Cases right | Median answer | Slowest answer | Note |
|---|---|---|---|---|
| `liquid/lfm-2.5-2.6b:free` | 9 of 9 | 9.1 s | 15.3 s | First place. Cannot switch its reasoning off and thinks for about 2000 tokens before a verdict, which is why `AUDIT_MAX_TOKENS` defaults to 4096 |
| `nvidia/nemotron-3-super-120b-a12b:free` | 8 of 9 | 9.7 s | 13.0 s | Second place. Called the compliant draft non-compliant once (a wording-only finding: a false positive, never a missed violation) |
| `dots-studio/dots-3-note-preview:free` | 7 of 9 | 11.0 s | 14.9 s | Left out. Two false positives on the compliant draft, and its end is announced for 31 December 2026 |
<!-- snapshot:end -->

**The three cases a candidate must pass** (`app/models/cases.py`). Being listed as free with
"structured output" proves nothing; the only evidence that counts is this service's own work on
the sample SKU `NR-MAG-200`, sent through the same prompts, schemas, parsing and validation a
real run uses:

| Case | What is sent | Right when |
|---|---|---|
| `synthesis` | The synthesis prompt with a neutral brief | The draft validates and `deterministic_findings` finds nothing to fix |
| `audit-catches` | The audit prompt with a planted draft: a cure claim in two fields and no licence number anywhere | One of the model's findings that survive `overrule_model_findings` names the cure claim or the NPN (a finding the pipeline discards is no catch) |
| `audit-passes` | The audit prompt with a compliant draft | The verdict is compliant after `overrule_model_findings` and `merge_audit`, as a run would act on it (a wording-only finding the pipeline discards does not fail the model) |

A candidate must pass every case (`MAX_MISSES = 0` in `app/models/rules.py`): one wrong or
unusable answer fails it, and its test stops there so no call is spent on a model that can no
longer pass. A call that OpenRouter turned away (a rate limit, a timeout, an upstream error)
leaves the model *not judged*, which says something about the moment and nothing about the
model. A model whose usual (median) answer takes more than half of `OPENROUTER_TIMEOUT_S` is
*slow*: right, but good for the last place only. At most four candidates are tested at once
(`CANDIDATE_LIMIT = 4`, so a test is 12 free requests at the very most), request starts are kept
six seconds apart so a visitor's run still finds room, and tests are kept fifteen minutes apart
whoever starts them.

**Repair, never rank** (`app/models/rules.py`, pure functions). A chain has three places
(`PLACES = 3`): two like the tested default, and one spare. A model leaves the chain only when
two sources agree: it gave no answer to a real request here, and OpenRouter's own page for it
says "Model not found". A free place is filled by a model that passed the test, behind the
models already there; then at most one slow model, if a place is still free. A model that is in
the chain keeps its place and its order whatever one test says about it, and a chain is never
emptied: three cases can rule a model out, they cannot rank two models that both pass. One
exception puts a newcomer in front, and it ranks nothing: a model of the measured default that
was withdrawn, comes back and passes goes back to the place a person measured it for.

**What happens after a failed request** (`app/models/chooser.py`). Nothing is tested while
requests succeed, and starting the service calls no model (any request wakes a sleeping host, a
crawler's included). After a request that a model of the chain did not answer, the chain client
tells the chooser, which looks into it in the background, at most once a minute, and asks the
cheap questions first: it reads the published list again (a repair may already be waiting
there), then OpenRouter's own page for each model that failed. Only "Model not found" counts: a
rate limit or an upstream error on a model OpenRouter still lists is taken for a passing
failure, and nothing is tested. A withdrawn model is taken out at once, and the free place is
filled by a test of the free models OpenRouter lists that are not in the chain. Unlike Project 1
there are no stand-ins: three cases take under a minute per candidate, so a model answers for
the service only once a test has judged it. The chain the service chose is kept in memory and,
where the disk allows, in `model_cache.json` (`app/models/cache.py`); that file is never
committed, and a host that forgets its files starts again from the published chain or the
tested default. Every id the service chooses has to match `vendor/model:free`: `cache.use()` and
`testing.run_test()` refuse anything else, and `OPENROUTER_FREE_ONLY=true` makes the chain client
itself refuse to call an id without the suffix, so a typo or a bad file can never spend money.
The *Models* section of the playground shows all of this (where the chain came from, a test
while it runs, what the last one found) and its button starts a test through
`POST /playground/models/test`, which answers `202` or `409` with the reason it cannot start now.

**The published chain** (`app/models/published.py`). A service that nobody uses tests nothing,
and on a free host it forgets what it knew at every sleep. So a scheduled job on GitHub
(`.github/workflows/model-check.yml`, shown as "Model check") runs `python -m
app.models.published` every day at 06:17 UTC, on a push to `main` that changes a file the test
is made of, and by hand. Deciding whether a test is due costs nothing and needs no key: one is
due when nothing usable is published, when the last test is six and a half days old, when the
last test could confirm nothing and 20 hours have passed, when OpenRouter's list of free models
changed in a way a test could act on, or when the run was started by hand with "force" ticked.
The test itself needs the repository secret `OPENROUTER_API_KEY` (only one step of the `test`
job sees it) and spends at most 12 free requests. A second job, which runs no downloaded code,
commits the chain it confirmed to `chain.json` on the branch `tested-models`; a third turns the
run red when the owner has something to do, and the first job's log then ends with a line that
begins `Failing this run on purpose:`. A service in automatic mode reads the file at start-up
and after a failed request, from `DEFAULT_PUBLISHED_URL` (`MODEL_PUBLISHED_URL` points
elsewhere; `null` switches it off): one request, no key, no free request spent. The file is a
claim like any other, so it is checked before it is used: at most 64 000 bytes, a version 1
object, a date that is not in the future, a chain of one to three different ids that all end in
`:free`, and a fingerprint of the test that made it (the cases, their schemas and messages, the
rules' constants, the default chain, the response format), so a fork that changes a prompt
ignores this project's file by itself. It is taken only when its test is newer than the one
behind the chain in use, and after a failed request not when it names the very chain that just
failed. The branch is not `main` on purpose: a host that deploys on every push to `main` would
restart the service at every publication. From 14 days without a publication the log says how
old the list is; GitHub pauses scheduled workflows in a public repository after 60 days without
activity, and the Actions tab's "Enable workflow" brings it back.

**How the default was chosen, and how to change it.** By hand, from measurements, by the
ordering rule `rank()` in `app/models/rules.py` applies to a test's rows: (1) leave out a model
that never answered, or whose usual answer takes more than half of the time limit; (2) of the
rest, the first place goes to the model with the fewest wrong answers; (3) then the fewest calls
with no answer; (4) speed only breaks ties. The reason is in how the chain works: it moves on
only when a call fails, so a fallback replaces a failed call, never a wrong answer. Changing the
order of two working models stays a person's job: measure again, change `DEFAULT_MODEL_CHAIN`
and `DEFAULT_MEASURED_ON`, and push. A service that saved a chain under the old default ignores
that file, a published list made under it fails the fingerprint, and the push itself starts the
scheduled test.

**Pacing and allowance.** Request starts are paced to `OPENROUTER_REQUESTS_PER_MINUTE` (20,
OpenRouter's free-model limit). In demo mode the budget also asks OpenRouter (`GET /key`) how
many free requests the account has left today and stops admitting runs at a reserve, and a test
of the models is refused by the same rule when too little of the day is left for it and for
visitors. Free endpoints may keep and train on what they are sent: use the sample SKUs, never a
real product dossier, while running on free models.

---

## Quick start

### Prerequisites

* Python 3.12 (the version in `.python-version`; `app/services/llm_client.py` uses 3.12 generic
  syntax)
* Docker with Docker Compose for PostgreSQL 16, or nothing at all with `CHECKPOINTER=memory`
* An OpenRouter key (<https://openrouter.ai/keys>) for the zero-cost stack; `ANTHROPIC_API_KEY`
  or an OpenAI-compatible key for the paid providers

### 1. Start PostgreSQL

```bash
docker compose up -d postgres
```

The container is `postgres:16` with a persistent `pgdata` volume, bound to `127.0.0.1:5432`,
credentials `ops / ops / ops` (`POSTGRES_*` in `.env` override them). On the first boot of the
volume it runs `sql/init_inventory.sql`, which creates the inventory and webhook tables and
seeds the four sample SKUs. The service applies the same idempotent statements at start-up
(`AUTO_CREATE_SCHEMA=true`) and then runs the checkpointer's own `setup()`, so a hosted database
needs no manual step. `python -m app.seed` regenerates the SQL file from `app/seed.py`.

### 2. Install and configure

```bash
python -m venv .venv
. .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # then write your key after OPENROUTER_API_KEY=
```

The app reads `.env` and then `.env.local` (values there override `.env`); both are
git-ignored. Leave `OPENROUTER_MODELS` blank: the service then starts from the tested default
chain and looks after it. `WEBHOOK_TARGET` defaults to `dry_run`, so a fresh checkout records
deliveries without calling out: signed when `WEBHOOK_SECRET` is set, marked `unsigned`
otherwise.

### 3. Run the API

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Interactive docs: <http://localhost:8000/docs> (hidden when `ENVIRONMENT=production`). The
playground at `/` is a handful of static files in `app/playground/static/` with no build step.
It reads `/playground/config` (the SKUs, channels, sample briefs, limits, rules, both system
prompts and the graph topology), starts a run, shows the nodes it visited on the graph, the
inventory facts the agents were given, a timeline of every node with its model and timing, each
compliance audit, the revision instructions, the current draft, and then asks for your decision:
approve, edit the copy field by field, or reject. After the decision it shows the delivery
result and the receiver's inbox, and the raw JSON of the thread. A *Models* section draws the
chain in use from `/playground/models` and can start a test of the free models.

To run the API in Docker as well (the image runs as a non-root user, one worker, with a
`HEALTHCHECK` on `/livez`):

```bash
docker compose --profile full up --build
```

### Laptop mode: no database

```bash
CHECKPOINTER=memory uvicorn app.main:app --port 8000
```

Checkpoints go to LangGraph's `InMemorySaver`, the inventory is served from `app/seed.py`, and
the webhook audit log and inbox live in memory. Everything is forgotten when the process stops;
the start-up log says so.

### The zero-cost stack

| Stage | Free option | Setting |
|---|---|---|
| Both model calls | Ordered chain of OpenRouter `:free` models | `LLM_PROVIDER=openrouter`, `OPENROUTER_API_KEY` |
| Database | PostgreSQL in Docker, or Neon's free plan | `DATABASE_URL` (or `DATABASE_URL_UNPOOLED`) |
| Webhook | The service's own HMAC-verified receiver, or a dry run | `WEBHOOK_TARGET=self` with `WEBHOOK_SECRET` and `PUBLIC_BASE_URL`, or `dry_run` |
| Hosting | Render free web service + GitHub Pages | `render.yaml`, `.github/workflows/pages.yml` |

To watch a delivery arrive on your own machine: `WEBHOOK_TARGET=self`, `WEBHOOK_SECRET=<any long
random string>`, `PUBLIC_BASE_URL=http://127.0.0.1:8000`, then approve a run and read
`/api/v1/webhooks/inbox`. Or run the showcase itself as it runs on Render:

```bash
python scripts/run_local_demo.py        # http://127.0.0.1:8011/, demo mode, self-addressed webhook
```

It reads the key and the database from `.env` / `.env.local`, sets `DEMO_MODE=true`,
`WEBHOOK_TARGET=self` and `PUBLIC_BASE_URL` for that process only, and generates a
`WEBHOOK_SECRET` when none is set. `render.yaml` is not read, so `/docs` stays visible.

---

## curl walkthrough

The examples use `jq` for readability and the default local settings (no `API_KEY`, so no
header). Outside demo mode, when `API_KEY` is set, every `/api/v1/workflows` request needs
`-H "X-API-Key: ..."`.

### Health

```bash
curl -s http://localhost:8000/health | jq
```

```json
{
  "status": "ok",
  "version": "1.0.0",
  "service": "multi-agent-ops-pipeline",
  "environment": "development",
  "demo_mode": false,
  "database": {"kind": "postgres", "connected": true, "checkpointer_ready": true, "inventory_skus": 4},
  "llm": {
    "provider": "openrouter",
    "configured": true,
    "models": ["liquid/lfm-2.5-2.6b:free", "nvidia/nemotron-3-super-120b-a12b:free"],
    "mode": "automatic"
  },
  "webhook": {"target": "dry_run", "configured": true, "url_host": null}
}
```

`/health` answers `503` when the database cannot be reached and `"status": "degraded"` when no
model key is configured or the webhook target has no address. `/livez` answers without touching
anything; point host probes at it. `GET /api/v1/skus` lists the four sample SKUs and
`GET /api/v1/rules` the six rules.

### Start a run

```bash
curl -s -X POST http://localhost:8000/api/v1/workflows/start \
  -H "Content-Type: application/json" \
  -d '{
        "sku_code": "NR-MAG-200",
        "target_channels": ["web_store", "marketplace", "social"],
        "campaign_angle": "Autumn restock: calm evenings and better sleep routines for busy parents.",
        "max_revisions": 2
      }' | jq
```

The call runs the graph to the breakpoint and answers `201` with the thread's state (tens of
seconds on free models: two to four model calls). Shortened; the full `draft_copy` is the one
shown under *Edit* below, `inventory_data` is the SKU record, and the `attempts` lists in the
trace are cut to their first entry.

```json
{
  "thread_id": "3f1c9a6e-2b7d-4e55-9c0a-8d2f6b1e4a77",
  "status": "awaiting_human_review",
  "sku_code": "NR-MAG-200",
  "target_channels": ["web_store", "marketplace", "social"],
  "campaign_angle": "Autumn restock: calm evenings and better sleep routines for busy parents.",
  "next_nodes": ["webhook_dispatch"],
  "awaiting_human_review": true,
  "revision_count": 1,
  "max_revisions": 2,
  "requires_human_escalation": false,
  "human_review_status": "pending",
  "human_reviewer_id": null,
  "human_reviewer_notes": null,
  "inventory_data": {"sku_code": "NR-MAG-200", "product_name": "Magnesium Bisglycinate 200 mg", "npn_number": "80098765", "...": "..."},
  "draft_copy": {"web_title": "Magnesium Bisglycinate 200 mg: gentle, chelated magnesium for daily use", "...": "..."},
  "compliance_report": {
    "is_compliant": true,
    "risk_level": "low",
    "violations": [],
    "revision_summary": "Every claim is one of the authorized claims, the NPN is shown in both required fields and both mandatory risk statements are included word for word."
  },
  "audit_history": [
    {
      "revision": 0,
      "is_compliant": false,
      "risk_level": "high",
      "violations": [
        {
          "field_name": "social_post_copy",
          "violating_text": "The NPN 80098765 does not appear in social_post_copy.",
          "regulation_rule": "Show the Natural Product Number",
          "required_fix": "Add the licence number, written as NPN 80098765, to social_post_copy."
        }
      ],
      "revision_summary": "The social post omits the licence number. Deterministic checks added 1 finding(s): social_post_copy: Show the Natural Product Number.",
      "deterministic_findings": 1,
      "discarded_findings": 0,
      "model": "liquid/lfm-2.5-2.6b:free",
      "checked_at": "2026-10-15T14:02:11Z"
    },
    {
      "revision": 1,
      "is_compliant": true,
      "risk_level": "low",
      "violations": [],
      "revision_summary": "Every claim is one of the authorized claims, the NPN is shown in both required fields and both mandatory risk statements are included word for word.",
      "deterministic_findings": 0,
      "discarded_findings": 0,
      "model": "liquid/lfm-2.5-2.6b:free",
      "checked_at": "2026-10-15T14:02:31Z"
    }
  ],
  "revision_instructions": [
    "Revision 1: the previous draft failed the compliance audit (risk high).\nThe social post omits the licence number. Deterministic checks added 1 finding(s): social_post_copy: Show the Natural Product Number.\n1. [social_post_copy] Show the Natural Product Number: \"The NPN 80098765 does not appear in social_post_copy.\" -> Add the licence number, written as NPN 80098765, to social_post_copy."
  ],
  "webhook_delivery_result": null,
  "trace": [
    {"node": "supervisor", "at": "2026-10-15T14:01:58Z", "ms": 0.1, "detail": {"outcome": "routed", "route": "inventory_sql_agent", "channels": ["web_store", "marketplace", "social"], "max_revisions": 2}},
    {"node": "inventory_sql_agent", "at": "2026-10-15T14:01:58Z", "ms": 4.2, "detail": {"tool": "lookup_sku", "outcome": "found", "product_name": "Magnesium Bisglycinate 200 mg", "npn_number": "80098765", "stock_quantity": 1240, "ingredients": 1, "authorized_claims": 4}},
    {"node": "copy_synthesis_agent", "at": "2026-10-15T14:02:07Z", "ms": 8900.5, "detail": {"revision": 0, "model": "liquid/lfm-2.5-2.6b:free", "fallback_triggered": false, "json_mode": false, "attempts": [{"model": "liquid/lfm-2.5-2.6b:free", "outcome": "answered", "seconds": 8.9, "detail": null}], "typography_fixes": 0}},
    {"node": "compliance_validator", "at": "2026-10-15T14:02:11Z", "ms": 4100.0, "detail": {"revision": 0, "model": "liquid/lfm-2.5-2.6b:free", "decision": "revise", "risk_level": "high", "violations": 1, "deterministic_findings": 1, "discarded_findings": [], "llm_said_compliant": true, "fallback_triggered": false, "attempts": [{"model": "liquid/lfm-2.5-2.6b:free", "outcome": "answered", "seconds": 4.1, "detail": null}]}},
    {"node": "copy_synthesis_agent", "at": "2026-10-15T14:02:27Z", "ms": 15600.0, "detail": {"revision": 1, "model": "liquid/lfm-2.5-2.6b:free", "fallback_triggered": false, "json_mode": false, "attempts": [{"model": "liquid/lfm-2.5-2.6b:free", "outcome": "answered", "seconds": 15.6, "detail": null}], "typography_fixes": 0}},
    {"node": "compliance_validator", "at": "2026-10-15T14:02:31Z", "ms": 3800.0, "detail": {"revision": 1, "model": "liquid/lfm-2.5-2.6b:free", "decision": "compliant", "risk_level": "low", "violations": 0, "deterministic_findings": 0, "discarded_findings": [], "llm_said_compliant": true, "fallback_triggered": false, "attempts": [{"model": "liquid/lfm-2.5-2.6b:free", "outcome": "answered", "seconds": 3.8, "detail": null}]}},
    {"node": "human_approval_gate", "at": "2026-10-15T14:02:31Z", "ms": 0.0, "detail": {"escalated": false, "reason": "the draft is compliant and waits for sign-off"}}
  ],
  "error": null,
  "created_at": "2026-10-15T14:01:58Z",
  "updated_at": "2026-10-15T14:02:31Z",
  "checkpoint_id": "1f0a9c2e-5b3d-6f10-8001-2c4e6a8b0d1f"
}
```

Note the first audit: the model said the draft was compliant (`llm_said_compliant: true`) and
the deterministic NPN check overruled it. `NR-VITD-1000` ends at once with `out_of_stock`, and an
unknown code with `sku_not_found` (both `201`, with `error` filled and `next_nodes` empty).

### Read a thread

```bash
curl -s http://localhost:8000/api/v1/workflows/3f1c9a6e-2b7d-4e55-9c0a-8d2f6b1e4a77 | jq
```

The same shape as above, read from the checkpoint. An unknown or malformed id answers `404`.

### Review: approve

```bash
curl -s -X POST http://localhost:8000/api/v1/workflows/3f1c9a6e-2b7d-4e55-9c0a-8d2f6b1e4a77/review \
  -H "Content-Type: application/json" \
  -d '{"action": "approve", "reviewer_id": "ops-lead", "notes": "Looks good."}' | jq
```

The response is the thread's state with the `action` added. With the default `dry_run` target
and `WEBHOOK_SECRET` set (without it `signature` reads `"unsigned"`; the state holds the start
of the signature, never the whole header):

```json
{
  "action": "approve",
  "thread_id": "3f1c9a6e-2b7d-4e55-9c0a-8d2f6b1e4a77",
  "status": "dispatched",
  "next_nodes": [],
  "awaiting_human_review": false,
  "human_review_status": "approved",
  "human_reviewer_id": "ops-lead",
  "human_reviewer_notes": "Looks good.",
  "webhook_delivery_result": {
    "delivered": false,
    "mode": "dry_run",
    "delivery_id": "9b4c1d2e3f4a5b6c7d8e9f0a1b2c3d4e",
    "url_host": null,
    "status_code": null,
    "attempts": 0,
    "sent_at": "2026-10-15T14:05:02Z",
    "duration_ms": 0.0,
    "signature": "sha256=4f2a0c7b9e1d...",
    "error": null
  },
  "...": "..."
}
```

A second review of the same thread answers `409`: it is no longer waiting.

### Review: edit

`edit` replaces the draft with the reviewer's own and resumes. `override_copy` must be a complete
`ChannelCopyDraft`. The one below is written for `NR-MAG-200` and passes every deterministic
check (the NPN in both required fields and in `npn_displayed`, both mandatory warnings word for
word, the licensed 200 mg next to the ingredient, none of the prohibited wordings):

```bash
curl -s -X POST http://localhost:8000/api/v1/workflows/3f1c9a6e-2b7d-4e55-9c0a-8d2f6b1e4a77/review \
  -H "Content-Type: application/json" \
  -d @- <<'JSON' | jq
{
  "action": "edit",
  "reviewer_id": "ops-lead",
  "notes": "Tightened the social post.",
  "override_copy": {
    "web_title": "Magnesium Bisglycinate 200 mg: gentle, chelated magnesium for daily use",
    "web_description_html": "<p>Magnesium Bisglycinate 200 mg gives you 200 mg of magnesium per vegetable capsule, in a chelated form chosen for tolerability. It helps to maintain proper muscle function, including the heart muscle, helps in the development and maintenance of bones and teeth, and helps to prevent magnesium deficiency.</p><ul><li>Serving: 1 capsule, 1 to 2 times daily</li><li>Vegan, gluten free, non-GMO</li></ul><p><strong>NPN 80098765</strong></p><p>Consult a health care practitioner prior to use if you are pregnant or breastfeeding. Some people may experience diarrhoea.</p>",
    "marketplace_bullets": [
      "Helps to maintain proper muscle function, including the heart muscle",
      "Helps in the development and maintenance of bones and teeth",
      "Helps to prevent magnesium deficiency and helps in energy metabolism",
      "200 mg of magnesium per vegetable capsule, 1 capsule 1 to 2 times daily",
      "Chelated bisglycinate form chosen for tolerability; vegan, gluten free, non-GMO"
    ],
    "social_post_copy": "Autumn evenings, steady routines. Magnesium Bisglycinate 200 mg helps to maintain proper muscle function and helps in energy metabolism, in a chelated form that is easy on the stomach. One capsule, once or twice a day. NPN 80098765 #magnesium #eveningroutine",
    "image_generation_prompt": "A matte white supplement bottle on a wooden bedside table at dusk, a folded linen blanket and a glass of water beside it. Soft warm lamplight, shallow depth of field, calm and unhurried. No text overlays.",
    "disclaimers_included": [
      "Consult a health care practitioner prior to use if you are pregnant or breastfeeding.",
      "Some people may experience diarrhoea."
    ],
    "npn_displayed": "NPN 80098765"
  }
}
JSON
```

The response has `"action": "edit"`, `"human_review_status": "edited"`, the new `draft_copy`,
and `status` `dispatched`. The edit path validates the shape and resumes the webhook node; it
does not run the audit again, so the deterministic checks are a promise about this example, not
something the API enforces on a reviewer. `action: "edit"` without `override_copy`, or
`override_copy` with any other action, is a `422`.

### Review: reject

```bash
curl -s -X POST http://localhost:8000/api/v1/workflows/3f1c9a6e-2b7d-4e55-9c0a-8d2f6b1e4a77/review \
  -H "Content-Type: application/json" \
  -d '{"action": "reject", "reviewer_id": "ops-lead", "notes": "Wrong tone for autumn."}' | jq
```

```json
{
  "action": "reject",
  "status": "rejected",
  "next_nodes": [],
  "awaiting_human_review": false,
  "human_review_status": "rejected",
  "human_reviewer_id": "ops-lead",
  "human_reviewer_notes": "Wrong tone for autumn.",
  "webhook_delivery_result": null,
  "...": "..."
}
```

`webhook_delivery_result` stays `null`: the node never ran.

### The receiver and the inbox

With `WEBHOOK_TARGET=self` (the public demo, or locally with `PUBLIC_BASE_URL`), an approval
posts the payload to the service's own `POST /api/v1/webhooks/receiver`. The receiver answers
`503` when `WEBHOOK_SECRET` is not set, recomputes `sha256=<hex>` of HMAC-SHA256 over
`timestamp + "." + delivery_id + "." + body` with that secret, compares it in constant time with
`X-Signature-256`, refuses a timestamp more than 300 seconds away or a header that is not plain
ASCII (`401` either way), and stores the delivery in `webhook_inbox` (a repeated
`X-Delivery-Id` is kept once; since the id is signed, a delivery cannot be replayed under a
fresh one). The payload is `build_payload` in
`app/services/webhook_dispatcher.py`: the event `listing.approved`, the thread, SKU, channels
and brief, the approved `copy`, a `compliance` summary (verdict, risk, revisions, escalation,
audits), the `review` (status, reviewer, notes) and the `inventory` essentials.

```bash
curl -s "http://localhost:8000/api/v1/webhooks/inbox?limit=5" | jq
```

```json
[
  {
    "delivery_id": "9b4c1d2e3f4a5b6c7d8e9f0a1b2c3d4e",
    "thread_id": "3f1c9a6e-2b7d-4e55-9c0a-8d2f6b1e4a77",
    "received_at": "2026-10-15T14:05:02Z",
    "signature_valid": true,
    "payload": {"event": "listing.approved", "thread_id": "3f1c9a6e-2b7d-4e55-9c0a-8d2f6b1e4a77", "sku_code": "NR-MAG-200", "...": "..."}
  }
]
```

To deliver to your own system instead, set `WEBHOOK_TARGET=url` and `WEBHOOK_URL` (https, or
http for a test), share `WEBHOOK_SECRET` with it, and verify the headers the same way (`verify`
in the dispatcher module is the reference implementation). Deliveries are retried on 429, 5xx,
timeouts and connection errors with a 0.5, 1, 2, 4, 8 second backoff, up to
`WEBHOOK_MAX_ATTEMPTS`; a 4xx is final; every attempt is a row in `webhook_audit_log`.

---

## Configuration

All settings live in `app/config.py` and are read from the environment, then `.env`, then
`.env.local` (`.env.example` documents every name). Blank values mean "unset" and the literal
`null` means `None`. The ones you will actually tune:

| Variable | Default | Notes |
|---|---|---|
| `CHECKPOINTER` | `postgres` | `memory` runs without a database and forgets every thread at restart |
| `DATABASE_URL` / `DATABASE_URL_UNPOOLED` | `postgresql://ops:ops@localhost:5432/ops` / unset | psycopg strings; a hosted `postgresql://...?sslmode=require` is accepted as is, quotes and a pasted `psql ` or `DATABASE_URL=` prefix are stripped. The unpooled one wins when both are set |
| `DB_POOL_MIN_SIZE` / `DB_POOL_MAX_SIZE` | `1` / `5` | The app runs its own pool; keep it small on a free database |
| `DB_STARTUP_RETRIES` / `DB_CONNECT_TIMEOUT_S` / `HEALTH_TIMEOUT_S` | `3` / `10` / `5` | Start-up retries while a serverless database resumes; a bad setting is never retried |
| `AUTO_CREATE_SCHEMA` | `true` | Create the tables, upsert the sample SKUs, run the checkpointer's `setup()` at start-up |
| `MAX_REVISIONS` | `3` | Default audit budget when a request names none (0 to 5); see the routing rules for what the number means |
| `SYNTHESIS_TEMPERATURE` / `AUDIT_TEMPERATURE` | `0.2` / `0.0` | Not sent to Claude, which takes `ANTHROPIC_EFFORT` instead |
| `SYNTHESIS_MAX_TOKENS` / `AUDIT_MAX_TOKENS` | `4096` / `4096` | Output caps; a truncated answer is a failed attempt. A model that reasons before it answers spends part of the cap on that |
| `OPENROUTER_REASONING` | `off` | OpenRouter's unified reasoning control, sent to every model of the chain: `default` sends nothing; `off`, `minimal`, `low`, `medium` or `high` is sent, and a model whose endpoint refuses the switch is called without it |
| `LLM_PROVIDER` | `openrouter` | `openai_compatible` (OpenAI, Gemini's compatibility endpoint, Groq: `OPENAI_BASE_URL`, `OPENAI_MODEL`) or `anthropic` (`ANTHROPIC_MODEL`, `ANTHROPIC_EFFORT`) |
| `OPENROUTER_API_KEY` | unset | The only key the free stack needs. Start-up refuses a value with `://` or whitespace in it, so a pasted database string never reaches a provider |
| `OPENROUTER_MODELS` | blank | Blank: automatic, from `DEFAULT_MODEL_CHAIN`. Any value pins the chain |
| `OPENROUTER_FREE_ONLY` | `true` | Refuses any chain entry without `:free`, at start-up and again at call time |
| `OPENROUTER_STRICT_JSON_SCHEMA` | `true` | `false` skips straight to JSON mode |
| `OPENROUTER_REQUESTS_PER_MINUTE` / `OPENROUTER_TIMEOUT_S` | `20` / `60` | Pacing of request starts (0 = none); per-call timeout, also used for the other providers |
| `MODEL_PUBLISHED_URL` / `MODEL_CATALOGUE_CHECK` | this project's `chain.json` / `true` | Where the published chain is read (`null` = never); `false` makes no request at start-up |
| `WEBHOOK_TARGET` | `dry_run` | `url` (needs `WEBHOOK_URL` and `WEBHOOK_SECRET`), `self` (needs `WEBHOOK_SECRET` and `PUBLIC_BASE_URL`, or Render's `RENDER_EXTERNAL_URL`), `dry_run` |
| `WEBHOOK_TIMEOUT_S` / `WEBHOOK_MAX_ATTEMPTS` | `10` / `3` | Per attempt; attempts with backoff |
| `API_KEY` | unset | Inbound `X-API-Key` on the `/api/v1/workflows` routes outside demo mode; not a provider key |
| `MAX_JSON_BODY_BYTES` | `32768` | Every POST, checked before the body is parsed (`413`) |
| `CORS_ALLOW_ORIGINS` | empty | Comma-separated origins (the GitHub Pages site); empty disables CORS headers |
| `ENVIRONMENT` / `LOG_LEVEL` | `development` / `INFO` | `production` hides `/docs` and makes the page's test route refuse non-browser callers |
| `PLAYGROUND_ENABLED` | `true` | Serve `/` and `/playground/*` |
| `DEMO_MODE` / `DEMO_MAX_REVISIONS` | `false` / `2` | The public showcase (next section) |
| `RUN_RATE_LIMIT_PER_MINUTE` / `RUN_DAILY_LIMIT_PER_CLIENT` | `2` / `12` | Per visitor address (demo mode) |
| `RUN_GLOBAL_LIMIT_PER_MINUTE` / `RUN_DAILY_LIMIT` / `RUN_MAX_INFLIGHT` | `6` / `120` / `2` | For everyone together, and load shedding (demo mode) |
| `OPENROUTER_ALLOWANCE_CHECK` / `OPENROUTER_ALLOWANCE_RESERVE` | `true` / `100` | Ask OpenRouter for the day's remaining free requests once a minute, stop at the reserve (demo mode) |
| `TRUST_PROXY_HEADERS` / `CLIENT_IP_HEADER` / `TRUSTED_PROXY_HOPS` | `false` / unset / `1` | Where the visitor address comes from behind a proxy |

Settings that contradict each other fail at start-up with a plain message and without the
value: a `url` target in demo mode, `url` without `WEBHOOK_URL`, `url` or `self` without
`WEBHOOK_SECRET`, a non-free id with `OPENROUTER_FREE_ONLY`, a non-https published URL, a pool
minimum above its maximum.

---

## Public showcase (demo mode)

`DEMO_MODE=true` turns the service into something that can be left on the open internet on a
free account. `render.yaml` sets it.

| | Normal mode | Demo mode |
|---|---|---|
| `POST /api/v1/workflows/start`, `GET .../{thread_id}`, `POST .../review` | Open, or `X-API-Key` when `API_KEY` is set | Open to everyone; `API_KEY` is ignored and the budget decides |
| `max_revisions` | 0 to 5, default `MAX_REVISIONS` | Capped at `DEMO_MAX_REVISIONS` (2 on Render: one redraft, four model calls at most) |
| Webhook | `url`, `self` or `dry_run` | `self` or `dry_run` only: a `url` target is refused at start-up |
| `/docs` | Served | Hidden, because `render.yaml` also sets `ENVIRONMENT=production` |

**The budget** (`app/budget.py`) refuses a run with `429` and a `Retry-After` header when one
visitor address passes 2 runs a minute or 12 a day, everyone together passes 6 a minute or 120 a
day, or OpenRouter's own count of the day's free requests (`GET /key`, read at most once a
minute) would fall to the reserve: 100, or a fifth of the account's daily allowance when that is
smaller. A run is admitted for `2 * (max_revisions + 1)` calls, counted against the reserve
until the next read. Before the budget, a cheaper check sheds load: a third run while two are
being executed gets `503` with `Retry-After: 5`. The allowance rule is the one a restart cannot
erase: a free host forgets everything in memory, OpenRouter does not.

**Who is a visitor.** Behind Render the address comes from `CF-Connecting-IP`, which Cloudflare
overwrites (`CLIENT_IP_HEADER`). The left-most `X-Forwarded-For` entry is never used, because the
client writes it; the fallback is the entry `TRUSTED_PROXY_HOPS` places from the right. IPv6
addresses count per /64 network. With `TRUST_PROXY_HEADERS=false` (the default) only the socket
address is used.

**The self target.** On Render `WEBHOOK_TARGET=self` and `RENDER_EXTERNAL_URL` is set by the
host, so every approved listing is posted to the service's own `/api/v1/webhooks/receiver`,
signed with the `WEBHOOK_SECRET` Render generated. The demo never calls out to a third party,
and visitors can read what arrived on `/api/v1/webhooks/inbox`.

**What protects the OpenRouter account.** The five budget rules above; the per-request cap on
`max_revisions`; the 300-character cap on the brief (so a prompt cannot be padded into a long
one); `OPENROUTER_FREE_ONLY=true`; request pacing at 20 a minute; one worker (`--workers 1` in
`render.yaml`), so the budget and the model chooser have one memory; the model test a visitor
can start from the page, which is admitted through the same budget as a run (for the 12 free
requests it sends at most), limited to four page-started tests a day for everyone together,
fifteen minutes apart, and refused when the day's allowance is low or unknown (in production
the route also wants the page's `Accept: application/json`, which keeps a plain form or a link
from starting one and is not a protection on its own); and the key itself, which is typed once
into Render and lives nowhere in the repository. Error responses never echo an upstream body:
clients get the provider's name, "temporary" or "permanent", and an incident id that matches
the log line.

---

## Tests

The suite is hermetic: no network, no PostgreSQL, no API key. `.github/workflows/tests.yml` runs
`ruff check`, `ruff format --check` and `python -m pytest -q` on every push to `main` and every
pull request, and `pyproject.toml` sets `asyncio_mode = "auto"` with a two-minute
`faulthandler_timeout` so a hanging test says where it stood.

```bash
pytest
```

What is covered:

* `tests/conftest.py` makes the suite hermetic before any other test module loads: every
  environment variable `Settings` reads is removed (a developer shell full of real keys cannot
  change the outcome; the suite runs as `ENVIRONMENT=test`), both `httpx` transports raise as
  soon as a request would leave the process and the real `AsyncOpenAI` and `AsyncAnthropic`
  constructors fail loudly, and the model-chain cache is pointed at a temporary file. It
  provides the vocabulary the other files share: `make_settings` (nothing read from the
  environment), `FakeChat` (a `StructuredChat` whose answers are scripted per schema, so
  "exactly three drafts were requested" is proven, not assumed), a compliant and a
  non-compliant draft for `NR-MAG-200`, `build_components` (the production topology over
  `InMemorySaver`, the in-memory inventory and webhook store, a recording dispatcher) and
  `build_stand` (the real FastAPI app over those, driven with `httpx.ASGITransport`).
* `tests/test_graph_flow.py` - the graph end to end: the happy path to the gate and through
  approval, the trace in node order, the two nodes' own temperatures and token budgets, the
  self-correction cycle (a lenient fake auditor waved the bad draft through and the
  deterministic checks did not), the revision prompt's content (previous draft, violations,
  notes, a static system prompt), the count of deterministic findings, escalation at
  `max_revisions` and at zero, the early ends (unknown SKU, a code the tool refuses, out of
  stock, invalid request) without a model call, upper-casing, and provider failures that leave
  the thread checkpointed at the failed node.
* `tests/test_hitl_interrupt.py` - the pause before `webhook_dispatch`, approve resuming and
  running the node exactly once (a second resume sends nothing), edit dispatching the
  reviewer's copy, reject ending the thread without running the node (the two together in
  `test_human_edit_and_reject`, on two threads of one graph, and each on its own in finer
  tests), the node's own refusal when nothing was decided, and the state history showing the
  interrupt checkpoint.
* `tests/test_api.py` - the routes over the real compiled graph: `/livez`, `/health` (and its
  degraded form), the SKU and rule listings, start to the gate, read-back, `404`s, the terminal
  `201`s, approve, edit (and its `422`s), reject then `409`, validation that never echoes the
  payload, nine malformed starts, the body limit, demo mode (the per-visitor budget, the
  revision cap, load shedding, the routes open without the key) and the key outside it, two
  overlapping approves dispatching once, a stored draft that no longer validates staying
  readable, the webhook receiver (valid, forged, unsigned, stale, non-object, no secret, a
  header that is not ASCII, a replay under a fresh delivery id) and the `self` target end to
  end through the app itself, provider failures mapped to `503`/`502` without the upstream
  text (the failed thread named in a header and its state), a checkpointer failure during
  start mapped to `503` and a fault of the service to a JSON `500` with an incident id, a
  draft that overflows its cap after typography fixes, a failure during the review resume
  that leaves the thread reviewable, the playground config, the model-test button admitted
  through the demo budget, the database start with and without `AUTO_CREATE_SCHEMA`, hidden
  docs in production, and CORS.
* `tests/test_compliance_rules.py` - every prohibited pattern fires on its own sentence and none
  fires on any SKU's licence text, one finding per pattern per field, the fields that are not
  scanned, the NPN checks, the dosage check (wrong amount, wrong unit, an amount far from the
  ingredient, a multi-strain total, a standardization percentage, and a regression where the
  context window once cut a number in half), the risk statements after normalization and
  against paraphrases, `overrule_model_findings` (a quote with a word the licence does not
  contain survives whatever the fix says, a wording-only finding on a restated claim is
  dropped, a mechanical finding stands only where the code found a problem), `merge_audit`
  (findings win, deduplication, risk escalation) and the revision instructions.
* `tests/test_webhook.py` - signing and verification (the timestamp and every byte of the body,
  the 300-second skew, constant-time comparison), the canonical encoding, the payload shape,
  and the dispatcher over `httpx.MockTransport`: a 200, retries on 500 and 429, a final 400,
  timeouts and connection errors until the attempts are spent, the backoff table, a broken
  audit sink that never breaks a delivery, the dry run (signed, recorded, nothing sent), the
  `self` target's address and its failure without one.
* `tests/test_anthropic_client.py` - the Claude client against a stub `messages.create`: the
  schema in `output_config`, no temperature, the joined system prompts, refusal, truncation,
  empty and invalid answers, fenced JSON, the key rejection, the SDK error mapping, and that the
  real SDK client cannot be constructed inside the suite.

* `tests/test_llm_client.py` - `OpenAICompatibleChain` against a stub `chat.completions.create`:
  every failure a free model produces in practice (rate limits, in-band upstream errors,
  withdrawal, timeouts, truncation, empty or invalid output) sends the call to the next model,
  strict schema falls back to JSON mode, and a rejected key stops the chain at once.
* `tests/test_budget.py` - the five budget rules in order, driven by a clock the test moves by
  hand and a fake OpenRouter allowance reader.
* `tests/test_config.py` - what blank and pasted values mean, the cross-field rules, and that
  the committed `.env.example` agrees with the settings and holds no secret.
* `tests/test_prompts.py` - the untrusted brief is fenced with a nonce the brief cannot guess,
  a revision call carries the previous draft and the violations, and no prompt holds an em dash.
* `tests/test_seed.py` - the committed `sql/init_inventory.sql` equals `render_sql()`, the four
  products validate, and the sample briefs are sound.
* `tests/test_models_cases.py`, `test_models_rules.py`, `test_models_testing.py`,
  `test_models_catalogue.py` - the three cases and their grading, the verdicts and the
  repair-never-rank chain as pure functions, one test of the models over a scripted chain
  factory, and the catalogue lookups over a fake `fetch`. Nothing calls a model or the network.
* `tests/test_models_chooser.py` - the chooser inside a running service, over a scripted world
  (the published file, the catalogue, the allowance and the models' answers as values): a blank
  setting means automatic and anything else says why not; start-up adopts a newer published
  chain, ignores an unusable one, never raises and never waits long, and reads nothing when the
  chain is pinned, the key is missing or the check is off; after a failed request a withdrawn
  model is dropped at once and a repair test fills its place, a rate-limited model that is
  still listed changes nothing, a burst of failures leads to one look a minute, the published
  list repairs the chain at no cost but is not taken when it names the chain that just failed,
  and a refused repair is owed and tried again; the button's test, its refusals (one test at a
  time, a quarter of an hour apart, the allowance until midnight), a test cut off by shutdown,
  and the picture the page draws from.
* `tests/test_models_published.py` - what the scheduled job writes is what a service reads
  (every check in `parse`, one bad id discarding the whole file, the verdicts worked out again
  by this service's rules), when a test is due, the job's exit codes and the lines that begin
  `Failing this run on purpose:`, and the workflow file itself (the paths a push starts it for,
  the two-job trust split).
* `tests/test_readme.py` - holds this document to the code it quotes: the status table above
  names every `STATUS_` constant in `app/graph/state.py` and nothing else, the default chain is
  written in the order `app/config.py` ships, the defaults in the configuration table are the
  settings' own, and every test file is described in this section.
* `tests/test_site.py` - holds the published pages to their sources: both pages declare a
  content security policy and use no inline code, the project page may load the demo's icon
  and nothing else from outside, no page uses an em dash, every asset and code link exists, the
  live demo's address is written once and well formed, the shared assets are the same bytes on
  both pages, `render.yaml` agrees with the settings, `.gitignore` protects the local secrets,
  and, once `site/recorded/runs.json` exists, every recorded run is a real API response (the
  schemas validate it, the prompts and the stack match the code, each scenario ended as its
  story says, no secret is in it). `.github/workflows/pages.yml` runs this file before
  publishing anything.

**Live PostgreSQL tests.** `tests/test_integration_db.py` is skipped unless `TEST_DATABASE_URL`
is set. It works inside its own schema (`ops_test`), created and dropped by the test, so it
never touches the tables a service uses: it opens the real `Database` (pool, DDL, seed,
checkpointer setup), reads the inventory, round-trips the webhook audit log and the inbox, and
runs the compiled graph with a small fake model over the real `AsyncPostgresSaver`: a run pauses
before the webhook node, a second `Database` and graph (as another process would build after a
restart) find the paused thread in PostgreSQL, approve it and resume it to a dispatch.

```bash
TEST_DATABASE_URL=postgresql://ops:ops@localhost:5432/ops pytest tests/test_integration_db.py -v
```

No Docker available? `python scripts/run_live_db_tests.py` installs `pgserver` on demand (it
ships PostgreSQL binaries inside the virtualenv, data in the git-ignored `.pgdata/`), starts an
embedded server, runs that file against it and stops it again; extra arguments go to pytest.

---

## Tradeoffs and limits

* **One driver instead of two.** The inventory, the webhook tables and the LangGraph saver all
  go through one psycopg pool opened with `autocommit=True`, `row_factory=dict_row` and
  `prepare_threshold=0` (what `AsyncPostgresSaver` needs, and what a transaction-mode pooler
  tolerates). It keeps the connection count to one small pool on a free database. The price is
  that the inventory queries run in autocommit too; they are read-only, so nothing is lost.
* **A static breakpoint instead of `interrupt()`.** The pause is `interrupt_before` on the
  webhook node, not a dynamic `interrupt()` call inside a node with `Command(resume=...)`. It is
  simpler to reason about and to test (the pause is a property of the graph, not of a code
  path), and the reviewer's decision is ordinary state. What it cannot do is pause in the middle
  of a node, or ask the human a question whose answer the node then continues with.
* **Deterministic checks are regex, not NLP.** They prove presence and absence of specific
  wordings and numbers. A disease claim phrased without any listed verb or disease word, or a
  benefit the licence never granted, is the model auditor's to catch, and a model can miss it.
* **Free models and strictness.** A strict auditor on a weak model produces false violations,
  and the loop then spends revisions on them; a lenient one lets claims through to the human.
  The prompts lean strict ("when in doubt, it is a violation") because a human reads every draft
  anyway.
* **One worker.** The demo budget, the in-flight counter and the model chooser live in process
  memory. A second worker would have a second budget. `render.yaml` and the `Dockerfile` run
  `--workers 1` on purpose.
* **In-memory budget.** A restart forgets today's counts; OpenRouter's own counter (rule 5) is
  the backstop, and the per-visitor fairness rules are only as good as the proxy header.
* **No listing endpoint.** Threads are addressed by id; there is no `GET /api/v1/workflows`.
  The inbox is the only list, and it is open.
* **Resuming a failed node is manual.** When a model call fails on every model of the chain,
  `start` answers `502` or `503` with an incident id and the thread stays at its last
  checkpoint. The thread id travels in the `X-Thread-Id` header of that error response and in
  the log line that carries the incident id, and the thread's `error` says what happened, so
  `GET /api/v1/workflows/{thread_id}` finds it; the review route refuses it (`409`) because it
  is not waiting at the gate. Nothing retries it later; `ainvoke(None, config)` from a Python
  shell would.
* **The rules are a demo, not legal advice.** The six rules, the sample claims and the NPNs are
  written for this project. Real NHP advertising review needs the actual licence and a person
  who knows the regulations.

---

## Project layout

```text
autonomous-operation-marketing/
├── docker-compose.yml           # postgres:16 (+ optional api profile), seeds sql/init_inventory.sql
├── Dockerfile                   # production image: non-root, one worker, HEALTHCHECK on /livez
├── render.yaml                  # Render Blueprint: one free web service in demo mode
├── requirements.txt             # exact versions; pgserver named as the optional live-test server
├── pyproject.toml               # pytest (asyncio auto mode) + ruff (100 columns, E F I UP B)
├── .env.example                 # every setting, documented; no secret
├── .python-version              # 3.12: the Python the host and the GitHub jobs run
├── LICENSE                      # MIT
├── sql/init_inventory.sql       # generated by `python -m app.seed`; a test keeps it current
├── benchmarks/                  # the measurements behind DEFAULT_MODEL_CHAIN: every call, saved
├── app/
│   ├── config.py                # pydantic-settings, DEFAULT_MODEL_CHAIN, DEFAULT_PUBLISHED_URL
│   ├── schemas.py               # the two model output schemas, the API models, SkuRecord
│   ├── errors.py                # PipelineError, ConfigurationError, ProviderError, InventoryError
│   ├── seed.py                  # the four sample SKUs and the SQL renderer
│   ├── database.py              # one psycopg pool: inventory, webhook tables, AsyncPostgresSaver
│   ├── budget.py                # demo mode: who may run a pipeline, how often, and the day's allowance
│   ├── main.py                  # FastAPI routes, body guard, auth, playground, lifespan
│   ├── compliance/rules.py      # the six rules, the prohibited patterns, the deterministic checks
│   ├── graph/
│   │   ├── state.py             # PipelineState, reducers, STATUS_ constants
│   │   ├── prompts.py           # the two system prompts, nonce-tagged user messages
│   │   ├── tools.py             # lookup_sku / list_skus with JSON-schema specs
│   │   ├── nodes.py             # the six nodes and the three routers
│   │   └── builder.py           # build_graph(nodes, checkpointer), interrupt_before, topology()
│   ├── services/
│   │   ├── llm_client.py        # OpenAICompatibleChain: strict schema -> JSON mode, fallback, pacing
│   │   ├── anthropic_client.py  # Claude through the official SDK, structured output
│   │   └── webhook_dispatcher.py # HMAC signing, retries, audit log, the three targets
│   ├── models/                  # the free-model chain that looks after itself
│   │   ├── cache.py             # the chosen chain in memory and in model_cache.json (never committed)
│   │   ├── cases.py             # the three cases a candidate must pass, built from the sample SKU
│   │   ├── catalogue.py         # asks OpenRouter which models are listed; reports only
│   │   ├── rules.py             # what a test concludes: verdicts and the chain after it
│   │   ├── testing.py           # one test of the free models: send the cases, record, decide
│   │   ├── chooser.py           # the only module that changes the chain
│   │   └── published.py         # chain.json: format, checks, and the scheduled job's command
│   └── playground/static/       # the browser playground served at / (no build step)
│       ├── index.html           # run, review, the models section, the graph, rules, prompts
│       └── assets/              # app.js, app.css, theme.js, favicon.svg
├── site/                        # the project page, published on GitHub Pages
│   ├── index.html               # the one place that holds the live demo's address
│   ├── assets/                  # site.js, site.css, theme.js, favicon.svg
│   └── recorded/                # runs.json goes here: real runs, saved as the service returned them
├── docs/DEPLOY.md               # putting it online, step by step
├── scripts/
│   ├── openrouter_smoke.py      # live check of the key and the chain (two free requests)
│   ├── record_runs.py           # drives the real app and writes site/recorded/runs.json
│   ├── run_live_db_tests.py     # the live-database tests on embedded PostgreSQL; --serve keeps it up
│   └── run_local_demo.py        # the showcase on your machine, in demo mode
├── tests/
│   ├── conftest.py              # environment scrubbed, network closed, fakes and builders
│   ├── test_graph_flow.py       # the graph end to end: happy path, cycle, escalation, early ends
│   ├── test_hitl_interrupt.py   # the pause, approve, edit, reject, the node's refusal
│   ├── test_api.py              # the routes over the real graph, demo mode, the receiver
│   ├── test_compliance_rules.py # the deterministic checks and the merge
│   ├── test_webhook.py          # signing, payload, retries, the three targets
│   ├── test_llm_client.py       # the OpenAI-compatible chain: fallback, JSON mode, the key
│   ├── test_anthropic_client.py # the Claude client against a stub SDK
│   ├── test_budget.py           # the five budget rules against a hand-moved clock
│   ├── test_config.py           # settings parsing, cross-field rules, .env.example
│   ├── test_prompts.py          # the fenced brief, revision evidence, no em dash
│   ├── test_seed.py             # the SQL file matches the generator
│   ├── test_models_*.py         # cases, rules, testing, catalogue, chooser, published: no network
│   ├── test_readme.py           # the README's tables against the code they quote
│   ├── test_site.py             # the pages and the recorded runs against their sources
│   └── test_integration_db.py   # opt-in: the real checkpointer persists a paused thread
└── .github/workflows/           # tests.yml ("Tests"), pages.yml ("Project page"),
                                 # model-check.yml ("Model check")
```

---

## License

The code in this repository is released under the [MIT License](LICENSE): anyone may use,
change, share and sell it, as long as copies keep the copyright notice and the license text. It
comes as is, without warranty.

The packages in `requirements.txt` are permissive (MIT, BSD or Apache 2.0) with one exception:
`psycopg`, `psycopg-pool` and `psycopg-binary` are distributed under the LGPL 3.0. This project
imports them unchanged as a library, which the LGPL allows without placing conditions on this
repository's own code; modifying psycopg itself would be different. Free models answer under
their providers' terms, and free endpoints may keep what they are sent.
