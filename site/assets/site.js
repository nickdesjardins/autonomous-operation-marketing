// The project page: shows the recorded runs from recorded/runs.json, one tab per scenario with
// the whole story (request, inventory, every audit, the human decision, the webhook and what the
// receiver stored), computes the tiles from the record, and wakes the live demo.
//
// The record is what scripts/record_runs.py writes: {recorded_at, stack, synthesis_prompt,
// audit_prompt, runs: [{id, why, expect, request, start_response, review_request,
// review_response, inbox}]}, where start_response and review_response are the API bodies as
// returned (PipelineStatusResponse / HumanReviewResponse) and inbox is the receiver's
// WebhookDelivery for that thread, or null.
(function () {
  "use strict";

  var root = document.documentElement;
  var $ = function (id) { return document.getElementById(id); };
  function el(tag, cls, text) {
    var node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }

  // ------------------------------------------------------------------ theme ---
  var toggle = $("theme-toggle");
  if (toggle) {
    toggle.addEventListener("click", function () {
      var dark = root.dataset.theme === "dark" ||
        (!root.dataset.theme && window.matchMedia("(prefers-color-scheme: dark)").matches);
      root.dataset.theme = dark ? "light" : "dark";
      try { localStorage.setItem("theme", root.dataset.theme); } catch (e) { /* ignore */ }
    });
  }

  // ---------------------------------------------------------------- helpers ---
  var NODE_TITLES = {
    supervisor: "Supervisor",
    inventory_sql_agent: "Inventory SQL Agent",
    copy_synthesis_agent: "Copy Synthesis Agent",
    compliance_validator: "Compliance Validator",
    human_approval_gate: "Human Approval Gate",
    webhook_dispatch: "Webhook Dispatch",
    human_review: "Human review"
  };
  var LLM_NODES = { copy_synthesis_agent: true, compliance_validator: true };
  var STATUS_WORDS = {
    started: ["Started", "muted"],
    invalid_request: ["Invalid request: the supervisor ended the run", "bad"],
    inventory_loaded: ["Inventory loaded", "muted"],
    sku_not_found: ["SKU not found: the run ended before any model was called", "bad"],
    out_of_stock: ["Out of stock: the run ended before any model was called", "warn"],
    drafted: ["Drafted", "muted"],
    compliant: ["Compliant", "ok"],
    revision_requested: ["Revision requested", "warn"],
    escalated_to_human: ["Escalated to a human", "warn"],
    awaiting_human_review: ["Paused: waiting for a human", "warn"],
    approved: ["Approved", "ok"],
    edited: ["Edited", "ok"],
    rejected: ["Rejected: nothing left the system", "bad"],
    dispatched: ["Dispatched: the signed webhook was delivered", "ok"],
    dispatch_failed: ["Dispatch failed after every attempt", "bad"],
    dispatch_refused: ["Dispatch refused: no human approval on record", "bad"]
  };
  function statusWords(status) { return (STATUS_WORDS[status] || [status, "muted"])[0]; }
  function statusTone(status) { return (STATUS_WORDS[status] || [status, "muted"])[1]; }
  function title(node) { return NODE_TITLES[node] || node; }
  function ms(value) {
    if (value === null || value === undefined) return "n/a";
    return value >= 1000 ? (value / 1000).toFixed(1) + " s" : Math.round(value) + " ms";
  }
  function median(values) {
    if (!values.length) return null;
    var sorted = values.slice().sort(function (a, b) { return a - b; });
    var middle = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
  }
  function yesNo(value) { return value ? "yes" : "no"; }
  function fact(list, term, value) {
    if (value === null || value === undefined || value === "") return;
    list.appendChild(el("dt", null, term));
    var dd = el("dd");
    if (Array.isArray(value)) {
      var ul = el("ul");
      value.forEach(function (item) { ul.appendChild(el("li", null, item)); });
      dd.appendChild(ul);
    } else {
      dd.textContent = String(value);
    }
    list.appendChild(dd);
  }
  function finalResponse(run) { return run.review_response || run.start_response; }
  function nodeTime(body) {
    return (body.trace || []).reduce(function (sum, e) { return sum + (typeof e.ms === "number" ? e.ms : 0); }, 0);
  }

  // -------------------------------------------------------------- the story ---
  function detailWords(event) {
    var d = event.detail || {};
    var parts = [];
    switch (event.node) {
      case "supervisor":
        parts.push("outcome: " + d.outcome);
        if (d.route) parts.push("routed to " + title(d.route));
        if (d.max_revisions !== undefined) parts.push("revision budget " + d.max_revisions);
        if (d.problems) parts.push("problems: " + d.problems.join("; "));
        break;
      case "inventory_sql_agent":
        parts.push("tool " + d.tool + ": " + d.outcome);
        if (d.product_name) parts.push(d.product_name);
        if (d.stock_quantity !== undefined) parts.push("stock " + d.stock_quantity);
        if (d.authorized_claims !== undefined) parts.push(d.authorized_claims + " authorized claims, " + d.ingredients + " ingredients");
        break;
      case "copy_synthesis_agent":
        parts.push(d.revision > 0 ? "revision " + d.revision + " of the draft" : "first draft");
        if (d.model) parts.push("model " + d.model);
        if (d.fallback_triggered) parts.push("fallback triggered: an earlier model of the chain did not answer");
        if (d.json_mode) parts.push("plain JSON mode (the model rejected the strict schema)");
        break;
      case "compliance_validator":
        parts.push("decision: " + d.decision);
        parts.push("risk " + d.risk_level);
        parts.push(d.violations + " violation" + (d.violations === 1 ? "" : "s") + ", " + d.deterministic_findings + " from the deterministic checks");
        if (d.llm_said_compliant && d.violations > 0) parts.push("the model said compliant; the checks disagreed");
        if (d.model) parts.push("model " + d.model);
        if (d.fallback_triggered) parts.push("fallback triggered");
        break;
      case "human_approval_gate":
        parts.push(d.reason || "waiting");
        if (d.escalated) parts.push("escalated: violations remain");
        break;
      case "webhook_dispatch":
        parts.push("outcome: " + d.outcome);
        if (d.mode) parts.push("target " + d.mode);
        if (d.attempts !== undefined) parts.push(d.attempts + " attempt" + (d.attempts === 1 ? "" : "s"));
        if (d.status_code) parts.push("HTTP " + d.status_code);
        if (d.delivery_id) parts.push("delivery " + d.delivery_id);
        break;
      case "human_review":
        parts.push("action: " + d.action + " by " + d.reviewer_id);
        break;
      default:
        Object.keys(d).forEach(function (key) { parts.push(key + ": " + JSON.stringify(d[key])); });
    }
    return parts.join(" · ");
  }

  function renderPath(body) {
    var box = el("div", "path");
    box.setAttribute("aria-label", "The nodes this run visited, in order");
    var previous = null;
    (body.trace || []).forEach(function (event) {
      var cycle = event.node === "copy_synthesis_agent" && previous === "compliance_validator";
      if (previous !== null) box.appendChild(el("span", "path__arrow" + (cycle ? " path__arrow--cycle" : ""), cycle ? "↺" : "→"));
      var chip = el("span", "path__node", title(event.node) + (cycle ? " (revision " + event.detail.revision + ")" : ""));
      if (LLM_NODES[event.node]) chip.className += " path__node--llm";
      if (cycle) chip.className += " path__node--cycle";
      if (event.node === "human_review") chip.className += " path__node--pause";
      box.appendChild(chip);
      previous = event.node;
    });
    if (body.awaiting_human_review) {
      box.appendChild(el("span", "path__arrow", "→"));
      box.appendChild(el("span", "path__node path__node--pause", "paused before Webhook Dispatch"));
    } else if (previous !== null) {
      box.appendChild(el("span", "path__arrow", "→"));
      box.appendChild(el("span", "path__node path__node--end", "END"));
    }
    return box;
  }

  function renderTimeline(trace) {
    var list = el("ol", "timeline");
    var previous = null;
    trace.forEach(function (event) {
      if (event.node === "copy_synthesis_agent" && previous === "compliance_validator") {
        list.appendChild(el("li", "tl--loop", "↺ Back to the synthesis agent: revision " + (event.detail && event.detail.revision) + ". The critic's findings became the next draft's instructions."));
      }
      var li = el("li");
      if (LLM_NODES[event.node]) li.className = "tl--llm";
      if (event.node === "human_approval_gate") li.className = "tl--pause";
      if (event.node === "human_review") li.className = "tl--human";
      var head = el("div", "tl__head");
      head.appendChild(el("span", "tl__node", title(event.node)));
      head.appendChild(el("span", "tl__ms", (event.ms === null || event.ms === undefined ? "" : ms(event.ms) + " · ") + event.at));
      li.appendChild(head);
      li.appendChild(el("p", "tl__detail", detailWords(event)));
      list.appendChild(li);
      previous = event.node;
    });
    return list;
  }

  function violationsTable(violations) {
    var wrap = el("div", "tablewrap");
    var table = el("table");
    var thead = el("thead");
    var hr = el("tr");
    ["Field", "Rule", "Text", "Required fix"].forEach(function (h) { hr.appendChild(el("th", null, h)); });
    thead.appendChild(hr);
    table.appendChild(thead);
    var tbody = el("tbody");
    violations.forEach(function (v) {
      var tr = el("tr");
      tr.appendChild(el("td", null, v.field_name));
      tr.appendChild(el("td", null, v.regulation_rule));
      tr.appendChild(el("td", null, v.violating_text));
      tr.appendChild(el("td", null, v.required_fix));
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    wrap.appendChild(table);
    return wrap;
  }

  function renderAudits(history) {
    var box = el("div");
    if (!history.length) { box.appendChild(el("p", "muted", "No audit ran: the run ended before a draft was written.")); return box; }
    history.forEach(function (entry) {
      var card = el("section", "audit");
      var head = el("div", "audit__head");
      var badge = el("span", "badge", entry.is_compliant ? "compliant" : "not compliant");
      badge.dataset.tone = entry.is_compliant ? "ok" : (entry.risk_level === "high" ? "bad" : "warn");
      head.appendChild(badge);
      head.appendChild(el("span", "run__meta", "audit of revision " + entry.revision + " · risk " + entry.risk_level
        + " · " + entry.violations.length + " violation" + (entry.violations.length === 1 ? "" : "s")
        + " (" + entry.deterministic_findings + " found by the deterministic checks)"
        + (entry.model ? " · " + entry.model : "")));
      card.appendChild(head);
      card.appendChild(el("p", "audit__summary", entry.revision_summary));
      if (entry.violations.length) card.appendChild(violationsTable(entry.violations));
      box.appendChild(card);
    });
    return box;
  }

  function renderDraft(draft, cls) {
    var box = el("div", "draft" + (cls ? " " + cls : ""));
    box.appendChild(el("p", "draft__title", draft.web_title));
    box.appendChild(el("p", "label", "Product page (the HTML the model wrote, shown as text)"));
    box.appendChild(el("pre", "draft__text", draft.web_description_html));
    box.appendChild(el("p", "label", "Marketplace bullets"));
    var ul = el("ul");
    draft.marketplace_bullets.forEach(function (b) { ul.appendChild(el("li", null, b)); });
    box.appendChild(ul);
    box.appendChild(el("p", "label", "Social post"));
    box.appendChild(el("pre", "draft__text", draft.social_post_copy));
    box.appendChild(el("p", "label", "Image generation prompt"));
    box.appendChild(el("pre", "draft__text", draft.image_generation_prompt));
    box.appendChild(el("p", "label", "Disclaimers included"));
    var dl = el("ul");
    draft.disclaimers_included.forEach(function (d) { dl.appendChild(el("li", null, d)); });
    box.appendChild(dl);
    box.appendChild(el("p", "label", "NPN displayed: " + draft.npn_displayed));
    return box;
  }

  var DRAFT_FIELDS = ["web_title", "web_description_html", "marketplace_bullets", "social_post_copy", "image_generation_prompt", "disclaimers_included", "npn_displayed"];
  function changedFields(before, after) {
    return DRAFT_FIELDS.filter(function (key) { return JSON.stringify(before[key]) !== JSON.stringify(after[key]); });
  }

  function renderRun(run) {
    var card = $("run");
    card.textContent = "";
    var start = run.start_response;
    var last = finalResponse(run);
    card.appendChild(el("p", "run__why", run.why));

    var head = el("div", "run__head");
    var badge = el("span", "badge", statusWords(last.status));
    badge.dataset.tone = statusTone(last.status);
    head.appendChild(badge);
    var models = {};
    (last.trace || []).forEach(function (e) { if (e.detail && e.detail.model) models[e.detail.model] = true; });
    head.appendChild(el("span", "run__meta", run.request.sku_code + " · " + last.revision_count + " of " + last.max_revisions + " loops used"
      + (last.requires_human_escalation ? " · escalated" : "")
      + " · " + ms(nodeTime(last)) + " in the nodes"
      + (Object.keys(models).length ? " · " + Object.keys(models).join(", ") : "")));
    card.appendChild(head);
    if (run.expect && run.expect !== last.status) {
      card.appendChild(el("p", "run__expect", "This scenario expected the run to end as \"" + run.expect + "\"; it ended as \"" + last.status + "\"."));
    }
    card.appendChild(renderPath(last));

    card.appendChild(el("h3", null, "The request"));
    var request = el("dl", "facts");
    fact(request, "SKU", run.request.sku_code);
    fact(request, "Channels", (run.request.target_channels || []).join(", "));
    fact(request, "Campaign brief", run.request.campaign_angle);
    fact(request, "Self-correction loops allowed", run.request.max_revisions);
    card.appendChild(request);

    card.appendChild(el("h3", null, "Inventory facts the agents were given"));
    var inventory = el("dl", "facts");
    var data = start.inventory_data;
    if (!data) {
      fact(inventory, "Inventory", "not loaded: the run ended before the lookup, or the lookup found nothing");
    } else {
      fact(inventory, "Product", data.product_name + " (" + data.sku_code + ")");
      fact(inventory, "NPN", data.npn_number);
      fact(inventory, "Form and serving", data.dosage_form + "; " + data.serving_size);
      fact(inventory, "Stock", data.stock_quantity);
      fact(inventory, "Active ingredients", (data.active_ingredients || []).map(function (i) { return i.name + ": " + i.amount + " " + i.unit + " per " + i.per; }));
      fact(inventory, "Authorized claims", data.authorized_claims || []);
      fact(inventory, "Risk information", (data.risk_information || []).map(function (r) { return r.text + (r.mandatory ? " (mandatory)" : ""); }));
      fact(inventory, "Contraindications", data.contraindications);
    }
    card.appendChild(inventory);
    if (start.error) card.appendChild(el("p", "run__note", "The service said: " + start.error));

    card.appendChild(el("h3", null, "Timeline: every node that ran"));
    card.appendChild(renderTimeline(last.trace || []));

    card.appendChild(el("h3", null, "Compliance audits"));
    card.appendChild(renderAudits(last.audit_history || []));

    if ((last.revision_instructions || []).length) {
      card.appendChild(el("h3", null, "Revision instructions the synthesis agent received"));
      last.revision_instructions.forEach(function (text) { card.appendChild(el("pre", "instr", text)); });
    }

    if (start.draft_copy) {
      card.appendChild(el("h3", null, last.revision_count > 0 ? "The draft after " + last.revision_count + " revision" + (last.revision_count === 1 ? "" : "s") : "The draft, right first time"));
      card.appendChild(renderDraft(start.draft_copy));
    }

    card.appendChild(el("h3", null, "The human decision"));
    var review = run.review_request;
    if (!review) {
      card.appendChild(el("p", "run__note", start.awaiting_human_review
        ? "The thread is still waiting: no decision was recorded."
        : "None: the run ended before the gate, so nothing was written and nothing could be sent."));
    } else {
      var decision = el("dl", "facts");
      fact(decision, "Action", review.action);
      fact(decision, "Reviewer", review.reviewer_id);
      fact(decision, "Notes", review.notes);
      card.appendChild(decision);
      if (review.action === "edit" && review.override_copy && start.draft_copy) {
        var changed = changedFields(start.draft_copy, review.override_copy);
        card.appendChild(el("h4", null, "What the human changed: " + (changed.length ? changed.join(", ") : "nothing")));
        if (changed.indexOf("web_title") >= 0) {
          var titles = el("dl", "facts");
          fact(titles, "The model wrote", start.draft_copy.web_title);
          fact(titles, "The human sent", review.override_copy.web_title);
          card.appendChild(titles);
        }
        if (changed.some(function (key) { return key !== "web_title"; })) card.appendChild(renderDraft(review.override_copy, "edited"));
      }
    }

    card.appendChild(el("h3", null, "The webhook"));
    var delivery = run.review_response && run.review_response.webhook_delivery_result;
    if (!delivery) {
      card.appendChild(el("p", "run__note", review && review.action === "reject"
        ? "Nothing was sent. The rejection was written to the thread as if the webhook node had run, so the node never executed."
        : "No delivery: the thread never reached the webhook node."));
    } else {
      var facts = el("dl", "facts");
      fact(facts, "Mode", delivery.mode);
      fact(facts, "Delivered", yesNo(delivery.delivered));
      fact(facts, "Attempts", delivery.attempts);
      fact(facts, "HTTP status", delivery.status_code);
      fact(facts, "Delivery id", delivery.delivery_id);
      fact(facts, "Receiver host", delivery.url_host);
      fact(facts, "Signature header", delivery.signature ? (delivery.signature.length > 22 ? delivery.signature.slice(0, 19) + "..." : delivery.signature) : null);
      fact(facts, "Error", delivery.error);
      card.appendChild(facts);
      if (run.inbox) {
        var payload = run.inbox.payload || {};
        card.appendChild(el("p", "run__note", "The receiver (the service's own /api/v1/webhooks/receiver) stored delivery " + run.inbox.delivery_id
          + " at " + run.inbox.received_at + ": signature valid " + yesNo(run.inbox.signature_valid)
          + ", event " + payload.event + ", review status " + (payload.review && payload.review.status)
          + ", " + (payload.compliance && payload.compliance.audits) + " audit(s) in the trail."));
      } else {
        card.appendChild(el("p", "run__note", "The receiver's inbox holds no entry for this thread."));
      }
    }
  }

  // -------------------------------------------------------------- the record ---
  function showRecord(record) {
    var runs = record.runs || [];
    if (!runs.length) return;
    var tabs = $("run-tabs");
    tabs.textContent = "";
    runs.forEach(function (run, index) {
      var tab = el("button", "tab", run.id + ": " + run.request.sku_code);
      tab.type = "button";
      tab.setAttribute("role", "tab");
      tab.setAttribute("aria-selected", index === 0 ? "true" : "false");
      tab.addEventListener("click", function () {
        Array.prototype.forEach.call(tabs.children, function (t) { t.setAttribute("aria-selected", "false"); });
        tab.setAttribute("aria-selected", "true");
        renderRun(run);
      });
      tabs.appendChild(tab);
    });
    renderRun(runs[0]);

    var audits = 0, violations = 0, deterministic = 0, calls = 0;
    runs.forEach(function (run) {
      var last = finalResponse(run);
      (last.audit_history || []).forEach(function (entry) {
        audits += 1;
        violations += entry.violations.length;
        deterministic += entry.deterministic_findings || 0;
      });
      (last.trace || []).forEach(function (e) { if (e.detail && e.detail.model) calls += 1; });
    });
    $("tile-runs").textContent = String(runs.length);
    $("tile-audits").textContent = String(audits);
    $("tile-violations").textContent = String(violations);
    $("tile-deterministic").textContent = String(deterministic);
    $("tile-time").textContent = ms(median(runs.map(function (run) { return nodeTime(finalResponse(run)); })));
    $("tiles").hidden = false;
    $("runs-note").hidden = false;

    var day = record.recorded_at ? record.recorded_at.slice(0, 10) : "";
    $("runs-meta").textContent = "Real runs of the service, saved as the API returned them" + (day ? " on " + day : "")
      + ": " + runs.length + " scenarios, " + calls + " model calls.";
    $("foot-recorded").textContent = day ? "Runs recorded " + day : "";

    if (record.synthesis_prompt && record.audit_prompt) {
      $("synthesis-prompt").textContent = record.synthesis_prompt;
      $("audit-prompt").textContent = record.audit_prompt;
      $("prompts").hidden = false;
    }
    var stack = record.stack || {};
    if (stack.chain && stack.chain.length) {
      $("stack-models").textContent = stack.provider === "openrouter"
        ? "Free OpenRouter models, tried in order: " + stack.chain.join(", then ")
        : "Provider " + stack.provider + ": " + stack.chain.join(", then ");
    }
    if (stack.checkpointer) {
      $("stack-database").textContent = "PostgreSQL on Neon (CHECKPOINTER=" + stack.checkpointer + "), one psycopg pool shared by the inventory, the webhook tables and the LangGraph saver";
    }
    if (stack.webhook_target) {
      $("stack-webhook").textContent = "HMAC-SHA256 signed, WEBHOOK_TARGET=" + stack.webhook_target
        + (stack.webhook_target === "self" ? ": delivered to the service's own receiver, which verifies the signature and keeps the payload" : "")
        + (stack.max_revisions !== undefined ? "; recorded with MAX_REVISIONS=" + stack.max_revisions : "");
    }
  }

  fetch("recorded/runs.json", { headers: { "Accept": "application/json" } })
    .then(function (r) { if (!r.ok) throw new Error("runs " + r.status); return r.json(); })
    .then(showRecord)
    .catch(function () { /* the fallback sentence in the page stays */ });

  // --------------------------------------------------------------- live demo ---
  // The demo sleeps on a free host after 15 minutes without a request; the next request wakes
  // it in about a minute. It lives at another address, so this page may not read its answers.
  // What it can do is load the demo's icon as a picture: "load" fires only when the service
  // itself sends the icon, never for the host's own "waking up" page.
  var link = $("demo-link");
  var words = $("demo-words");
  var chip = $("demo-chip");
  if (!link || !words) return;

  var address;
  try { address = new URL(link.href); } catch (e) { address = null; }
  if (!address || address.protocol !== "https:" || /\.invalid$/.test(address.hostname)) {
    root.dataset.demo = "unset";
    // Not a link until the demo exists: no dead click, and nothing to reach with the keyboard.
    link.removeAttribute("href");
    link.setAttribute("aria-disabled", "true");
    words.textContent = "The live demo is not online yet. This page shows recorded runs instead.";
    return;
  }
  if (chip) { chip.href = address.href; chip.hidden = false; }

  var ICON = new URL("/playground/assets/favicon.svg", address).href;
  var RETRY_MS = 4000;
  var GIVE_UP_MS = 180000;
  var startedAt = Date.now();
  var done = false;

  function say(state, text) { root.dataset.demo = state; words.textContent = text; }

  function ask() {
    if (done) return;
    var image = new Image();
    var settled = false;
    var timer = setTimeout(function () { if (!settled) { settled = true; again(); } }, 20000);
    image.onload = function () {
      if (settled) return;
      settled = true; done = true; clearTimeout(timer);
      say("awake", "The live demo is awake. Run the pipeline on a sample SKU and decide what ships.");
    };
    image.onerror = function () { if (settled) return; settled = true; clearTimeout(timer); again(); };
    image.src = ICON + "?t=" + Date.now();
  }
  function again() {
    var waited = Date.now() - startedAt;
    if (waited > GIVE_UP_MS) {
      done = true;
      say("asleep", "The live demo has not answered for three minutes. The recorded runs below show what it does.");
      return;
    }
    if (waited > 4000) {
      say("waking", "Waking the live demo. It runs on free hosting and sleeps between visits, so this takes about a minute.");
    }
    setTimeout(ask, RETRY_MS);
  }
  ask();
})();
