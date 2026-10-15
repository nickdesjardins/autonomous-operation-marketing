// The playground: one form that starts a pipeline run, the run rendered node by node, the
// review panel that decides what leaves, and the picture of the model chain. Every value on the
// page comes from the service's own answers (GET /playground/config, POST /api/v1/workflows/start,
// POST /api/v1/workflows/{id}/review, GET /api/v1/webhooks/inbox, GET /playground/models).
(function () {
  "use strict";

  var $ = function (id) { return document.getElementById(id); };
  var config = null; // the last answer to GET /playground/config
  var current = null; // the last PipelineStatusResponse shown
  var nodeTitles = {}; // node id -> title, from config.topology
  var nodeKinds = {}; // node id -> kind, from config.topology

  // ------------------------------------------------------------------ theme ---
  var toggle = $("theme-toggle");
  if (toggle) {
    toggle.addEventListener("click", function () {
      var root = document.documentElement;
      var dark = root.dataset.theme === "dark" ||
        (!root.dataset.theme && window.matchMedia("(prefers-color-scheme: dark)").matches);
      root.dataset.theme = dark ? "light" : "dark";
      try { localStorage.setItem("theme", root.dataset.theme); } catch (e) { /* ignore */ }
    });
  }

  // ---------------------------------------------------------------- helpers ---
  function el(tag, cls, text) {
    var node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined && text !== null) node.textContent = String(text);
    return node;
  }
  function clear(node) { node.textContent = ""; }
  function setStatus(node, text, kind) {
    node.textContent = text || "";
    if (kind) { node.dataset.kind = kind; } else { delete node.dataset.kind; }
  }
  function ms(value) {
    if (value === null || value === undefined) return "n/a";
    return value >= 1000 ? (value / 1000).toFixed(1) + " s" : Math.round(value) + " ms";
  }
  function yesNo(value) { return value ? "yes" : "no"; }
  function lines(text) {
    return text.split("\n").map(function (s) { return s.trim(); }).filter(function (s) { return s.length > 0; });
  }
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
  // The review is not a node of the graph (the API writes it into the trace), so it has no title
  // in the topology.
  function title(node) { return nodeTitles[node] || (node === "human_review" ? "Human review" : node); }

  // The status values come from app/graph/state.py; each gets a sentence and a tone.
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
    awaiting_human_review: ["Paused: waiting for your decision", "warn"],
    approved: ["Approved", "ok"],
    edited: ["Edited", "ok"],
    rejected: ["Rejected: nothing left the system", "bad"],
    dispatched: ["Dispatched: the signed webhook was delivered", "ok"],
    dispatch_failed: ["Dispatch failed after every attempt", "bad"],
    dispatch_refused: ["Dispatch refused: no human approval on record", "bad"]
  };
  function statusWords(status) { return (STATUS_WORDS[status] || [status, "muted"])[0]; }
  function statusTone(status) { return (STATUS_WORDS[status] || [status, "muted"])[1]; }

  // ------------------------------------------------------------------ errors ---
  // The API never echoes a request back; its "detail" is a sentence (or, for 422, a list of
  // schema errors), so the page can show it as it is.
  function explainError(status, body, headers) {
    var detail = body && body.detail;
    var text = typeof detail === "string" ? detail : "";
    if (Array.isArray(detail)) {
      text = detail.map(function (e) {
        var where = (e.loc || []).filter(function (part) { return part !== "body"; }).join(".");
        return (where ? where + ": " : "") + (e.msg || e.type || "invalid");
      }).join("; ");
    }
    var retry = headers && headers.get ? headers.get("Retry-After") : null;
    if (status === 429) return (text || "Too many runs right now.") + (retry ? " Try again in " + retry + " s." : "");
    if (status === 503) return (text || "The service is temporarily unavailable.") + (retry ? " Try again in " + retry + " s." : "");
    if (status === 502) return text || "The model provider failed permanently for this request.";
    if (status === 422) return "The request was rejected by the schema: " + (text || "check the fields") + ".";
    if (status === 409) return text || "This thread is not waiting for a review.";
    if (status === 404) return text || "Unknown thread.";
    if (status === 413) return text || "The request body is too large.";
    if (status === 401) return text || "This route needs the service's API key (X-API-Key).";
    if (status === 403) return text || "This route is not open from here.";
    return text || ("Request failed (HTTP " + status + ").");
  }
  function readJson(response) {
    return response.json().catch(function () { return null; }).then(function (body) {
      return { status: response.status, body: body, headers: response.headers };
    });
  }

  // ------------------------------------------------------------------ config ---
  function loadConfig() {
    return fetch("/playground/config", { headers: { "Accept": "application/json" } })
      .then(function (r) { if (!r.ok) throw new Error("config " + r.status); return r.json(); })
      .then(function (cfg) {
        config = cfg;
        if (cfg.demo_mode) $("demo-note").hidden = false;
        (cfg.topology && cfg.topology.nodes || []).forEach(function (n) { nodeTitles[n.id] = n.title; nodeKinds[n.id] = n.kind; });

        var sku = $("sku");
        clear(sku);
        cfg.skus.forEach(function (s) {
          var opt = el("option", null, s.sku_code + ": " + s.product_name + (s.stock_quantity > 0 ? " (" + s.stock_quantity + " in stock)" : " (out of stock)"));
          opt.value = s.sku_code;
          sku.appendChild(opt);
        });

        var channels = $("channels");
        clear(channels);
        cfg.channels.forEach(function (c) {
          var label = el("label");
          var box = el("input");
          box.type = "checkbox";
          box.name = "channel";
          box.value = c;
          box.checked = true;
          label.appendChild(box);
          label.appendChild(el("span", null, c));
          channels.appendChild(label);
        });

        var revisions = $("max-revisions");
        clear(revisions);
        for (var n = 0; n <= cfg.max_revisions; n += 1) {
          var opt = el("option", null, n === 0 ? "0: audit once, never redraft" : n + (n === 1 ? " loop" : " loops"));
          opt.value = String(n);
          if (n === cfg.max_revisions) opt.selected = true;
          revisions.appendChild(opt);
        }

        var chips = $("chips");
        clear(chips);
        cfg.sample_angles.forEach(function (angle) {
          var b = el("button", "chip", angle);
          b.type = "button";
          b.addEventListener("click", function () { $("brief").value = angle; $("brief").focus(); });
          chips.appendChild(b);
        });

        var limits = cfg.limits || {};
        if (cfg.demo_mode && limits.per_client_per_minute) {
          var left = limits.free_requests_remaining_today;
          $("limits").textContent = "Limits: " + limits.per_client_per_minute + " runs per minute and "
            + limits.per_client_per_day + " per day for each visitor, " + limits.per_day + " per day for everyone"
            + (left !== null && left !== undefined ? " · free-model requests left today: " + left : "") + ".";
        } else {
          $("limits").textContent = "One run is a draft and an audit per attempt, at most max(1, loops) attempts: up to 2 × max(1, loops) model calls.";
        }

        var llm = cfg.llm || {};
        $("foot-provider").textContent = llm.configured
          ? "Models: " + llm.provider + " (" + llm.mode + ") · " + (llm.models || []).join(", then ")
            + " · webhook target: " + (cfg.webhook && cfg.webhook.target)
          : "Models: " + llm.provider + " is not configured (no API key); runs answer 503";

        renderRules(cfg.rules || []);
        $("synthesis-prompt").textContent = cfg.synthesis_prompt || "";
        $("audit-prompt").textContent = cfg.audit_prompt || "";
        renderTopology(cfg.topology);
      })
      .catch(function () {
        setStatus($("status"), "The service is waking up or its database is not ready; refresh in a moment.", "error");
      });
  }

  function renderRules(rules) {
    var box = $("rules-cards");
    clear(box);
    rules.forEach(function (rule) {
      var card = el("article", "card");
      var sev = el("span", "sev", rule.severity + " severity");
      sev.dataset.sev = rule.severity;
      card.appendChild(sev);
      card.appendChild(el("h3", null, rule.title));
      card.appendChild(el("p", null, rule.summary));
      card.appendChild(el("p", "small", rule.id));
      box.appendChild(card);
    });
  }

  function renderTopology(topology) {
    if (!topology || !topology.edges) return;
    var list = $("topology");
    clear(list);
    topology.edges.forEach(function (edge) {
      list.appendChild(el("li", null, title(edge.from) + " → " + title(edge.to) + (edge.when ? " (" + edge.when + ")" : "")));
    });
    var interrupt = (topology.interrupt_before || []).map(title).join(", ");
    if (interrupt) list.appendChild(el("li", null, "interrupt_before: " + interrupt));
    $("topology-box").hidden = false;
  }

  // ---------------------------------------------------------------- the run ---
  function selectedChannels() {
    return Array.prototype.map.call(document.querySelectorAll('#channels input:checked'), function (box) { return box.value; });
  }

  $("run-form").addEventListener("submit", function (event) {
    event.preventDefault();
    var status = $("status");
    var brief = $("brief").value.trim();
    var channels = selectedChannels();
    if (!channels.length) { setStatus(status, "Pick at least one channel.", "error"); return; }
    if (brief.length < 3) { setStatus(status, "The brief needs at least three characters.", "error"); return; }
    var payload = {
      sku_code: $("sku").value,
      target_channels: channels,
      campaign_angle: brief,
      max_revisions: Number($("max-revisions").value)
    };
    var button = $("run-btn");
    button.disabled = true;
    $("result").hidden = true;
    $("review-result").hidden = true;
    setStatus(status, "Running: inventory lookup, a draft, an audit, and a redraft for every violation… free models take 10 to 60 seconds per call.");
    var started = performance.now();
    fetch("/api/v1/workflows/start", {
      method: "POST",
      headers: { "Content-Type": "application/json", "Accept": "application/json" },
      body: JSON.stringify(payload)
    })
      .then(readJson)
      .then(function (res) {
        if (res.status === 201 && res.body) {
          var seconds = Math.round((performance.now() - started) / 100) / 10;
          setStatus(status, "Answered in " + seconds + " s round trip: " + statusWords(res.body.status) + ".", res.body.awaiting_human_review ? "ok" : undefined);
          render(res.body);
        } else {
          setStatus(status, explainError(res.status, res.body, res.headers), "error");
        }
      })
      .catch(function () { setStatus(status, "The request did not reach the service. Is it awake?", "error"); })
      .finally(function () { button.disabled = false; });
  });

  function render(body) {
    current = body;
    var result = $("result");
    result.hidden = false;
    var badge = $("result-badge");
    badge.dataset.tone = statusTone(body.status);
    badge.textContent = statusWords(body.status);
    var models = {};
    (body.trace || []).forEach(function (e) { if (e.detail && e.detail.model) models[e.detail.model] = true; });
    $("result-meta").textContent = "thread " + body.thread_id + " · " + body.revision_count + " of " + body.max_revisions + " loops used"
      + (body.requires_human_escalation ? " · escalated" : "")
      + (Object.keys(models).length ? " · " + Object.keys(models).join(", ") : "");
    var error = $("result-error");
    error.hidden = !body.error;
    error.textContent = body.error || "";

    renderPath(body);
    renderInventory(body.inventory_data);
    renderTimeline(body.trace || []);
    renderAudits(body.audit_history || []);
    renderInstructions(body.revision_instructions || []);
    renderDraft(body.draft_copy);
    renderReviewPanel(body);
    lightGraph(body);
    $("result-json").textContent = JSON.stringify(body, null, 2);
    result.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function renderPath(body) {
    var box = $("result-path");
    clear(box);
    var previous = null;
    (body.trace || []).forEach(function (event) {
      var cycle = event.node === "copy_synthesis_agent" && previous === "compliance_validator";
      if (previous !== null) box.appendChild(el("span", "path__arrow" + (cycle ? " path__arrow--cycle" : ""), cycle ? "↺" : "→"));
      var chip = el("span", "path__node", title(event.node) + (cycle ? " (revision " + event.detail.revision + ")" : ""));
      if (nodeKinds[event.node] === "llm" || nodeKinds[event.node] === "llm+checks") chip.className += " path__node--llm";
      if (cycle) chip.className += " path__node--cycle";
      if (event.node === "human_review") chip.className += " path__node--pause";
      box.appendChild(chip);
      previous = event.node;
    });
    if (body.awaiting_human_review) {
      box.appendChild(el("span", "path__arrow", "→"));
      box.appendChild(el("span", "path__node path__node--pause", "paused before Webhook Dispatch: your decision"));
    } else if (previous !== null && body.next_nodes && !body.next_nodes.length) {
      box.appendChild(el("span", "path__arrow", "→"));
      box.appendChild(el("span", "path__node path__node--end", "END"));
    }
  }

  function renderInventory(data) {
    var list = $("inventory");
    clear(list);
    if (!data) { fact(list, "Inventory", "not loaded: the run ended before the lookup, or the lookup found nothing"); return; }
    fact(list, "Product", data.product_name + " (" + data.sku_code + ")");
    fact(list, "NPN", data.npn_number);
    fact(list, "Form and serving", data.dosage_form + "; " + data.serving_size);
    fact(list, "Stock", data.stock_quantity);
    fact(list, "Active ingredients", (data.active_ingredients || []).map(function (i) { return i.name + ": " + i.amount + " " + i.unit + " per " + i.per; }));
    fact(list, "Authorized claims", data.authorized_claims || []);
    fact(list, "Risk information", (data.risk_information || []).map(function (r) { return r.text + (r.mandatory ? " (mandatory)" : ""); }));
    fact(list, "Contraindications", data.contraindications);
    fact(list, "Brand notes", data.brand_notes);
  }

  // What each node's trace detail says, in words. Keys are the ones app/graph/nodes.py writes.
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
        if (d.problems) parts.push("problems: " + d.problems.join("; "));
        break;
      case "copy_synthesis_agent":
        parts.push(d.revision > 0 ? "revision " + d.revision + " of the draft" : "first draft");
        if (d.model) parts.push("model " + d.model);
        if (d.fallback_triggered) parts.push("fallback triggered: an earlier model of the chain did not answer");
        if (d.json_mode) parts.push("plain JSON mode (the model rejected the strict schema)");
        if (d.attempts && d.attempts.length > 1) parts.push(d.attempts.length + " attempts");
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
        if (d.review) parts.push("review status " + d.review);
        break;
      case "human_review":
        parts.push("action: " + d.action + " by " + d.reviewer_id);
        break;
      default:
        Object.keys(d).forEach(function (key) { parts.push(key + ": " + JSON.stringify(d[key])); });
    }
    return parts.join(" · ");
  }

  function renderTimeline(trace) {
    var list = $("timeline");
    clear(list);
    var previous = null;
    trace.forEach(function (event) {
      if (event.node === "copy_synthesis_agent" && previous === "compliance_validator") {
        list.appendChild(el("li", "tl--loop", "↺ Back to the synthesis agent: revision " + (event.detail && event.detail.revision) + ". The cycle is the point: the critic's findings become the next draft's instructions."));
      }
      var li = el("li");
      var kind = nodeKinds[event.node];
      if (kind === "llm" || kind === "llm+checks") li.className = "tl--llm";
      if (event.node === "human_approval_gate") li.className = "tl--pause";
      if (event.node === "human_review") li.className = "tl--human";
      var head = el("div", "tl__head");
      head.appendChild(el("span", "tl__node", event.node === "human_review" ? "Human review" : title(event.node)));
      head.appendChild(el("span", "tl__ms", (event.ms === null || event.ms === undefined ? "" : ms(event.ms) + " · ") + event.at));
      li.appendChild(head);
      li.appendChild(el("p", "tl__detail", detailWords(event)));
      list.appendChild(li);
      previous = event.node;
    });
    if (!trace.length) list.appendChild(el("li", "muted", "No node has run."));
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
    var box = $("audits");
    clear(box);
    if (!history.length) { box.appendChild(el("p", "muted", "No audit ran: the run ended before a draft was written.")); return; }
    history.forEach(function (entry) {
      var card = el("section", "audit");
      var head = el("div", "audit__head");
      var badge = el("span", "badge", entry.is_compliant ? "compliant" : "not compliant");
      badge.dataset.tone = entry.is_compliant ? "ok" : (entry.risk_level === "high" ? "bad" : "warn");
      head.appendChild(badge);
      head.appendChild(el("span", "result__meta", "audit of revision " + entry.revision + " · risk " + entry.risk_level
        + " · " + entry.violations.length + " violation" + (entry.violations.length === 1 ? "" : "s")
        + " (" + entry.deterministic_findings + " found by the deterministic checks)"
        + (entry.model ? " · " + entry.model : "") + " · " + entry.checked_at));
      card.appendChild(head);
      card.appendChild(el("p", "audit__summary", entry.revision_summary));
      if (entry.violations.length) card.appendChild(violationsTable(entry.violations));
      box.appendChild(card);
    });
  }

  function renderInstructions(instructions) {
    var block = $("instructions-block");
    var box = $("instructions");
    clear(box);
    block.hidden = !instructions.length;
    instructions.forEach(function (text) {
      box.appendChild(el("pre", "instr", text));
    });
  }

  function renderDraft(draft) {
    var block = $("draft-block");
    var box = $("draft");
    clear(box);
    block.hidden = !draft;
    if (!draft) return;
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
  }

  function lightGraph(body) {
    var ran = {};
    (body.trace || []).forEach(function (e) { ran[e.node] = true; });
    Array.prototype.forEach.call(document.querySelectorAll("#graph .node"), function (node) {
      node.classList.toggle("is-run", Boolean(ran[node.dataset.node]));
    });
    $("graph-cycle").classList.toggle("is-run", body.revision_count > 0);
    $("graph-interrupt").classList.toggle("is-paused", Boolean(body.awaiting_human_review));
  }

  // ----------------------------------------------------------------- review ---
  function renderReviewPanel(body) {
    var panel = $("review");
    panel.hidden = !body.awaiting_human_review;
    if (!body.awaiting_human_review) return;
    $("review-lede").textContent = body.requires_human_escalation
      ? "The self-correction budget is spent and violations remain, so the draft was escalated to you. Approving ships it as it is; editing lets you fix it yourself; rejecting ends the thread with nothing sent."
      : "The draft passed the audit and the graph stopped on a checkpoint before the webhook node. Approve to dispatch it, edit it first, or reject it.";
    showEdit(false);
    setStatus($("review-status"), "");
  }

  function showEdit(on) {
    $("edit-fields").hidden = !on;
    $("edit-send-btn").hidden = !on;
    $("edit-cancel-btn").hidden = !on;
    $("approve-btn").hidden = on;
    $("edit-btn").hidden = on;
    $("reject-btn").hidden = on;
    if (on && current && current.draft_copy) {
      var d = current.draft_copy;
      $("edit-title").value = d.web_title;
      $("edit-description").value = d.web_description_html;
      $("edit-bullets").value = d.marketplace_bullets.join("\n");
      $("edit-social").value = d.social_post_copy;
      $("edit-image").value = d.image_generation_prompt;
      $("edit-disclaimers").value = d.disclaimers_included.join("\n");
      $("edit-npn").value = d.npn_displayed;
      $("edit-title").focus();
    }
  }

  function overrideCopy() {
    return {
      web_title: $("edit-title").value.trim(),
      web_description_html: $("edit-description").value.trim(),
      marketplace_bullets: lines($("edit-bullets").value),
      social_post_copy: $("edit-social").value.trim(),
      image_generation_prompt: $("edit-image").value.trim(),
      disclaimers_included: lines($("edit-disclaimers").value),
      npn_displayed: $("edit-npn").value.trim()
    };
  }

  function review(action) {
    if (!current) return;
    var status = $("review-status");
    var reviewer = $("reviewer-id").value.trim();
    if (!reviewer) { setStatus(status, "A reviewer id is required: the decision is recorded with it.", "error"); return; }
    var payload = { action: action, reviewer_id: reviewer };
    var notes = $("reviewer-notes").value.trim();
    if (notes) payload.notes = notes;
    if (action === "edit") {
      payload.override_copy = overrideCopy();
      if (payload.override_copy.marketplace_bullets.length !== 5) {
        setStatus(status, "The marketplace bullets must be exactly five, one per line.", "error");
        return;
      }
    }
    var buttons = Array.prototype.slice.call(document.querySelectorAll("#review-form button"));
    buttons.forEach(function (b) { b.disabled = true; });
    setStatus(status, action === "reject"
      ? "Rejecting: the thread ends and the webhook node never runs."
      : "Resuming the graph from its checkpoint: the webhook node signs and delivers the payload…");
    fetch("/api/v1/workflows/" + encodeURIComponent(current.thread_id) + "/review", {
      method: "POST",
      headers: { "Content-Type": "application/json", "Accept": "application/json" },
      body: JSON.stringify(payload)
    })
      .then(readJson)
      .then(function (res) {
        if (res.status === 200 && res.body) {
          setStatus(status, "");
          render(res.body);
          renderReviewResult(res.body);
        } else {
          setStatus(status, explainError(res.status, res.body, res.headers), "error");
        }
      })
      .catch(function () { setStatus(status, "The request did not reach the service. Is it awake?", "error"); })
      .finally(function () { buttons.forEach(function (b) { b.disabled = false; }); });
  }

  $("approve-btn").addEventListener("click", function () { review("approve"); });
  $("reject-btn").addEventListener("click", function () { review("reject"); });
  $("edit-btn").addEventListener("click", function () { showEdit(true); });
  $("edit-cancel-btn").addEventListener("click", function () { showEdit(false); });
  $("edit-send-btn").addEventListener("click", function () { review("edit"); });
  $("review-form").addEventListener("submit", function (event) { event.preventDefault(); });

  function renderReviewResult(body) {
    var section = $("review-result");
    section.hidden = false;
    var line = $("review-line");
    var who = body.human_reviewer_id ? " by " + body.human_reviewer_id : "";
    var notes = body.human_reviewer_notes ? " Notes: " + body.human_reviewer_notes : "";
    line.textContent = "Review: " + body.human_review_status + who + ". " + statusWords(body.status) + "." + notes;
    var facts = $("delivery");
    clear(facts);
    var inbox = $("inbox-line");
    inbox.textContent = "";
    var result = body.webhook_delivery_result;
    if (!result) {
      facts.appendChild(el("dt", null, "Webhook"));
      facts.appendChild(el("dd", null, body.human_review_status === "rejected"
        ? "Nothing was sent. The rejection was written as if the webhook node had run, so the thread is over and the node never executes."
        : "No delivery result on record."));
      return;
    }
    fact(facts, "Mode", result.mode);
    fact(facts, "Delivered", yesNo(result.delivered) + (result.mode === "dry_run" ? " (dry run: recorded, nothing sent; signed when WEBHOOK_SECRET is set)" : ""));
    fact(facts, "Attempts", result.attempts);
    fact(facts, "HTTP status", result.status_code);
    fact(facts, "Delivery id", result.delivery_id);
    fact(facts, "Receiver host", result.url_host);
    // The state holds a fingerprint of the signature (the start of it), never the whole header.
    fact(facts, "Signature header", result.signature ? (result.signature.length > 22 ? result.signature.slice(0, 19) + "..." : result.signature) : null);
    fact(facts, "Duration", result.duration_ms !== undefined ? ms(result.duration_ms) : null);
    fact(facts, "Error", result.error);
    if (config && config.webhook && config.webhook.target === "self" && result.delivered) {
      inbox.textContent = "Asking the receiver what it got…";
      fetch("/api/v1/webhooks/inbox?limit=50", { headers: { "Accept": "application/json" } })
        .then(function (r) { if (!r.ok) throw new Error("inbox " + r.status); return r.json(); })
        .then(function (deliveries) {
          var mine = deliveries.filter(function (d) { return d.delivery_id === result.delivery_id || d.thread_id === body.thread_id; })[0];
          if (!mine) { inbox.textContent = "The receiver's inbox does not list this delivery yet."; return; }
          var payload = mine.payload || {};
          inbox.textContent = "The receiver (this service's own /api/v1/webhooks/receiver) stored delivery " + mine.delivery_id
            + " at " + mine.received_at + ": signature valid " + yesNo(mine.signature_valid)
            + ", event " + payload.event + ", review status " + (payload.review && payload.review.status)
            + ", " + (payload.compliance && payload.compliance.audits) + " audit(s) in the trail.";
        })
        .catch(function () { inbox.textContent = "The receiver's inbox could not be read."; });
    }
  }

  // ----------------------------------------------------------------- models ---
  // GET /playground/models answers with codes, dates and lists (app/models/chooser.py); the
  // words for each are here, so one place holds the wording.
  var MODE_WORDS = {
    automatic: "Automatic: the service looks after the chain itself.",
    pinned: "Pinned: OPENROUTER_MODELS names the chain, so the service never tests or replaces a model.",
    no_key: "No OpenRouter key is set, so nothing is tested or replaced.",
    not_openrouter: "The base URL is not OpenRouter's, so nothing is tested or replaced."
  };
  var ORIGIN_WORDS = {
    pinned: function () { return "Named in the settings."; },
    "default": function (o) { return "The tested default" + (o.tested ? ", measured on " + o.tested.slice(0, 10) : "") + "."; },
    published: function (o) { return "From this project's scheduled test" + (o.tested ? " of " + o.tested.slice(0, 10) : "") + (o.age_days >= 14 ? ", " + o.age_days + " days ago" : "") + "."; },
    own: function (o) {
      var why = o.why || {};
      if (why.code === "gone") return "Chosen by the service itself: OpenRouter no longer offers " + (why.models || []).join(", ") + ".";
      if (why.code === "first_answer") return "A stand-in answers until a test decides: " + (why.models || []).join(", ") + ".";
      if (why.code === "test") return "Chosen by the service's own test" + (o.tested ? ", which ended on " + o.tested : "") + ".";
      return "Chosen by the service itself.";
    }
  };
  var PUBLISHED_WORDS = {
    off: "The published list is never read here (MODEL_PUBLISHED_URL is null).",
    used: "The published list is in use.",
    older: "The published list is not newer than the chain in use.",
    same: "The published list names the very chain that just failed, so it was not taken.",
    missing: "No list has been published yet.",
    unreachable: "The published list could not be reached.",
    invalid: "The published list could not be read.",
    foreign: "The published list was made by another version of the test."
  };
  var REFUSAL_WORDS = {
    running: "The service is busy: looking into a failed request, or testing.",
    pause: "A test ran a short while ago.",
    page_limit: "This button has started its tests for today.",
    list_unreadable: "OpenRouter's list of free models could not be read just now.",
    nothing_to_test: "OpenRouter lists no other free model to test right now.",
    allowance: "Too little of today's free allowance is left.",
    allowance_unknown: "The service cannot check its free allowance right now.",
    pinned: "The chain is pinned.",
    no_key: "No key is set.",
    not_openrouter: "The base URL is not OpenRouter's."
  };
  var VERDICT_TONE = { passed: "ok", slow: "warn", failed: "bad", not_judged: "muted" };
  var VERDICT_WORDS = { passed: "passed", slow: "right but slow", failed: "failed", not_judged: "not judged" };
  var WHY_WORDS = {
    all_right: "every case answered right",
    wrong_answers: "a wrong answer",
    unusable_answers: "answers that did not fit the schema",
    unanswered: "calls went unanswered",
    no_answers: "no answer at all"
  };
  var PLACING_WORDS = {
    kept: "keeps its place",
    dropped: "left the chain",
    added: "added to the chain",
    no_place: "passed, but no place was free"
  };
  var pollTimer = null;

  function renderModels(p) {
    var box = $("models-body");
    clear(box);
    var linesBox = el("div", "models__lines");
    linesBox.appendChild(el("p", null, MODE_WORDS[p.mode] || ("Mode: " + p.mode)));
    var chain = el("ol", "chain");
    (p.chain || []).forEach(function (model, index) {
      var li = el("li");
      li.appendChild(el("span", "place", String(index + 1) + "."));
      li.appendChild(el("code", null, model));
      if ((p.gone || []).indexOf(model) >= 0) li.appendChild(el("span", "gone", "no longer offered by OpenRouter"));
      chain.appendChild(li);
    });
    if (!(p.chain || []).length) chain.appendChild(el("li", "muted", "The chain is empty."));
    box.appendChild(linesBox);
    box.appendChild(chain);
    var origin = p.origin || {};
    var originWords = ORIGIN_WORDS[origin.kind];
    if (originWords) linesBox.appendChild(el("p", null, "Origin: " + originWords(origin)));
    if (p.published && PUBLISHED_WORDS[p.published.state]) {
      linesBox.appendChild(el("p", null, PUBLISHED_WORDS[p.published.state] + (p.published.tested ? " (tested " + p.published.tested + ")" : "")));
    }
    if (p.doing) linesBox.appendChild(el("p", null, "Right now: " + p.doing + "."));
    var test = p.test || p.last;
    if (test) {
      var after = el("div");
      after.appendChild(el("h3", null, (p.test ? "The test that is running" : "The last test") + (test.id ? " (" + test.id + ")" : "")));
      var words = [];
      if (test.by) words.push("started by: " + test.by);
      if (test.started) words.push("started " + test.started);
      if (test.finished) words.push("finished " + test.finished);
      if (test.outcome) words.push("outcome: " + test.outcome);
      if (test.requests !== undefined) words.push(test.requests + " free requests spent");
      if (test.cases) words.push(test.cases.length + " cases per model");
      after.appendChild(el("p", "muted small", words.join(" · ")));
      var rows = test.models || [];
      if (rows.length) {
        var wrap = el("div", "tablewrap");
        var table = el("table");
        var hr = el("tr");
        ["Model", "Verdict", "Why", "Right"].forEach(function (h) { hr.appendChild(el("th", null, h)); });
        var thead = el("thead");
        thead.appendChild(hr);
        table.appendChild(thead);
        var tbody = el("tbody");
        rows.forEach(function (row) {
          var tr = el("tr");
          tr.appendChild(el("td", null, row.model));
          var td = el("td");
          var verdict = el("span", "verdict", VERDICT_WORDS[row.verdict] || (row.verdict || "being tested"));
          verdict.dataset.tone = VERDICT_TONE[row.verdict] || "muted";
          td.appendChild(verdict);
          tr.appendChild(td);
          tr.appendChild(el("td", null, (WHY_WORDS[row.why] || row.why || "") + (row.placing ? " (" + (PLACING_WORDS[row.placing] || row.placing) + ")" : "")));
          tr.appendChild(el("td", "num", (row.correct !== undefined ? row.correct : "") + (row.calls !== undefined ? " of " + row.calls : "")));
          tbody.appendChild(tr);
        });
        table.appendChild(tbody);
        wrap.appendChild(table);
        after.appendChild(wrap);
      }
      box.appendChild(after);
    }
    var button = $("models-test");
    var line = $("models-line");
    var b = p.button || {};
    button.hidden = !b.offered;
    button.disabled = !b.ready;
    if (b.offered) {
      line.textContent = b.ready
        ? "A test sends each free model this service's own three cases (a draft and two audits) from the demo's free allowance, and takes a few minutes."
        : (REFUSAL_WORDS[b.reason] || "The button is not available right now.") + (b.again_at ? " Possible again at " + b.again_at + "." : "");
    } else {
      line.textContent = "";
    }
  }

  function loadModels() {
    return fetch("/playground/models", { headers: { "Accept": "application/json" } })
      .then(function (r) { if (!r.ok) throw new Error("models " + r.status); return r.json(); })
      .then(function (p) {
        renderModels(p);
        if (p.busy && !pollTimer) {
          pollTimer = setTimeout(function () { pollTimer = null; loadModels(); }, 20000);
        }
      })
      .catch(function () {
        var box = $("models-body");
        clear(box);
        box.appendChild(el("p", "muted", "The picture of the chain could not be read (the service may be starting)."));
      });
  }

  $("models-test").addEventListener("click", function () {
    var button = $("models-test");
    var line = $("models-line");
    button.disabled = true;
    line.textContent = "Asking the service to start a test…";
    fetch("/playground/models/test", { method: "POST", headers: { "Accept": "application/json" } })
      .then(readJson)
      .then(function (res) {
        if (res.status === 202) {
          line.textContent = "A test started. This panel refreshes every 20 seconds while it runs.";
          setTimeout(loadModels, 3000);
        } else if (res.status === 409 && res.body) {
          line.textContent = (REFUSAL_WORDS[res.body.reason] || "No test can start now.") + (res.body.again_at ? " Possible again at " + res.body.again_at + "." : "");
          setTimeout(loadModels, 3000);
        } else {
          line.textContent = explainError(res.status, res.body, res.headers);
          button.disabled = false;
        }
      })
      .catch(function () { line.textContent = "The request did not reach the service."; button.disabled = false; });
  });

  loadConfig();
  loadModels();
})();
