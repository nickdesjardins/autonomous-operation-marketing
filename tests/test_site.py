"""The published pages are checked against their sources, so they cannot quietly go stale.

Covers the project page (``site/``, served by GitHub Pages), the playground (served by the API at
``/``), the recorded runs when they exist, and the deployment files that have to agree with the
settings. Everything here is hermetic: no network, no database, no model.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from app.config import Settings
from app.graph.builder import NODE_ORDER
from app.graph.prompts import AUDIT_SYSTEM_PROMPT, SYNTHESIS_SYSTEM_PROMPT
from app.schemas import (
    HumanReviewRequest,
    HumanReviewResponse,
    PipelineStartRequest,
    PipelineStatusResponse,
)

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"
PLAYGROUND = ROOT / "app" / "playground" / "static"
RECORD = SITE / "recorded" / "runs.json"
PAGES = [SITE / "index.html", PLAYGROUND / "index.html"]

#: The six scenarios scripts/record_runs.py records, by id.
SCENARIOS = {"happy-path", "self-correction", "escalation", "edit", "reject", "out-of-stock"}
RUN_KEYS = {
    "id",
    "why",
    "expect",
    "request",
    "start_response",
    "review_request",
    "review_response",
    "inbox",
}
SECRET_MARKERS = ("sk-or-", "neon.tech", "postgresql://", "Bearer ")
TRACE_NODES = set(NODE_ORDER) | {"human_review"}


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ------------------------------------------------------------------------- static pages


@pytest.mark.parametrize("page", PAGES, ids=["site", "playground"])
def test_pages_declare_a_policy_and_use_no_inline_code(page: Path) -> None:
    html = _read(page)
    csp = re.search(r'http-equiv="Content-Security-Policy" content="([^"]+)"', html)
    assert csp, "every page declares a content security policy"
    policy = csp.group(1)
    for directive in (
        "default-src 'none'",
        "script-src 'self'",
        "style-src 'self'",
        "connect-src 'self'",
    ):
        assert directive in policy, directive
    assert "unsafe-inline" not in policy and "unsafe-eval" not in policy
    assert not re.search(r"<script(?![^>]*\bsrc=)[^>]*>", html), "no inline scripts"
    assert not re.search(r"\son[a-z]+\s*=", html), "no inline event handlers"
    assert "<style" not in html and not re.search(r'\sstyle="', html), "no inline styles"


def test_project_page_may_load_the_demo_icon_and_nothing_else_from_outside() -> None:
    html = _read(SITE / "index.html")
    policy = re.search(r'content="([^"]+)"', html).group(1)  # type: ignore[union-attr]
    assert "img-src 'self' https://*.onrender.com" in policy
    playground = _read(PLAYGROUND / "index.html")
    assert "onrender.com" not in playground


def test_pages_never_use_an_em_dash() -> None:
    em_dash = chr(0x2014)  # spelled as a code point: the character itself may not appear in code
    for path in (*PAGES, SITE / "assets" / "site.js", PLAYGROUND / "assets" / "app.js"):
        assert em_dash not in _read(path), path.name


def test_project_page_assets_exist() -> None:
    html = _read(SITE / "index.html")
    for relative in re.findall(r'(?:href|src)="((?!https?:|#|\./)[^"]+)"', html):
        if relative.startswith("recorded/"):
            continue  # the recorded runs come later, and the page degrades without them
        assert (SITE / relative).is_file(), relative


def test_project_page_degrades_without_recorded_runs() -> None:
    html = _read(SITE / "index.html")
    assert 'href="recorded/runs.json"' in html  # the raw link, in the one fallback sentence
    script = _read(SITE / "assets" / "site.js")
    assert 'fetch("recorded/runs.json"' in script
    assert "the fallback sentence in the page stays" in script


def test_playground_assets_exist() -> None:
    html = _read(PLAYGROUND / "index.html")
    found = re.findall(r'(?:href|src)="(/playground/assets/[^"]+)"', html)
    assert len(found) >= 4
    for absolute in found:
        assert (PLAYGROUND / "assets" / absolute.rsplit("/", 1)[1]).is_file(), absolute


def test_live_demo_address_is_written_once_and_is_well_formed() -> None:
    html = _read(SITE / "index.html")
    links = re.findall(r'id="demo-link" href="([^"]+)"', html)
    assert len(links) == 1
    assert re.fullmatch(r"https://[a-z0-9-]+\.(onrender\.com|invalid)/#run", links[0]), links[0]
    # No other element of the page may hold a Render address.
    assert html.count(".onrender.com/") <= 1


@pytest.mark.parametrize("name", ["theme.js", "favicon.svg"])
def test_shared_assets_are_the_same_bytes_on_both_pages(name: str) -> None:
    assert (SITE / "assets" / name).read_bytes() == (PLAYGROUND / "assets" / name).read_bytes(), (
        f"copy app/playground/static/assets/{name} to site/assets/{name}"
    )


def test_code_links_point_to_files_that_exist() -> None:
    html = _read(SITE / "index.html")
    linked = set(re.findall(r"blob/main/((?:app|scripts|tests)/[a-z_/]+\.py)", html))
    assert len(linked) >= 6, "every step of the flow links to the file that implements it"
    for path in linked:
        assert (ROOT / path).is_file(), path
    assert "autonomous-operation-marketing#readme" in html
    assert "blob/main/docs/DEPLOY.md" in html


def test_pages_link_the_repository_and_the_sibling_projects() -> None:
    repository = "https://github.com/nickdesjardins/autonomous-operation-marketing"
    assert repository in _read(PLAYGROUND / "index.html")
    html = _read(SITE / "index.html")
    assert repository in html
    assert "https://nickdesjardins.github.io/Invoice-Extraction-Service/" in html
    assert "https://nickdesjardins.github.io/hybrid-knowledge-rag/" in html


def test_wake_up_icon_path_is_served_by_the_playground() -> None:
    script = _read(SITE / "assets" / "site.js")
    match = re.search(r'new URL\("(/playground/assets/[^"]+)"', script)
    assert match
    assert (PLAYGROUND / "assets" / match.group(1).rsplit("/", 1)[1]).is_file()
    assert "GIVE_UP_MS = 180000" in script  # three minutes, as the page says


def test_graph_drawing_names_the_six_nodes_on_both_pages() -> None:
    for page in PAGES:
        html = _read(page)
        drawn = re.findall(r'data-node="([a-z_]+)"', html)
        assert drawn == list(NODE_ORDER), page.name
        assert 'id="graph-cycle"' in html and 'id="graph-interrupt"' in html


# --------------------------------------------------------------------- deployment files


def test_render_blueprint_agrees_with_the_settings() -> None:
    blueprint = _read(ROOT / "render.yaml")

    def value(key: str) -> str:
        match = re.search(rf"- key: {key}\n\s+value: \"?([^\"\n#]+?)\"?\s*(?:#.*)?\n", blueprint)
        assert match, key
        return match.group(1).strip()

    assert "plan: free" in blueprint and "runtime: python" in blueprint
    assert "healthCheckPath: /livez" in blueprint  # liveness that never queries the database
    assert value("DEMO_MODE") == "true"
    assert value("WEBHOOK_TARGET") == "self"
    assert value("LLM_PROVIDER") == "openrouter"
    assert value("TRUST_PROXY_HEADERS") == "true" and value("CLIENT_IP_HEADER")
    # Secrets are asked for by Render or generated by it, never written in the file.
    for secret in ("OPENROUTER_API_KEY", "DATABASE_URL"):
        assert re.search(rf"- key: {secret}\n\s+sync: false", blueprint), secret
    assert re.search(r"- key: WEBHOOK_SECRET\n\s+generateValue: true", blueprint)
    for marker in SECRET_MARKERS:
        assert marker not in blueprint, marker
    # Every key the blueprint sets is a real setting.
    fields = {name.upper() for name in Settings.model_fields}
    for key in re.findall(r"- key: ([A-Z_]+)", blueprint):
        assert key in fields, key
    hermetic = Settings(_env_file=None, _env_prefix="HERMETIC_TEST_")
    assert int(value("DEMO_MAX_REVISIONS")) <= hermetic.max_revisions


def test_gitignore_protects_local_secrets() -> None:
    lines = _read(ROOT / ".gitignore").splitlines()
    assert ".env" in lines and ".env.*" in lines and "!.env.example" in lines
    assert ".pgdata/" in lines and "model_cache.json" in lines
    dockerignore = _read(ROOT / ".dockerignore").splitlines()
    assert ".env" in dockerignore and ".env.*" in dockerignore


# ----------------------------------------------------------------------- recorded runs


@pytest.fixture(scope="module")
def record() -> dict:
    if not RECORD.is_file():
        pytest.skip("run scripts/record_runs.py")
    return json.loads(_read(RECORD))


def _by_id(record: dict) -> dict[str, dict]:
    return {run["id"]: run for run in record["runs"]}


def test_recorded_runs_are_real_api_responses(record: dict) -> None:
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", record["recorded_at"])
    ids = [run["id"] for run in record["runs"]]
    assert set(ids) <= SCENARIOS and len(ids) == len(set(ids))
    assert len(ids) >= 1
    for run in record["runs"]:
        assert set(run) == RUN_KEYS, run["id"]
        assert run["why"] and run["expect"]
        request = PipelineStartRequest.model_validate(run["request"])
        start = PipelineStatusResponse.model_validate(run["start_response"])
        assert start.sku_code == request.sku_code
        assert start.target_channels == list(request.target_channels)
        assert start.campaign_angle == request.campaign_angle
        assert start.max_revisions == request.max_revisions
        assert {event.node for event in start.trace} <= TRACE_NODES
        if run["review_request"] is not None:
            review = HumanReviewRequest.model_validate(run["review_request"])
            response = HumanReviewResponse.model_validate(run["review_response"])
            PipelineStatusResponse.model_validate(run["review_response"])
            assert response.action == review.action
            assert response.thread_id == start.thread_id
            assert response.human_reviewer_id == review.reviewer_id
            assert len(response.trace) > len(start.trace)
        else:
            assert run["review_response"] is None and run["inbox"] is None


def test_recorded_prompts_and_stack_match_the_code(record: dict) -> None:
    assert record["synthesis_prompt"] == SYNTHESIS_SYSTEM_PROMPT  # re-record after changing them
    assert record["audit_prompt"] == AUDIT_SYSTEM_PROMPT
    stack = record["stack"]
    assert stack["checkpointer"] == "postgres"
    assert stack["webhook_target"] == "self"
    assert stack["provider"] in {"openrouter", "openai_compatible", "anthropic"}
    assert stack["chain"], "the chain in use was recorded"
    if stack["provider"] == "openrouter":
        assert all(model.endswith(":free") for model in stack["chain"])
    for run in record["runs"]:
        assert run["request"]["max_revisions"] <= stack["max_revisions"]


def test_recorded_happy_path_was_dispatched_and_received(record: dict) -> None:
    runs = _by_id(record)
    for run_id in ("happy-path", "edit"):
        run = runs.get(run_id)
        if run is None:
            continue
        assert run["expect"] == "dispatched"
        assert run["start_response"]["awaiting_human_review"] is True
        response = run["review_response"]
        assert response["status"] == "dispatched"
        delivery = response["webhook_delivery_result"]
        assert delivery["mode"] == "self" and delivery["delivered"] is True
        assert delivery["attempts"] >= 1 and 200 <= delivery["status_code"] < 300
        inbox = run["inbox"]
        assert inbox and inbox["signature_valid"] is True
        assert inbox["thread_id"] == run["start_response"]["thread_id"]
        assert inbox["payload"]["event"] == "listing.approved"
        assert inbox["payload"]["copy"] == response["draft_copy"]
    assert "happy-path" in runs or "edit" in runs


def test_recorded_edit_changed_the_title_before_dispatch(record: dict) -> None:
    run = _by_id(record).get("edit")
    if run is None:
        pytest.skip("the edit scenario was not recorded")
    review = run["review_request"]
    assert review["action"] == "edit"
    assert review["override_copy"]["web_title"] != run["start_response"]["draft_copy"]["web_title"]
    assert run["review_response"]["draft_copy"] == review["override_copy"]
    assert run["review_response"]["human_review_status"] == "edited"


def test_recorded_reject_sent_nothing(record: dict) -> None:
    run = _by_id(record).get("reject")
    if run is None:
        pytest.skip("the reject scenario was not recorded")
    assert run["review_request"]["action"] == "reject"
    response = run["review_response"]
    assert response["status"] == "rejected"
    assert response["webhook_delivery_result"] is None
    assert run["inbox"] is None
    assert "webhook_dispatch" not in {event["node"] for event in response["trace"]}


def test_recorded_out_of_stock_called_no_model(record: dict) -> None:
    run = _by_id(record).get("out-of-stock")
    if run is None:
        pytest.skip("the out-of-stock scenario was not recorded")
    start = run["start_response"]
    assert start["status"] == "out_of_stock" and run["expect"] == "out_of_stock"
    assert start["draft_copy"] is None and start["audit_history"] == []
    assert [event["node"] for event in start["trace"]] == ["supervisor", "inventory_sql_agent"]
    assert run["review_request"] is None


def test_recorded_escalation_expectation_matches_what_happened(record: dict) -> None:
    run = _by_id(record).get("escalation")
    if run is None:
        pytest.skip("the escalation scenario was not recorded")
    start = run["start_response"]
    assert start["max_revisions"] == 1
    final = run["review_response"] or start
    assert final["status"] == run["expect"]
    if start["requires_human_escalation"]:
        assert run["review_request"]["action"] == "reject"
        assert start["audit_history"][-1]["is_compliant"] is False


def test_recorded_audit_trail_is_consistent(record: dict) -> None:
    for run in record["runs"]:
        final = run["review_response"] or run["start_response"]
        history = final["audit_history"]
        assert [entry["revision"] for entry in history] == list(range(len(history)))
        for entry in history:
            assert 0 <= entry["deterministic_findings"] <= len(entry["violations"])
            if entry["violations"]:
                assert not entry["is_compliant"]  # merge_audit: any violation means not compliant
        synthesis = [e for e in final["trace"] if e["node"] == "copy_synthesis_agent"]
        assert len(synthesis) == len(history) or final["status"] == "out_of_stock"
        assert final["revision_count"] <= final["max_revisions"]


def test_recorded_runs_hold_no_secrets(record: dict) -> None:
    blob = json.dumps(record)
    for marker in SECRET_MARKERS:
        assert marker not in blob, marker
