"""StateGraph assembly: the nodes, the edges, the cycle, and the interrupt before the webhook.

The topology is the whole point of the project, so it is spelled out once here and exposed as
data (:func:`topology`) for the README, the playground and a test that holds them together.
"""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.graph.nodes import PipelineNodes
from app.graph.state import PipelineState

NODE_SUPERVISOR = "supervisor"
NODE_INVENTORY = "inventory_sql_agent"
NODE_SYNTHESIS = "copy_synthesis_agent"
NODE_COMPLIANCE = "compliance_validator"
NODE_GATE = "human_approval_gate"
NODE_WEBHOOK = "webhook_dispatch"

NODE_ORDER = (
    NODE_SUPERVISOR,
    NODE_INVENTORY,
    NODE_SYNTHESIS,
    NODE_COMPLIANCE,
    NODE_GATE,
    NODE_WEBHOOK,
)


def build_graph(nodes: PipelineNodes, checkpointer: BaseCheckpointSaver) -> CompiledStateGraph:
    """Compile the pipeline with ``checkpointer`` and the breakpoint before the webhook.

    Tests compile the very same topology with an in-memory saver; production compiles it with
    the PostgreSQL saver. Nothing else differs.
    """
    graph = StateGraph(PipelineState)
    graph.add_node(NODE_SUPERVISOR, nodes.supervisor)
    graph.add_node(NODE_INVENTORY, nodes.inventory_sql_agent)
    graph.add_node(NODE_SYNTHESIS, nodes.copy_synthesis_agent)
    graph.add_node(NODE_COMPLIANCE, nodes.compliance_validator)
    graph.add_node(NODE_GATE, nodes.human_approval_gate)
    graph.add_node(NODE_WEBHOOK, nodes.webhook_dispatch)

    graph.add_edge(START, NODE_SUPERVISOR)
    graph.add_conditional_edges(
        NODE_SUPERVISOR,
        nodes.route_after_supervisor,
        {NODE_INVENTORY: NODE_INVENTORY, NODE_SYNTHESIS: NODE_SYNTHESIS, END: END},
    )
    graph.add_conditional_edges(
        NODE_INVENTORY,
        nodes.route_after_inventory,
        {NODE_SYNTHESIS: NODE_SYNTHESIS, END: END},
    )
    graph.add_edge(NODE_SYNTHESIS, NODE_COMPLIANCE)
    graph.add_conditional_edges(
        NODE_COMPLIANCE,
        nodes.route_after_compliance,
        {NODE_SYNTHESIS: NODE_SYNTHESIS, NODE_GATE: NODE_GATE},
    )
    graph.add_edge(NODE_GATE, NODE_WEBHOOK)
    graph.add_edge(NODE_WEBHOOK, END)

    return graph.compile(checkpointer=checkpointer, interrupt_before=[NODE_WEBHOOK])


def topology() -> dict[str, Any]:
    """The graph as data: nodes in order, edges with their conditions, and the breakpoint."""
    return {
        "nodes": [
            {"id": NODE_SUPERVISOR, "title": "Supervisor", "kind": "deterministic"},
            {"id": NODE_INVENTORY, "title": "Inventory SQL Agent", "kind": "tool"},
            {"id": NODE_SYNTHESIS, "title": "Copy Synthesis Agent", "kind": "llm"},
            {"id": NODE_COMPLIANCE, "title": "Compliance Validator", "kind": "llm+checks"},
            {"id": NODE_GATE, "title": "Human Approval Gate", "kind": "deterministic"},
            {"id": NODE_WEBHOOK, "title": "Webhook Dispatch", "kind": "mutation"},
        ],
        "edges": [
            {"from": "START", "to": NODE_SUPERVISOR},
            {"from": NODE_SUPERVISOR, "to": NODE_INVENTORY, "when": "inventory not loaded yet"},
            {"from": NODE_SUPERVISOR, "to": NODE_SYNTHESIS, "when": "inventory already loaded"},
            {"from": NODE_SUPERVISOR, "to": "END", "when": "invalid request"},
            {"from": NODE_INVENTORY, "to": NODE_SYNTHESIS, "when": "SKU found and in stock"},
            {"from": NODE_INVENTORY, "to": "END", "when": "SKU unknown or out of stock"},
            {"from": NODE_SYNTHESIS, "to": NODE_COMPLIANCE},
            {"from": NODE_COMPLIANCE, "to": NODE_GATE, "when": "compliant"},
            {
                "from": NODE_COMPLIANCE,
                "to": NODE_SYNTHESIS,
                "when": "violations and revision_count < max_revisions",
            },
            {
                "from": NODE_COMPLIANCE,
                "to": NODE_GATE,
                "when": "violations and revision_count >= max_revisions (escalation)",
            },
            {"from": NODE_GATE, "to": NODE_WEBHOOK, "when": "after human approval or edit"},
            {"from": NODE_WEBHOOK, "to": "END"},
        ],
        "interrupt_before": [NODE_WEBHOOK],
    }
