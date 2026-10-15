"""Autonomous multi-agent operations and marketing pipeline.

A LangGraph state machine drafts channel marketing copy for a Natural Health Product SKU, audits
it against Health Canada NHP rules, loops back on violations, pauses for a human before an
external webhook fires, and keeps every step in PostgreSQL.
"""

__version__ = "1.0.0"
