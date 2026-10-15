"""The README is held to the code it quotes, so the two cannot quietly drift apart.

``tests/test_site.py`` does this for the published pages. This file does it for the parts of
``README.md`` that repeat something the code defines: the status values, the default model
chain, the defaults in the configuration table, and the list of test files. Everything here is
a read of two files; nothing calls a model or the network.
"""

from __future__ import annotations

import re
from pathlib import Path

from app import config
from app.graph import state

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")


def section(heading: str) -> str:
    """The text under ``heading`` up to the next heading of the same or a higher level."""
    level = len(heading) - len(heading.lstrip("#"))
    match = re.search(rf"^{re.escape(heading)}$", README, re.MULTILINE)
    assert match, f"the README has no heading {heading!r}"
    rest = README[match.end() :]
    following = re.search(rf"^#{{1,{level}}} ", rest, re.MULTILINE)
    return rest[: following.start()] if following else rest


def test_status_table_names_every_status_constant_and_nothing_else() -> None:
    constants = {
        value
        for name, value in vars(state).items()
        if name.startswith("STATUS_") and isinstance(value, str)
    }
    text = section("### Status values")
    body = text[text.index("|---|") :]  # the rows after the header
    rows = re.findall(r"^\| `([a-z_]+)` \|", body, re.MULTILINE)
    assert len(rows) == len(set(rows)), "a status is listed twice"
    assert set(rows) == constants


def test_default_chain_is_written_in_the_shipped_order() -> None:
    chain = config.parse_model_chain(config.DEFAULT_MODEL_CHAIN)
    text = section("## Model selection: free models and the chain that repairs itself")
    words = re.search(r"\*\*The default chain, in words\.\*\* First `([^`]+)`[^`]*`([^`]+)`", text)
    assert words and [words.group(1), words.group(2)] == chain
    snapshot = text[text.index("<!-- snapshot:start -->") : text.index("<!-- snapshot:end -->")]
    assert re.findall(r"^\| `([^`]+)` \|", snapshot, re.MULTILINE)[: len(chain)] == chain
    # The /health example lists the chain in use, which is the default on a fresh service.
    health = re.search(r'"models": \[([^\]]+)\]', section("### Health"))
    assert health and re.findall(r'"([^"]+)"', health.group(1)) == chain


def _same(written: str, default: object) -> bool:
    """Whether a value written in the table is the setting's default."""
    if isinstance(default, bool):
        return written == str(default).lower()
    if isinstance(default, float):
        return float(written) == default
    return written == str(default)


def test_configuration_table_quotes_the_settings_own_defaults() -> None:
    fields = config.Settings.model_fields
    rows = re.findall(r"^\| (`[^|]+`) \| ([^|]+) \|", section("## Configuration"), re.MULTILINE)
    assert len(rows) >= 20
    compared = 0
    for names_cell, defaults_cell in rows:
        names = re.findall(r"`([A-Z_]+)`", names_cell)
        defaults = [cell.strip() for cell in defaults_cell.split(" / ")]
        assert len(names) == len(defaults), names_cell
        for name, written in zip(names, defaults, strict=True):
            assert name.lower() in fields, f"{name} is not a setting"
            if not (written.startswith("`") and written.endswith("`")):
                continue  # "unset", "blank", "empty": nothing to compare with
            assert _same(written.strip("`"), fields[name.lower()].default), name
            compared += 1
    assert compared >= 30


def test_every_test_file_is_described() -> None:
    text = section("## Tests")
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        assert path.name in text, f"{path.name} is not described in the README"
