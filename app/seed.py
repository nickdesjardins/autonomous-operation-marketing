"""The sample inventory: four fictional Natural Health Product SKUs, one of them out of stock.

This module is the single source of truth for the sample data. The database layer upserts it
at start-up (``AUTO_CREATE_SCHEMA=true``), the in-memory inventory serves it when there is no
database, and ``sql/init_inventory.sql`` (what docker-compose runs on first boot) is rendered
from it with ``python -m app.seed``. A test checks that the committed SQL file matches.

The products, licence numbers, claims and warnings are written for this demo. The claims are
modelled on the structure-function wording Health Canada licenses for these ingredient classes;
the NPNs are not real licences.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from app.schemas import SkuRecord

SQL_PATH = Path(__file__).resolve().parents[1] / "sql" / "init_inventory.sql"

SKU_SEED: list[dict[str, Any]] = [
    {
        "sku_code": "NR-MAG-200",
        "product_name": "Magnesium Bisglycinate 200 mg",
        "npn_number": "80098765",
        "dosage_form": "Vegetable capsule",
        "stock_quantity": 1240,
        "serving_size": "1 capsule, 1 to 2 times daily",
        "active_ingredients": [
            {
                "name": "Magnesium (magnesium bisglycinate)",
                "amount": 200,
                "unit": "mg",
                "per": "capsule",
            }
        ],
        "authorized_claims": [
            "Helps to maintain proper muscle function, including the heart muscle.",
            "Helps in the development and maintenance of bones and teeth.",
            "Helps to prevent magnesium deficiency.",
            "Helps in energy metabolism and in tissue formation.",
        ],
        "risk_information": [
            {
                "text": "Consult a health care practitioner prior to use if you are pregnant or "
                "breastfeeding.",
                "mandatory": True,
            },
            {
                "text": "Some people may experience diarrhoea.",
                "mandatory": True,
            },
            {
                "text": "Keep out of reach of children.",
                "mandatory": False,
            },
        ],
        "contraindications": "Do not use if you have kidney disease unless directed by a health "
        "care practitioner.",
        "brand_notes": "Chelated form chosen for tolerability. Vegan, gluten free, non-GMO.",
    },
    {
        "sku_code": "NR-ORE-C80",
        "product_name": "Wild Oregano Oil C80",
        "npn_number": "80087654",
        "dosage_form": "Oil in a glass dropper bottle",
        "stock_quantity": 310,
        "serving_size": "4 drops, 1 to 3 times daily, held under the tongue",
        "active_ingredients": [
            {
                "name": "Oregano oil (Origanum vulgare, leaf), standardized to 80% carvacrol",
                "amount": 100,
                "unit": "mg",
                "per": "4 drops",
            }
        ],
        "authorized_claims": [
            "Used in Herbal Medicine as an expectorant to help relieve the symptoms of bronchitis.",
            "Used in Herbal Medicine to help relieve cough and cold symptoms.",
            "Source of antioxidants.",
        ],
        "risk_information": [
            {
                "text": "Consult a health care practitioner prior to use if you are pregnant or "
                "breastfeeding.",
                "mandatory": True,
            },
            {
                "text": "Consult a health care practitioner if symptoms persist or worsen.",
                "mandatory": True,
            },
            {
                "text": "Hypersensitivity, such as allergy, has been known to occur, in which "
                "case discontinue use.",
                "mandatory": True,
            },
        ],
        "contraindications": "Do not use if you are allergic to plants of the Lamiaceae "
        "(mint) family.",
        "brand_notes": "Wild-crafted Mediterranean oregano in organic olive oil.",
    },
    {
        "sku_code": "NR-PRO-50B",
        "product_name": "Probiotic 50 Billion CFU",
        "npn_number": "80076543",
        "dosage_form": "Delayed-release vegetable capsule",
        "stock_quantity": 865,
        "serving_size": "1 capsule daily with food",
        "active_ingredients": [
            {
                "name": "Lactobacillus acidophilus",
                "amount": 20,
                "unit": "billion CFU",
                "per": "capsule",
            },
            {
                "name": "Bifidobacterium lactis",
                "amount": 20,
                "unit": "billion CFU",
                "per": "capsule",
            },
            {
                "name": "Lactobacillus rhamnosus",
                "amount": 10,
                "unit": "billion CFU",
                "per": "capsule",
            },
        ],
        "authorized_claims": [
            "Helps support intestinal and gastrointestinal health.",
            "Could promote a favourable gut flora.",
        ],
        "risk_information": [
            {
                "text": "If you have fever, vomiting, bloody diarrhoea or severe abdominal pain, "
                "consult a health care practitioner prior to use.",
                "mandatory": True,
            },
            {
                "text": "If symptoms of digestive upset occur, worsen or persist beyond 3 days, "
                "discontinue use and consult a health care practitioner.",
                "mandatory": True,
            },
            {
                "text": "Do not use if you have an immune-compromised condition.",
                "mandatory": True,
            },
        ],
        "contraindications": "Do not use if you have an immune-compromised condition (for "
        "example AIDS, lymphoma, patients undergoing long-term corticosteroid treatment).",
        "brand_notes": "Shelf stable. Delayed-release capsule so the strains reach the intestine.",
    },
    {
        "sku_code": "NR-VITD-1000",
        "product_name": "Vitamin D3 1000 IU",
        "npn_number": "80065432",
        "dosage_form": "Softgel",
        "stock_quantity": 0,
        "serving_size": "1 softgel daily",
        "active_ingredients": [
            {"name": "Vitamin D3 (cholecalciferol)", "amount": 1000, "unit": "IU", "per": "softgel"}
        ],
        "authorized_claims": [
            "Helps in the development and maintenance of bones and teeth.",
            "Helps in the absorption and use of calcium and phosphorus.",
        ],
        "risk_information": [
            {"text": "Keep out of reach of children.", "mandatory": False},
        ],
        "contraindications": None,
        "brand_notes": "Back-ordered until the next production run.",
    },
]

#: Campaign briefs the playground offers as one-click samples (visitors may type their own).
SAMPLE_ANGLES: list[str] = [
    "Autumn restock: calm evenings and better sleep routines for busy parents.",
    "Winter wellness kit for people who travel a lot and want to stay on top of their routine.",
    "New year, steady habits: a daily ritual that is easy to keep.",
    "Back to school: digestive comfort for the whole family's busy weeks.",
]


def seed_records() -> list[SkuRecord]:
    """The seed as validated records: a typo in this file fails at import time of the tests."""
    return [SkuRecord.model_validate(row) for row in SKU_SEED]


def _sql_text(value: str | None) -> str:
    if value is None:
        return "NULL"
    return "'" + value.replace("'", "''") + "'"


def _sql_jsonb(value: object) -> str:
    return _sql_text(json.dumps(value, ensure_ascii=False)) + "::jsonb"


SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS inventory_skus (
    sku_code            TEXT PRIMARY KEY,
    product_name        TEXT NOT NULL,
    npn_number          TEXT NOT NULL,
    dosage_form         TEXT NOT NULL,
    stock_quantity      INTEGER NOT NULL CHECK (stock_quantity >= 0),
    serving_size        TEXT NOT NULL,
    active_ingredients  JSONB NOT NULL,
    authorized_claims   JSONB NOT NULL,
    risk_information    JSONB NOT NULL,
    contraindications   TEXT,
    brand_notes         TEXT,
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS webhook_audit_log (
    id              BIGSERIAL PRIMARY KEY,
    delivery_id     TEXT NOT NULL,
    thread_id       TEXT NOT NULL,
    attempt         INTEGER NOT NULL,
    target_mode     TEXT NOT NULL,
    url_host        TEXT,
    status_code     INTEGER,
    ok              BOOLEAN NOT NULL,
    error           TEXT,
    duration_ms     DOUBLE PRECISION,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_webhook_audit_thread ON webhook_audit_log (thread_id, created_at);

CREATE TABLE IF NOT EXISTS webhook_inbox (
    delivery_id     TEXT PRIMARY KEY,
    thread_id       TEXT,
    received_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    signature_valid BOOLEAN NOT NULL,
    payload         JSONB NOT NULL
);
"""


def schema_statements() -> list[str]:
    """The DDL as separate statements, for a driver that runs one statement per execute."""
    return [part.strip() for part in SCHEMA_SQL.split(";") if part.strip()]


def seed_statements() -> list[str]:
    """One idempotent upsert per SKU (``ON CONFLICT DO UPDATE`` keeps the stock as shipped)."""
    statements = []
    for row in SKU_SEED:
        statements.append(
            "INSERT INTO inventory_skus (sku_code, product_name, npn_number, dosage_form, "
            "stock_quantity, serving_size, active_ingredients, authorized_claims, "
            "risk_information, contraindications, brand_notes)\nVALUES ("
            + ", ".join(
                [
                    _sql_text(row["sku_code"]),
                    _sql_text(row["product_name"]),
                    _sql_text(row["npn_number"]),
                    _sql_text(row["dosage_form"]),
                    str(int(row["stock_quantity"])),
                    _sql_text(row["serving_size"]),
                    _sql_jsonb(row["active_ingredients"]),
                    _sql_jsonb(row["authorized_claims"]),
                    _sql_jsonb(row["risk_information"]),
                    _sql_text(row["contraindications"]),
                    _sql_text(row["brand_notes"]),
                ]
            )
            + ")\nON CONFLICT (sku_code) DO UPDATE SET\n"
            "    product_name = EXCLUDED.product_name,\n"
            "    npn_number = EXCLUDED.npn_number,\n"
            "    dosage_form = EXCLUDED.dosage_form,\n"
            "    stock_quantity = EXCLUDED.stock_quantity,\n"
            "    serving_size = EXCLUDED.serving_size,\n"
            "    active_ingredients = EXCLUDED.active_ingredients,\n"
            "    authorized_claims = EXCLUDED.authorized_claims,\n"
            "    risk_information = EXCLUDED.risk_information,\n"
            "    contraindications = EXCLUDED.contraindications,\n"
            "    brand_notes = EXCLUDED.brand_notes,\n"
            "    updated_at = now();"
        )
    return statements


def render_sql() -> str:
    """The content of ``sql/init_inventory.sql``: schema plus the seed, idempotent."""
    header = (
        "-- Generated by `python -m app.seed` from app/seed.py. Do not edit by hand: a test\n"
        "-- compares this file with the generator. docker-compose runs it on the first boot of\n"
        "-- the database; the service applies the same statements at start-up when\n"
        "-- AUTO_CREATE_SCHEMA=true, so a hosted database (Neon) needs no manual step.\n"
        "-- The LangGraph checkpoint tables are created by langgraph-checkpoint-postgres itself\n"
        "-- (checkpointer.setup()).\n\n"
    )
    return header + SCHEMA_SQL + "\n" + "\n\n".join(seed_statements()) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Write ``sql/init_inventory.sql`` (or print it with ``--print``)."""
    args = list(sys.argv[1:] if argv is None else argv)
    text = render_sql()
    if "--print" in args:
        sys.stdout.write(text)
        return 0
    SQL_PATH.parent.mkdir(parents=True, exist_ok=True)
    SQL_PATH.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {SQL_PATH}")
    return 0


if __name__ == "__main__":  # pragma: no cover - command-line entry
    sys.exit(main())
