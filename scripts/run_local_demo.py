"""Run the public showcase on this machine, in demo mode, without Docker.

    python scripts/run_local_demo.py            # http://127.0.0.1:8011/
    python scripts/run_local_demo.py --port 9000

Reads ``.env`` / ``.env.local`` for the OpenRouter key and the database URL, and sets three
values for this process only: ``DEMO_MODE=true`` (runs budgeted, revisions capped),
``WEBHOOK_TARGET=self`` and ``PUBLIC_BASE_URL=http://127.0.0.1:8011`` (the service posts every
approved listing to its own receiver, so nothing calls out). A ``WEBHOOK_SECRET`` is required to
sign those deliveries; when neither the environment nor the dotenv files hold one, a random one
is generated for this process. ``render.yaml`` is not read, so production settings such as
``ENVIRONMENT=production`` (which hides /docs) apply only if ``.env.local`` sets them too.

On Windows the service runs on a selector event loop: psycopg's async connections cannot use the
proactor loop uvicorn would pick by itself.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import pathlib
import secrets
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)


def _dotenv_has(name: str) -> bool:
    """Whether .env or .env.local sets ``name`` (the files are read, never printed)."""
    from dotenv import dotenv_values

    for file in (".env", ".env.local"):
        path = ROOT / file
        if path.is_file() and (dotenv_values(path).get(name) or "").strip():
            return True
    return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    os.environ["DEMO_MODE"] = "true"
    os.environ["WEBHOOK_TARGET"] = "self"
    os.environ["PUBLIC_BASE_URL"] = f"http://{args.host}:{args.port}"
    if not os.environ.get("WEBHOOK_SECRET") and not _dotenv_has("WEBHOOK_SECRET"):
        os.environ["WEBHOOK_SECRET"] = secrets.token_hex(32)
        print("WEBHOOK_SECRET is not set; a random one signs the deliveries of this process.")

    import uvicorn

    config = uvicorn.Config(
        "app.main:app", host=args.host, port=args.port, workers=1, log_level="info"
    )
    server = uvicorn.Server(config)
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(server.serve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
