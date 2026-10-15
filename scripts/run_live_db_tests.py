"""Run the live PostgreSQL tests against an embedded server, or keep that server up for recording.

Useful on machines without Docker: ``pgserver`` ships PostgreSQL binaries inside the virtualenv,
no admin rights required. It is installed on demand (``pip install pgserver``) the first time
this script runs; its data directory is ``.pgdata/`` in the project root (git-ignored).

    python scripts/run_live_db_tests.py            # runs tests/test_integration_db.py
    python scripts/run_live_db_tests.py -k filter  # extra args are passed to pytest
    python scripts/run_live_db_tests.py --serve    # prints the URL and keeps the server up

``--serve`` is for ``scripts/record_runs.py``, which needs a PostgreSQL checkpointer: start this
in one terminal, then in another one point the service at the printed URL and record:

    PowerShell:  $env:DATABASE_URL = "<the printed URL>"; python scripts/record_runs.py
    bash:        DATABASE_URL="<the printed URL>" python scripts/record_runs.py

If ``.env.local`` holds ``DATABASE_URL_UNPOOLED`` (Neon's direct string), set that name instead
of ``DATABASE_URL``: the unpooled one wins when both are set (``app/config.py``).

Press Ctrl+C here to stop the server when the recording is done. The printed URL is local
(user postgres, no password) and holds no secret.
"""

from __future__ import annotations

import asyncio
import logging
import os
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / ".pgdata"
PGSERVER_REQUIREMENT = "pgserver==0.1.4"


def _import_pgserver():  # type: ignore[no-untyped-def]
    try:
        import pgserver
    except ImportError:
        print(f"pgserver is not installed; installing {PGSERVER_REQUIREMENT} into this environment")
        subprocess.check_call([sys.executable, "-m", "pip", "install", PGSERVER_REQUIREMENT])
        import pgserver
    return pgserver


def main() -> int:
    args = [arg for arg in sys.argv[1:] if arg != "--serve"]
    serve = "--serve" in sys.argv[1:]
    pgserver = _import_pgserver()
    server = pgserver.get_server(DATA_DIR)
    try:
        uri = server.get_uri()
        print(f"Embedded PostgreSQL ready: {uri}")
        if serve:
            print("Keeping it up for scripts/record_runs.py (DATABASE_URL above). Ctrl+C stops it.")
            try:
                while True:
                    time.sleep(1.0)
            except KeyboardInterrupt:
                print("Stopping the embedded server.")
            return 0
        import pytest

        os.environ["TEST_DATABASE_URL"] = uri
        if sys.platform == "win32":
            # psycopg's async connections cannot run on Windows' default proactor loop.
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        os.chdir(ROOT)
        return int(pytest.main(["tests/test_integration_db.py", "-v", *args]))
    finally:
        # pytest closes the captured streams pgserver's logger writes to; silence it before stopping.
        logging.getLogger("pgserver").disabled = True
        server.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
