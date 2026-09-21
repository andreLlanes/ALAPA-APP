#Read-only Postgres access shared by the O2 experiments.

import os
import time

import psycopg2
from dotenv import load_dotenv

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
load_dotenv(os.path.join(_REPO_ROOT, ".env"))

MAX_RETRIES = 8
BACKOFF_FACTOR = 2

_CONN = None


def resolve_pg_dsn():
    """Build the Postgres DSN from PG_DSN, or from the PG_* component vars."""
    dsn = os.environ.get("PG_DSN")
    if dsn:
        return dsn
    host = os.environ.get("PG_HOST")
    port = os.environ.get("PG_PORT", "5432")
    dbname = os.environ.get("PG_DB")
    user = os.environ.get("PG_USER")
    password = os.environ.get("PG_PASSWORD")
    if not all([host, dbname, user, password]):
        raise ValueError("Missing PG_DSN or PG_HOST/PG_DB/PG_USER/PG_PASSWORD.")
    return f"postgresql://{user}:{password}@{host}:{port}/{dbname}"


def get_conn():
    """Return the shared read-only connection, opening it if needed."""
    global _CONN
    if _CONN is None or _CONN.closed != 0:
        _CONN = psycopg2.connect(resolve_pg_dsn())
        _CONN.set_session(readonly=True, autocommit=True)
    return _CONN


def _reset():
    """Drop the shared connection so the next call reopens it."""
    global _CONN
    if _CONN is not None:
        try:
            _CONN.close()
        except psycopg2.Error:
            pass
    _CONN = None


def fetch(query, params=None):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with get_conn().cursor() as cur:
                cur.execute(query, params)
                return [d[0] for d in cur.description], cur.fetchall()
        except (psycopg2.OperationalError, psycopg2.InterfaceError) as exc:
            _reset()
            if attempt == MAX_RETRIES:
                raise
            wait = BACKOFF_FACTOR ** attempt
            print(f"  connection error (attempt {attempt}/{MAX_RETRIES}): "
                  f"{str(exc).strip().splitlines()[0]}; retrying in {wait}s")
            time.sleep(wait)
