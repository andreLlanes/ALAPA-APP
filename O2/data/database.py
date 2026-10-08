""" Copy the database to local files (one per city), so runs read the copy and never query the
    database again. Recopy only after the database changes.
    python data/database.py             copy if missing
    python data/database.py --refresh   recopy everything
"""

import argparse
import os
import sys
import tempfile
from contextlib import closing
from pathlib import Path

import pandas as pd
import psycopg2
from dotenv import load_dotenv

# Put the O2 root on the path so this file imports the same whether it is run as a script
# (python data/database.py) or as a module (python -m data.database).
sys.path.append(str(Path(__file__).resolve().parents[1]))

from Common.schema import KEY_COL, TIME_COL
from utils import runlog
from utils.artifacts import COPY_DIR

CITIES = {"mm": "Metro Manila", "bk": "Bangkok", "la": "Los Angeles"}
TABLE = "openaq.merged_clean"

def copy_path(city: str) -> Path:
    """ Return the path of a city's copy of the database (data/cache/<city>.parquet).
    """
    return COPY_DIR / f"{city}.parquet"

def database_url() -> str:
    """ Build the Postgres connection string from .env: PG_DSN, or
        PG_HOST/PG_PORT/PG_DB/PG_USER/PG_PASSWORD.
    """
    env = os.environ
    if env.get("PG_DSN"):
        return env["PG_DSN"]
    return (f"postgresql://{env['PG_USER']}:{env['PG_PASSWORD']}"
            f"@{env['PG_HOST']}:{env.get('PG_PORT', '5432')}/{env['PG_DB']}")

def fetch_city_rows(cur, city: str) -> pd.DataFrame:
    """ Fetch every row and column of one city, ordered by station then time.
    """
    query = cur.mogrify(f"SELECT * FROM {TABLE} WHERE city = %s ORDER BY {KEY_COL}, {TIME_COL}",
                        (CITIES[city],)).decode()
    # Bulk export streamed to a temporary binary file (psycopg2 writes bytes; a text-mode file
    # fails on Windows), so the CSV text never sits in memory.
    COPY_DIR.mkdir(exist_ok=True)
    with tempfile.TemporaryFile("w+b", dir=COPY_DIR) as tmp:
        cur.copy_expert(f"COPY ({query}) TO STDOUT WITH CSV HEADER", tmp)
        tmp.seek(0)
        df = pd.read_csv(tmp)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], utc=True)
    return df

def copy_database(refresh: bool = False):
    """ Copy every city from the database to data/cache/, in one connection.
        Cities already copied are skipped unless refresh is set.
    """
    todo = [c for c in CITIES if refresh or not copy_path(c).exists()]
    if not todo:
        return
    load_dotenv()
    COPY_DIR.mkdir(exist_ok=True)
    with closing(psycopg2.connect(database_url())) as conn, conn.cursor() as cur:
        for city in todo:
            df = fetch_city_rows(cur, city)
            df.to_parquet(copy_path(city), index=False)
            print(f"    Copied {CITIES[city]}: {len(df):,} rows")

def read_copy(city: str, columns=None) -> pd.DataFrame:
    """ Read a city from the local copy, making the copy first if it is missing.
    """
    if not copy_path(city).exists():
        copy_database()
    return pd.read_parquet(copy_path(city), columns=columns)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="recopy after the database changed")
    args = ap.parse_args()
    with runlog.logged("database", args):
        print("[DATA] Copying Database")
        copy_database(args.refresh)
