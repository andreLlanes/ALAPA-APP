import os
from dotenv import load_dotenv
import psycopg2

load_dotenv()


def resolve_pg_dsn() -> str:
    dsn = os.environ.get("PG_DSN")
    if dsn:
        return dsn

    host = os.environ.get("PG_HOST")
    port = os.environ.get("PG_PORT", "5432")
    dbname = os.environ.get("PG_DB")
    user = os.environ.get("PG_USER")
    password = os.environ.get("PG_PASSWORD")

    if not all([host, dbname, user, password]):
        raise ValueError("Missing PG_DSN or PG_HOST/PG_DB/PG_USER/PG_PASSWORD env vars.")

    return f"postgresql://{user}:{password}@{host}:{port}/{dbname}"


def add_coordinate_columns(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            ALTER TABLE openaq.meteo_hourly
            ADD COLUMN IF NOT EXISTS latitude double precision,
            ADD COLUMN IF NOT EXISTS longitude double precision;
            """
        )
    conn.commit()


def backfill_coordinates(conn) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE openaq.meteo_hourly m
            SET latitude = l.latitude,
                longitude = l.longitude
            FROM openaq.locations l
            WHERE m.location_key = l.location_key
              AND (
                  m.latitude IS DISTINCT FROM l.latitude
                  OR m.longitude IS DISTINCT FROM l.longitude
              );
            """
        )
        conn.commit()
        return cur.rowcount


def main() -> None:
    dsn = resolve_pg_dsn()
    with psycopg2.connect(dsn) as conn:
        add_coordinate_columns(conn)
        updated_rows = backfill_coordinates(conn)
        print(f"Updated {updated_rows} meteo_hourly rows with coordinates from openaq.locations")


if __name__ == "__main__":
    main()
