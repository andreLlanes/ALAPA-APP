import os
import time
from datetime import date, timedelta

import pandas as pd
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

import meteo_etl as meteo_etl_mod

load_dotenv()

DEFAULT_START_DATE = os.environ.get("METEO_START_DATE", "2021-06-30")
DEFAULT_END_DATE = os.environ.get("METEO_END_DATE", "2026-06-30")
GRID_STEP_DEG = float(os.environ.get("METEO_GRID_STEP_DEG", "0.25"))
CHUNK_SIZE = int(os.environ.get("METEO_UPSERT_CHUNK_SIZE", "1000"))

CITY_BBOXES = {
    "Metro Manila": (14.316284, 120.868835, 14.781522, 121.143494),
    "Bangkok": (13.600000, 100.400000, 13.950000, 100.850000),
    "Singapore": (1.200000, 103.600000, 1.450000, 104.000000),
}

METEO_COLUMNS = [
    "temperature_c",
    "humidity_pct",
    "wind_speed_ms",
    "wind_gusts_ms",
    "wind_dir_deg",
    "surface_pressure_hpa",
    "precipitation_mm",
    "boundary_layer_height_m",
]


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


def generate_grid_centers(bbox: tuple[float, float, float, float]) -> list[tuple[float, float]]:
    min_lat, min_lon, max_lat, max_lon = bbox
    centers = []
    lat = min_lat
    while lat <= max_lat + 1e-9:
        lon = min_lon
        while lon <= max_lon + 1e-9:
            centers.append((round(lat, 4), round(lon, 4)))
            lon += GRID_STEP_DEG
        lat += GRID_STEP_DEG
    return centers


def fetch_weather_for_grid_cell(city: str, lat: float, lon: float, start_date: date, end_date: date) -> pd.DataFrame:
    raw_df = meteo_etl_mod.fetch_open_meteo_data(lat, lon, start_date, end_date)
    if raw_df.empty:
        return pd.DataFrame()

    df = raw_df.rename(columns=meteo_etl_mod.RENAMES)
    if "time" in df.columns:
        df = df.rename(columns={"time": "timestamp_utc"})
    if "timestamp_utc" not in df.columns:
        return pd.DataFrame()

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df = df.set_index("timestamp_utc")
    expected_index = pd.date_range(start=start_date, end=end_date + timedelta(days=0), freq="h", tz="UTC")
    df = df.reindex(expected_index)
    df.index.name = "timestamp_utc"
    df = df.reset_index()

    df["city"] = city
    df["latitude"] = lat
    df["longitude"] = lon
    return df


def create_table_if_needed(conn) -> None:
    ddl = """
        CREATE TABLE IF NOT EXISTS openaq.meteo_hourly_grid (
            city text NOT NULL,
            latitude double precision NOT NULL,
            longitude double precision NOT NULL,
            timestamp_utc timestamptz NOT NULL,
            temperature_c double precision,
            humidity_pct double precision,
            wind_speed_ms double precision,
            wind_gusts_ms double precision,
            wind_dir_deg double precision,
            surface_pressure_hpa double precision,
            precipitation_mm double precision,
            boundary_layer_height_m double precision,
            source text NOT NULL DEFAULT 'open-meteo',
            inserted_at timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (city, latitude, longitude, timestamp_utc)
        )
    """
    with conn.cursor() as cur:
        cur.execute(ddl)
    conn.commit()
    print("Ensured openaq.meteo_hourly_grid exists.")


def _na_to_none(value):
    try:
        if value is None or pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def upsert_grid_weather(conn, df: pd.DataFrame) -> None:
    if df.empty:
        print("No grid weather rows to upsert.")
        return

    required = {"city", "latitude", "longitude", "timestamp_utc"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing required columns: {required - set(df.columns)}")

    rows = []
    for row in df.itertuples(index=False):
        rows.append(
            (
                row.city,
                float(row.latitude),
                float(row.longitude),
                row.timestamp_utc,
                _na_to_none(getattr(row, "temperature_c", None)),
                _na_to_none(getattr(row, "humidity_pct", None)),
                _na_to_none(getattr(row, "wind_speed_ms", None)),
                _na_to_none(getattr(row, "wind_gusts_ms", None)),
                _na_to_none(getattr(row, "wind_dir_deg", None)),
                _na_to_none(getattr(row, "surface_pressure_hpa", None)),
                _na_to_none(getattr(row, "precipitation_mm", None)),
                _na_to_none(getattr(row, "boundary_layer_height_m", None)),
                "open-meteo",
            )
        )

    query = """
        INSERT INTO openaq.meteo_hourly_grid (
            city, latitude, longitude, timestamp_utc,
            temperature_c, humidity_pct, wind_speed_ms, wind_gusts_ms, wind_dir_deg,
            surface_pressure_hpa, precipitation_mm, boundary_layer_height_m, source
        ) VALUES %s
        ON CONFLICT (city, latitude, longitude, timestamp_utc)
        DO UPDATE SET
            temperature_c          = EXCLUDED.temperature_c,
            humidity_pct           = EXCLUDED.humidity_pct,
            wind_speed_ms          = EXCLUDED.wind_speed_ms,
            wind_gusts_ms          = EXCLUDED.wind_gusts_ms,
            wind_dir_deg           = EXCLUDED.wind_dir_deg,
            surface_pressure_hpa   = EXCLUDED.surface_pressure_hpa,
            precipitation_mm       = EXCLUDED.precipitation_mm,
            boundary_layer_height_m= EXCLUDED.boundary_layer_height_m,
            source                 = EXCLUDED.source,
            inserted_at            = now()
    """

    with conn.cursor() as cur:
        for i in range(0, len(rows), CHUNK_SIZE):
            chunk = rows[i : i + CHUNK_SIZE]
            psycopg2.extras.execute_values(cur, query, chunk, page_size=CHUNK_SIZE)
    conn.commit()
    print(f"Committed {len(rows)} grid-weather rows.")


def main() -> None:
    start_date = date.fromisoformat(DEFAULT_START_DATE)
    end_date = date.fromisoformat(DEFAULT_END_DATE)

    frames = []
    for city, bbox in CITY_BBOXES.items():
        centers = generate_grid_centers(bbox)
        print(f"Processing {city} with {len(centers)} grid centers")
        for lat, lon in centers:
            df = fetch_weather_for_grid_cell(city, lat, lon, start_date, end_date)
            if not df.empty:
                frames.append(df)

    if not frames:
        print("No grid weather data fetched.")
        return

    final_df = pd.concat(frames, ignore_index=True)
    final_df = final_df.sort_values(["city", "latitude", "longitude", "timestamp_utc"])

    dsn = resolve_pg_dsn()
    with psycopg2.connect(dsn) as conn:
        create_table_if_needed(conn)
        upsert_grid_weather(conn, final_df)


if __name__ == "__main__":
    main()
