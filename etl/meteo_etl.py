
import os
import time
from datetime import datetime, date, timedelta
from pathlib import Path

import pandas as pd
import psycopg2
import psycopg2.extras
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

load_dotenv()

ARCHIVE_API_BASE = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_API_BASE = "https://api.open-meteo.com/v1/forecast"
HOURLY_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_gusts_10m",
    "wind_direction_10m",
    "surface_pressure",
    "precipitation",
    "boundary_layer_height",
]
RENAMES = {
    "temperature_2m": "temperature_c",
    "relative_humidity_2m": "humidity_pct",
    "wind_speed_10m": "wind_speed_ms",
    "wind_gusts_10m": "wind_gusts_ms",
    "wind_direction_10m": "wind_dir_deg",
    "surface_pressure": "surface_pressure_hpa",
    "precipitation": "precipitation_mm",
    "boundary_layer_height": "boundary_layer_height_m",
}

EXPECTED_UNITS = {
    "temperature_2m": "°C",
    "relative_humidity_2m": "%",
    "wind_speed_10m": "m/s",
    "wind_gusts_10m": "m/s",
    "wind_direction_10m": "°",
    "surface_pressure": "hPa",
    "precipitation": "mm",
    "boundary_layer_height": "m",
}

DEFAULT_START_DATE = os.environ.get("METEO_START_DATE", "2021-06-30")
DEFAULT_END_DATE = os.environ.get("METEO_END_DATE", "2026-06-30")
LOCATIONS_CSV = os.environ.get("METEO_LOCATIONS_CSV")
CHUNK_SIZE = int(os.environ.get("METEO_UPSERT_CHUNK_SIZE", "1000"))
CHUNK_DAYS = int(os.environ.get("METEO_CHUNK_DAYS", "0"))
MAX_RETRIES = int(os.environ.get("METEO_MAX_RETRIES", "20"))
BACKOFF_FACTOR = float(os.environ.get("METEO_BACKOFF_FACTOR", "1"))
REQUEST_TIMEOUT = (15, 180)

retry_strategy = Retry(
    total=MAX_RETRIES,
    backoff_factor=BACKOFF_FACTOR,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["HEAD", "GET", "OPTIONS"],
    raise_on_status=False,
)
SESSION = requests.Session()
adapter = HTTPAdapter(max_retries=retry_strategy)
SESSION.mount("https://", adapter)
SESSION.mount("http://", adapter)


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


def fetch_openaq_locations() -> pd.DataFrame:
    csv_path = Path(LOCATIONS_CSV) if LOCATIONS_CSV else Path(__file__).resolve().parent.parent / "locations.csv"
    df = pd.read_csv(csv_path)
    return df[["location_key", "latitude", "longitude"]].dropna()


def build_request_params(lat: float, lon: float, start_date: date, end_date: date) -> dict:
    return {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": "UTC",
        "temperature_unit": "celsius",
        "windspeed_unit": "ms",
        "precipitation_unit": "mm",
        "pressure_unit": "hpa",
    }


def validate_open_meteo_units(hourly_units: dict) -> None:
    if not hourly_units:
        raise ValueError("Open-Meteo response missing hourly units metadata.")

    mismatches = []
    for variable, expected in EXPECTED_UNITS.items():
        actual = hourly_units.get(variable)
        if actual != expected:
            mismatches.append(f"{variable}: expected {expected}, got {actual}")

    if mismatches:
        details = "; ".join(mismatches)
        raise ValueError(f"Open-Meteo returned unexpected units: {details}")


def fetch_open_meteo_data(lat: float, lon: float, start_date: date, end_date: date) -> pd.DataFrame:
    today = date.today()
    frames = []

    if end_date < start_date:
        return pd.DataFrame()

    if start_date <= today and end_date > today:
        historical_end = min(today, end_date)
        forecast_start = today + timedelta(days=1)
        frames.extend(chunked_open_meteo_fetch(lat, lon, start_date, historical_end, ARCHIVE_API_BASE))
        if end_date >= forecast_start:
            frames.extend(chunked_open_meteo_fetch(lat, lon, forecast_start, end_date, FORECAST_API_BASE))
    elif end_date <= today:
        frames.extend(chunked_open_meteo_fetch(lat, lon, start_date, end_date, ARCHIVE_API_BASE))
    else:
        frames.extend(chunked_open_meteo_fetch(lat, lon, start_date, end_date, FORECAST_API_BASE))

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df = df.loc[:, ~df.columns.duplicated()]
    return df


def chunked_open_meteo_fetch(lat: float, lon: float, start_date: date, end_date: date, endpoint: str) -> list[pd.DataFrame]:
    frames = []
    current_start = start_date
    total_days = (end_date - start_date).days + 1
    effective_chunk_days = total_days if CHUNK_DAYS <= 0 else CHUNK_DAYS

    while current_start <= end_date:
        current_end = min(current_start + timedelta(days=effective_chunk_days - 1), end_date)
        print(f"Request chunk {current_start} to {current_end} against {endpoint}")
        attempt = 0
        while attempt < MAX_RETRIES:
            try:
                frames.append(fetch_open_meteo_slice(lat, lon, current_start, current_end, endpoint))
                break
            except requests.exceptions.RequestException as exc:
                attempt += 1
                wait = 2 ** attempt
                print(f"Open-Meteo request failed (attempt {attempt}/{MAX_RETRIES}) for {current_start}-{current_end}: {exc}")
                if attempt >= MAX_RETRIES:
                    raise
                time.sleep(wait)
        current_start = current_end + timedelta(days=1)
    return frames


def fetch_open_meteo_slice(lat: float, lon: float, start_date: date, end_date: date, endpoint: str) -> pd.DataFrame:
    params = build_request_params(lat, lon, start_date, end_date)
    response = SESSION.get(endpoint, params=params, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    data = response.json()
    if "hourly" not in data:
        return pd.DataFrame()

    validate_open_meteo_units(data.get("hourly_units", {}))
    df = pd.DataFrame(data["hourly"])
    if df.empty:
        return df

    df = df.rename(columns=RENAMES)
    df = df.rename(columns={"time": "timestamp_utc"}) if "time" in df.columns else df
    if "timestamp_utc" not in df.columns:
        return pd.DataFrame()

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    return df


def align_hourly_data(df: pd.DataFrame, location_key: str, start_date: date, end_date: date) -> pd.DataFrame:
    if df.empty:
        index = pd.date_range(start=start_date, end=end_date + timedelta(days=0), freq="h", tz="UTC")
        aligned = pd.DataFrame(index=index)
        aligned.index.name = "timestamp_utc"
        aligned = aligned.reset_index()
        aligned["location_key"] = location_key
        return aligned

    df = df.set_index("timestamp_utc")
    expected_index = pd.date_range(start=start_date, end=end_date + timedelta(days=0), freq="h", tz="UTC")
    df = df.reindex(expected_index)
    df.index.name = "timestamp_utc"
    df = df.reset_index()
    df["location_key"] = location_key
    return df


def _na_to_none(value):
    try:
        if value is None or pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def upsert_meteo_hourly(conn, df: pd.DataFrame) -> None:
    if df.empty:
        print("No meteo rows to upsert.")
        return

    required = {"location_key", "timestamp_utc"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing required columns: {required - set(df.columns)}")

    rows = []
    for row in df.itertuples(index=False):
        rows.append(
            (
                row.location_key,
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
        INSERT INTO openaq.meteo_hourly (
            location_key, timestamp_utc,
            temperature_c, humidity_pct,
            wind_speed_ms, wind_gusts_ms, wind_dir_deg,
            surface_pressure_hpa, precipitation_mm,
            boundary_layer_height_m, source
        ) VALUES %s
        ON CONFLICT (location_key, timestamp_utc)
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
    print(f"Committed {len(rows)} meteo rows.")


def main() -> None:
    start_date = date.fromisoformat(DEFAULT_START_DATE)
    end_date = date.fromisoformat(DEFAULT_END_DATE)
    locations_df = fetch_openaq_locations()

    records = []
    for _, row in locations_df.iterrows():
        station = row["location_key"]
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        print(f"Fetching {station} {lat},{lon} from {start_date} to {end_date}")
        hourly_df = fetch_open_meteo_data(lat, lon, start_date, end_date)
        hourly_df = align_hourly_data(hourly_df, station, start_date, end_date)
        records.append(hourly_df)

    if not records:
        print("No station data fetched.")
        return

    final_weather_df = pd.concat(records, ignore_index=True)
    final_weather_df = final_weather_df.sort_values(["location_key", "timestamp_utc"])

    dsn = resolve_pg_dsn()
    with psycopg2.connect(dsn) as conn:
        upsert_meteo_hourly(conn, final_weather_df)


if __name__ == "__main__":
    main()
