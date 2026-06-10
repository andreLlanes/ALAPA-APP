DROP SCHEMA IF EXISTS openaq CASCADE;
CREATE SCHEMA openaq;

CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE openaq.locations (
    location_key text PRIMARY KEY,
    source text NOT NULL,
    external_id text,
    name text,
    locality text,
    country_iso text,
    owner_name text,
    provider_name text,
    is_monitor boolean,
    is_mobile boolean,
    latitude double precision,
    longitude double precision,
    geom geography(Point, 4326),
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE openaq.measurements (
    id bigserial PRIMARY KEY,
    location_key text NOT NULL REFERENCES openaq.locations(location_key),
    timestamp_utc timestamptz NOT NULL,
    pm25 double precision,
    temperature_c double precision,
    humidity_pct double precision,
    pm1 double precision,
    pm10 double precision,
    co2 double precision,
    tvoc double precision,
    has_pm1 boolean NOT NULL DEFAULT false,
    has_pm10 boolean NOT NULL DEFAULT false,
    has_co2 boolean NOT NULL DEFAULT false,
    has_tvoc boolean NOT NULL DEFAULT false,
    source text NOT NULL,
    raw_payload jsonb,
    inserted_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (location_key, timestamp_utc)
);

CREATE INDEX idx_locations_geom ON openaq.locations USING GIST (geom);
CREATE INDEX idx_measurements_ts ON openaq.measurements (timestamp_utc);
CREATE INDEX idx_measurements_loc ON openaq.measurements (location_key);
-- Compound index for fast per-location watermark queries (MAX(timestamp_utc) GROUP BY location_key)
CREATE INDEX idx_measurements_loc_ts ON openaq.measurements (location_key, timestamp_utc DESC);

-- ---------------------------------------------------------------------------
-- Model-ready hourly view (the "merged dataset" for transfer-learning + forecasting).
-- One row per (sensor location, hour) with the 5 forecast features, a city label, and
-- coordinates so the modelling step can pool Manila + Singapore + Bangkok. Export with:
--     \copy (SELECT * FROM openaq.training_hourly ORDER BY city, location_key, timestamp_utc) TO 'merged.csv' CSV HEADER
-- or read directly with pandas.read_sql("SELECT * FROM openaq.training_hourly", conn).
-- ---------------------------------------------------------------------------
CREATE VIEW openaq.training_hourly AS
SELECT
    m.location_key,
    m.timestamp_utc,
    CASE l.country_iso
        WHEN 'PH' THEN 'Manila'
        WHEN 'SG' THEN 'Singapore'
        WHEN 'TH' THEN 'Bangkok'
        ELSE COALESCE(l.locality, l.country_iso, 'unknown')
    END                          AS city,
    l.country_iso,
    l.source,
    l.latitude,
    l.longitude,
    m.pm25,
    m.pm1,
    m.pm10,
    m.temperature_c,
    m.humidity_pct
FROM openaq.measurements m
JOIN openaq.locations l USING (location_key);
