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
