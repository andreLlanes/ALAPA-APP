
CREATE TABLE openaq.meteo_hourly (
    location_key text NOT NULL REFERENCES openaq.locations(location_key),
    latitude double precision,
    longitude double precision,
    timestamp_utc timestamptz NOT NULL,
    temperature_c   double precision,
    humidity_pct    double precision,
    wind_speed_ms   double precision,
    wind_gusts_ms   double precision,
    wind_dir_deg    double precision,
    surface_pressure_hpa    double precision,
    precipitation_mm    double precision,
    boundary_layer_height_m double precision,

    source text NOT NULL DEFAULT 'open-meteo',
    inserted_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (location_key, timestamp_utc)
);