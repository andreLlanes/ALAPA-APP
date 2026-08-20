-- Run once before etl/openaq_clean.py.
-- Creates the cleaned measurements table and the model-ready training view.

CREATE TABLE IF NOT EXISTS openaq.measurements_clean (
    location_key   text             NOT NULL REFERENCES openaq.locations(location_key),
    timestamp_utc  timestamptz      NOT NULL,
    pm1            double precision,
    pm25           double precision,
    pm10           double precision,
    temperature_c  double precision,
    humidity_pct   double precision,
    co2            double precision,
    tvoc           double precision,
    inserted_at    timestamptz      NOT NULL DEFAULT now(),
    UNIQUE (location_key, timestamp_utc)
);

CREATE INDEX IF NOT EXISTS idx_measurements_clean_loc_ts
    ON openaq.measurements_clean (location_key, timestamp_utc DESC);

-- Duplicate table for Los Angeles data (for validation and comparison).
-- Same schema as measurements_clean but populated with LA-specific data.
CREATE TABLE IF NOT EXISTS openaq.measurements_withLA (
    location_key   text             NOT NULL REFERENCES openaq.locations(location_key),
    timestamp_utc  timestamptz      NOT NULL,
    pm1            double precision,
    pm25           double precision,
    pm10           double precision,
    temperature_c  double precision,
    humidity_pct   double precision,
    co2            double precision,
    tvoc           double precision,
    inserted_at    timestamptz      NOT NULL DEFAULT now(),
    UNIQUE (location_key, timestamp_utc)
);

CREATE INDEX IF NOT EXISTS idx_measurements_withla_loc_ts
    ON openaq.measurements_withLA (location_key, timestamp_utc DESC);

-- Model-ready view: joins cleaned measurements to locations and adds
-- city label, local-time features (local_hour, day_of_week).
-- Timestamps stay in UTC; local time is computed at query time.
CREATE OR REPLACE VIEW openaq.training_clean AS
SELECT
    mc.location_key,
    mc.timestamp_utc,
    mc.pm1,
    mc.pm25,
    mc.pm10,
    mc.temperature_c,
    mc.humidity_pct,
    mc.co2,
    mc.tvoc,
    CASE l.country_iso
        WHEN 'PH' THEN 'Manila'
        WHEN 'TH' THEN 'Bangkok'
        WHEN 'US' THEN 'Los Angeles'
    END                                                    AS city,
    l.country_iso,
    l.source,
    l.latitude,
    l.longitude,
    EXTRACT(hour FROM mc.timestamp_utc AT TIME ZONE
        CASE l.country_iso
            WHEN 'PH' THEN 'Asia/Manila'
            WHEN 'TH' THEN 'Asia/Bangkok'
            WHEN 'US' THEN 'America/Los_Angeles'
        END
    )::int                                                 AS local_hour,
    EXTRACT(dow FROM mc.timestamp_utc AT TIME ZONE
        CASE l.country_iso
            WHEN 'PH' THEN 'Asia/Manila'
            WHEN 'TH' THEN 'Asia/Bangkok'
            WHEN 'US' THEN 'America/Los_Angeles'
        END
    )::int                                                 AS day_of_week
FROM openaq.measurements_clean mc
JOIN openaq.locations           l  USING (location_key);

-- View for Los Angeles data with local-time features.
CREATE OR REPLACE VIEW openaq.training_withLA AS
SELECT
    mla.location_key,
    mla.timestamp_utc,
    mla.pm1,
    mla.pm25,
    mla.pm10,
    mla.temperature_c,
    mla.humidity_pct,
    mla.co2,
    mla.tvoc,
    CASE l.country_iso
        WHEN 'PH' THEN 'Manila'
        WHEN 'TH' THEN 'Bangkok'
        WHEN 'US' THEN 'Los Angeles'
    END                                                    AS city,
    l.country_iso,
    l.source,
    l.latitude,
    l.longitude,
    EXTRACT(hour FROM mla.timestamp_utc AT TIME ZONE
        CASE l.country_iso
            WHEN 'PH' THEN 'Asia/Manila'
            WHEN 'TH' THEN 'Asia/Bangkok'
            WHEN 'US' THEN 'America/Los_Angeles'
        END
    )::int                                                 AS local_hour,
    EXTRACT(dow FROM mla.timestamp_utc AT TIME ZONE
        CASE l.country_iso
            WHEN 'PH' THEN 'Asia/Manila'
            WHEN 'TH' THEN 'Asia/Bangkok'
            WHEN 'US' THEN 'America/Los_Angeles'
        END
    )::int                                                 AS day_of_week
FROM openaq.measurements_withLA mla
JOIN openaq.locations           l  USING (location_key);
