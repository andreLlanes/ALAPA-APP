-- Populate measurements_withLA table with data from measurements_clean
-- Run this script AFTER running the LA ETL (openaq_etl.py and openaq_clean.py for US/LA data)
-- This will copy all Manila + Bangkok data from measurements_clean and add LA data

-- Step 1: Copy all existing measurements_clean data to measurements_withLA
INSERT INTO openaq.measurements_withLA 
    (location_key, timestamp_utc, pm1, pm25, pm10, temperature_c, humidity_pct, co2, tvoc, inserted_at)
SELECT 
    location_key, timestamp_utc, pm1, pm25, pm10, temperature_c, humidity_pct, co2, tvoc, inserted_at
FROM openaq.measurements_clean
ON CONFLICT (location_key, timestamp_utc) DO NOTHING;

-- Step 2: Verify the data was copied
SELECT 
    l.country_iso,
    COUNT(*) as measurement_count,
    MIN(mc.timestamp_utc) as earliest_record,
    MAX(mc.timestamp_utc) as latest_record
FROM openaq.measurements_withLA mc
JOIN openaq.locations l USING (location_key)
GROUP BY l.country_iso
ORDER BY l.country_iso;
