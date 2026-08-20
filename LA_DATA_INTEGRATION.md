# Los Angeles Data Integration Guide

## Overview

This guide explains how to extract, process, and integrate Los Angeles (LA) air quality data into your ALAPA project, replacing the Singapore data that was previously out of scope.

## Quick Start

### Prerequisites

- PostgreSQL database set up with OpenAQ schema
- Valid OpenAQ API key
- Python 3.8+
- Environment variables configured (see Configuration section)

### Step 1: Update Database Schema

First, create the new `measurements_withLA` table and views:

```bash
psql $PG_DSN -f sql/openaq_clean_schema.sql
```

This command creates:

- `openaq.measurements_withLA` - Table for all measurements including LA
- `openaq.training_withLA` - View with local timezone features for LA

### Step 2: Extract LA Data

Run the complete ETL pipeline:

```bash
# Option A: Use the automated script (Linux/Mac)
bash run_la_etl.sh

# Option B: Run steps manually
python etl/openaq_etl.py          # Extract OpenAQ measurements
python etl/meteo_grid.py          # Extract meteorological data
METEO_BBOX="33.800000,-118.500000,34.300000,-117.900000" python etl/OM_PM10_ETL.py  # Extract PM10
python etl/openaq_clean.py        # Clean and validate data
```

### Step 3: Populate measurements_withLA

After extraction, populate the new combined measurements table:

```bash
psql $PG_DSN -f sql/populate_measurements_withLA.sql
```

### Step 4: Verify Data

Check that LA data was extracted and integrated correctly:

```bash
python verify_la_data.py
```

## Configuration

### Environment Variables

Set these before running the ETL scripts:

```bash
# Database connection (choose one method)
export PG_DSN="postgresql://user:password@localhost:5432/alapa_db"

# OR individual components
export PG_HOST="localhost"
export PG_PORT="5432"
export PG_DB="alapa_db"
export PG_USER="postgres"
export PG_PASSWORD="your_password"

# OpenAQ API
export OPENAQ_API_KEY="your_api_key"

# Data extraction windows (optional, defaults to 2021-06-30 to 2026-06-30)
export OPENAQ_START_DATE="2021-06-30"
export OPENAQ_END_DATE="2026-06-30"
export METEO_START_DATE="2021-06-30"
export METEO_END_DATE="2026-06-30"
```

### Regional Bounding Boxes

The following bounding boxes are used for data extraction (format: min_lat, min_lon, max_lat, max_lon):

**Manila, Philippines (PH)**

```
bbox: 14.316284,120.868835,14.781522,121.143494
```

**Bangkok, Thailand (TH)**

```
bbox: 13.600000,100.400000,13.950000,100.850000
```

**Los Angeles, USA (US)**

```
bbox: 33.800000,-118.500000,34.300000,-117.900000
Covers: Greater LA metropolitan area including downtown LA, Long Beach, and Pasadena
```

## Data Sources

### OpenAQ Measurements

- **Service**: OpenAQ v3 API
- **Parameters**: PM2.5, PM10, Temperature, Humidity
- **Data Type**: Hourly aggregated from ground sensors (AirGradient, AirNow, Clarity, etc.)
- **Rate Limit**: 60 requests/min, 2000 requests/hour

### Meteorological Data

- **Service**: Open-Meteo Archive API
- **Parameters**: Temperature, Humidity, Wind Speed, Wind Gust, Wind Direction, Surface Pressure, Precipitation, Boundary Layer Height
- **Grid Resolution**: 0.25° × 0.25°
- **Data Type**: Hourly historical and forecast

### Air Quality (PM10)

- **Service**: Open-Meteo Air Quality API
- **Domain**: CAMS Global
- **Parameter**: PM10
- **Grid Resolution**: Varies by domain

## Database Schema

### Tables

#### `openaq.locations`

Sensor metadata including:

- `location_key`: Unique sensor identifier
- `country_iso`: Country code (PH, TH, US)
- `city`: City name (Manila, Bangkok, Los Angeles)
- `latitude`, `longitude`: Coordinates
- `source`: Data provider (airgradient, air4thai, clarity, etc.)

#### `openaq.measurements` (raw)

Raw measurements from OpenAQ API with quality flags and metadata.

#### `openaq.measurements_clean`

Cleaned measurements for Manila and Bangkok:

- Physical bounds checking (e.g., PM2.5: 0-500 µg/m³)
- Flatline detection (removes sensor malfunctions)
- Data quality validation

#### `openaq.measurements_withLA` ✨ (NEW)

Combined cleaned measurements including Los Angeles:

- All rows from `measurements_clean` (Manila + Bangkok)
- Plus cleaned LA measurements
- Same schema: location_key, timestamp_utc, pm1, pm25, pm10, temperature_c, humidity_pct, co2, tvoc

#### `openaq.meteo_hourly_grid`

Gridded meteorological data for all regions.

#### `openaq.meteo_pm10_hourly_grid`

Gridded PM10 air quality data for all regions.

### Views

#### `openaq.training_clean`

Training data view for Manila + Bangkok:

- Joins measurements_clean with locations
- Adds city labels
- Computes local time features (local_hour, day_of_week)
- Uses appropriate timezone for each city

#### `openaq.training_withLA` ✨ (NEW)

Training data view for all three cities:

- Joins measurements_withLA with locations
- Includes Manila (Asia/Manila tz)
- Includes Bangkok (Asia/Bangkok tz)
- Includes Los Angeles (America/Los_Angeles tz)
- Computes local time features per timezone

**View Columns**:

- `location_key`, `timestamp_utc`, `pm1`, `pm25`, `pm10`
- `temperature_c`, `humidity_pct`, `co2`, `tvoc`
- `city`: City name
- `country_iso`: Country code
- `source`: Data provider
- `latitude`, `longitude`: Coordinates
- `local_hour`: Hour in local timezone (0-23)
- `day_of_week`: Day of week in local timezone (0-6, where 0=Sunday)

## Data Quality

### Cleaning Rules

**Physical Bounds** (applied during `openaq_clean.py`):

```
PM2.5:        0 - 500 µg/m³
PM10:         0 - 500 µg/m³
Temperature: -10 - 60°C
Humidity:     0 - 100%
```

**Flatline Detection**:

- Removes ≥3 consecutive identical hourly values (suspicious sensor reading)
- Treats NaN values as breaks in the run
- Common causes: sensor malfunction, calibration pause

### Coverage Statistics

Use the verification script to check data completeness:

```bash
python verify_la_data.py
```

Output includes:

- Sensor counts per country
- Date ranges (earliest to latest record)
- Data coverage percentages (% non-null values per parameter)

## Workflow

### Phase 1: Initial Setup (Run Once)

```bash
# 1. Create schema
psql $PG_DSN -f sql/openaq_clean_schema.sql

# 2. Export environment variables
export OPENAQ_API_KEY="..."
export PG_DSN="..."
```

### Phase 2: LA Data Extraction (First Run)

```bash
# 3. Extract LA data
python etl/openaq_etl.py          # 1-2 hours
python etl/meteo_grid.py          # 5-10 minutes
METEO_BBOX="33.800000,-118.500000,34.300000,-117.900000" \
  python etl/OM_PM10_ETL.py       # 5-10 minutes
python etl/openaq_clean.py        # 5-10 minutes

# 4. Populate combined table
psql $PG_DSN -f sql/populate_measurements_withLA.sql

# 5. Verify
python verify_la_data.py
```

### Phase 3: Ongoing Updates (Weekly/Monthly)

```bash
# Re-run ETL to add new measurements
python etl/openaq_etl.py
python etl/openaq_clean.py

# Optional: Update meteorological data
python etl/meteo_grid.py
METEO_BBOX="33.800000,-118.500000,34.300000,-117.900000" \
  python etl/OM_PM10_ETL.py

# Note: populate_measurements_withLA.sql uses ON CONFLICT DO NOTHING,
# so you can run it repeatedly without duplication
psql $PG_DSN -f sql/populate_measurements_withLA.sql
```

## Queries

### Query All LA Data

```sql
SELECT * FROM openaq.training_withLA
WHERE country_iso = 'US'
ORDER BY timestamp_utc DESC
LIMIT 100;
```

### Compare Cities

```sql
SELECT
    city,
    COUNT(*) as record_count,
    AVG(pm25) as avg_pm25,
    STDDEV(pm25) as std_pm25,
    MIN(temperature_c) as min_temp,
    MAX(temperature_c) as max_temp
FROM openaq.training_withLA
GROUP BY city
ORDER BY city;
```

### Time-based Analysis

```sql
SELECT
    city,
    local_hour,
    AVG(pm25) as avg_pm25,
    STDDEV(pm25) as std_pm25
FROM openaq.training_withLA
WHERE timestamp_utc >= NOW() - INTERVAL '30 days'
GROUP BY city, local_hour
ORDER BY city, local_hour;
```

### Sensor Health Check

```sql
SELECT
    l.location_key,
    l.city,
    l.source,
    COUNT(*) as total_records,
    COUNT(CASE WHEN pm25 IS NOT NULL THEN 1 END) as pm25_records,
    COUNT(CASE WHEN pm10 IS NOT NULL THEN 1 END) as pm10_records,
    MAX(timestamp_utc) as last_update
FROM openaq.measurements_withLA m
JOIN openaq.locations l USING (location_key)
WHERE l.country_iso = 'US'
GROUP BY l.location_key, l.city, l.source
ORDER BY l.location_key;
```

## Code Changes Summary

### Modified Files

1. **etl/openaq_etl.py**
   - Added `"US"` country filter with LA bounding box
   - Accepts all providers (unlike Bangkok which is restricted to AirGradient + air4thai)

2. **etl/meteo_grid.py**
   - Added `"Los Angeles"` to `CITY_BBOXES` dictionary
   - Grid data extraction now covers LA metropolitan area

3. **etl/openaq_clean.py**
   - Updated comments to reflect LA inclusion
   - Filter still excludes SG (Singapore) as before
   - LA data now cleaned alongside Manila/Bangkok

4. **etl/OM_PM10_ETL.py**
   - Added comments documenting regional bbox options
   - Environment variable `METEO_BBOX` can be set to LA coordinates for PM10 extraction

5. **sql/openaq_clean_schema.sql**
   - Added `measurements_withLA` table (duplicate of `measurements_clean` structure)
   - Added `training_withLA` view with LA timezone support
   - Updated `training_clean` view to include LA city label

### New Files

1. **sql/populate_measurements_withLA.sql**
   - SQL migration script to populate `measurements_withLA`
   - Uses ON CONFLICT DO NOTHING for idempotent execution

2. **run_la_etl.sh**
   - Automated bash script for complete LA ETL pipeline
   - Provides step-by-step execution with progress indicators
   - Handles all database setup and data extraction

3. **verify_la_data.py**
   - Python verification utility for data quality checks
   - Compares measurements_clean vs measurements_withLA
   - Provides summary statistics and coverage percentages

## Troubleshooting

### Issue: "duplicate key value violates unique constraint"

**Cause**: Running populate_measurements_withLA.sql multiple times with overlapping data
**Solution**: The script uses ON CONFLICT DO NOTHING, so rerun is safe; or drop and recreate the table

### Issue: No LA locations found after running openaq_etl.py

**Cause**: OpenAQ API may have no sensors in the specified bbox, or API key is invalid
**Solution**:

- Verify OPENAQ_API_KEY is valid
- Check OpenAQ website to confirm sensors exist in LA area
- Review API logs: `tail -f etl/openaq_etl.log`

### Issue: measurements_withLA remains empty

**Cause**: populate_measurements_withLA.sql not run, or measurements_clean is empty
**Solution**:

- Ensure openaq_clean.py completed successfully
- Run: `psql $PG_DSN -f sql/populate_measurements_withLA.sql`
- Verify with: `SELECT COUNT(*) FROM openaq.measurements_withLA;`

### Issue: Time zone issues in local_hour calculation

**Cause**: PostgreSQL timezone not set correctly
**Solution**: Verify timezone in training_withLA view uses correct abbreviation:

- Manila: `'Asia/Manila'`
- Bangkok: `'Asia/Bangkok'`
- Los Angeles: `'America/Los_Angeles'`

## Performance Notes

- **OpenAQ API extraction**: ~2 hours for 5-year backfill (rate-limited to 60 req/min)
- **Meteorological grid**: ~5-10 minutes per city (depends on grid resolution and date range)
- **Data cleaning**: ~5-10 minutes for 5+ years of 3-city data
- **Database population**: ~1 minute for measurements_withLA

## Next Steps

1. Review LA data quality using `verify_la_data.py`
2. Update any machine learning models to use `training_withLA` view
3. Adjust model features/preprocessing if needed for US timezone/data patterns
4. Document any LA-specific data characteristics or limitations

## Support

For issues or questions:

- Check OpenAQ API documentation: https://docs.openaq.org/
- Review Open-Meteo API: https://open-meteo.com/
- See main project README.md for overall project structure
