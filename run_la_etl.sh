#!/bin/bash
# Script to run the complete ETL pipeline for Los Angeles (LA) data extraction and cleaning
# This script assumes you have already set up the PostgreSQL database and have valid OpenAQ API credentials

# Configuration - Set these environment variables before running:
# export OPENAQ_API_KEY="your-api-key"
# export PG_DSN="postgresql://user:password@host:port/dbname"
# OR
# export PG_HOST="localhost"
# export PG_PORT="5432"
# export PG_DB="your_db"
# export PG_USER="your_user"
# export PG_PASSWORD="your_password"

set -e

echo "=========================================="
echo "Los Angeles (LA) Air Quality Data ETL"
echo "=========================================="
echo ""

# Step 1: Create schema if not exists
echo "[1/5] Setting up database schema..."
psql ${PG_DSN:-postgresql://$PG_USER:$PG_PASSWORD@$PG_HOST:$PG_PORT/$PG_DB} -f sql/openaq_clean_schema.sql

# Step 2: Extract OpenAQ raw measurements for LA
echo "[2/5] Extracting OpenAQ measurements for Los Angeles (US)..."
echo "     (This will fetch historical hourly data from OpenAQ API)"
python etl/openaq_etl.py

# Step 3: Extract meteorological data for LA
echo "[3/5] Extracting meteorological data for Los Angeles..."
echo "     (This includes temperature, humidity, wind, pressure, etc.)"
python etl/meteo_grid.py

# Step 4: Extract PM10 air quality data for LA
echo "[4/5] Extracting PM10 air quality data for Los Angeles..."
echo "     (Set METEO_BBOX='33.800000,-118.500000,34.300000,-117.900000' for LA)"
METEO_BBOX="33.800000,-118.500000,34.300000,-117.900000" python etl/OM_PM10_ETL.py

# Step 5: Clean measurements (apply bounds and flatline detection)
echo "[5/5] Cleaning and applying quality checks..."
echo "     (This removes outliers and suspicious patterns)"
python etl/openaq_clean.py

# Step 6: Populate measurements_withLA table
echo ""
echo "[6/6] Populating measurements_withLA table with combined data..."
psql ${PG_DSN:-postgresql://$PG_USER:$PG_PASSWORD@$PG_HOST:$PG_PORT/$PG_DB} -f sql/populate_measurements_withLA.sql

echo ""
echo "=========================================="
echo "ETL Complete!"
echo "=========================================="
echo ""
echo "Summary:"
echo "  - Raw OpenAQ data: openaq.measurements"
echo "  - Cleaned data (Manila + Bangkok): openaq.measurements_clean"
echo "  - Cleaned data with LA (all three cities): openaq.measurements_withLA"
echo "  - Training view (original): openaq.training_clean"
echo "  - Training view with LA: openaq.training_withLA"
echo ""
echo "Next steps:"
echo "  1. Verify data quality with: SELECT * FROM openaq.training_withLA LIMIT 10;"
echo "  2. Check data distribution: SELECT city, COUNT(*) FROM openaq.training_withLA GROUP BY city;"
echo ""
