"""
verify_la_data.py - Verify Los Angeles data extraction and quality

This script provides utilities to:
1. Check if LA data has been extracted
2. Verify data completeness and quality
3. Compare LA data statistics with other regions
4. Ensure measurements_withLA is properly populated
"""

import os
import sys
from datetime import datetime
import psycopg2
import psycopg2.extras
from tabulate import tabulate

def resolve_pg_dsn() -> str:
    """Build PostgreSQL DSN from environment variables."""
    dsn = os.environ.get("PG_DSN")
    if dsn:
        return dsn
    host = os.environ.get("PG_HOST", "localhost")
    port = os.environ.get("PG_PORT", "5432")
    dbname = os.environ.get("PG_DB")
    user = os.environ.get("PG_USER")
    password = os.environ.get("PG_PASSWORD")
    if not all([host, dbname, user, password]):
        raise ValueError("Missing PG_DSN or PG_HOST/PG_DB/PG_USER/PG_PASSWORD env vars.")
    return f"postgresql://{user}:{password}@{host}:{port}/{dbname}"


def check_la_locations(conn) -> dict:
    """Check if LA locations have been extracted."""
    query = """
        SELECT 
            country_iso,
            COUNT(*) as sensor_count,
            MIN(latitude) as min_lat,
            MAX(latitude) as max_lat,
            MIN(longitude) as min_lon,
            MAX(longitude) as max_lon
        FROM openaq.locations
        WHERE country_iso IS NOT NULL
        GROUP BY country_iso
        ORDER BY country_iso
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(query)
        return cur.fetchall()


def check_measurements_clean(conn) -> dict:
    """Check data in measurements_clean table."""
    query = """
        SELECT 
            l.country_iso,
            COUNT(*) as measurement_count,
            MIN(mc.timestamp_utc) as earliest,
            MAX(mc.timestamp_utc) as latest,
            COUNT(DISTINCT l.location_key) as unique_locations,
            ROUND(100.0 * COUNT(CASE WHEN mc.pm25 IS NOT NULL THEN 1 END) / COUNT(*), 2) as pm25_coverage_pct,
            ROUND(100.0 * COUNT(CASE WHEN mc.pm10 IS NOT NULL THEN 1 END) / COUNT(*), 2) as pm10_coverage_pct,
            ROUND(100.0 * COUNT(CASE WHEN mc.temperature_c IS NOT NULL THEN 1 END) / COUNT(*), 2) as temp_coverage_pct
        FROM openaq.measurements_clean mc
        JOIN openaq.locations l USING (location_key)
        GROUP BY l.country_iso
        ORDER BY l.country_iso
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(query)
        return cur.fetchall()


def check_measurements_withLA(conn) -> dict:
    """Check data in measurements_withLA table."""
    query = """
        SELECT 
            l.country_iso,
            COUNT(*) as measurement_count,
            MIN(mla.timestamp_utc) as earliest,
            MAX(mla.timestamp_utc) as latest,
            COUNT(DISTINCT l.location_key) as unique_locations,
            ROUND(100.0 * COUNT(CASE WHEN mla.pm25 IS NOT NULL THEN 1 END) / COUNT(*), 2) as pm25_coverage_pct,
            ROUND(100.0 * COUNT(CASE WHEN mla.pm10 IS NOT NULL THEN 1 END) / COUNT(*), 2) as pm10_coverage_pct,
            ROUND(100.0 * COUNT(CASE WHEN mla.temperature_c IS NOT NULL THEN 1 END) / COUNT(*), 2) as temp_coverage_pct
        FROM openaq.measurements_withLA mla
        JOIN openaq.locations l USING (location_key)
        GROUP BY l.country_iso
        ORDER BY l.country_iso
    """
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(query)
        return cur.fetchall()


def main():
    """Run all verification checks."""
    try:
        dsn = resolve_pg_dsn()
        conn = psycopg2.connect(dsn)
        
        print("=" * 80)
        print("Los Angeles Data Extraction Verification Report")
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print("=" * 80)
        print()
        
        # Check 1: Locations
        print("\n[1] SENSOR LOCATIONS BY COUNTRY")
        print("-" * 80)
        locations = check_la_locations(conn)
        if locations:
            headers = ["Country", "Sensor Count", "Min Lat", "Max Lat", "Min Lon", "Max Lon"]
            rows = [
                [
                    l["country_iso"],
                    l["sensor_count"],
                    f"{l['min_lat']:.4f}",
                    f"{l['max_lat']:.4f}",
                    f"{l['min_lon']:.4f}",
                    f"{l['max_lon']:.4f}",
                ]
                for l in locations
            ]
            print(tabulate(rows, headers=headers, tablefmt="grid"))
            if any(loc["country_iso"] == "US" for loc in locations):
                print("✓ Los Angeles (US) locations found!")
            else:
                print("✗ Los Angeles (US) locations NOT found - LA ETL may not have run yet")
        print()
        
        # Check 2: measurements_clean data
        print("\n[2] DATA QUALITY - measurements_clean (Manila + Bangkok)")
        print("-" * 80)
        clean_data = check_measurements_clean(conn)
        if clean_data:
            headers = ["Country", "Records", "Earliest", "Latest", "Locations", "PM2.5 %", "PM10 %", "Temp %"]
            rows = [
                [
                    d["country_iso"],
                    f"{d['measurement_count']:,}",
                    d["earliest"].strftime("%Y-%m-%d") if d["earliest"] else "N/A",
                    d["latest"].strftime("%Y-%m-%d") if d["latest"] else "N/A",
                    d["unique_locations"],
                    d["pm25_coverage_pct"],
                    d["pm10_coverage_pct"],
                    d["temp_coverage_pct"],
                ]
                for d in clean_data
            ]
            print(tabulate(rows, headers=headers, tablefmt="grid"))
        print()
        
        # Check 3: measurements_withLA data
        print("\n[3] DATA QUALITY - measurements_withLA (Manila + Bangkok + LA)")
        print("-" * 80)
        withla_data = check_measurements_withLA(conn)
        if withla_data:
            headers = ["Country", "Records", "Earliest", "Latest", "Locations", "PM2.5 %", "PM10 %", "Temp %"]
            rows = [
                [
                    d["country_iso"],
                    f"{d['measurement_count']:,}",
                    d["earliest"].strftime("%Y-%m-%d") if d["earliest"] else "N/A",
                    d["latest"].strftime("%Y-%m-%d") if d["latest"] else "N/A",
                    d["unique_locations"],
                    d["pm25_coverage_pct"],
                    d["pm10_coverage_pct"],
                    d["temp_coverage_pct"],
                ]
                for d in withla_data
            ]
            print(tabulate(rows, headers=headers, tablefmt="grid"))
            
            # Summary
            total_withla = sum(d["measurement_count"] for d in withla_data)
            total_clean = sum(d["measurement_count"] for d in clean_data)
            la_count = sum(d["measurement_count"] for d in withla_data if d["country_iso"] == "US")
            
            print()
            print(f"SUMMARY:")
            print(f"  Total records in measurements_clean: {total_clean:,}")
            print(f"  Total records in measurements_withLA: {total_withla:,}")
            print(f"  New LA records added: {la_count:,}")
            print(f"  Growth: {((total_withla - total_clean) / total_clean * 100):.1f}% increase" if total_clean > 0 else "  No data in measurements_clean yet")
        else:
            print("⚠ measurements_withLA is empty - run populate_measurements_withLA.sql after LA ETL")
        
        print()
        print("=" * 80)
        print("Verification Complete")
        print("=" * 80)
        
        conn.close()
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
