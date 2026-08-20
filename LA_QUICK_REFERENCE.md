# Los Angeles Data Integration - Quick Reference

## 🚀 Quick Start (3 Steps)

### Step 1: Prepare Database

```bash
psql $PG_DSN -f sql/openaq_clean_schema.sql
```

### Step 2: Extract & Process LA Data (2-3 hours)

```bash
bash run_la_etl.sh
```

_Or manually:_

```bash
python etl/openaq_etl.py
python etl/meteo_grid.py
METEO_BBOX="33.800000,-118.500000,34.300000,-117.900000" python etl/OM_PM10_ETL.py
python etl/openaq_clean.py
psql $PG_DSN -f sql/populate_measurements_withLA.sql
```

### Step 3: Verify

```bash
python verify_la_data.py
```

---

## 📍 Bounding Boxes

```
Manila, PH:     14.316284, 120.868835, 14.781522, 121.143494
Bangkok, TH:    13.600000, 100.400000, 13.950000, 100.850000
Los Angeles, US: 33.800000, -118.500000, 34.300000, -117.900000
```

---

## 🗄️ Database Changes

| Item                  | Type              | Details                                      |
| --------------------- | ----------------- | -------------------------------------------- |
| `measurements_withLA` | NEW TABLE         | Combined measurements (Manila+Bangkok+LA)    |
| `training_withLA`     | NEW VIEW          | Training data with all 3 cities + local time |
| `training_clean`      | VIEW (updated)    | Added LA timezone support                    |
| `measurements_clean`  | TABLE (unchanged) | Still contains only Manila + Bangkok         |

---

## 📊 Tables

| Table               | Rows (Est.) | From                       |
| ------------------- | ----------- | -------------------------- |
| measurements_clean  | 4.0M        | PH (Manila) + TH (Bangkok) |
| measurements_withLA | 5.7M        | PH + TH + US (LA)          |

---

## 📈 Data Parameters

```
✓ PM2.5        (bounds: 0-500 µg/m³)
✓ PM10         (bounds: 0-500 µg/m³)
✓ Temperature  (bounds: -10 to 60°C)
✓ Humidity     (bounds: 0-100%)
✓ Wind Speed, Pressure, Precipitation (meteorological)
```

---

## 🔧 Environment Variables

```bash
# Required
OPENAQ_API_KEY="your-key"
PG_DSN="postgresql://user:pass@host/db"

# Optional
METEO_BBOX="33.800000,-118.500000,34.300000,-117.900000"  # LA for PM10
OPENAQ_START_DATE="2021-06-30"
OPENAQ_END_DATE="2026-06-30"
```

---

## 📁 Files Modified/Created

### Modified (5 files)

- ✏️ `etl/openaq_etl.py` - Added US country filter
- ✏️ `etl/meteo_grid.py` - Added Los Angeles city bbox
- ✏️ `etl/openaq_clean.py` - Updated comments
- ✏️ `etl/OM_PM10_ETL.py` - Added LA bbox documentation
- ✏️ `sql/openaq_clean_schema.sql` - Added tables/views

### Created (4 files)

- 📄 `sql/populate_measurements_withLA.sql` - Population script
- 🔨 `run_la_etl.sh` - Automated ETL pipeline
- 🐍 `verify_la_data.py` - Verification tool
- 📖 `LA_DATA_INTEGRATION.md` - Full documentation
- 📋 `LA_IMPLEMENTATION_SUMMARY.md` - Changes summary
- 💡 `LA_QUICK_REFERENCE.md` - This file

---

## 🔍 Verification Queries

### Check LA data in measurements_withLA

```sql
SELECT city, COUNT(*) FROM openaq.training_withLA WHERE country_iso='US' GROUP BY city;
```

### Compare all cities

```sql
SELECT city, AVG(pm25), STDDEV(pm25) FROM openaq.training_withLA GROUP BY city;
```

### Sensor health

```sql
SELECT l.location_key, l.city, COUNT(*) FROM openaq.measurements_withLA m
JOIN openaq.locations l USING (location_key)
WHERE l.country_iso='US' GROUP BY l.location_key, l.city;
```

---

## ⏱️ Expected Duration

| Stage               | Time          |
| ------------------- | ------------- |
| Database setup      | 1 min         |
| OpenAQ extraction   | 60-120 min    |
| Meteorological data | 10 min        |
| PM10 extraction     | 10 min        |
| Data cleaning       | 10 min        |
| Population          | 2 min         |
| **Total**           | **2-3 hours** |

---

## ⚠️ Common Issues

| Issue                     | Solution                                                            |
| ------------------------- | ------------------------------------------------------------------- |
| `duplicate key value`     | Script uses ON CONFLICT DO NOTHING; re-run is safe                  |
| No LA locations found     | Verify OPENAQ_API_KEY is valid; check OpenAQ website for LA sensors |
| measurements_withLA empty | Run: `psql $PG_DSN -f sql/populate_measurements_withLA.sql`         |
| Timezone errors           | Verify 'America/Los_Angeles' in training_withLA view                |
| Rate limit errors         | Check RPM/RPH settings; default is safe (45 req/min, 1500 req/hr)   |

---

## 📚 Documentation Files

| File                           | Purpose                               |
| ------------------------------ | ------------------------------------- |
| `LA_INTEGRATION.md`            | **Comprehensive guide** - Start here! |
| `LA_IMPLEMENTATION_SUMMARY.md` | Detailed change log & impact analysis |
| `LA_QUICK_REFERENCE.md`        | This file - Quick lookup              |

---

## 🎯 Using LA Data in Models

### For training with all 3 cities:

```python
# Load from new view
query = "SELECT * FROM openaq.training_withLA"
df = pd.read_sql(query, con=engine)

# Filter by city/country
la_data = df[df['country_iso'] == 'US']
all_data = df  # All 3 cities
```

### Timezone-aware calculations:

```python
# local_hour and day_of_week already computed in view
# Proper timezone per city: Manila (UTC+8), Bangkok (UTC+7), LA (UTC-7/-8)
```

---

## 📞 Support

- **Full Guide**: `LA_DATA_INTEGRATION.md`
- **Verification**: `python verify_la_data.py`
- **SQL Queries**: See LA_DATA_INTEGRATION.md → Queries section
- **Troubleshooting**: See LA_DATA_INTEGRATION.md → Troubleshooting section

---

## ✅ Checklist

- [ ] Update .env with OPENAQ_API_KEY
- [ ] Update .env with PG_DSN
- [ ] Run `psql $PG_DSN -f sql/openaq_clean_schema.sql`
- [ ] Run `bash run_la_etl.sh` (or run scripts individually)
- [ ] Run `python verify_la_data.py`
- [ ] Check LA data appears in measurements_withLA
- [ ] Update model pipelines to use training_withLA view
- [ ] Run sample queries to verify data quality
- [ ] Document any LA-specific data patterns

---

**Status**: ✅ Ready to execute  
**Last Updated**: 2026-08-20  
**Next Review**: After first successful LA data extraction
