import os
import time
import json
import shutil
import schedule
import logging
import boto3
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime, date
from dateutil.relativedelta import relativedelta
from sentinelhub import(
    SHConfig,
    DataCollection,
    SentinelHubCatalog,
    SentinelHubRequest,
    SentinelHubStatistical,
    BBox,
    bbox_to_dimensions,
    CRS,
    MimeType,
    Geometry,
)

load_dotenv(dotenv_path=Path(__file__).parent / ".env")
logger = logging.getLogger(__name__)

config = SHConfig()
config.sh_client_id = os.environ["SH_CLIENT_ID"]
config.sh_client_secret = os.environ["SH_CLIENT_SECRET"]
config.sh_base_url = os.environ["SH_BASE_URL"]
config.sh_token_url = os.environ["SH_TOKEN_URL"]

OUTPUT_DIR = Path("outputs/ndvi")
STATE_FILE = Path("outputs/ndvi/.collected.json")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

S3_BUCKET = os.environ["S3_BUCKET"]
S3_PREFIX = "ndvi"

RESOLUTION = 100
CLOUD_COVER_THRESHOLD = 80
BACKFILL_YEARS = 5
BACKFILL_WINDOW_MONTHS = 4

aoi_bbox = BBox(bbox=[120.868835,14.316284,121.143494,14.781522], crs=CRS.WGS84)
aoi_size = bbox_to_dimensions(aoi_bbox, resolution=RESOLUTION)

evalscript = """
    function setup(){
        return{
            input: [{
                bands: ["B04", "B08", "SCL", "dataMask"]
            }],
            mosaicking: "ORBIT",
            output: {
                bands: 2,
                sampleType: "FLOAT32"
            }
        }
    }
    function evaluatePixel(samples){
        var ndvi = [];
    
        for (var i = 0; i < samples.length; i++){
            var scl = samples[i].SCL;

            if (scl === 3 || (scl >= 8 && scl <= 11)) continue;
            if (samples[i].dataMask === 0) continue;

            var b8 = samples[i].B08, b4 = samples[i].B04;
            if (b8 + b4 === 0) continue;
            ndvi.push((b8 - b4) / (b8 + b4));
        }

        if (ndvi.length === 0) return [0, 0];

        ndvi.sort((a,b) => a - b);
        var mid = Math.floor(ndvi.length / 2);
        var median = ndvi.length % 2 !== 0 ? ndvi[mid] : (ndvi[mid-1] + ndvi[mid]) / 2;
        return [median, 1];
    }
"""

def load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"collected": []}


def save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def get_last_quarter_window() -> tuple[str, str]:
    end = date.today().replace(day=1)
    start = end - relativedelta(months=4)
    return str(start), str(end)


def get_backfill_windows(years: int = BACKFILL_YEARS, window_months: int = BACKFILL_WINDOW_MONTHS) -> list[tuple[date, date]]:
    end = date.today().replace(day=1)
    n_windows = (years * 12) // window_months

    windows = []
    for _ in range(n_windows):
        start = end - relativedelta(months=window_months)
        windows.append((start, end))
        end = start
    return windows


def build_ndvi_request(start: str, end: str, maxCloudCoverage = CLOUD_COVER_THRESHOLD) -> SentinelHubRequest:
    return SentinelHubRequest(
        evalscript=evalscript,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A.define_from(
                    name="s2l2a", service_url="https://sh.dataspace.copernicus.eu"
                ),
                time_interval=(start, end),
                other_args={
                    "dataFilter": {
                        #"mosaickingOrder": "leastCC",
                        "maxCloudCoverage": maxCloudCoverage
                    }
                }
            )
        ], 
        responses = [SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox = aoi_bbox,
        size = aoi_size,
        config = config,
        data_folder = str(OUTPUT_DIR)
    )

def save_request_output(request: SentinelHubRequest, filename: Path):
    saved = OUTPUT_DIR / request.get_filename_list()[0]
    filename.parent.mkdir(parents=True, exist_ok=True)
    saved.replace(filename)
    shutil.rmtree(saved.parent)


def upload_to_s3(local_path: Path):
    s3 = boto3.client("s3")
    key = f"{S3_PREFIX}/{local_path.name}"
    s3.upload_file(str(local_path), S3_BUCKET, key)
    logger.info(f"Uploaded to s3://{S3_BUCKET}/{key}")


def collect_ndvi():
    start, end = get_last_quarter_window()
    logger.info(f"Collecting Sentinel-2 NDVI: {start} to {end}.")
    request = build_ndvi_request(start, end, CLOUD_COVER_THRESHOLD)

    data = request.get_data(save_data=True)
    filename = OUTPUT_DIR / f"S2_NDVI_{start}_{end}.tif"
    save_request_output(request, filename)
    logger.info(f"NDVI GeoTIFF saved {filename}.")
    upload_to_s3(filename)
    return data

def backfill_ndvi(years: int = BACKFILL_YEARS):
    logger.info(f"=== NDVI backfill start (last {years} year(s)) ===")
    state = load_state()
    collected = state.get("collected", [])
    total = 0

    for start, end in get_backfill_windows(years):
        label = f"{start}_{end}"
        if label in collected:
            logger.info(f"Already collected: {label}")
            continue

        try:
            logger.info(f"Backfill collecting Sentinel-2 NDVI: {start} to {end}.")
            request = build_ndvi_request(str(start), str(end), CLOUD_COVER_THRESHOLD)
            request.get_data(save_data=True)
            filename = OUTPUT_DIR / f"S2_NDVI_{start}_{end}.tif"
            save_request_output(request, filename)
            logger.info(f"NDVI GeoTIFF saved {filename}.")
            upload_to_s3(filename)
            collected.append(label)
            total += 1
        except Exception as e:
            logger.error(f"Backfill failed for {start} to {end}: {e}")

    state["collected"] = collected
    save_state(state)
    logger.info(f"=== Backfill done. {total} new window(s) collected. ===")

def run_scheduler():
    trigger_months = {1,5,9}
    def job():
        if datetime.utcnow().month in trigger_months and datetime.utcnow().day == 1:
            try:
                collect_ndvi()
            except Exception as e:
                logger.exception(f"NDVI collection failed: {e}")
    schedule.every().day.at("06:00").do(job)
    logger.info(f"NDVI Scheduler running.")
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["scheduler", "now", "backfill"], default="scheduler",
                        help="now = run once immediately; scheduler = run on 4-month cycle; backfill = collect past N years")
    parser.add_argument("--years", type=int, default=BACKFILL_YEARS, help="How many years back to backfill")
    args = parser.parse_args()

    if args.mode == "now":
        collect_ndvi()
    elif args.mode == "backfill":
        backfill_ndvi(args.years)
    else:
        run_scheduler()