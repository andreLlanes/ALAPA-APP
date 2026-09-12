import os
import time
import schedule
import logging
import json
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime, timezone, timedelta
from sentinelhub import(
    SentinelHubRequest,
    SentinelHubCatalog,
    DataCollection,
    MimeType,
    BBox,
    CRS,
    SHConfig,
    bbox_to_dimensions
)

load_dotenv(dotenv_path=Path(__file__).parent / ".env")
logger = logging.getLogger(__name__)

config = SHConfig()
config.sh_client_id = os.environ["SH_CLIENT_ID"]
config.sh_client_secret = os.environ["SH_CLIENT_SECRET"]
config.sh_base_url = os.environ["SH_BASE_URL"]
config.sh_token_url = os.environ["SH_TOKEN_URL"]

OUTPUT_DIR = Path("outputs/tropomi")
STATE_FILE = Path("outputs/tropomi/.downloaded.json")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

aoi_bbox = BBox(bbox=[120.868835,14.316284,121.143494,14.781522], crs=CRS.WGS84)

S5P_COLLECTION = DataCollection.define(
    name = "sentinel_5p",
    api_id = "sentinel-5p-l2",
    service_url = "https://sh.dataspace.copernicus.eu"
)

PRODUCTS = {
    "NO2": {"band": "NO2", "resolution": 5500, "granule_tag": "L2__NO2___"},
    "CO": {"band": "CO", "resolution": 7000, "granule_tag": "L2__CO____"},
    "SO2": {"band": "SO2", "resolution": 5500, "granule_tag": "L2__SO2___"},
    "O3": {"band": "O3", "resolution": 5500, "granule_tag": "L2__O3____"}
}
def make_evalscript(band: str) -> str:
    return f"""
        function setup(){{
            return{{
                input: [{{
                    bands: ["{band}", "dataMask"]
                }}],
                output: {{bands: 2, sampleType: "FLOAT32"}}
            }}
        }}

        function evaluatePixel(sample){{
            if (sample.dataMask === 0) return [0, 0];
            return [sample.{band}, 1];
        }}
    """
def load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {p: [] for p in PRODUCTS}

def save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

def get_new_granules(product_key: str, granule_tag:str, since: datetime) -> list[dict]:
    catalog = SentinelHubCatalog(config=config)
    now = datetime.now(timezone.utc)

    search_iterator = catalog.search(
        collection=S5P_COLLECTION,
        bbox=aoi_bbox,
        time=(since,now),
        fields={"include": ["id", "properties.datetime"], "exclude": []}
    )

    granules = []
    for item in search_iterator:
        granule_id = item["id"]
        if granule_tag not in granule_id:
            continue
        try:
            parts = granule_id.split("_")
            dt_parts = [p for p in parts if len(p) == 15 and "T" in p]
            start = datetime.strptime(dt_parts[0], "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
            end   = datetime.strptime(dt_parts[1], "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
        except Exception:
            dt_str = item["properties"]["datetime"]
            dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
            start = dt - timedelta(minutes=30)
            end = dt + timedelta(minutes=30)
        granules.append({"id": granule_id, "start": start, "end": end})
    logger.info(f"{product_key}: {len(granules)} new granule(s) since {since.date()}")
    return granules

def download_granule(product_key: str, band: str, granule: dict, resolution: int):
    start = granule["start"].strftime("%Y-%m-%dT%H:%M:%S")
    end = granule["end"].strftime("%Y-%m-%dT%H:%M:%S")
    dt = granule["start"].strftime("%Y%m%dT%H%M%S")

    out_dir = OUTPUT_DIR / product_key
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = out_dir / f"S5P_{product_key}_{dt}.tif"

    if filename.exists():
        logger.info(f"Already exists, skipping: {filename}")
        return
    
    size = bbox_to_dimensions(aoi_bbox, resolution=resolution)
    request = SentinelHubRequest(
        evalscript=make_evalscript(band),
        input_data  = [
            SentinelHubRequest.input_data(
                data_collection = S5P_COLLECTION,
                time_interval = (start, end)
            )
        ],
        responses = [SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox = aoi_bbox,
        size = size,
        config = config,
        data_folder = str(out_dir)
    )
    data = request.get_data(save_data=True)
    arr = data[0]
    logger.info(f"Saved:{filename.name} shape={arr.shape}")

def collect_product(product_key:str, state:dict):
    product = PRODUCTS[product_key]
    downloaded = state.get(product_key, [])
    since = datetime.now(timezone.utc) - timedelta(days=2)
    granules = get_new_granules(product_key, product["granule_tag"], since)

    new_ids = []
    for granule in granules:
        if granule["id"] in downloaded:
            logger.info(f"{product_key}: already downloaded {granule['id']}")
            continue
        try:
            download_granule(product_key, product["band"], granule, product["resolution"])
            new_ids.append(granule["id"])
        except Exception as e:
            logger.error(f"{product_key}: failed on {granule["id"]} - {e}")
    state[product_key] = downloaded + new_ids
    return new_ids

def run_scheduler():
    state = load_state()
    total = 0
    for product_key in PRODUCTS:
        try:
            new = collect_product(product_key, state)
            total += len(new)
        except Exception as e:
            logger.exception(f"Collection failed for {product_key}: e")
    
    save_state(state)
    logger.info(f"{total} new granules saved")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["scheduler", "now"], default="scheduler")
    args = parser.parse_args()

    if args.mode == "now":
        run_scheduler()
    else:
        schedule.every().day.at("14:00").do(run_daily)
        logger.info("Scheduler running — TROPOMI collection daily at 14:00 UTC")
        while True:
            schedule.run_pending()
            time.sleep(60)
