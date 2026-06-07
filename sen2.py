import os
import time
import schedule
import logging
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
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

aoi_bbox = BBox(bbox=[268000, 1582000, 298000, 1634000], crs=CRS.UTM_51N)
aoi_size = bbox_to_dimensions(aoi_bbox, resolution=100)

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

def get_last_quarter_window() -> tuple[str, str]:
    end = date.today().replace(day=1)
    start = end - relativedelta(months=4)
    return str(start), str(end)

def build_ndvi_request(start: str, end: str, maxCloudCoverage = 5) -> SentinelHubRequest:
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

def collect_ndvi():
    start, end = get_last_quarter_window()
    logger.info(f"Collecting Sentinel-2 NDVI: {start} to {end}.")
    request = build_ndvi_request(start, end, 80)

    data = request.get_data(save_data=True)
    filename = OUTPUT_DIR / f"S2_NDVI_{start}_{end}.tif"
    logger.info(f"NDVI GeoTIFF saved {filename}.")
    return data

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
    parser.add_argument("--mode", choices=["scheduler", "now"], default="scheduler",
                        help="now = run once immediately; scheduler = run on 4-month cycle")
    args = parser.parse_args()

    if args.mode == "now":
        collect_ndvi()
    else:
        run_scheduler()
