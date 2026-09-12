"""
Sentinel-2 quarterly NDVI composites for Metro Manila on the study's common grid
(Section: NDVI Compositing). Backfill version: computes all 12 offset quarters
from 2023-08-30 to 2026-08-30 in one run. The automated scheduler is a separate
script; this one is the one-off historical ETL.

Method:
  - NDVI = (B08 - B04) / (B08 + B04).
  - Scenes with >5% cloud cover excluded at request time (maxCloudCoverage).
  - Per-pixel SCL masking of cloud shadow (3), cloud medium/high (8, 9), thin
    cirrus (10) and snow/ice (11); invalid pixels dropped via dataMask.
  - Quarterly per-pixel MEDIAN of all valid acquisitions in each 3-month period.
  - Requested in UTM Zone 51N (EPSG:32651) on the common 100 m grid
    (origin 276700/1635100, 242 x 490), downsampled from the 10 m native B04/B08.

Output: one 2-band GeoTIFF per quarter (band 1 = median NDVI, band 2 = valid
flag: 1 where at least one clear acquisition existed, 0 otherwise), 12 files in
all, each aligned cell-for-cell with the other covariate layers.
"""

import os
import logging
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from dateutil.relativedelta import relativedelta
from datetime import date
from dotenv import load_dotenv
from sentinelhub import (
    SHConfig, DataCollection, SentinelHubRequest, BBox, CRS, MimeType,
)

load_dotenv(dotenv_path=Path(__file__).parent / ".env")
logger = logging.getLogger(__name__)

GRID_CRS = "EPSG:32651"
ORIGIN_E = 276700.0
ORIGIN_N = 1635100.0
CELL = 100.0
NCOLS = 242
NROWS = 490
GRID_TRANSFORM = Affine(CELL, 0, ORIGIN_E, 0, -CELL, ORIGIN_N)

AOI_BBOX = BBox(bbox=[ORIGIN_E, ORIGIN_N - NROWS * CELL,
                      ORIGIN_E + NCOLS * CELL, ORIGIN_N], crs=CRS("32651"))
AOI_SIZE = (NCOLS, NROWS)

BACKFILL_START = date(2023, 8, 30)
N_QUARTERS = 12
MAX_CLOUD = 5

OUTPUT_DIR = Path("outputs/ndvi")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

config = SHConfig()
config.sh_client_id = os.environ["SH_CLIENT_ID"]
config.sh_client_secret = os.environ["SH_CLIENT_SECRET"]
config.sh_base_url = os.environ["SH_BASE_URL"]
config.sh_token_url = os.environ["SH_TOKEN_URL"]

EVALSCRIPT = """
    function setup(){
        return{
            input: [{ bands: ["B04", "B08", "SCL", "dataMask"] }],
            mosaicking: "ORBIT",
            output: { bands: 2, sampleType: "FLOAT32" }
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


def quarter_windows():
    """The 12 offset quarters as (label, start, end_exclusive) date tuples."""
    out = []
    for i in range(N_QUARTERS):
        q0 = BACKFILL_START + relativedelta(months=3 * i)
        q1 = BACKFILL_START + relativedelta(months=3 * (i + 1))
        out.append((f"Q{i + 1:02d}_{q0.isoformat()}", q0, q1))
    return out


def build_request(start, end):
    return SentinelHubRequest(
        evalscript=EVALSCRIPT,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A.define_from(
                    name="s2l2a", service_url="https://sh.dataspace.copernicus.eu"
                ),
                time_interval=(start.isoformat(), end.isoformat()),
                other_args={"dataFilter": {"maxCloudCoverage": MAX_CLOUD}},
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=AOI_BBOX,
        size=AOI_SIZE,
        config=config,
    )


def write_quarter(arr, label):
    """arr shape (NROWS, NCOLS, 2): band0 NDVI, band1 valid flag."""
    path = OUTPUT_DIR / f"S2_NDVI_{label}.tif"
    with rasterio.open(
        path, "w", driver="GTiff",
        height=NROWS, width=NCOLS, count=2, dtype="float32",
        crs=GRID_CRS, transform=GRID_TRANSFORM, compress="deflate",
    ) as dst:
        dst.write(arr[:, :, 0].astype("float32"), 1)
        dst.write(arr[:, :, 1].astype("float32"), 2)
        dst.set_band_description(1, "ndvi_median")
        dst.set_band_description(2, "valid")
    return path


def collect_all():
    quarters = quarter_windows()
    logger.info(f"Backfilling {len(quarters)} quarters "
                f"{quarters[0][1]} .. {quarters[-1][2]}")

    for qi, (label, start, end) in enumerate(quarters):
        logger.info(f"[{qi + 1}/{len(quarters)}] {label}: {start} -> {end}")
        arr = build_request(start, end).get_data()[0]
        valid_frac = float((arr[:, :, 1] > 0).mean())
        logger.info(f"    valid pixels: {valid_frac * 100:.1f}%")
        path = write_quarter(arr, label)
        logger.info(f"    wrote {path}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )
    collect_all()