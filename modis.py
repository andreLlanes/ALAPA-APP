import os
import logging
import schedule
import time
import json
import tempfile
import requests
import boto3
import numpy as np
import rasterio
from rasterio.warp import transform_geom
from rasterio.crs import CRS
from rasterio.mask import mask as rio_mask
from shapely.geometry import box, shape
from datetime import date, timedelta
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env")
logger = logging.getLogger(__name__)

LAADS_TOKEN = os.environ["LAADS_TOKEN"]
LAADS_BASE = "https://ladsweb.modaps.eosdis.nasa.gov"
PRODUCT = "MCD19A2"
COLLECTION = "61"
TILE = "h29v07"

S3_BUCKET = os.environ["S3_BUCKET"]
S3_PREFIX = "modis"

OUTPUT_DIR = Path("outputs/modis")
STATE_FILE = Path("outputs/modis/.downloaded.json")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

AOI_BOUNDS = [120.868835, 14.316284, 121.143494, 14.781522]
AOI_SHAPE = [box(*AOI_BOUNDS).__geo_interface__]
NODATA = -9999.0

BAND_ORDER = ["AOD_047", "AOD_055", "CWV", "QA"]
SUBDATASETS = {
    "AOD_047": {
        "hdf_name": "Optical_Depth_047",
        "scale": 0.001,
        "fill_value": -28672,
        "description": "Aerosol Optical Depth at 470nm",
    },
    "AOD_055": {
        "hdf_name": "Optical_Depth_055",
        "scale": 0.001,
        "fill_value": -28672,
        "description": "Aerosol Optical Depth at 550nm",
    },
    "CWV": {
        "hdf_name": "Column_WV",
        "scale": 0.001,
        "fill_value": -28672,
        "description": "Column Water Vapour",
    },
    "QA": {
        "hdf_name": "AOD_QA",
        "scale": None,
        "fill_value": 65535,
        "description": "QA Bitmask",
    },
}


def load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"downloaded": []}


def save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def build_qa_mask(qa_array: np.ndarray) -> np.ndarray:
    cloud_bits = qa_array & 0b00000011
    adjacent_cloud = qa_array & 0b00000100
    return (cloud_bits <= 1) & (adjacent_cloud == 0)


def trim_to_valid_data(clipped: dict[str, np.ndarray], transform):
    valid_mask = np.zeros_like(clipped["AOD_047"], dtype=bool)
    for band_key in ("AOD_047", "AOD_055", "CWV"):
        valid_mask |= ~np.isnan(clipped[band_key])

    if not valid_mask.any():
        return None, None

    rows, cols = np.where(valid_mask)
    row_min, row_max = rows.min(), rows.max()
    col_min, col_max = cols.min(), cols.max()

    trimmed = {
        band_key: band_data[row_min:row_max + 1, col_min:col_max + 1]
        for band_key, band_data in clipped.items()
    }
    trimmed_transform = rasterio.transform.Affine(
        transform.a,
        transform.b,
        transform.c + col_min * transform.a + row_min * transform.b,
        transform.d,
        transform.e,
        transform.f + col_min * transform.d + row_min * transform.e,
    )
    return trimmed, trimmed_transform


def upload_to_s3(local_path: Path):
    s3 = boto3.client("s3")
    key = f"{S3_PREFIX}/{local_path.name}"
    s3.upload_file(str(local_path), S3_BUCKET, key)
    logger.info(f"Uploaded to s3://{S3_BUCKET}/{key}")


# LAADS Download
def laads_headers() -> dict:
    return {
        "Authorization": f"Bearer {LAADS_TOKEN}",
        "Cookie": f"Authorization=Bearer {LAADS_TOKEN}",
    }


def list_granules(target_date: date) -> list[dict]:
    year = target_date.year
    doy = target_date.timetuple().tm_yday
    url = f"{LAADS_BASE}/api/v2/content/details/allData/{COLLECTION}/{PRODUCT}/{year}/{doy:03d}/"

    resp = requests.get(url, headers=laads_headers(), timeout=30)
    resp.raise_for_status()

    granules = []
    for item in resp.json().get("content", []):
        name = item.get("name", "")
        if not name.endswith(".hdf") or TILE not in name:
            continue
        url = f"{LAADS_BASE}/archive/allData/{COLLECTION}/{PRODUCT}/{year}/{doy:03d}/{name}"
        granules.append({"name": name, "url": url})

    logger.info(f"MODIS: {len(granules)} granule(s) found for {target_date} (DOY {doy:03d})")
    return granules


def download_hdf(url: str, dest: Path):
    session = requests.Session()
    session.cookies.set("Authorization", f"Bearer {LAADS_TOKEN}")

    resp = session.get(url, headers=laads_headers(), stream=True, timeout=120, allow_redirects=True)
    resp.raise_for_status()

    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)

    size_mb = dest.stat().st_size / 1e6
    logger.info(f"Downloaded: {dest.name} ({size_mb:.1f} MB)")
    if size_mb < 0.1:
        raise ValueError(f"File too small ({size_mb:.3f} MB) - auth failed")


# HDF4 Extraction
def extract_subdataset(hdf_path: Path, hdf_name: str):
    hdf_str = str(hdf_path).replace("\\", "/")
    sd_path = f'HDF4_EOS:EOS_GRID:"{hdf_str}":grid1km:{hdf_name}'

    with rasterio.open(sd_path) as src:
        data = src.read(1)
        src_crs = src.crs
        src_trans = src.transform
        src_w = src.width
        src_h = src.height

    return data, src_crs, src_trans, src_w, src_h


def build_transformed_aoi(raster_crs):
    aoi_shape = transform_geom(CRS.from_epsg(4326), raster_crs, AOI_SHAPE[0])
    aoi_box = box(*shape(aoi_shape).bounds)
    return aoi_shape, aoi_box


def clip_to_aoi(data, transform, width, height, raster_crs, aoi_shape, aoi_box):
    from rasterio.transform import array_bounds

    bounds = array_bounds(height, width, transform)
    raster_box = box(bounds[0], bounds[1], bounds[2], bounds[3])
    if not raster_box.intersects(aoi_box):
        raise ValueError(f"Raster bounds {bounds} do not overlap transformed AOI")

    with rasterio.MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype="float32",
            crs=raster_crs,
            transform=transform,
        ) as ds:
            ds.write(data, 1)
        with memfile.open() as ds:
            clipped, clipped_transform = rio_mask(ds, [aoi_shape], crop=True, nodata=np.nan)

    return clipped[0], clipped_transform


# Process Granule
def process_granule(hdf_path: Path, target_date: date):
    date_str = target_date.strftime("%Y%m%d")
    out_path = OUTPUT_DIR / f"MODIS_{date_str}_{hdf_path.stem}.tif"

    if out_path.exists():
        logger.info(f"Already processed, skipping: {out_path.name}")
        return

    # Extract all Bands
    bands = {}
    qa_int = None
    crs_info = None

    for band_key, band_def in SUBDATASETS.items():
        raw, src_crs, src_trans, src_w, src_h = extract_subdataset(
            hdf_path, band_def["hdf_name"]
        )
        if band_def["scale"] is not None:
            fill_mask = raw == band_def["fill_value"]
            data = raw.astype("float32") * band_def["scale"]
            data[fill_mask] = np.nan
        else:
            data = raw.astype("int32")

        if band_key == "QA":
            qa_int = data.copy()
        if crs_info is None:
            crs_info = (src_crs, src_trans, src_w, src_h)

        bands[band_key] = data

    # Apply QA Mask
    qa_safe = np.where(qa_int == 65535, 99, qa_int)
    qa_mask = build_qa_mask(qa_safe)

    for band_key in ("AOD_047", "AOD_055", "CWV"):
        bands[band_key] = np.where(
            qa_mask & ~np.isnan(bands[band_key]),
            bands[band_key],
            np.nan,
        )

    logger.info(f"Valid AOD047 after QA: {int(np.sum(~np.isnan(bands['AOD_047'])))}")

    # Clip all bands to AOI
    src_crs, src_trans, src_w, src_h = crs_info
    aoi_shape, aoi_box = build_transformed_aoi(src_crs)
    clipped = {}
    clip_transform = None

    for band_key, data in bands.items():
        clp, clip_transform = clip_to_aoi(
            data, src_trans, src_w, src_h, src_crs, aoi_shape, aoi_box
        )
        clipped[band_key] = clp

    trimmed, clip_transform = trim_to_valid_data(clipped, clip_transform)
    if trimmed is None:
        logger.warning(f"No valid science pixels after QA/clip for {hdf_path.name}; skipping save")
        return

    clipped = trimmed
    clip_h, clip_w = clipped["AOD_047"].shape

    # Write multi-band GeoTIFF
    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=clip_h,
        width=clip_w,
        count=len(BAND_ORDER),
        dtype="float32",
        crs=src_crs,
        transform=clip_transform,
        nodata=NODATA,
    ) as dst:
        for i, band_key in enumerate(BAND_ORDER, start=1):
            band_data = np.where(np.isnan(clipped[band_key]), NODATA, clipped[band_key])
            dst.write(band_data.astype("float32"), i)
            dst.update_tags(
                i,
                name=band_key,
                description=SUBDATASETS[band_key]["description"],
            )

    valid = int(np.sum(~np.isnan(clipped["AOD_047"])))
    logger.info(f"Saved: {out_path.name}  shape=({clip_h},{clip_w})  valid_AOD047={valid}")

    upload_to_s3(out_path)


def run_daily(days: int = 7):
    logger.info(f"=== MODIS daily collection start (lookback {days} day(s)) ===")
    state = load_state()
    downloaded = state.get("downloaded", [])
    total = 0

    for days_back in range(days):
        check_date = date.today() - timedelta(days=days_back)
        try:
            granules = list_granules(check_date)
        except Exception as e:
            logger.error(f"Directory listing failed for {check_date}: {e}")
            continue

        for granule in granules:
            name = granule["name"]
            if name in downloaded:
                logger.info(f"Already downloaded: {name}")
                continue

            with tempfile.TemporaryDirectory() as tmpdir:
                hdf_path = Path(tmpdir) / name
                try:
                    download_hdf(granule["url"], hdf_path)
                    process_granule(hdf_path, check_date)
                    downloaded.append(name)
                    total += 1
                except Exception as e:
                    logger.error(f"Failed on {name}: {e}")

    state["downloaded"] = downloaded
    save_state(state)
    logger.info(f"=== Done. {total} new granule(s) processed. ===")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["scheduler", "now"], default="scheduler")
    parser.add_argument("--days", type=int, default=7, help="How many days back to check")
    args = parser.parse_args()

    if args.mode == "now":
        run_daily(args.days)
    else:
        schedule.every().day.at("10:00").do(run_daily, days=args.days)
        logger.info("Scheduler running - MODIS collection daily at 10:00 UTC")
        while True:
            schedule.run_pending()
            time.sleep(60)
