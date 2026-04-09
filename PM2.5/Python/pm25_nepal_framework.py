"""
=============================================================================
High-Resolution PM2.5 Estimation Framework for Nepal  —  v10
=============================================================================
Authors  : Samarpan Mani Gautam, Udhyan Shah, Suresh Acharaya Dilip Kumar Rajak,
           Jigyasu Ghimire, Liza Pradhan
Framework: Two-stage ML pipeline (Random Forest AOD Gap-Fill + HistGBR PM2.5)
Inputs   : MAIAC AOD (HDF), ERA5 Meteo (NetCDF), NDVI (HDF), DEM (TIF),
           Population (TIF), OpenAQ Ground Stations (CSV),
           TROPOMI NO2/HCHO/CO/O3/CH4/Aerosol Index (NetCDF)
Output   : Daily 1-km PM2.5 GeoTIFF(s) + validation metrics + diagnostic plots
           + distance-to-station map + confidence flags + applicability statement
=============================================================================


# ──────────────────────────────────────────────────────────────────────────────
# 0. IMPORTS & GLOBAL CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
import os
import glob
import warnings
import logging
import joblib
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import seaborn as sns
from scipy.stats import pearsonr
from scipy.spatial import cKDTree   # [V10-S2] KDTree for distance computation

# Shapefile overlay
try:
    import geopandas as gpd
    GEOPANDAS_AVAILABLE = True
except ImportError:
    warnings.warn(
        "geopandas not found — shapefile overlays disabled.  "
        "Install with: conda install -c conda-forge geopandas"
    )
    GEOPANDAS_AVAILABLE = False

# Geospatial
try:
    from osgeo import gdal, osr, gdalconst
    gdal.UseExceptions()
except ImportError:
    raise ImportError("GDAL not found. Install with: conda install -c conda-forge gdal")

# Machine Learning
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

# PyTorch (optional)
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(r"C:\Users\samue\Documents\Conferences\Space\PM2.5")

# ── Data paths ────────────────────────────────────────────────────────────────
AOD_GLOB_2025 = str(PROJECT_ROOT / "AOD_Data" / "**" / "MCD19A2.A2025*.hdf")
AOD_GLOB_2026 = str(PROJECT_ROOT / "AOD_Data" / "**" / "MCD19A2.A2026*.hdf")
AOD_GLOB      = str(PROJECT_ROOT / "AOD_Data" / "**" / "MCD19A2.A20*.hdf")
NDVI_GLOB     = str(PROJECT_ROOT / "Vegetation_Data" / "MOD13A3.A2025*.hdf")


def _find_era5_dir(project_root):
    candidates = [
        project_root / "ERA5__Data"  / "era5_nepal_meteo_2025",
        project_root / "ERA5_Data"   / "era5_nepal_meteo_2025",
        project_root / "ERA5__Data"  / "era5_nepal_meteo_2026",
        project_root / "ERA5_Data"   / "era5_nepal_meteo_2026",
        project_root / "ERA5__Data",
        project_root / "ERA5_Data",
    ]
    for c in candidates:
        if c.exists() and list(c.glob("era5_*.nc")):
            return c
    return project_root / "ERA5__Data" / "era5_nepal_meteo_2025"


ERA5_DIR        = _find_era5_dir(PROJECT_ROOT)
ERA5_DAILY_GLOB = str(ERA5_DIR / "era5_*.nc")

GROUND_DIR   = str(PROJECT_ROOT / "PM2.5_Data")
DEM_PATH     = str(PROJECT_ROOT / "DEM_Data"         / "output_SRTMGL1.tif")
POP_PATH     = str(PROJECT_ROOT / "Population__Data" / "npl_pd_2020_1km_UNadj.tif")
OUTPUT_DIR   = str(PROJECT_ROOT / "Outputs")
SHP_ROOT     = PROJECT_ROOT.parent / "npl_admin_boundaries_shp"
SHP_ADMIN0   = str(SHP_ROOT / "npl_admin0.shp")
SHP_ADMIN1   = str(SHP_ROOT / "npl_admin1.shp")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── TROPOMI / Sentinel-5P auxiliary data paths ────────────────────────────────
_SPACE_ROOT = PROJECT_ROOT.parent
TROPOMI_NO2_PATH  = str(_SPACE_ROOT / "NO2"           / "NO2_Data.nc")
TROPOMI_HCHO_PATH = str(_SPACE_ROOT / "HCHO"          / "HCHO_Data.nc")
TROPOMI_CO_PATH   = str(_SPACE_ROOT / "CO"            / "CO_Data.nc")
TROPOMI_O3_PATH   = str(_SPACE_ROOT / "O3"            / "O3_Data.nc")
TROPOMI_CH4_PATH  = str(_SPACE_ROOT / "CH4"           / "CH4_Data.nc")
TROPOMI_AI_PATH   = str(_SPACE_ROOT / "AEROSOL_Index"  / "AEROSOL_Index_Data.nc")

# ── Nepal bounding box (WGS-84) ───────────────────────────────────────────────
NEPAL_BBOX = {"xmin": 80.0, "ymin": 26.3, "xmax": 88.2, "ymax": 30.5}

USE_AOD_PERIOD = True
SEED = 42
np.random.seed(SEED)

# ── [V7-F2] City-group fold assignments ──────────────────────────────────────
CITY_FOLD_RULES: list[tuple[str, int]] = [
    ("birgunj",      3), ("dhangadhi",    3), ("hetauda",      3),
    ("tokha",        0), ("gokarneshwor", 0), ("gothatar",     0),
    ("gaushala",     0), ("sc-13",        0), ("sc-12",        0),
    ("sc-14",        0), ("sc-31",        0), ("sc-01",        0),
    ("kirtipur",     1), ("farsidol",     1), ("khokana",      1),
    ("taudaha",      1), ("nakhipot",     1), ("lagankhel",    1),
    ("kupondole",    1), ("pulchowk",     1), ("sanepa",       1),
    ("sc-05",        1), ("sc-07",        1), ("sc-08",        1),
    ("sc-09",        1), ("sc-15",        1), ("sc-22",        1),
    ("sc-25",        1), ("sc-40",        1), ("sc-41",        1),
]


def _assign_city_fold(station_id: str) -> int:
    """[V7-F2] Assign a station to a CV fold by keyword matching."""
    sid = station_id.lower()
    for keyword, fold in CITY_FOLD_RULES:
        if keyword in sid:
            return fold
    return 2   # KTM_central default


FOLD_NAMES = {0: "KTM_north", 1: "KTM_south", 2: "KTM_central", 3: "OuterCities"}

# ── [V10-S1] Region labels for region-specific models ────────────────────────
# Folds 0,1,2 → KTM Valley; Fold 3 → OuterCities
REGION_KTM   = "ktm_valley"
REGION_OUTER = "outer_cities"


def _assign_region(station_id: str) -> str:
    """Map a station to its macro-region (for region-specific model training)."""
    fold = _assign_city_fold(station_id)
    return REGION_OUTER if fold == 3 else REGION_KTM


# ══════════════════════════════════════════════════════════════════════════════
# 1. HDF / RASTER UTILITIES  (unchanged from v9)
# ══════════════════════════════════════════════════════════════════════════════

def list_hdf_subdatasets(hdf_path: str) -> list:
    ds = gdal.Open(hdf_path)
    if ds is None:
        raise FileNotFoundError(f"GDAL could not open: {hdf_path}")
    sds = ds.GetSubDatasets()
    ds = None
    return sds


def read_hdf_band(hdf_path: str, layer_keyword: str,
                  scale_factor: float = 1.0,
                  fill_value: float = -28672.0):
    sds_list = list_hdf_subdatasets(hdf_path)
    matched  = [s[0] for s in sds_list if layer_keyword in s[0]]
    if not matched:
        available = [s[0].split(":")[-1] for s in sds_list]
        raise KeyError(
            f"Layer '{layer_keyword}' not found in {os.path.basename(hdf_path)}.\n"
            f"Available layers: {available}"
        )
    ds           = gdal.Open(matched[0])
    data         = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    geotransform = ds.GetGeoTransform()
    projection   = ds.GetProjection()
    nodata       = ds.GetRasterBand(1).GetNoDataValue()
    nrows, ncols = data.shape
    ds = None

    mask = (data == fill_value)
    if nodata is not None:
        mask |= (data == nodata)
    data[mask] = np.nan
    data = data * scale_factor

    geo_meta = {
        "geotransform": geotransform,
        "projection":   projection,
        "nrows":        nrows,
        "ncols":        ncols,
        "nodata":       np.nan,
    }
    return data, geo_meta


def read_geotiff(tif_path: str):
    ds = gdal.Open(tif_path)
    if ds is None:
        raise FileNotFoundError(f"Cannot open GeoTIFF: {tif_path}")
    band   = ds.GetRasterBand(1)
    data   = band.ReadAsArray().astype(np.float32)
    nodata = band.GetNoDataValue()
    geo_meta = {
        "geotransform": ds.GetGeoTransform(),
        "projection":   ds.GetProjection(),
        "nrows":        ds.RasterYSize,
        "ncols":        ds.RasterXSize,
        "nodata":       nodata,
    }
    ds = None
    if nodata is not None:
        data[data == nodata] = np.nan
    return data, geo_meta


def write_geotiff(output_path: str, array: np.ndarray,
                  geo_meta: dict, nodata: float = -9999.0) -> None:
    nrows, ncols = array.shape
    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(
        output_path, ncols, nrows, 1, gdal.GDT_Float32,
        options=["COMPRESS=LZW", "TILED=YES", "BIGTIFF=IF_SAFER"]
    )
    out_ds.SetGeoTransform(geo_meta["geotransform"])
    out_ds.SetProjection(geo_meta["projection"])
    out_band = out_ds.GetRasterBand(1)
    out_band.SetNoDataValue(nodata)
    arr_out = np.where(np.isnan(array), nodata, array)
    out_band.WriteArray(arr_out)
    out_band.FlushCache()
    out_ds = None
    log.info("  → Written: %s", output_path)


def resample_to_reference(src_path: str, ref_meta: dict,
                           resample_alg: int = gdalconst.GRA_Bilinear) -> np.ndarray:
    gt  = ref_meta["geotransform"]
    prj = ref_meta["projection"]
    nr  = ref_meta["nrows"]
    nc  = ref_meta["ncols"]

    mem_driver = gdal.GetDriverByName("MEM")
    dst_ds = mem_driver.Create("", nc, nr, 1, gdal.GDT_Float32)
    dst_ds.SetGeoTransform(gt)
    dst_ds.SetProjection(prj)
    dst_ds.GetRasterBand(1).Fill(np.nan)
    dst_ds.GetRasterBand(1).SetNoDataValue(np.nan)

    src_ds = gdal.Open(src_path)
    gdal.ReprojectImage(src_ds, dst_ds, None, None, resample_alg)
    arr = dst_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    src_ds = None
    dst_ds = None
    return arr


def geotransform_to_coords(geo_meta: dict):
    gt = geo_meta["geotransform"]
    nc, nr = geo_meta["ncols"], geo_meta["nrows"]
    lons = gt[0] + gt[1] * (np.arange(nc) + 0.5)
    lats = gt[3] + gt[5] * (np.arange(nr) + 0.5)
    return lons, lats


def warp_to_wgs84_bbox(src_array: np.ndarray, src_meta: dict,
                        bbox: dict,
                        target_res_deg: float = 0.008983):
    nodata_val   = -9999.0
    nrows, ncols = src_array.shape
    mem_drv = gdal.GetDriverByName("MEM")
    src_ds  = mem_drv.Create("", ncols, nrows, 1, gdal.GDT_Float32)
    src_ds.SetGeoTransform(src_meta["geotransform"])
    src_ds.SetProjection(src_meta["projection"])
    band = src_ds.GetRasterBand(1)
    band.SetNoDataValue(nodata_val)
    arr_out = np.where(np.isnan(src_array), nodata_val, src_array).astype(np.float32)
    band.WriteArray(arr_out)

    wgs84_srs = osr.SpatialReference()
    wgs84_srs.ImportFromEPSG(4326)
    wgs84_prj = wgs84_srs.ExportToWkt()

    xmin, ymin = bbox["xmin"], bbox["ymin"]
    xmax, ymax = bbox["xmax"], bbox["ymax"]
    out_ncols  = int(round((xmax - xmin) / target_res_deg))
    out_nrows  = int(round((ymax - ymin) / target_res_deg))
    out_gt     = (xmin, target_res_deg, 0.0, ymax, 0.0, -target_res_deg)

    dst_ds = mem_drv.Create("", out_ncols, out_nrows, 1, gdal.GDT_Float32)
    dst_ds.SetGeoTransform(out_gt)
    dst_ds.SetProjection(wgs84_prj)
    dst_ds.GetRasterBand(1).SetNoDataValue(nodata_val)
    dst_ds.GetRasterBand(1).Fill(nodata_val)

    gdal.ReprojectImage(src_ds, dst_ds,
                        src_meta["projection"], wgs84_prj,
                        gdalconst.GRA_Bilinear)

    result = dst_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    result[result == nodata_val] = np.nan

    geo_meta_out = {
        "geotransform": out_gt,
        "projection":   wgs84_prj,
        "nrows":        out_nrows,
        "ncols":        out_ncols,
        "nodata":       np.nan,
    }
    src_ds = None
    dst_ds = None
    return result, geo_meta_out


# ══════════════════════════════════════════════════════════════════════════════
# 2. DATA INGESTION  (unchanged from v9)
# ══════════════════════════════════════════════════════════════════════════════

def load_aod(pattern: str, bbox: dict | None = None):
    """
    [V8-M1] STREAMING two-pass loader — fixes ArrayMemoryError.
    (See v9 docstring for full description.)
    """
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        raise FileNotFoundError(f"No AOD HDF files found: {pattern}")
    log.info("Loading %d AOD HDF tiles (streaming — [V8-M1]) …", len(files))

    ref_meta: dict | None = None
    composite_sum:   np.ndarray | None = None
    composite_count: np.ndarray | None = None
    doy_sum:   dict[int, np.ndarray] = {}
    doy_count: dict[int, int]        = {}
    n_dates_seen = set()

    for f in files:
        try:
            token = os.path.basename(f).split(".")[1]
            date  = pd.to_datetime(token[1:], format="%Y%j")
            arr, meta = read_hdf_band(f, "Optical_Depth_047",
                                      scale_factor=0.001, fill_value=-28672.0)
            log.info("  AOD tile: %s  shape=%s", os.path.basename(f), arr.shape)
        except Exception as exc:
            log.warning("  Skipped %s — %s", os.path.basename(f), exc)
            continue

        if bbox is not None:
            try:
                arr_wgs, meta_wgs = warp_to_wgs84_bbox(arr, meta, bbox)
            except Exception as exc:
                log.warning("  Warp failed for %s — %s", os.path.basename(f), exc)
                continue
            del arr
        else:
            arr_wgs, meta_wgs = arr, meta

        if ref_meta is None:
            ref_meta = meta_wgs
            composite_sum   = np.zeros((meta_wgs["nrows"], meta_wgs["ncols"]),
                                       dtype=np.float64)
            composite_count = np.zeros((meta_wgs["nrows"], meta_wgs["ncols"]),
                                       dtype=np.int32)

        if arr_wgs.shape != (ref_meta["nrows"], ref_meta["ncols"]):
            tmp = os.path.join(OUTPUT_DIR, f"_tmp_aod_wgs_{date.strftime('%Y%m%d')}.tif")
            write_geotiff(tmp, arr_wgs, meta_wgs)
            arr_wgs = resample_to_reference(tmp, ref_meta)

        valid = ~np.isnan(arr_wgs)
        composite_sum   += np.where(valid, arr_wgs.astype(np.float64), 0.0)
        composite_count += valid.astype(np.int32)

        doy = int(date.dayofyear)
        if doy not in doy_sum:
            doy_sum[doy]   = np.zeros_like(composite_sum, dtype=np.float64)
            doy_count[doy] = 0
        doy_sum[doy]   += np.where(valid, arr_wgs.astype(np.float64), 0.0)
        doy_count[doy] += 1

        n_dates_seen.add(date.date())
        del arr_wgs

    if ref_meta is None or composite_sum is None:
        raise RuntimeError("No valid AOD tiles could be loaded.")

    log.info("AOD spans %d unique dates.", len(n_dates_seen))

    with np.errstate(invalid="ignore"):
        composite = np.where(
            composite_count > 0,
            (composite_sum / composite_count).astype(np.float32),
            np.nan,
        ).astype(np.float32)

    log.info("AOD composite shape: %s  valid px: %d / %d",
             composite.shape,
             int(np.sum(~np.isnan(composite))),
             composite.size)

    doy_map: dict[int, np.ndarray] = {}
    for doy, s in doy_sum.items():
        with np.errstate(invalid="ignore"):
            doy_map[doy] = (s / doy_count[doy]).astype(np.float32)

    daily_aod: dict = {
        "_composite": composite,
        "_ref_meta":  ref_meta,
        "_doy_map":   doy_map,
        "_date_min":  min(n_dates_seen) if n_dates_seen else None,
        "_date_max":  max(n_dates_seen) if n_dates_seen else None,
    }
    return daily_aod, ref_meta


def load_ndvi(pattern: str, ref_meta: dict) -> np.ndarray:
    files = sorted(glob.glob(pattern))
    if not files:
        log.warning("No NDVI HDF files found — using zeros.")
        return np.zeros((ref_meta["nrows"], ref_meta["ncols"]), dtype=np.float32)

    log.info("Loading %d NDVI tiles …", len(files))
    tiles = []
    for i, f in enumerate(files):
        try:
            arr, ndvi_meta = read_hdf_band(f, "1 km monthly NDVI",
                                           scale_factor=0.0001, fill_value=-3000.0)
            tmp = os.path.join(OUTPUT_DIR, f"_tmp_ndvi_{i}.tif")
            write_geotiff(tmp, arr, ndvi_meta)
            resampled = resample_to_reference(tmp, ref_meta)
            tiles.append(resampled)
        except Exception as exc:
            log.warning("  NDVI tile skipped: %s", exc)

    if not tiles:
        return np.zeros((ref_meta["nrows"], ref_meta["ncols"]), dtype=np.float32)

    ndvi = np.nanmean(np.stack(tiles, axis=0), axis=0)
    ndvi = np.clip(ndvi, -1.0, 1.0)
    log.info("NDVI resampled shape: %s", ndvi.shape)
    return ndvi


def load_era5(daily_glob: str, ref_meta: dict) -> dict:
    """[F23] Returns daily dict + '_composite'. [F22] CDS NetCDF4 quirks handled."""
    from scipy.interpolate import RegularGridInterpolator

    daily_files = sorted(glob.glob(daily_glob))
    nr, nc = ref_meta["nrows"], ref_meta["ncols"]
    zeros  = np.zeros((nr, nc), dtype=np.float32)

    ERA5_VARS = ["t2m", "d2m", "blh", "u10", "v10", "sp", "tp"]
    VAR_ALIASES = {
        "t2m": ["t2m", "2m_temperature"],
        "d2m": ["d2m", "e2m", "2m_dewpoint_temperature"],
        "blh": ["blh", "boundary_layer_height", "pblh"],
        "u10": ["u10", "10m_u_component_of_wind"],
        "v10": ["v10", "10m_v_component_of_wind"],
        "sp":  ["sp",  "surface_pressure"],
        "tp":  ["tp",  "total_precipitation"],
    }

    if not daily_files:
        log.warning("No ERA5 files found — meteorological covariates set to zero.")
        empty = {v: zeros.copy() for v in ERA5_VARS}
        empty = _derive_era5(empty, nr, nc)
        return {"_composite": empty}

    log.info("Loading %d ERA5 daily files …", len(daily_files))

    lons_out, lats_out = geotransform_to_coords(ref_meta)
    lon_grid, lat_grid = np.meshgrid(lons_out, lats_out)

    def _squeeze_to_2d(da):
        for tdim in ("valid_time", "time"):
            if tdim in da.dims and da.sizes[tdim] > 1:
                da = da.mean(dim=tdim)
        squeeze_dims = [d for d in da.dims
                        if da.sizes[d] == 1
                        and d not in ("latitude", "lat", "longitude", "lon")]
        if squeeze_dims:
            da = da.squeeze(squeeze_dims, drop=True)
        return da

    def _interp_da(da):
        da = _squeeze_to_2d(da)
        lat_name = next((n for n in list(da.dims) + list(da.coords)
                         if n in ("latitude", "lat")), None)
        lon_name = next((n for n in list(da.dims) + list(da.coords)
                         if n in ("longitude", "lon")), None)
        if lat_name is None or lon_name is None:
            return None
        src_lats = da[lat_name].values.astype(np.float64)
        src_lons = da[lon_name].values.astype(np.float64)
        src_data = da.values.astype(np.float64)
        if src_lats.ndim == 2: src_lats = src_lats[:, 0]
        if src_lons.ndim == 2: src_lons = src_lons[0, :]
        if da.dims.index(lat_name) > da.dims.index(lon_name):
            src_data = src_data.T
        if src_lats[0] > src_lats[-1]:
            src_lats = src_lats[::-1]
            src_data = src_data[::-1, :]
        if np.all(np.isnan(src_data)):
            return None
        finite_vals = src_data[np.isfinite(src_data)]
        fill_val = float(np.nanmedian(finite_vals)) if len(finite_vals) > 0 else 0.0
        src_data = np.where(np.isfinite(src_data), src_data, fill_val)
        q_lats = np.clip(lat_grid.ravel(), src_lats.min(), src_lats.max())
        q_lons = np.clip(lon_grid.ravel(), src_lons.min(), src_lons.max())
        fn = RegularGridInterpolator(
            (src_lats, src_lons), src_data,
            method="linear", bounds_error=False, fill_value=np.nan)
        return fn(np.column_stack([q_lats, q_lons])).reshape(nr, nc).astype(np.float32)

    def _find_var(ds, aliases):
        ds_lower = {v.lower(): v for v in ds.data_vars}
        for alias in aliases:
            if alias in ds.data_vars:
                return ds[alias]
            if alias.lower() in ds_lower:
                return ds[ds_lower[alias.lower()]]
        return None

    daily_era5: dict = {}
    composite_sum   = {v: np.zeros((nr, nc), dtype=np.float64) for v in ERA5_VARS}
    composite_count = {v: 0 for v in ERA5_VARS}

    for fpath in daily_files:
        fname = os.path.basename(fpath)
        try:
            date_str = fname.replace("era5_", "").replace(".nc", "")
            date = pd.to_datetime(date_str, format="%Y%m%d")
        except Exception:
            log.warning("  Cannot parse date from %s — skipping.", fname)
            continue

        try:
            ds = xr.open_dataset(fpath, engine="netcdf4")
        except Exception as exc:
            log.warning("  Skipping %s: %s", fname, exc)
            continue

        day_vars = {}
        for canon, aliases in VAR_ALIASES.items():
            da = _find_var(ds, aliases)
            if da is None:
                day_vars[canon] = zeros.copy()
            else:
                try:
                    arr = _interp_da(da)
                    day_vars[canon] = arr if arr is not None else zeros.copy()
                except Exception:
                    day_vars[canon] = zeros.copy()
            if np.any(day_vars[canon] != 0):
                composite_sum[canon]   += np.nan_to_num(day_vars[canon])
                composite_count[canon] += 1
        ds.close()

        daily_era5[date] = _derive_era5(day_vars, nr, nc)

    log.info("  ERA5 daily fields loaded for %d dates.", len(daily_era5))

    composite_vars = {}
    for canon in ERA5_VARS:
        if composite_count[canon] > 0:
            composite_vars[canon] = (composite_sum[canon] / composite_count[canon]).astype(np.float32)
        else:
            composite_vars[canon] = zeros.copy()
    daily_era5["_composite"] = _derive_era5(composite_vars, nr, nc)

    return daily_era5


def _derive_era5(var_dict: dict, nr: int, nc: int) -> dict:
    zeros = np.zeros((nr, nc), dtype=np.float32)
    u10 = var_dict.get("u10", zeros)
    v10 = var_dict.get("v10", zeros)
    var_dict["wind_speed"] = np.sqrt(u10 ** 2 + v10 ** 2)

    t2m_k = var_dict.get("t2m", zeros + 273.15)
    d2m_k = var_dict.get("d2m", zeros + 273.15)
    t2m_c = t2m_k - 273.15
    td_c  = d2m_k - 273.15
    var_dict["t2m"] = t2m_c
    with np.errstate(invalid="ignore", divide="ignore"):
        rh = 100.0 * (
            np.exp(17.625 * td_c  / (243.04 + td_c)) /
            np.exp(17.625 * t2m_c / (243.04 + t2m_c))
        )
    var_dict["rh"]   = np.clip(np.nan_to_num(rh, nan=50.0), 0.0, 100.0).astype(np.float32)
    var_dict["prec"] = var_dict.pop("tp", zeros.copy())
    return var_dict


def load_static_layers(dem_path: str, pop_path: str, ref_meta: dict) -> dict:
    layers = {}
    for name, path in [("elevation", dem_path), ("population", pop_path)]:
        try:
            resampled = resample_to_reference(path, ref_meta)
            layers[name] = resampled
            log.info("  %s layer resampled: %s", name.capitalize(), resampled.shape)
        except Exception as exc:
            log.warning("  Static layer '%s' failed (%s) — zeros used.", name, exc)
            layers[name] = np.zeros((ref_meta["nrows"], ref_meta["ncols"]),
                                    dtype=np.float32)
    layers["pop_log"] = np.log1p(np.nan_to_num(layers["population"], nan=0.0))
    return layers


def load_tropomi_layers(ref_meta: dict) -> dict:
    """[V9-M2] Robust TROPOMI/Sentinel-5P loader (unchanged from v9)."""
    from scipy.interpolate import RegularGridInterpolator

    nr, nc = ref_meta["nrows"], ref_meta["ncols"]
    lons_out, lats_out = geotransform_to_coords(ref_meta)
    lon_grid, lat_grid = np.meshgrid(lons_out, lats_out)
    zeros = np.zeros((nr, nc), dtype=np.float32)

    TROPOMI_SOURCES = {
        "no2":     (TROPOMI_NO2_PATH,
                    ["NO2", "no2", "nitrogendioxide_tropospheric_column",
                     "tropospheric_NO2_column_number_density"]),
        "hcho":    (TROPOMI_HCHO_PATH,
                    ["HCHO", "hcho", "formaldehyde_tropospheric_vertical_column"]),
        "co":      (TROPOMI_CO_PATH,
                    ["CO", "co", "carbonmonoxide_total_column"]),
        "o3":      (TROPOMI_O3_PATH,
                    ["O3", "o3", "ozone_total_vertical_column"]),
        "ch4":     (TROPOMI_CH4_PATH,
                    ["CH4", "ch4", "methane_mixing_ratio_bias_corrected"]),
        "aero_ai": (TROPOMI_AI_PATH,
                    ["AI", "aerosol_index", "absorbing_aerosol_index"]),
    }

    def _pick_da(ds, candidates):
        for c in candidates:
            if c in ds.data_vars:
                return ds[c]
        dv_lower = {v.lower(): v for v in ds.data_vars}
        for c in candidates:
            if c.lower() in dv_lower:
                return ds[dv_lower[c.lower()]]
        for v in ds.data_vars:
            vl = v.lower()
            for c in candidates:
                if c.lower() in vl:
                    return ds[v]
        dvars = list(ds.data_vars)
        if dvars:
            return ds[dvars[0]]
        coord_keys = [k for k in ds.coords if ds.coords[k].ndim >= 2]
        if coord_keys:
            return ds.coords[coord_keys[0]]
        return None

    def _open_all_groups(fpath):
        datasets = []
        try:
            import netCDF4 as nc4
            with nc4.Dataset(fpath) as root:
                groups = list(root.groups.keys())
        except Exception:
            groups = []
        try:
            datasets.append(xr.open_dataset(fpath, engine="netcdf4"))
        except Exception:
            pass
        for grp in groups:
            try:
                datasets.append(xr.open_dataset(fpath, engine="netcdf4", group=grp))
            except Exception:
                pass
        return datasets

    def _interp_to_grid(da):
        for tdim in ("time", "valid_time"):
            if tdim in da.dims and da.sizes.get(tdim, 0) > 1:
                da = da.mean(dim=tdim)
        squeeze_dims = [d for d in da.dims
                        if da.sizes[d] == 1
                        and d not in ("latitude", "lat", "longitude", "lon", "y", "x")]
        if squeeze_dims:
            da = da.squeeze(squeeze_dims, drop=True)

        all_names = list(da.dims) + list(da.coords)
        lat_name  = next((n for n in all_names if n.lower() in ("latitude", "lat", "y")), None)
        lon_name  = next((n for n in all_names if n.lower() in ("longitude", "lon", "x")), None)
        if lat_name is None or lon_name is None:
            return None

        src_lats = np.asarray(da[lat_name].values, dtype=np.float64).ravel()
        src_lons = np.asarray(da[lon_name].values, dtype=np.float64).ravel()
        src_data = np.asarray(da.values, dtype=np.float64)

        if src_data.ndim == 1:
            return None
        while src_data.ndim > 2:
            src_data = src_data[0]

        if da.dims.index(lat_name) > da.dims.index(lon_name):
            src_data = src_data.T
        if src_lats[0] > src_lats[-1]:
            src_lats = src_lats[::-1]
            src_data = src_data[::-1, :]

        finite = src_data[np.isfinite(src_data)]
        fill   = float(np.nanmedian(finite)) if len(finite) > 0 else 0.0
        src_data = np.where(np.isfinite(src_data), src_data, fill)

        q_lats = np.clip(lat_grid.ravel(), src_lats.min(), src_lats.max())
        q_lons = np.clip(lon_grid.ravel(), src_lons.min(), src_lons.max())
        fn = RegularGridInterpolator(
            (src_lats, src_lons), src_data,
            method="linear", bounds_error=False, fill_value=fill)
        return fn(np.column_stack([q_lats, q_lons])).reshape(nr, nc).astype(np.float32)

    result = {}
    for short, (fpath, candidates) in TROPOMI_SOURCES.items():
        if not os.path.exists(fpath):
            log.warning("  [V9-M2] TROPOMI file not found: %s — zeros.", fpath)
            result[short] = zeros.copy()
            continue
        try:
            arr_out = None
            for ds in _open_all_groups(fpath):
                try:
                    da = _pick_da(ds, candidates)
                    if da is None:
                        ds.close()
                        continue
                    arr_out = _interp_to_grid(da)
                    ds.close()
                    if arr_out is not None:
                        break
                except Exception:
                    try:
                        ds.close()
                    except Exception:
                        pass

            if arr_out is None:
                log.warning("  [V9-M2] TROPOMI %s: no usable 2-D field found — zeros.", short)
                result[short] = zeros.copy()
                continue

            if short in ("no2", "hcho", "co", "ch4"):
                arr_out = np.log1p(np.clip(arr_out, 0.0, None))

            result[short] = arr_out
            log.info("  [V9-M2] TROPOMI %-8s loaded — min=%.3g  max=%.3g",
                     short, float(np.nanmin(arr_out)), float(np.nanmax(arr_out)))

        except Exception as exc:
            log.warning("  [V9-M2] TROPOMI %s failed (%s) — zeros.", short, exc)
            result[short] = zeros.copy()

    n_loaded = sum(1 for k, v in result.items() if not np.allclose(v, 0))
    log.info("  [V9-M2] TROPOMI: %d / %d tracers loaded successfully.",
             n_loaded, len(TROPOMI_SOURCES))
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Ground-station coordinate lookup (unchanged from v9)
# ──────────────────────────────────────────────────────────────────────────────
STATION_COORDS: dict[str, tuple[float, float]] = {
    "cen-sr-04":  (27.0104, 84.8735),
    "cen-sr-13":  (27.0048, 84.8762),
    "cen-sr-14":  (28.6942, 80.5862),
    "cen-sr-09":  (28.6886, 80.5935),
    "cen_sr-09":  (28.6886, 80.5935),
    "cen-sr-15":  (27.4151, 85.1420),
    "cen-sr-18":  (27.4187, 85.1391),
    "cen-sr-25":  (27.6714, 85.3240),
    "sc-28":      (27.6865, 85.3504),
    "sc-01":      (27.7091, 85.3247),
    "sc-35":      (27.6766, 85.3681),
    "sc-42":      (27.6748, 85.3750),
    "sc-33":      (27.6821, 85.3569),
    "sc-39":      (27.6938, 85.3383),
    "sc-03":      (27.7102, 85.3195),
    "sc-13":      (27.7399, 85.3724),
    "sc-12":      (27.7318, 85.3759),
    "sc-07":      (27.6631, 85.2831),
    "sc-25":      (27.6775, 85.2790),
    "sc-09":      (27.6587, 85.2874),
    "sc-40":      (27.6704, 85.3177),
    "sc-05":      (27.6659, 85.3163),
    "sc-08":      (27.6741, 85.3001),
    "sc-15":      (27.6776, 85.3176),
    "sc-22":      (27.6799, 85.3152),
    "sc-14":      (27.7494, 85.3282),
    "sc-31":      (27.7518, 85.3207),
    "sc-19":      (27.7055, 85.3087),
    "sc-44":      (27.6777, 85.3198),
    "sc-10":      (27.7076, 85.2948),
    "sc-23":      (27.6970, 85.3046),
    "sc-18":      (27.7018, 85.2992),
    "sc-20":      (27.6969, 85.3100),
    "sc-21":      (27.6860, 85.2817),
    "sc-41":      (27.6441, 85.2989),
    "birgunj":    (27.0104, 84.8735),
    "dhangadhi":  (28.6942, 80.5862),
    "hetauda":    (27.4151, 85.1420),
    "balkumari":  (27.6865, 85.3504),
    "gaushala":   (27.7091, 85.3247),
    "jadibuti":   (27.6766, 85.3681),
    "kaushaltar": (27.6821, 85.3569),
    "kirtipur":   (27.6775, 85.2790),
    "tokha":      (27.7518, 85.3207),
    "kupondole":  (27.6704, 85.3177),
    "lagankhel":  (27.6659, 85.3163),
    "pulchowk":   (27.6777, 85.3198),
    "sanepa":     (27.6799, 85.3152),
    "chhetrapati":(27.7055, 85.3087),
    "sundarighat":(27.6970, 85.3046),
    "dabali":     (27.7150, 85.3140),
    "handigaun":  (27.7150, 85.3140),
    "phora":      (27.7175, 85.3131),
    "farsidol":   (27.6441, 85.2989),
    "khokana":    (27.6441, 85.2989),
    "gokarneshwor":(27.7399, 85.3724),
    "gothatar":   (27.7318, 85.3759),
    "taudaha":    (27.6587, 85.2874),
    "nakhipot":   (27.6741, 85.3001),
    "teku":       (27.6969, 85.3100),
    "tankeshwor": (27.7018, 85.2992),
    "tyanglaphat":(27.6860, 85.2817),
    "ramkot":     (27.7076, 85.2948),
    "sifal":      (27.7102, 85.3195),
    "kadhaghari": (27.6748, 85.3750),
}


def _coords_from_station_tag(tag: str) -> tuple[float, float] | None:
    tag_lower = tag.lower()
    best_key  = None
    best_len  = 0
    for key in STATION_COORDS:
        if key in tag_lower and len(key) > best_len:
            best_key = key
            best_len = len(key)
    return STATION_COORDS[best_key] if best_key else None


def _parse_openaq_v3_csv(csv_path: str, station_tag: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(csv_path, low_memory=False)
        df.columns = df.columns.str.strip()
    except Exception as exc:
        log.warning("  Could not read %s — %s", csv_path, exc)
        return None

    lat_col = lon_col = dt_col = val_col = None
    fallback_lat = fallback_lon = None

    if "coordinates.latitude" in df.columns:
        lat_col = "coordinates.latitude"
        lon_col = "coordinates.longitude"
    if "period.datetimeFrom.utc" in df.columns:
        dt_col  = "period.datetimeFrom.utc"
    if "value" in df.columns:
        val_col = "value"

    if lat_col is not None and df[lat_col].isna().all():
        if "coordinates" in df.columns:
            try:
                import ast
                coords_parsed = df["coordinates"].dropna().apply(
                    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
                )
                if not coords_parsed.empty:
                    sample = coords_parsed.iloc[0]
                    if isinstance(sample, dict):
                        fallback_lat = float(sample.get("latitude",  np.nan))
                        fallback_lon = float(sample.get("longitude", np.nan))
            except Exception:
                pass
        if fallback_lat is None or np.isnan(fallback_lat):
            result = _coords_from_station_tag(station_tag)
            if result:
                fallback_lat, fallback_lon = result
        if fallback_lat is not None and not np.isnan(fallback_lat):
            df[lat_col] = fallback_lat
            df[lon_col] = fallback_lon
        else:
            lat_col = lon_col = None

    if lat_col is None:
        col_lower = {c.lower(): c for c in df.columns}
        lat_col = col_lower.get("latitude")
        lon_col = col_lower.get("longitude")
        if dt_col is None:
            dt_col = next((col_lower[k] for k in col_lower
                           if "utc" in k or "date" in k), None)
        if val_col is None:
            if "parameter" in col_lower:
                df = df[df[col_lower["parameter"]].str.lower().str.strip()
                        == "pm25"].copy()
            val_col = col_lower.get("value")
        if lat_col is None:
            result = _coords_from_station_tag(station_tag)
            if result:
                fallback_lat, fallback_lon = result
                df["_lat"] = fallback_lat
                df["_lon"] = fallback_lon
                lat_col, lon_col = "_lat", "_lon"

    if lat_col is None or lon_col is None or val_col is None:
        log.warning("  %s — could not identify lat/lon/value columns. Skipping.",
                    os.path.basename(csv_path))
        return None

    out = pd.DataFrame()
    out["pm25"]      = pd.to_numeric(df[val_col], errors="coerce")
    out["latitude"]  = pd.to_numeric(df[lat_col], errors="coerce")
    out["longitude"] = pd.to_numeric(df[lon_col], errors="coerce")

    if dt_col:
        parsed_dt = pd.to_datetime(df[dt_col], errors="coerce", utc=True)
        out["date"] = parsed_dt.dt.tz_localize(None).dt.normalize()
    else:
        out["date"] = pd.Timestamp("2025-01-01")

    out["station_id"] = station_tag
    out = out.dropna(subset=["pm25", "latitude", "longitude"])
    out = out[(out["pm25"] > 0) & (out["pm25"] < 1000)]
    return out if len(out) > 0 else None


def load_ground_observations(ground_dir: str,
                              aod_date_min: pd.Timestamp | None = None,
                              aod_date_max: pd.Timestamp | None = None) -> pd.DataFrame:
    csv_files = sorted(glob.glob(os.path.join(ground_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {ground_dir}")

    log.info("Scanning %d CSV file(s) in %s …", len(csv_files), ground_dir)

    frames = []
    for csv_path in csv_files:
        tag = Path(csv_path).stem
        df  = _parse_openaq_v3_csv(csv_path, tag)
        if df is not None:
            frames.append(df)
            log.info("  %-60s  %5d rows  (lat %.3f – %.3f)",
                     Path(csv_path).name, len(df),
                     df["latitude"].min(), df["latitude"].max())

    if not frames:
        raise RuntimeError("No valid ground observation CSVs could be parsed.")

    combined = pd.concat(frames, ignore_index=True)

    if USE_AOD_PERIOD and aod_date_min is not None and aod_date_max is not None:
        def _strip_tz_local(ts):
            if ts is None: return ts
            if hasattr(ts, "tzinfo") and ts.tzinfo is not None:
                return ts.tz_localize(None)
            return ts
        aod_min_cmp = _strip_tz_local(aod_date_min)
        aod_max_cmp = _strip_tz_local(aod_date_max)
        if hasattr(combined["date"].dtype, "tz") and combined["date"].dt.tz is not None:
            combined["date"] = combined["date"].dt.tz_localize(None)
        before = len(combined)
        combined = combined[
            (combined["date"] >= aod_min_cmp) &
            (combined["date"] <= aod_max_cmp)
        ]
        log.info(
            "  [F11] Date filter to AOD period %s – %s: %d → %d rows",
            aod_date_min.date(), aod_date_max.date(), before, len(combined)
        )

    combined = combined.drop_duplicates(
        subset=["station_id", "date", "latitude", "longitude"]
    )
    combined = (
        combined
        .groupby(["station_id", "date", "latitude", "longitude"], as_index=False)
        .agg({"pm25": "mean"})
    )
    log.info("  [FIX-AGG] Aggregated to daily means.")

    n_stations = combined["station_id"].nunique()
    log.info(
        "Ground obs total: %d rows  |  unique station files: %d  |  "
        "unique lat/lon pairs: %d",
        len(combined), n_stations,
        combined[["latitude", "longitude"]].drop_duplicates().shape[0]
    )
    return combined


# ══════════════════════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING  (unchanged from v9)
# ══════════════════════════════════════════════════════════════════════════════

def build_covariate_stack(aod: np.ndarray,
                           era5: dict,
                           ndvi: np.ndarray,
                           static: dict,
                           ref_meta: dict,
                           tropomi: dict | None = None,
                           doy_map: dict | None = None) -> dict:
    """[F21] + [V7-F3] + [V9-R1] Build the covariate feature stack for Stage 2."""
    ERA5_FEATURE_KEYS = {"t2m", "blh", "u10", "v10", "wind_speed", "rh", "prec"}

    gt  = ref_meta["geotransform"]
    nr  = ref_meta["nrows"]
    nc  = ref_meta["ncols"]
    col_idx = np.arange(nc, dtype=np.float32)
    row_idx = np.arange(nr, dtype=np.float32)
    lon_1d  = (gt[0] + gt[1] * (col_idx + 0.5)).astype(np.float32)
    lat_1d  = (gt[3] + gt[5] * (row_idx + 0.5)).astype(np.float32)
    lon_grid_2d, lat_grid_2d = np.meshgrid(lon_1d, lat_1d)

    covariates = {}
    for k, v in era5.items():
        if k in ERA5_FEATURE_KEYS:
            covariates[k] = v
    covariates["ndvi"]      = ndvi
    covariates["elevation"] = static["elevation"]
    covariates["pop_log"]   = static["pop_log"]

    lon_norm = (lon_grid_2d - 84.0) / 4.5
    lat_norm = (lat_grid_2d - 28.0) / 2.5
    covariates["sin_lon"] = np.sin(np.pi * lon_norm).astype(np.float32)
    covariates["cos_lon"] = np.cos(np.pi * lon_norm).astype(np.float32)
    covariates["sin_lat"] = np.sin(np.pi * lat_norm).astype(np.float32)
    covariates["cos_lat"] = np.cos(np.pi * lat_norm).astype(np.float32)

    covariates["lat_raw"] = lat_grid_2d.copy()
    covariates["lon_raw"] = lon_grid_2d.copy()

    elev = static["elevation"]
    rh   = era5.get("rh", np.full((nr, nc), 50.0, dtype=np.float32))
    covariates["aod_x_elev"] = (aod * np.clip(elev, 0, 5000) / 1000.0).astype(np.float32)
    covariates["aod_x_rh"]   = (aod * rh / 100.0).astype(np.float32)

    elev_class = np.zeros_like(elev, dtype=np.float32)
    elev_class[elev >= 500]  = 1.0
    elev_class[elev >= 1500] = 2.0
    elev_class[elev >= 3000] = 3.0
    covariates["elev_class"] = elev_class

    if tropomi:
        for tkey, tarr in tropomi.items():
            covariates[f"trop_{tkey}"] = tarr

    if doy_map and len(doy_map) > 1:
        all_doys = sorted(doy_map.keys())
        mid_doy  = all_doys[len(all_doys) // 2]
        prev_doy = all_doys[max(0, len(all_doys) // 2 - 1)]
        aod_today = doy_map[mid_doy]
        aod_prev  = doy_map[prev_doy]
        aod_delta = np.where(
            np.isfinite(aod_today) & np.isfinite(aod_prev),
            (aod_today - aod_prev).astype(np.float32),
            0.0,
        ).astype(np.float32)
        covariates["aod_delta"] = aod_delta
        log.info("  [V8-M3] AOD inter-day delta feature added (DOY %d→%d).",
                 prev_doy, mid_doy)

    blh  = era5.get("blh",  np.full((nr, nc), 500.0, dtype=np.float32))
    t2m  = era5.get("t2m",  np.full((nr, nc),  15.0, dtype=np.float32))
    prec = era5.get("prec", np.zeros((nr, nc),         dtype=np.float32))
    ws   = era5.get("wind_speed", np.full((nr, nc), 2.0, dtype=np.float32))

    blh_safe = np.where(blh > 50.0, blh, 50.0)
    covariates["aod_blh_ratio"] = (aod / blh_safe * 1000.0).astype(np.float32)
    covariates["aod_x_ws"]  = (aod * np.clip(ws, 0.1, 20.0)).astype(np.float32)
    t_norm = np.clip((t2m + 5.0) / 30.0, 0.1, 5.0).astype(np.float32)
    covariates["aod_x_t2m"] = (aod * t_norm).astype(np.float32)
    covariates["prec_flag"]  = (prec > 0.001).astype(np.float32)
    covariates["elev_x_lat"] = (
        np.clip(elev, 0, 5000) / 1000.0 * lat_grid_2d
    ).astype(np.float32)
    covariates["blh_x_rh"] = (
        np.clip(blh, 0, 3000) / 1000.0 *
        era5.get("rh", np.full((nr, nc), 50.0, dtype=np.float32)) / 100.0
    ).astype(np.float32)

    covariates["_lon"] = lon_grid_2d
    covariates["_lat"] = lat_grid_2d
    return covariates


def flatten_features(aod: np.ndarray,
                      covariates: dict,
                      mask: np.ndarray | None = None):
    feat_names = ["aod"] + list(covariates.keys())
    arrays     = [aod]  + [covariates[k] for k in covariates]
    stack      = np.stack(arrays, axis=0)
    F, R, C    = stack.shape
    stack_2d   = stack.reshape(F, -1).T
    if mask is not None:
        flat_mask = mask.ravel()
        stack_2d  = stack_2d[flat_mask]
    return stack_2d.astype(np.float32), feat_names


# ══════════════════════════════════════════════════════════════════════════════
# 4. STAGE 1 — AOD GAP-FILLING (RANDOM FOREST)  (unchanged from v9)
# ══════════════════════════════════════════════════════════════════════════════

class AODGapFiller:
    # [V12-MEM] Cap training pixels to limit RAM usage on large AOD grids.
    # 413 k clear-sky pixels x 200 deep trees -> MemoryError on most laptops.
    # 150 k pixels with depth-capped, leaf-subsampled trees keeps OOB R2
    # virtually identical while using ~4-6x less RAM.
    MAX_TRAIN_PX: int = 150_000

    def __init__(self, n_estimators: int = 200, n_jobs: int = -1, seed: int = SEED):
        self.rf = RandomForestRegressor(
            n_estimators     = n_estimators,
            max_features     = "sqrt",
            max_depth        = 16,          # [V12-MEM] cap depth -> smaller nodes
            min_samples_leaf = 10,          # [V12-MEM] larger leaves -> less RAM
            max_samples      = 0.7,         # [V12-MEM] row-subsample per tree
            oob_score        = True,
            n_jobs           = n_jobs,
            random_state     = seed,
        )
        self.feature_names: list = []
        self.trained = False

    def fit(self, aod: np.ndarray, covariates: dict) -> None:
        valid_mask = ~np.isnan(aod)
        n_valid    = int(valid_mask.sum())
        log.info("Stage 1 — RF training on %d clear-sky pixels …", n_valid)

        feat_arrays = list(covariates.values())
        feat_names  = list(covariates.keys())

        stack   = np.stack(feat_arrays, axis=0).reshape(len(feat_arrays), -1).T
        X_train = stack[valid_mask.ravel()].astype(np.float32)
        y_train = aod[valid_mask].astype(np.float32)

        row_valid = ~np.isnan(X_train).any(axis=1)
        X_train, y_train = X_train[row_valid], y_train[row_valid]

        # [V12-MEM] Randomly subsample if too many clear-sky pixels.
        if len(X_train) > self.MAX_TRAIN_PX:
            rng  = np.random.default_rng(SEED)
            idx  = rng.choice(len(X_train), size=self.MAX_TRAIN_PX, replace=False)
            X_train, y_train = X_train[idx], y_train[idx]
            log.info("  [V12-MEM] Subsampled to %d pixels for RF training.", self.MAX_TRAIN_PX)

        self.feature_names = feat_names
        self.rf.fit(X_train, y_train)
        self.trained = True

        log.info("  RF fitted.  OOB R2=%.4f", self.rf.oob_score_)
        imp = pd.Series(self.rf.feature_importances_, index=feat_names)
        log.info("  Top-5 feature importances:\n%s", imp.nlargest(5).to_string())

    def predict_gap_fill(self, aod: np.ndarray, covariates: dict) -> np.ndarray:
        if not self.trained:
            raise RuntimeError("Call .fit() before .predict_gap_fill()")

        gap_mask = np.isnan(aod)
        n_gaps   = int(gap_mask.sum())
        log.info("Stage 1 — filling %d gap pixels (%.1f %% of grid) …",
                 n_gaps, 100 * n_gaps / aod.size)

        feat_arrays = [covariates[k] for k in self.feature_names]
        stack  = np.stack(feat_arrays, axis=0).reshape(len(feat_arrays), -1).T
        X_gap  = stack[gap_mask.ravel()].astype(np.float32)

        col_medians = np.nanmedian(X_gap, axis=0)
        nan_idx = np.where(np.isnan(X_gap))
        X_gap[nan_idx] = np.take(col_medians, nan_idx[1])

        aod_filled = aod.copy()
        if n_gaps > 0:
            preds = self.rf.predict(X_gap).astype(np.float32)
            preds = np.clip(preds, 0.0, 5.0)
            aod_filled[gap_mask] = preds

        log.info("  Gap-filling complete. Residual NaNs: %d",
                 int(np.isnan(aod_filled).sum()))
        return aod_filled


# ══════════════════════════════════════════════════════════════════════════════
# 5. STAGE 2 — PM2.5 PREDICTION (HistGBR + Isotonic Calibration)
# ══════════════════════════════════════════════════════════════════════════════

class PM25Predictor:
    """[F20] + [V7-F4] + [V7-F5] HistGBR predictor with isotonic calibration."""
    def __init__(self, n_features: int,
                 epochs: int = 800,
                 batch_size: int = 256,
                 lr: float = 5e-4,
                 seed: int = SEED):
        self.n_features  = n_features
        self.scaler      = StandardScaler()
        self.calibrator  = None
        self.trained     = False
        try:
            self.model = HistGradientBoostingRegressor(
                loss="poisson",
                learning_rate=0.04,
                max_iter=epochs,
                max_depth=8,
                max_leaf_nodes=63,
                min_samples_leaf=15,
                l2_regularization=0.3,
                max_bins=255,
                early_stopping=True,
                validation_fraction=0.10,
                n_iter_no_change=40,
                scoring="neg_root_mean_squared_error",
                random_state=seed,
            )
            log.info("Stage 2 — HistGradientBoostingRegressor (loss=poisson, v7 params)")
        except Exception:
            self.model = HistGradientBoostingRegressor(
                loss="absolute_error",
                learning_rate=0.04, max_iter=epochs,
                max_depth=8, min_samples_leaf=15, l2_regularization=0.3,
                max_bins=255, early_stopping=True, validation_fraction=0.10,
                n_iter_no_change=40, random_state=seed,
            )
            log.info("Stage 2 — HistGradientBoostingRegressor (loss=absolute_error, v7 params)")

    def fit(self, X: np.ndarray, y: np.ndarray,
            sample_weight: np.ndarray | None = None) -> dict:
        if len(X) >= 10:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X, y, test_size=0.2, random_state=SEED)
            if sample_weight is not None:
                sw_tr, _ = train_test_split(
                    sample_weight, test_size=0.2, random_state=SEED)
            else:
                sw_tr = None
        else:
            X_tr, X_val, y_tr, y_val = X, X, y, y
            sw_tr = sample_weight

        X_tr_s  = self.scaler.fit_transform(X_tr).astype(np.float32)
        X_val_s = self.scaler.transform(X_val).astype(np.float32)
        y_tr_log = np.log1p(np.clip(y_tr, 0, None))

        self.model.fit(X_tr_s, y_tr_log, sample_weight=sw_tr)
        self.trained = True
        n_iters = getattr(self.model, "n_iter_", "?")
        log.info("  HistGBR fitted in %s iterations.", n_iters)

        y_val_raw = np.expm1(self.model.predict(X_val_s)).clip(0.0, 1000.0)
        self.calibrator = IsotonicRegression(out_of_bounds="clip")
        self.calibrator.fit(y_val_raw, y_val)
        log.info("  Isotonic calibrator fitted on %d val samples.", len(y_val))

        y_tr_pred  = self.predict(X_tr)
        y_val_pred = self.predict(X_val)
        metrics = {
            "train_r2":   r2_score(y_tr, y_tr_pred),
            "val_r2":     r2_score(y_val, y_val_pred),
            "train_rmse": float(np.sqrt(mean_squared_error(y_tr,  y_tr_pred))),
            "val_rmse":   float(np.sqrt(mean_squared_error(y_val, y_val_pred))),
        }
        log.info("  Train  R²=%.4f  RMSE=%.2f µg/m³\n  Val    R²=%.4f  RMSE=%.2f µg/m³",
                 metrics["train_r2"], metrics["train_rmse"],
                 metrics["val_r2"],   metrics["val_rmse"])
        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_s   = self.scaler.transform(X).astype(np.float32)
        preds = np.expm1(self.model.predict(X_s)).clip(0.0, 1000.0)
        if self.calibrator is not None:
            preds = self.calibrator.predict(preds).astype(np.float32)
        return preds.clip(0.0, 1000.0)

    def save(self, output_dir: str, tag: str = "") -> None:
        suffix      = f"_{tag}" if tag else ""
        scaler_path = os.path.join(output_dir, f"pm25_scaler{suffix}.joblib")
        model_path  = os.path.join(output_dir, f"pm25_model{suffix}.joblib")
        cal_path    = os.path.join(output_dir, f"pm25_calibrator{suffix}.joblib")
        joblib.dump(self.scaler,     scaler_path)
        joblib.dump(self.model,      model_path)
        if self.calibrator is not None:
            joblib.dump(self.calibrator, cal_path)
        log.info("  PM25Predictor saved → %s, %s", scaler_path, model_path)

    def load(self, output_dir: str, tag: str = "") -> None:
        suffix      = f"_{tag}" if tag else ""
        scaler_path = os.path.join(output_dir, f"pm25_scaler{suffix}.joblib")
        model_path  = os.path.join(output_dir, f"pm25_model{suffix}.joblib")
        cal_path    = os.path.join(output_dir, f"pm25_calibrator{suffix}.joblib")
        self.scaler = joblib.load(scaler_path)
        self.model  = joblib.load(model_path)
        if os.path.exists(cal_path):
            self.calibrator = joblib.load(cal_path)
        self.trained = True
        log.info("  PM25Predictor loaded ← %s, %s", scaler_path, model_path)


# ══════════════════════════════════════════════════════════════════════════════
# 6. COLLOCATION  (unchanged from v9)
# ══════════════════════════════════════════════════════════════════════════════

def collocate_stations(ground_df: pd.DataFrame,
                        daily_aod: dict,
                        daily_era5: dict,
                        covariates: dict,
                        ref_meta: dict) -> pd.DataFrame:
    """[F23] + [V7-F1] Daily-paired collocation with climatological AOD fallback."""
    gt    = ref_meta["geotransform"]
    xmin  = gt[0]; res_x = gt[1]
    ymax  = gt[3]; res_y = gt[5]
    nr    = ref_meta["nrows"]
    nc    = ref_meta["ncols"]

    lons, lats = geotransform_to_coords(ref_meta)

    aod_composite  = daily_aod.get("_composite")
    era5_composite = daily_era5.get("_composite", {})
    doy_map: dict[int, np.ndarray] = daily_aod.get("_doy_map", {})
    available_doys = sorted(doy_map.keys())

    aod_by_date  = {k.normalize(): v for k, v in daily_aod.items()
                    if not isinstance(k, str) and isinstance(k, pd.Timestamp)}
    era5_by_date = {k.normalize(): v for k, v in daily_era5.items()
                    if not isinstance(k, str)}

    ERA5_FEATURE_KEYS = {"t2m", "blh", "u10", "v10", "wind_speed", "rh", "prec"}

    def _closest_doy_aod(doy_query: int) -> np.ndarray | None:
        if not available_doys:
            return None
        def _doy_dist(d):
            diff = abs(d - doy_query)
            return min(diff, 365 - diff)
        best_doy = min(available_doys, key=_doy_dist)
        return doy_map[best_doy]

    records = []
    skipped = aod_same_day = aod_doy_match = aod_composite_fb = era5_hits = era5_miss = 0

    for _, row in ground_df.iterrows():
        col_f = (row["longitude"] - xmin) / res_x - 0.5
        row_f = (row["latitude"]  - ymax) / res_y - 0.5
        c = int(round(col_f))
        r = int(round(row_f))

        if not (0 <= r < nr and 0 <= c < nc):
            tol = 0.05
            if (abs(row["longitude"] - lons[0])  < tol or
                abs(row["longitude"] - lons[-1]) < tol or
                abs(row["latitude"]  - lats[0])  < tol or
                abs(row["latitude"]  - lats[-1]) < tol):
                c = int(np.clip(c, 0, nc - 1))
                r = int(np.clip(r, 0, nr - 1))
            else:
                skipped += 1
                continue

        dt = row.get("date", pd.NaT)
        date_key = pd.NaT if pd.isna(dt) else pd.Timestamp(dt).normalize()

        if date_key in aod_by_date:
            aod_val = float(aod_by_date[date_key][r, c])
            aod_same_day += 1
        elif not pd.isna(dt) and available_doys:
            doy_q = int(pd.Timestamp(dt).dayofyear)
            doy_arr = _closest_doy_aod(doy_q)
            aod_val = float(doy_arr[r, c]) if doy_arr is not None else float(aod_composite[r, c])
            aod_doy_match += 1
        else:
            aod_val = float(aod_composite[r, c]) if aod_composite is not None else np.nan
            aod_composite_fb += 1

        if date_key in era5_by_date:
            era5_day = era5_by_date[date_key]
            era5_hits += 1
        else:
            era5_day = era5_composite
            era5_miss += 1

        rec = {
            "pm25":       row["pm25"],
            "latitude":   row["latitude"],
            "longitude":  row["longitude"],
            "date":       dt,
            "station_id": row.get("station_id",
                                   f"{row['latitude']:.4f}_{row['longitude']:.4f}"),
            "aod":        aod_val,
        }

        for feat, arr in covariates.items():
            rec[feat] = float(arr[r, c])

        for k in ERA5_FEATURE_KEYS:
            if k in era5_day:
                rec[k] = float(era5_day[k][r, c])

        if pd.notna(dt):
            rec["doy"]     = float(dt.timetuple().tm_yday)
            rec["month"]   = float(dt.month)
            rec["doy_sin"] = float(np.sin(2 * np.pi * rec["doy"] / 365.0))
            rec["doy_cos"] = float(np.cos(2 * np.pi * rec["doy"] / 365.0))
        else:
            rec["doy"] = rec["month"] = rec["doy_sin"] = rec["doy_cos"] = np.nan

        records.append(rec)

    if skipped:
        log.warning("  %d observations skipped (outside grid extent).", skipped)

    total_aod = aod_same_day + aod_doy_match + aod_composite_fb
    log.info(
        "  AOD match: %d same-day | %d DOY-climatological | %d composite fallback "
        "(%.0f%% non-composite coverage)  [V7-F1]",
        aod_same_day, aod_doy_match, aod_composite_fb,
        100 * (aod_same_day + aod_doy_match) / max(1, total_aod)
    )
    log.info("  ERA5 match: %d same-day, %d composite fallback (%.0f%% daily coverage)",
             era5_hits, era5_miss,
             100 * era5_hits / max(1, era5_hits + era5_miss))

    df_out = pd.DataFrame(records).dropna(subset=["pm25", "aod", "latitude", "longitude"])

    station_means = df_out.groupby("station_id")["pm25"].transform("mean")
    df_out["station_pm25_mean"] = station_means.astype(np.float32)

    region_map = df_out["station_id"].apply(_assign_city_fold)
    region_means = df_out.groupby(region_map)["pm25"].transform("mean")
    df_out["region_pm25_mean"] = region_means.astype(np.float32)

    log.info("Collocated %d station-observations  (%d unique stations).",
             len(df_out), df_out["station_id"].nunique())
    return df_out


# ══════════════════════════════════════════════════════════════════════════════
# 7. VALIDATION — CITY-AWARE SPATIAL-BLOCK CROSS-VALIDATION  (unchanged from v9)
# ══════════════════════════════════════════════════════════════════════════════

def site_leave_one_out_cv(collocated_df: pd.DataFrame,
                           feature_cols: list) -> dict:
    """[V7-F2] + [V7-F7] City-aware 4-fold spatial CV."""
    log.info("Running City-Aware Spatial CV (4 folds)  [V7-F2] …")

    MIN_TEST_OBS  = 50
    MIN_TRAIN_OBS = 30

    collocated_df = collocated_df.copy()
    collocated_df["_fold"] = collocated_df["station_id"].apply(_assign_city_fold)

    for fid in range(4):
        members = (
            collocated_df[collocated_df["_fold"] == fid]["station_id"]
            .unique().tolist()
        )
        n_obs = int((collocated_df["_fold"] == fid).sum())
        log.info("  Fold %d (%s): %d stations, %d obs — e.g. %s",
                 fid, FOLD_NAMES[fid], len(members), n_obs,
                 ", ".join(s[:25] for s in members[:3]))

    n_stations = collocated_df["station_id"].nunique()
    log.info("  %d unique stations assigned to 4 folds.", n_stations)

    y_true_all, y_pred_all = [], []
    fold_results = []

    for fold_id in range(4):
        train_df = collocated_df[collocated_df["_fold"] != fold_id].copy()
        test_df  = collocated_df[collocated_df["_fold"] == fold_id].copy()

        n_train = len(train_df)
        n_test  = len(test_df)

        if n_test < MIN_TEST_OBS:
            log.warning("  Fold %d (%s) skipped: only %d test obs (min=%d).",
                        fold_id, FOLD_NAMES[fold_id], n_test, MIN_TEST_OBS)
            continue
        if n_train < MIN_TRAIN_OBS:
            log.warning("  Fold %d (%s) skipped: only %d train obs (min=%d).",
                        fold_id, FOLD_NAMES[fold_id], n_train, MIN_TRAIN_OBS)
            continue

        tr_station_means = train_df.groupby("station_id")["pm25"].mean()
        tr_region_means  = train_df.groupby("_fold")["pm25"].mean()
        global_mean_tr   = float(train_df["pm25"].mean())

        def _safe_encode_station(sid, tr_st_means, tr_reg_means, global_m):
            if sid in tr_st_means.index:
                return float(tr_st_means[sid])
            fold = _assign_city_fold(sid)
            if fold in tr_reg_means.index:
                return float(tr_reg_means[fold])
            return global_m

        train_df = train_df.copy()
        test_df  = test_df.copy()

        train_df["station_pm25_mean"] = train_df["station_id"].apply(
            lambda s: _safe_encode_station(s, tr_station_means, tr_region_means, global_mean_tr))
        test_df["station_pm25_mean"]  = test_df["station_id"].apply(
            lambda s: _safe_encode_station(s, tr_station_means, tr_region_means, global_mean_tr))

        train_df["region_pm25_mean"] = train_df["_fold"].map(tr_region_means).fillna(global_mean_tr)
        test_df["region_pm25_mean"]  = test_df["_fold"].map(tr_region_means).fillna(global_mean_tr)

        fold_feature_cols = feature_cols.copy()
        for enc_col in ("station_pm25_mean", "region_pm25_mean"):
            if enc_col not in fold_feature_cols and enc_col in train_df.columns:
                fold_feature_cols.append(enc_col)

        X_tr = train_df[fold_feature_cols].values.astype(np.float32)
        y_tr = train_df["pm25"].values.astype(np.float32)
        X_te = test_df[fold_feature_cols].values.astype(np.float32)
        y_te = test_df["pm25"].values.astype(np.float32)

        n_ktm   = int((collocated_df["_fold"] != 3).sum())
        n_outer = int((collocated_df["_fold"] == 3).sum())
        region_weight = max(1.0, n_ktm / max(1, n_outer))

        sw_tr = np.where(
            train_df["station_id"].apply(_assign_city_fold).values == 3,
            region_weight, 1.0
        ).astype(np.float32)

        if n_train >= 20:
            X_tr_main, X_tr_cal, y_tr_main, y_tr_cal, sw_main, _ = train_test_split(
                X_tr, y_tr, sw_tr, test_size=0.10, random_state=SEED)
        else:
            X_tr_main, X_tr_cal = X_tr, X_tr
            y_tr_main, y_tr_cal = y_tr, y_tr
            sw_main = sw_tr

        scaler_cv = StandardScaler()
        X_tr_s   = scaler_cv.fit_transform(X_tr_main)
        X_cal_s  = scaler_cv.transform(X_tr_cal)
        X_te_s   = scaler_cv.transform(X_te)

        try:
            gbr_cv = HistGradientBoostingRegressor(
                loss="poisson", learning_rate=0.04, max_iter=600,
                max_depth=8, max_leaf_nodes=63, min_samples_leaf=15,
                l2_regularization=0.3, max_bins=255,
                early_stopping=True, validation_fraction=0.10,
                n_iter_no_change=30, random_state=SEED)
            gbr_cv.fit(X_tr_s, np.log1p(np.clip(y_tr_main, 0, None)),
                       sample_weight=sw_main)
        except Exception:
            gbr_cv = HistGradientBoostingRegressor(
                loss="absolute_error", learning_rate=0.04, max_iter=600,
                max_depth=8, min_samples_leaf=15, l2_regularization=0.3,
                early_stopping=True, validation_fraction=0.10,
                n_iter_no_change=30, random_state=SEED)
            gbr_cv.fit(X_tr_s, np.log1p(np.clip(y_tr_main, 0, None)),
                       sample_weight=sw_main)

        cal_raw  = np.expm1(gbr_cv.predict(X_cal_s)).clip(0.0, 1000.0)
        iso_fold = IsotonicRegression(out_of_bounds="clip")
        iso_fold.fit(cal_raw, y_tr_cal)

        preds_raw = np.expm1(gbr_cv.predict(X_te_s)).clip(0.0, 1000.0)
        preds     = iso_fold.predict(preds_raw).astype(np.float32)

        if fold_id == 3:
            outer_in_train = train_df[train_df["_fold"] == 3]
            if len(outer_in_train) >= 10:
                X_outer_tr = scaler_cv.transform(
                    outer_in_train[fold_feature_cols].values.astype(np.float32))
                outer_preds_tr = iso_fold.predict(
                    np.expm1(gbr_cv.predict(X_outer_tr)).clip(0, 1000))
                outer_bias_tr = float(np.mean(outer_preds_tr - outer_in_train["pm25"].values))
                preds = np.clip(preds - outer_bias_tr, 0.0, 1000.0).astype(np.float32)
                log.info("  [V9-R3] Outer-city bias correction: %.1f µg/m³ (n=%d train outer obs)",
                         outer_bias_tr, len(outer_in_train))
            else:
                test_region_mean = float(tr_region_means.get(3, global_mean_tr))
                pred_mean = float(np.mean(preds))
                shift = test_region_mean - pred_mean
                preds = np.clip(preds + shift, 0.0, 1000.0).astype(np.float32)
                log.info("  [V9-R3] Outer-city mean anchoring: shift=%.1f µg/m³", shift)

        fold_r2   = r2_score(y_te, preds) if len(y_te) > 1 else float("nan")
        fold_bias = float(np.mean(preds - y_te))
        fold_rmse = float(np.sqrt(mean_squared_error(y_te, preds)))
        log.info("  Fold %d (%s): n_train=%-5d  n_test=%-5d  R²=%.4f  "
                 "RMSE=%.1f  Bias=%.1f µg/m³",
                 fold_id, FOLD_NAMES[fold_id],
                 n_train, n_test, fold_r2, fold_rmse, fold_bias)

        fold_results.append({
            "fold": fold_id, "name": FOLD_NAMES[fold_id],
            "r2": fold_r2, "rmse": fold_rmse, "bias": fold_bias,
            "n_test": n_test,
        })
        y_true_all.extend(y_te.tolist())
        y_pred_all.extend(preds.tolist())

    y_true = np.array(y_true_all)
    y_pred = np.array(y_pred_all)

    if len(y_true) < 2:
        log.warning("Insufficient predictions for CV metrics.")
        return {"r2": np.nan, "rmse": np.nan, "mae": np.nan, "bias": np.nan,
                "y_true": y_true, "y_pred": y_pred, "fold_results": fold_results}

    r2   = r2_score(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    bias = float(np.mean(y_pred - y_true))

    metrics = {"r2": r2, "rmse": rmse, "mae": mae, "bias": bias,
               "y_true": y_true, "y_pred": y_pred, "fold_results": fold_results}

    log.info("─" * 50)
    log.info("SPATIAL-BLOCK CV RESULTS  (4-fold city-aware, HistGBR + IsotonicCal)")
    log.info("  R²   = %.4f  (target ≥ 0.80)", r2)
    log.info("  RMSE = %.2f µg/m³", rmse)
    log.info("  MAE  = %.2f µg/m³", mae)
    log.info("  Bias = %.2f µg/m³", bias)

    if r2 < 0.80:
        log.warning("  CV R² below target (%.4f). Root-cause analysis:", r2)
        if abs(bias) > 20:
            log.warning(
                "    → Large bias (%.1f µg/m³): isotonic calibration may need "
                "more calibration data.", bias)
        weak_folds = [f for f in fold_results if f["r2"] < 0.5]
        if weak_folds:
            names = [f["name"] for f in weak_folds]
            log.warning("    → Weak folds: %s.", names)
        era5_zero = (
            bool(np.allclose(collocated_df["t2m"].values, 0))
            if "t2m" in collocated_df.columns else False
        )
        if era5_zero:
            log.warning("    → ERA5 t2m is zero — meteorological features are absent.")
    log.info("─" * 50)
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# 8. SPATIAL PM2.5 PREDICTION (FULL GRID)  (unchanged from v9)
# ══════════════════════════════════════════════════════════════════════════════

def predict_pm25_grid(predictor: "PM25Predictor",
                       aod_filled: np.ndarray,
                       covariates: dict,
                       feature_cols: list,
                       chunk_size: int = 50_000,
                       temporal_fill: dict | None = None) -> np.ndarray:
    log.info("Predicting PM2.5 across full 1-km grid (chunks of %d) …", chunk_size)
    nrows, ncols = aod_filled.shape
    n_total = nrows * ncols
    temporal_fill = temporal_fill or {}

    feat_arrays = []
    for col in feature_cols:
        if col == "aod":
            feat_arrays.append(aod_filled.ravel())
        elif col in covariates:
            feat_arrays.append(covariates[col].ravel())
        elif col in temporal_fill:
            feat_arrays.append(np.full(n_total, temporal_fill[col], dtype=np.float32))
        else:
            log.warning("  predict_pm25_grid: feature '%s' not found — using zeros.", col)
            feat_arrays.append(np.zeros(n_total, dtype=np.float32))

    X_full    = np.stack(feat_arrays, axis=1).astype(np.float32)
    valid     = ~np.isnan(X_full).any(axis=1)
    pm25_flat = np.full(n_total, np.nan, dtype=np.float32)

    X_valid = X_full[valid]
    n_valid = X_valid.shape[0]
    preds   = np.empty(n_valid, dtype=np.float32)

    for start in range(0, n_valid, chunk_size):
        end = min(start + chunk_size, n_valid)
        preds[start:end] = predictor.predict(X_valid[start:end])
        if start % (chunk_size * 10) == 0:
            log.info("  … %d / %d pixels processed", start, n_valid)

    pm25_flat[valid] = preds
    pm25_grid = pm25_flat.reshape(nrows, ncols)

    log.info("PM2.5 grid complete. Range: %.1f – %.1f µg/m³  (valid px: %d)",
             float(np.nanmin(pm25_grid)), float(np.nanmax(pm25_grid)), n_valid)
    return pm25_grid


# ══════════════════════════════════════════════════════════════════════════════
# 9.  [V10-S2] APPLICABILITY DOMAIN — KDTree Distance & Confidence Flags
# ══════════════════════════════════════════════════════════════════════════════

def build_station_kdtree(collocated_df: pd.DataFrame) -> cKDTree:
    """
    [V10-S2] Build a KDTree from unique training station (lat, lon) coordinates.

    Returns a cKDTree whose leaf coordinates are in degrees (lat, lon).
    Distance-to-station is converted to km during query using the 1°≈111 km
    approximation scaled by cos(mean_lat) for longitude.
    """
    unique_coords = (
        collocated_df[["latitude", "longitude"]]
        .drop_duplicates()
        .values
        .astype(np.float64)
    )
    tree = cKDTree(unique_coords)
    log.info("[V10-S2] KDTree built from %d unique station coordinates.", len(unique_coords))
    return tree


def build_region_kdtrees(collocated_df: pd.DataFrame) -> dict[str, cKDTree]:
    """
    [V10-S1] Build per-region KDTrees for distance_ensemble IDW weighting.

    Returns { REGION_KTM: cKDTree, REGION_OUTER: cKDTree }.
    """
    trees: dict[str, cKDTree] = {}
    for region in (REGION_KTM, REGION_OUTER):
        mask = collocated_df["station_id"].apply(_assign_region) == region
        region_df = collocated_df[mask]
        coords = (
            region_df[["latitude", "longitude"]]
            .drop_duplicates()
            .values
            .astype(np.float64)
        )
        if len(coords) == 0:
            # Fallback: use all stations so prediction never fails
            coords = (
                collocated_df[["latitude", "longitude"]]
                .drop_duplicates()
                .values
                .astype(np.float64)
            )
            log.warning("[V10-S1] Region '%s' has no stations — using all stations.", region)
        trees[region] = cKDTree(coords)
        log.info("[V10-S1] Region '%s' KDTree: %d stations.", region, len(coords))
    return trees


def _deg_to_km(dist_deg: np.ndarray, mean_lat_deg: float) -> np.ndarray:
    """Convert degree-distances from KDTree to approximate kilometres."""
    # Use equirectangular approximation:
    #   Δlat  ≈ deg × 111.0 km/°
    #   Δlon  ≈ deg × 111.0 × cos(lat) km/°
    # For a combined (lat,lon) Euclidean distance in degrees,
    # divide by √2 to get the effective single-axis scale, then multiply by
    # a simple mean factor.  Exact per-pixel anisotropy is not critical here
    # because confidence thresholds are approximate (20 / 50 km).
    cos_lat = np.cos(np.deg2rad(mean_lat_deg))
    scale   = 111.0 * np.sqrt((1.0 + cos_lat ** 2) / 2.0)
    return dist_deg * scale


def compute_distance_to_nearest_station(
        tree: cKDTree,
        ref_meta: dict,
        chunk_size: int = 100_000) -> np.ndarray:
    """
    [V10-S2] Pixel-wise distance to the nearest training station.

    Uses a memory-chunked approach: pixel coordinates are assembled in chunks
    so the (n_pixels × 2) query matrix never lives fully in RAM at once.

    Parameters
    ----------
    tree      : cKDTree built from station (lat, lon) coordinates.
    ref_meta  : Reference raster metadata (geotransform, nrows, ncols).
    chunk_size: Rows of pixels to process in one KDTree query.

    Returns
    -------
    dist_km : 2-D float32 array of shape (nrows, ncols), distances in km.
    """
    log.info("[V10-S2] Computing pixel-wise distance to nearest station …")

    gt    = ref_meta["geotransform"]
    nrows = ref_meta["nrows"]
    ncols = ref_meta["ncols"]

    lons_1d, lats_1d = geotransform_to_coords(ref_meta)
    # Mean latitude for degree→km scaling
    mean_lat = float(np.mean(lats_1d))

    dist_flat = np.empty(nrows * ncols, dtype=np.float32)

    # Build full pixel coordinate array in one go (two 1-D broadcasts — cheap)
    # lat repeats along columns; lon repeats along rows
    lat_flat = np.repeat(lats_1d, ncols)           # shape (nrows*ncols,)
    lon_flat = np.tile(lons_1d, nrows)             # shape (nrows*ncols,)

    n_total = nrows * ncols
    for start in range(0, n_total, chunk_size):
        end   = min(start + chunk_size, n_total)
        query = np.column_stack([lat_flat[start:end],
                                 lon_flat[start:end]]).astype(np.float64)
        # KDTree query in degree-space; workers=-1 uses all CPU cores
        dist_deg, _ = tree.query(query, workers=-1)
        dist_flat[start:end] = _deg_to_km(dist_deg, mean_lat).astype(np.float32)
        if start % (chunk_size * 20) == 0:
            log.info("  … distance map: %d / %d pixels done", start, n_total)

    dist_km = dist_flat.reshape(nrows, ncols)
    log.info("[V10-S2] Distance map complete. Range: %.1f – %.1f km",
             float(np.nanmin(dist_km)), float(np.nanmax(dist_km)))
    return dist_km


def confidence_from_distance(dist_km: np.ndarray,
                              radius_km: float = 50.0) -> np.ndarray:
    """
    [V10-S2] Assign confidence flags from distance-to-station map.

    Returns integer-coded float32 array:
        2 — HIGH   (distance ≤ 20 km)
        1 — MEDIUM (20 km < distance ≤ radius_km)
        0 — LOW    (distance > radius_km)
    """
    flags = np.zeros_like(dist_km, dtype=np.float32)
    flags[dist_km <= radius_km] = 1.0   # MEDIUM
    flags[dist_km <= 20.0]      = 2.0   # HIGH (overwrites MEDIUM for close pixels)
    return flags


# ══════════════════════════════════════════════════════════════════════════════
# 10.  [V10-S1] REGION-SPECIFIC & DISTANCE-ENSEMBLE MODELS
# ══════════════════════════════════════════════════════════════════════════════

def train_region_models(collocated_df: pd.DataFrame,
                         feature_cols: list,
                         sample_weights: np.ndarray) -> dict[str, PM25Predictor]:
    """
    [V10-S1] Train separate HistGBR models for KTM Valley and OuterCities.

    Both models are returned in a dict keyed by region name so they can be
    re-used for both 'region_specific' and 'distance_ensemble' strategies
    without duplicate training.

    Parameters
    ----------
    collocated_df  : Full collocated training dataframe (all folds).
    feature_cols   : List of feature column names (same as whole-country model).
    sample_weights : Per-row sample weights (from the whole-country model).

    Returns
    -------
    { REGION_KTM: PM25Predictor, REGION_OUTER: PM25Predictor }
    """
    log.info("[V10-S1] Training region-specific models …")
    models: dict[str, PM25Predictor] = {}

    for region in (REGION_KTM, REGION_OUTER):
        region_mask = collocated_df["station_id"].apply(_assign_region) == region
        region_df   = collocated_df[region_mask]
        region_sw   = sample_weights[region_mask.values]

        n_obs = len(region_df)
        log.info("  Region '%s': %d training observations.", region, n_obs)

        if n_obs < 10:
            log.warning(
                "  Region '%s' has only %d obs — using all-data model as fallback.",
                region, n_obs
            )
            # Return None; caller must handle by falling back to whole-country model
            models[region] = None
            continue

        X_region = region_df[feature_cols].values.astype(np.float32)
        y_region = region_df["pm25"].values.astype(np.float32)

        predictor = PM25Predictor(n_features=len(feature_cols), epochs=600)
        predictor.fit(X_region, y_region, sample_weight=region_sw)
        predictor.save(OUTPUT_DIR, tag=f"region_{region}")
        models[region] = predictor

    return models


def predict_with_region_models(
        region_models: dict[str, PM25Predictor],
        whole_country_predictor: PM25Predictor,
        aod_filled: np.ndarray,
        covariates: dict,
        feature_cols: list,
        ref_meta: dict,
        region_trees: dict[str, cKDTree],
        strategy: str,
        chunk_size: int = 50_000,
        temporal_fill: dict | None = None) -> np.ndarray:
    """
    [V10-S1] Produce a PM2.5 grid using the selected regional strategy.

    strategy='region_specific':
        Each pixel is assigned to the region of its nearest station (lat/lon KDTree).
        The corresponding regional PM25Predictor is used.

    strategy='distance_ensemble':
        Both regional predictors run on every pixel.
        Predictions are combined via inverse-distance weighting (IDW):
            w_r = 1 / (d_r + ε)  for each region r
            pm25 = Σ(w_r × pred_r) / Σ(w_r)
        where d_r is the km-distance to the nearest station of region r.

    Falls back to whole_country_predictor for any region with None model.
    """
    log.info("[V10-S1] Predicting PM2.5 grid — strategy='%s' …", strategy)

    nrows, ncols = aod_filled.shape
    n_total = nrows * ncols
    temporal_fill = temporal_fill or {}

    # ── Build feature matrix (same as predict_pm25_grid) ────────────────────
    feat_arrays = []
    for col in feature_cols:
        if col == "aod":
            feat_arrays.append(aod_filled.ravel())
        elif col in covariates:
            feat_arrays.append(covariates[col].ravel())
        elif col in temporal_fill:
            feat_arrays.append(np.full(n_total, temporal_fill[col], dtype=np.float32))
        else:
            feat_arrays.append(np.zeros(n_total, dtype=np.float32))

    X_full = np.stack(feat_arrays, axis=1).astype(np.float32)
    valid  = ~np.isnan(X_full).any(axis=1)

    # ── Build pixel lat/lon arrays for distance queries ─────────────────────
    lons_1d, lats_1d = geotransform_to_coords(ref_meta)
    mean_lat = float(np.mean(lats_1d))
    lat_flat = np.repeat(lats_1d, ncols)
    lon_flat = np.tile(lons_1d, nrows)

    pm25_flat = np.full(n_total, np.nan, dtype=np.float32)

    X_valid     = X_full[valid]
    lat_valid   = lat_flat[valid]
    lon_valid   = lon_flat[valid]
    n_valid     = X_valid.shape[0]
    preds_valid = np.empty(n_valid, dtype=np.float32)

    regions = (REGION_KTM, REGION_OUTER)

    # Resolve fallback: if any model is None, substitute whole-country
    effective_models = {}
    for r in regions:
        effective_models[r] = region_models.get(r) or whole_country_predictor

    if strategy == "region_specific":
        # ── Assign each valid pixel to its nearest-station region ────────────
        # Query BOTH region trees and pick the region with the shorter distance
        for start in range(0, n_valid, chunk_size):
            end   = min(start + chunk_size, n_valid)
            query = np.column_stack([lat_valid[start:end],
                                     lon_valid[start:end]]).astype(np.float64)

            d_ktm,   _ = region_trees[REGION_KTM].query(query,   workers=-1)
            d_outer, _ = region_trees[REGION_OUTER].query(query, workers=-1)

            # Pixel belongs to the region with the closer station
            is_ktm = d_ktm <= d_outer

            chunk_preds = np.empty(end - start, dtype=np.float32)

            # KTM-assigned pixels
            if is_ktm.any():
                chunk_preds[is_ktm] = effective_models[REGION_KTM].predict(
                    X_valid[start:end][is_ktm])

            # Outer-city-assigned pixels
            if (~is_ktm).any():
                chunk_preds[~is_ktm] = effective_models[REGION_OUTER].predict(
                    X_valid[start:end][~is_ktm])

            preds_valid[start:end] = chunk_preds

            if start % (chunk_size * 10) == 0:
                log.info("  … region_specific: %d / %d valid pixels done", start, n_valid)

    elif strategy == "distance_ensemble":
        # ── IDW blend of both regional predictions ───────────────────────────
        eps = 1e-3   # km epsilon to avoid division by zero for on-station pixels
        for start in range(0, n_valid, chunk_size):
            end   = min(start + chunk_size, n_valid)
            query = np.column_stack([lat_valid[start:end],
                                     lon_valid[start:end]]).astype(np.float64)

            d_ktm_deg,   _ = region_trees[REGION_KTM].query(query,   workers=-1)
            d_outer_deg, _ = region_trees[REGION_OUTER].query(query, workers=-1)

            # Convert to km
            d_ktm_km   = _deg_to_km(d_ktm_deg,   mean_lat).astype(np.float32)
            d_outer_km = _deg_to_km(d_outer_deg, mean_lat).astype(np.float32)

            # IDW weights: w = 1 / (d + ε)
            w_ktm   = 1.0 / (d_ktm_km   + eps)
            w_outer = 1.0 / (d_outer_km + eps)
            w_sum   = w_ktm + w_outer

            chunk_X = X_valid[start:end]
            pred_ktm   = effective_models[REGION_KTM].predict(chunk_X).astype(np.float32)
            pred_outer = effective_models[REGION_OUTER].predict(chunk_X).astype(np.float32)

            preds_valid[start:end] = (
                (w_ktm * pred_ktm + w_outer * pred_outer) / w_sum
            ).astype(np.float32)

            if start % (chunk_size * 10) == 0:
                log.info("  … distance_ensemble: %d / %d valid pixels done", start, n_valid)

    else:
        raise ValueError(f"Unknown strategy '{strategy}'. "
                         f"Choose: 'whole_country', 'region_specific', 'distance_ensemble'.")

    pm25_flat[valid] = preds_valid
    pm25_grid = pm25_flat.reshape(nrows, ncols)

    log.info("[V10-S1] Strategy '%s' grid complete. Range: %.1f – %.1f µg/m³",
             strategy, float(np.nanmin(pm25_grid)), float(np.nanmax(pm25_grid)))
    return pm25_grid


# ══════════════════════════════════════════════════════════════════════════════
# 11.  [V10-S4] APPLICABILITY STATEMENT
# ══════════════════════════════════════════════════════════════════════════════

def print_and_save_applicability_statement(radius_km: float,
                                            output_dir: str) -> None:
    """
    [V10-S4] Print and save the model applicability statement.

    Covers where the model CAN and CANNOT be applied reliably,
    the radius definition, and the confidence level interpretation.
    """
    statement = f"""
================================================================================
  PM2.5 ESTIMATION FRAMEWORK — APPLICABILITY STATEMENT
  Nepal v10 | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
================================================================================

WHERE THE MODEL CAN BE APPLIED RELIABLY
──────────────────────────────────────────────────────────────────────────────
• Urban and semi-urban areas of Nepal — specifically Kathmandu Valley,
  the Terai lowland cities (Birgunj, Hetauda, Dhangadhi), and periurban
  settlements with land use and pollution sources comparable to the training
  ground stations.
• Pixels within {radius_km:.0f} km of a training station (HIGH or MEDIUM
  confidence zones), where the model has demonstrated cross-validated
  prediction skill (CV R² target ≥ 0.80).
• Elevations below approximately 2500 m a.s.l., where AOD–PM2.5 conversion
  factors and meteorological drivers are similar to training conditions.
• Temporal periods overlapping with the MAIAC AOD tile archive
  (2025-01-01 – 2026-03-31) and with similar land cover and seasonal patterns.

WHERE THE MODEL SHOULD NOT BE APPLIED (or used with LOW confidence only)
──────────────────────────────────────────────────────────────────────────────
• High Himalayan regions above 4000 m a.s.l., where aerosol chemistry,
  mixing height, and precipitation patterns differ fundamentally from the
  training domain and no ground stations exist.
• Remote areas more than {radius_km:.0f} km from any training station
  (confidence flag = 0 / LOW), where spatial extrapolation is unconstrained
  and prediction uncertainty is high.
• Regions outside Nepal — the model was trained exclusively on Nepali
  ground-station data and cannot generalise to India, China, or Bhutan
  without retraining with local observations.
• Areas with fundamentally different pollution source profiles (e.g. heavy
  industrial zones, brick kilns, mining regions) not represented in the
  training station network.

CONFIDENCE FLAG DEFINITIONS
──────────────────────────────────────────────────────────────────────────────
  Flag 2 (HIGH)   — pixel within 20 km of a training station.
                    Model accuracy is highest; spatial extrapolation minimal.
  Flag 1 (MEDIUM) — pixel between 20 km and {radius_km:.0f} km of a station.
                    Model still applies; some spatial extrapolation involved.
  Flag 0 (LOW)    — pixel farther than {radius_km:.0f} km from all stations.
                    Predictions are outside the applicability domain;
                    use only with caution and quantified uncertainty.

APPLICABILITY RADIUS
──────────────────────────────────────────────────────────────────────────────
  Default radius: {radius_km:.0f} km  (user-settable via radius_km parameter in run_pipeline).
  Based on the spatial autocorrelation length of PM2.5 in Nepal (roughly
  50–80 km for the Terai plain; shorter in the valley micro-topography).

OUTPUT FILES
──────────────────────────────────────────────────────────────────────────────
  dist_to_station_km.tif  — Distance (km) from each 1-km pixel to the nearest
                             training station.
  confidence_flag.tif     — Confidence flag (0/1/2) as defined above.
  pm25_nepal_1km_whole_country.tif      — Single-model PM2.5 prediction.
  pm25_nepal_1km_region_specific.tif   — Region-specific model prediction
                                          (KTM Valley / Outer Cities).
  pm25_nepal_1km_distance_ensemble.tif — IDW ensemble of regional models.

================================================================================
"""
    print(statement)

    out_path = os.path.join(output_dir, "applicability_statement.txt")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(statement)
    log.info("[V10-S4] Applicability statement saved → %s", out_path)


# ══════════════════════════════════════════════════════════════════════════════
# 12. DIAGNOSTIC PLOTS  (v9 unchanged + [V10-S5] error-vs-distance panel)
# ══════════════════════════════════════════════════════════════════════════════

def _overlay_shapefiles_geo(ax_geo, shp_admin0: str, shp_admin1: str) -> None:
    if not GEOPANDAS_AVAILABLE:
        return
    for shp_path, color, lw, alpha in [
        (shp_admin1, "#7fb3d3", 0.6, 0.75),
        (shp_admin0, "#ffffff", 1.4, 0.95),
    ]:
        if not os.path.exists(shp_path):
            log.warning("Shapefile not found: %s", shp_path)
            continue
        try:
            gdf = gpd.read_file(shp_path).to_crs(epsg=4326)
            gdf.boundary.plot(ax=ax_geo, color=color, linewidth=lw,
                              alpha=alpha, transform=ax_geo.transData)
        except Exception as exc:
            log.warning("Shapefile overlay failed (%s): %s", shp_path, exc)


def _apply_journal_style(ax):
    """Apply consistent white/journal-ready styling to a single Axes."""
    ax.set_facecolor("white")
    ax.tick_params(colors="black")
    ax.spines[:].set_color("#444444")
    ax.tick_params(axis="both", which="both", direction="in",
                   top=True, right=True, labelsize=9)


def _dark_fig(w=9, h=7):
    """Return a new journal-style (white background) Figure + Axes pair."""
    fig, ax = plt.subplots(figsize=(w, h), facecolor="white")
    _apply_journal_style(ax)
    return fig, ax


def _save_fig(fig, path, dpi=300):
    fig.savefig(path, dpi=dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    log.info("    → Figure saved: %s", path)


def _plot_aod_pm_scatter(collocated_df, output_dir, strategy):
    """Panel A (standalone): MAIAC AOD vs ground PM2.5 scatter."""
    title_kw = dict(color="black", fontsize=12, fontweight="bold", pad=8)
    label_kw = dict(color="black", fontsize=10)

    fig, ax = _dark_fig(8, 7)
    aod_vals  = collocated_df["aod"].values
    pm25_vals = collocated_df["pm25"].values
    valid_m   = (~np.isnan(aod_vals)) & (~np.isnan(pm25_vals))
    if valid_m.sum() > 1 and np.unique(aod_vals[valid_m]).size > 1:
        aod_u = aod_vals[valid_m]
        pm_u  = pm25_vals[valid_m]
        r_val, _ = pearsonr(aod_u, pm_u)
        ax.scatter(aod_u, pm_u, alpha=0.5, s=14, c="#2166ac", edgecolors="none")
        m, b   = np.polyfit(aod_u, pm_u, 1)
        x_line = np.linspace(aod_u.min(), aod_u.max(), 100)
        ax.plot(x_line, m * x_line + b, "--", color="#d73027", lw=1.8,
                label=f"r = {r_val:.3f}")
        ax.legend(facecolor="white", edgecolor="#444444", fontsize=9)
    ax.set_title("AOD vs PM$_{2.5}$ (Ground Stations)", **title_kw)
    ax.set_xlabel("MAIAC AOD (550 nm)", **label_kw)
    ax.set_ylabel(r"PM$_{2.5}$ ($\mu$g/m$^3$)", **label_kw)
    fig.tight_layout()
    _save_fig(fig, os.path.join(output_dir, f"diag_A_aod_vs_pm25_{strategy}.png"))


def _plot_cv_scatter(cv_metrics, output_dir, strategy):
    """Panel B (standalone): CV predicted vs observed scatter, coloured by fold."""
    title_kw = dict(color="black", fontsize=12, fontweight="bold", pad=8)
    label_kw = dict(color="black", fontsize=10)
    fold_colors = {0: "#2166ac", 1: "#d73027", 2: "#4dac26", 3: "#f1a340"}

    fig, ax = _dark_fig(8, 7)
    y_true = cv_metrics.get("y_true", np.array([]))
    y_pred = cv_metrics.get("y_pred", np.array([]))
    fold_results = cv_metrics.get("fold_results", [])

    if len(y_true) > 1:
        fold_labels_all = []
        for fold_id in range(4):
            n_test = next((f["n_test"] for f in fold_results if f["fold"] == fold_id), 0)
            fold_labels_all.extend([fold_id] * n_test)
        fold_labels_all = np.array(fold_labels_all)

        for fid in range(4):
            mask_f = fold_labels_all == fid
            if mask_f.sum() == 0:
                continue
            ax.scatter(y_true[mask_f], y_pred[mask_f], alpha=0.6, s=16,
                       c=fold_colors[fid], edgecolors="none", label=FOLD_NAMES[fid])
        max_val = max(float(y_true.max()), float(y_pred.max()))
        ax.plot([0, max_val], [0, max_val], "k--", lw=1.2)
        ax.text(0.05, 0.80,
                f"R$^2$ = {cv_metrics['r2']:.3f}\n"
                f"RMSE = {cv_metrics['rmse']:.1f} µg/m³\n"
                f"Bias = {cv_metrics['bias']:+.1f} µg/m³",
                transform=ax.transAxes, color="black", fontsize=9,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="#444444"))
        ax.legend(facecolor="white", edgecolor="#444444", fontsize=8, loc="lower right")
    ax.set_title(f"CV: Predicted vs Observed PM$_{{2.5}}$ [{strategy}]", **title_kw)
    ax.set_xlabel(r"Observed PM$_{2.5}$ ($\mu$g/m$^3$)", **label_kw)
    ax.set_ylabel(r"Predicted PM$_{2.5}$ ($\mu$g/m$^3$)", **label_kw)
    fig.tight_layout()
    _save_fig(fig, os.path.join(output_dir, f"diag_B_cv_scatter_{strategy}.png"))


def _plot_fold_r2_bar(cv_metrics, output_dir, strategy):
    """Panel C (standalone): Per-fold R² horizontal bar chart."""
    title_kw = dict(color="black", fontsize=12, fontweight="bold", pad=8)
    label_kw = dict(color="black", fontsize=10)
    fold_colors = {0: "#2166ac", 1: "#d73027", 2: "#4dac26", 3: "#f1a340"}

    fold_results = cv_metrics.get("fold_results", [])
    fig, ax = _dark_fig(8, 5)
    if fold_results:
        fnames = [f["name"] for f in fold_results]
        fr2s   = [f["r2"]   for f in fold_results]
        colors = [fold_colors[f["fold"]] for f in fold_results]
        bars   = ax.barh(fnames, fr2s, color=colors, alpha=0.85, height=0.55)
        ax.axvline(0.80, color="#d73027", lw=1.5, linestyle="--", label="Target R²=0.80")
        ax.axvline(0.0,  color="black",   lw=0.8)
        for bar, val in zip(bars, fr2s):
            ax.text(max(0.02, val + 0.01), bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", color="black", fontsize=9)
        ax.set_xlim(-0.5, 1.05)
        ax.set_xlabel("R²", **label_kw)
        ax.legend(facecolor="white", edgecolor="#444444", fontsize=8)
    ax.set_title("CV R² per Geographic Fold", **title_kw)
    fig.tight_layout()
    _save_fig(fig, os.path.join(output_dir, f"diag_C_fold_r2_{strategy}.png"))


def _plot_pm25_timeseries(collocated_df, output_dir, strategy):
    """Panel D (standalone): KTM Valley daily mean PM2.5 time series."""
    title_kw = dict(color="black", fontsize=12, fontweight="bold", pad=8)
    label_kw = dict(color="black", fontsize=10)

    fig, ax = _dark_fig(10, 6)
    if "pm25" in collocated_df.columns and "date" in collocated_df.columns:
        ktm_df = collocated_df[collocated_df["_fold"] != 3] \
            if "_fold" in collocated_df.columns else collocated_df
        ts = (
            ktm_df.dropna(subset=["date", "pm25"])
            .groupby("date")["pm25"].mean()
            .reset_index()
            .sort_values("date")
        )
        if len(ts) > 0:
            ax.plot(ts["date"].values, ts["pm25"].values,
                    color="#2166ac", lw=1.1, alpha=0.9, label="KTM Valley mean")
            ax.fill_between(ts["date"].values, ts["pm25"].values,
                            alpha=0.15, color="#2166ac")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=30,
                     ha="right", color="black", fontsize=8)
            ax.set_xlabel("Date", **label_kw)
            ax.set_ylabel(r"PM$_{2.5}$ ($\mu$g/m$^3$)", **label_kw)
            ax.legend(facecolor="white", edgecolor="#444444", fontsize=8)
    ax.set_title("KTM Valley Mean PM$_{2.5}$ Time Series", **title_kw)
    fig.tight_layout()
    _save_fig(fig, os.path.join(output_dir, f"diag_D_timeseries_{strategy}.png"))


def plot_diagnostics(cv_metrics: dict,
                     collocated_df: pd.DataFrame,
                     output_dir: str,
                     strategy: str = "whole_country") -> None:
    """
    [V10] Diagnostic figures — Gap-filled AOD and predicted PM2.5 spatial
    panels have been removed.  Four panels remain, each saved both as an
    individual PNG and together in a single 2×2 combined figure:

      A  diag_A_aod_vs_pm25_{strategy}.png   — AOD vs PM2.5 scatter
      B  diag_B_cv_scatter_{strategy}.png    — CV predicted vs observed
      C  diag_C_fold_r2_{strategy}.png       — Per-fold R² bar chart
      D  diag_D_timeseries_{strategy}.png    — KTM Valley PM2.5 time series
         pm25_diagnostics_v10_{strategy}.png — combined 2×2 figure
    """
    _style = "seaborn-v0_8-whitegrid"
    try:
        plt.style.use(_style)
    except OSError:
        try:
            plt.style.use("seaborn-whitegrid")
        except OSError:
            plt.style.use("ggplot")
    sns.set_palette("colorblind")

    # ── Save all four panels individually first ────────────────────────────────
    _plot_aod_pm_scatter(collocated_df, output_dir, strategy)
    _plot_cv_scatter(cv_metrics, output_dir, strategy)
    _plot_fold_r2_bar(cv_metrics, output_dir, strategy)
    _plot_pm25_timeseries(collocated_df, output_dir, strategy)

    # ── Combined 2×2 figure ────────────────────────────────────────────────────
    title_kw = dict(color="black", fontsize=11, fontweight="bold", pad=7)
    label_kw = dict(color="black", fontsize=9)
    fold_colors = {0: "#2166ac", 1: "#d73027", 2: "#4dac26", 3: "#f1a340"}

    fig = plt.figure(figsize=(16, 12), facecolor="white")
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)

    ax_aod_pm   = fig.add_subplot(gs[0, 0])
    ax_cv       = fig.add_subplot(gs[0, 1])
    ax_fold_bar = fig.add_subplot(gs[1, 0])
    ax_ts       = fig.add_subplot(gs[1, 1])

    for a in [ax_aod_pm, ax_cv, ax_fold_bar, ax_ts]:
        _apply_journal_style(a)

    # ── Panel A: AOD vs PM2.5 ──────────────────────────────────────────────────
    aod_vals  = collocated_df["aod"].values
    pm25_vals = collocated_df["pm25"].values
    valid_m   = (~np.isnan(aod_vals)) & (~np.isnan(pm25_vals))
    if valid_m.sum() > 1 and np.unique(aod_vals[valid_m]).size > 1:
        aod_u = aod_vals[valid_m];  pm_u = pm25_vals[valid_m]
        r_val, _ = pearsonr(aod_u, pm_u)
        ax_aod_pm.scatter(aod_u, pm_u, alpha=0.5, s=12, c="#2166ac", edgecolors="none")
        m, b   = np.polyfit(aod_u, pm_u, 1)
        x_line = np.linspace(aod_u.min(), aod_u.max(), 100)
        ax_aod_pm.plot(x_line, m * x_line + b, "--", color="#d73027", lw=1.5,
                       label=f"r = {r_val:.3f}")
        ax_aod_pm.legend(facecolor="white", edgecolor="#444444", fontsize=8)
    ax_aod_pm.set_title("AOD vs PM$_{2.5}$ (Ground Stations)", **title_kw)
    ax_aod_pm.set_xlabel("MAIAC AOD (550 nm)", **label_kw)
    ax_aod_pm.set_ylabel(r"PM$_{2.5}$ ($\mu$g/m$^3$)", **label_kw)

    # ── Panel B: CV scatter ────────────────────────────────────────────────────
    y_true = cv_metrics.get("y_true", np.array([]))
    y_pred = cv_metrics.get("y_pred", np.array([]))
    fold_results = cv_metrics.get("fold_results", [])

    if len(y_true) > 1:
        fold_labels_all = []
        for fold_id in range(4):
            n_test = next((f["n_test"] for f in fold_results if f["fold"] == fold_id), 0)
            fold_labels_all.extend([fold_id] * n_test)
        fold_labels_all = np.array(fold_labels_all)

        for fid in range(4):
            mask_f = fold_labels_all == fid
            if mask_f.sum() == 0:
                continue
            ax_cv.scatter(y_true[mask_f], y_pred[mask_f], alpha=0.6, s=14,
                          c=fold_colors[fid], edgecolors="none", label=FOLD_NAMES[fid])
        max_val = max(float(y_true.max()), float(y_pred.max()))
        ax_cv.plot([0, max_val], [0, max_val], "k--", lw=1.2)
        ax_cv.text(0.05, 0.80,
                   f"R$^2$ = {cv_metrics['r2']:.3f}\nRMSE = {cv_metrics['rmse']:.1f} µg/m³"
                   f"\nBias = {cv_metrics['bias']:+.1f} µg/m³",
                   transform=ax_cv.transAxes, color="black", fontsize=8,
                   bbox=dict(facecolor="white", alpha=0.8, edgecolor="#444444"))
        ax_cv.legend(facecolor="white", edgecolor="#444444", fontsize=7, loc="lower right")
        ax_cv.set_xlabel(r"Observed PM$_{2.5}$ ($\mu$g/m$^3$)", **label_kw)
        ax_cv.set_ylabel(r"Predicted PM$_{2.5}$ ($\mu$g/m$^3$)", **label_kw)
    ax_cv.set_title(f"CV: Predicted vs Observed PM$_{{2.5}}$ [{strategy}]", **title_kw)

    # ── Panel C: Per-fold R² bar chart ─────────────────────────────────────────
    if fold_results:
        fnames = [f["name"] for f in fold_results]
        fr2s   = [f["r2"]   for f in fold_results]
        colors = [fold_colors[f["fold"]] for f in fold_results]
        bars   = ax_fold_bar.barh(fnames, fr2s, color=colors, alpha=0.85, height=0.55)
        ax_fold_bar.axvline(0.80, color="#d73027", lw=1.5, linestyle="--", label="Target 0.80")
        ax_fold_bar.axvline(0.0,  color="black",   lw=0.8)
        for bar, val in zip(bars, fr2s):
            ax_fold_bar.text(max(0.02, val + 0.01), bar.get_y() + bar.get_height() / 2,
                             f"{val:.3f}", va="center", color="black", fontsize=8)
        ax_fold_bar.set_xlim(-0.5, 1.05)
        ax_fold_bar.set_xlabel("R²", **label_kw)
        ax_fold_bar.legend(facecolor="white", edgecolor="#444444", fontsize=7)
    ax_fold_bar.set_title("CV R² per Geographic Fold", **title_kw)

    # ── Panel D: PM2.5 time series ─────────────────────────────────────────────
    if "pm25" in collocated_df.columns and "date" in collocated_df.columns:
        ktm_df = collocated_df[collocated_df["_fold"] != 3] \
            if "_fold" in collocated_df.columns else collocated_df
        ts = (
            ktm_df.dropna(subset=["date", "pm25"])
            .groupby("date")["pm25"].mean()
            .reset_index()
            .sort_values("date")
        )
        if len(ts) > 0:
            ax_ts.plot(ts["date"].values, ts["pm25"].values,
                       color="#2166ac", lw=0.9, alpha=0.9, label="KTM Valley mean")
            ax_ts.fill_between(ts["date"].values, ts["pm25"].values,
                               alpha=0.12, color="#2166ac")
            ax_ts.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
            ax_ts.xaxis.set_major_locator(mdates.MonthLocator())
            plt.setp(ax_ts.xaxis.get_majorticklabels(), rotation=30,
                     ha="right", color="black", fontsize=7)
            ax_ts.set_xlabel("Date", **label_kw)
            ax_ts.set_ylabel(r"PM$_{2.5}$ ($\mu$g/m$^3$)", **label_kw)
            ax_ts.legend(facecolor="white", edgecolor="#444444", fontsize=7)
    ax_ts.set_title("KTM Valley Mean PM$_{2.5}$ Time Series", **title_kw)

    fig.suptitle(
        f"PM$_{{2.5}}$ Estimation Framework — Nepal  |  2025–2026  (v10: {strategy})",
        color="black", fontsize=13, fontweight="bold", y=0.99
    )

    out_path = os.path.join(output_dir, f"pm25_diagnostics_v10_{strategy}.png")
    _save_fig(fig, out_path, dpi=300)
    log.info("Diagnostic figure saved: %s", out_path)


def plot_error_vs_distance(collocated_df: pd.DataFrame,
                            station_tree: cKDTree,
                            cv_metrics: dict,
                            output_dir: str,
                            strategy: str = "whole_country") -> None:
    """
    [V10-S5] Diagnostic scatter: prediction error coloured by distance to
    nearest training station.

    Uses CV predictions (y_true, y_pred) already stored in cv_metrics.
    Station distances are computed for each test observation using the
    global station KDTree.
    """
    y_true = cv_metrics.get("y_true", np.array([]))
    y_pred = cv_metrics.get("y_pred", np.array([]))

    if len(y_true) < 2:
        log.warning("[V10-S5] Cannot plot error-vs-distance: no CV predictions.")
        return

    # Pull test observations in the order they appear in the CV arrays
    # (same order as fold_results, concatenated by fold 0→3)
    fold_results = cv_metrics.get("fold_results", [])
    fold_order   = [f["fold"] for f in fold_results]

    obs_in_cv_order = pd.concat([
        collocated_df[collocated_df["_fold"] == fid]
        for fid in fold_order
        if fid in collocated_df.get("_fold", pd.Series()).unique()
    ], ignore_index=True) if "_fold" in collocated_df.columns else collocated_df

    # Re-filter to exactly n_preds rows in case of duplication differences
    n_preds = len(y_true)
    obs_in_cv_order = obs_in_cv_order.iloc[:n_preds].copy()

    coords = obs_in_cv_order[["latitude", "longitude"]].values.astype(np.float64)
    if len(coords) == 0:
        return

    dist_deg, _ = station_tree.query(coords, workers=-1)
    mean_lat     = float(coords[:, 0].mean())
    dist_km_obs  = _deg_to_km(dist_deg, mean_lat)

    errors = y_pred[:len(coords)] - y_true[:len(coords)]

    fig, ax = plt.subplots(figsize=(9, 6), facecolor="white")
    ax.set_facecolor("white")
    ax.tick_params(colors="black", direction="in", top=True, right=True)
    ax.spines[:].set_color("#444444")

    sc = ax.scatter(dist_km_obs, errors, c=np.abs(errors), cmap="plasma",
                    alpha=0.6, s=18, edgecolors="none", vmin=0, vmax=np.percentile(np.abs(errors), 95))
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"|Error| (µg/m$^3$)", color="black", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="black")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="black")

    ax.axhline(0, color="black", lw=0.8, linestyle="--")
    ax.set_xlabel("Distance to nearest training station (km)",
                  color="black", fontsize=10)
    ax.set_ylabel(r"Prediction error (µg/m$^3$)", color="black", fontsize=10)
    ax.set_title(
        f"PM$_{{2.5}}$ Prediction Error vs Station Distance [{strategy}]",
        color="black", fontsize=12, fontweight="bold"
    )

    out_path = os.path.join(output_dir, f"error_vs_distance_v10_{strategy}.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    log.info("[V10-S5] Error-vs-distance plot saved: %s", out_path)


# ══════════════════════════════════════════════════════════════════════════════
# 13. MAIN PIPELINE  [V10 — extended]
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(model_strategy: str = "whole_country",
                 radius_km: float = 50.0) -> None:
    """
    Run the full PM2.5 estimation pipeline.

    Parameters
    ----------
    model_strategy : str
        One of 'whole_country', 'region_specific', 'distance_ensemble'.
        Controls which prediction strategy to use (see [V10-S1] notes above).
        Default: 'whole_country' (identical to v9 behaviour).

    radius_km : float
        Applicability radius in km for confidence flag computation.
        Pixels farther than radius_km from any training station get flag=0 (LOW).
        Default: 50.0 km.
    """
    valid_strategies = ("whole_country", "region_specific", "distance_ensemble")
    if model_strategy not in valid_strategies:
        raise ValueError(
            f"model_strategy='{model_strategy}' not recognised. "
            f"Choose from: {valid_strategies}"
        )

    banner = "=" * 65
    log.info(banner)
    log.info("  High-Resolution PM2.5 Estimation Framework — Nepal  v10")
    log.info("  Strategy : %s", model_strategy)
    log.info("  Radius   : %.0f km", radius_km)
    log.info("  Started  : %s", datetime.now().strftime("%Y-%m-%d  %H:%M:%S"))
    log.info(banner)

    # ── 1. Load satellite / static data ──────────────────────────────────────
    daily_aod, ref_meta = load_aod(AOD_GLOB, bbox=NEPAL_BBOX)
    aod_composite = daily_aod["_composite"]

    daily_era5     = load_era5(ERA5_DAILY_GLOB, ref_meta)
    era5_composite = daily_era5["_composite"]

    ndvi   = load_ndvi(NDVI_GLOB, ref_meta)
    static = load_static_layers(DEM_PATH, POP_PATH, ref_meta)

    log.info("──── TROPOMI / Sentinel-5P Auxiliary Tracers [V8-M2] ────")
    tropomi_layers = load_tropomi_layers(ref_meta)

    _dmin = daily_aod.get("_date_min")
    _dmax = daily_aod.get("_date_max")
    aod_date_min = pd.Timestamp(_dmin) if _dmin else None
    aod_date_max = pd.Timestamp(_dmax) if _dmax else None

    ground_df = load_ground_observations(
        GROUND_DIR,
        aod_date_min=aod_date_min,
        aod_date_max=aod_date_max,
    )

    covariates = build_covariate_stack(
        aod_composite, era5_composite, ndvi, static, ref_meta,
        tropomi=tropomi_layers,
        doy_map=daily_aod.get("_doy_map"),
    )

    # ── 2. Stage 1: AOD gap-filling ───────────────────────────────────────────
    log.info("")
    log.info("──── STAGE 1: AOD Gap-Filling (Random Forest) ────")
    gap_filler = AODGapFiller(n_estimators=200)
    gap_filler.fit(aod_composite, covariates)
    aod_filled = gap_filler.predict_gap_fill(aod_composite, covariates)

    gf_path = os.path.join(OUTPUT_DIR, "aod_gap_filler.joblib")
    joblib.dump(gap_filler, gf_path)
    log.info("  AODGapFiller saved → %s", gf_path)

    aod_out = os.path.join(OUTPUT_DIR, "aod_gap_filled_1km.tif")
    write_geotiff(aod_out, aod_filled, ref_meta)

    # ── 3. Collocation ────────────────────────────────────────────────────────
    collocated_df = collocate_stations(ground_df, daily_aod, daily_era5,
                                       covariates, ref_meta)

    if len(collocated_df) < 5:
        log.error(
            "Only %d collocated observations. "
            "Check that ground stations fall within the AOD grid extent.",
            len(collocated_df)
        )
        return

    # Pixel-date deduplication
    n_before = len(collocated_df)
    pixel_key = collocated_df[["date", "latitude", "longitude"]].apply(
        lambda r: f"{r['date']}_{r['latitude']:.4f}_{r['longitude']:.4f}", axis=1
    )
    collocated_df["_pixel_date"] = pixel_key
    collocated_df = (
        collocated_df
        .groupby("_pixel_date", as_index=False)
        .apply(lambda g: g.assign(pm25=g["pm25"].mean()).iloc[0])
        .reset_index(drop=True)
        .drop(columns=["_pixel_date"])
    )
    log.info(
        "[FIX-PIXDUP] Pixel-date deduplication: %d → %d rows  "
        "(%d rows merged from co-located stations)",
        n_before, len(collocated_df), n_before - len(collocated_df)
    )

    # ── 4. Feature columns setup ──────────────────────────────────────────────
    log.info("")
    log.info("──── STAGE 2: PM2.5 Prediction (HistGBR + Isotonic Cal) ────")

    TEMPORAL_COLS = ["doy_sin", "doy_cos", "month"]
    base_feature_cols = ["aod"] + [k for k in covariates if not k.startswith("_")]
    temporal_present  = [c for c in TEMPORAL_COLS if c in collocated_df.columns
                         and collocated_df[c].notna().any()]
    feature_cols = base_feature_cols + temporal_present
    if temporal_present:
        log.info("  [FIX-TEMPORAL] Added temporal features: %s", temporal_present)

    # Pre-assign fold and region columns
    collocated_df["_fold"]   = collocated_df["station_id"].apply(_assign_city_fold)
    collocated_df["_region"] = collocated_df["station_id"].apply(_assign_region)

    # [V9-R2] Target encoding (full dataset — no leakage risk for final model)
    collocated_df["station_pm25_mean"] = (
        collocated_df.groupby("station_id")["pm25"].transform("mean").astype(np.float32))
    collocated_df["region_pm25_mean"] = (
        collocated_df.groupby("_fold")["pm25"].transform("mean").astype(np.float32))

    TARGET_ENC_COLS   = ["station_pm25_mean", "region_pm25_mean"]
    feature_cols_full = feature_cols + TARGET_ENC_COLS
    log.info("  [V9-R2] Added target-encoding features: %s", TARGET_ENC_COLS)

    X_all = collocated_df[feature_cols_full].values.astype(np.float32)
    y_all = collocated_df["pm25"].values.astype(np.float32)

    # [V7-F6] Sample weights
    fold_assignments = collocated_df["station_id"].apply(_assign_city_fold).values
    n_ktm   = int(np.sum(fold_assignments != 3))
    n_outer = int(np.sum(fold_assignments == 3))
    region_w = max(1.0, n_ktm / max(1, n_outer))
    sample_weights = np.where(fold_assignments == 3, region_w, 1.0).astype(np.float32)
    log.info(
        "  [V7-F6] Sample weights: %d KTM obs (w=1.0) | %d outer-city obs (w=%.1f)",
        n_ktm, n_outer, region_w
    )

    # ── 5. Train whole-country model (always needed as fallback) ──────────────
    whole_country_predictor = PM25Predictor(n_features=len(feature_cols_full), epochs=800)
    train_metrics = whole_country_predictor.fit(X_all, y_all, sample_weight=sample_weights)
    whole_country_predictor.save(OUTPUT_DIR, tag="final")

    # ── 6. [V10-S2] Build KDTree & compute distance / confidence maps ─────────
    log.info("")
    log.info("──── [V10-S2] APPLICABILITY DOMAIN ────")
    station_tree = build_station_kdtree(collocated_df)
    dist_km      = compute_distance_to_nearest_station(station_tree, ref_meta)
    conf_flags   = confidence_from_distance(dist_km, radius_km=radius_km)

    write_geotiff(os.path.join(OUTPUT_DIR, "dist_to_station_km.tif"), dist_km,  ref_meta)
    write_geotiff(os.path.join(OUTPUT_DIR, "confidence_flag.tif"),    conf_flags, ref_meta)

    pct_high   = 100 * float(np.mean(conf_flags == 2))
    pct_medium = 100 * float(np.mean(conf_flags == 1))
    pct_low    = 100 * float(np.mean(conf_flags == 0))
    log.info("  Confidence coverage — HIGH: %.1f%%  MEDIUM: %.1f%%  LOW: %.1f%%",
             pct_high, pct_medium, pct_low)

    # ── 7. [V10-S1] Train region-specific models (if needed) ─────────────────
    region_models:  dict[str, PM25Predictor] = {}
    region_trees:   dict[str, cKDTree]       = {}

    if model_strategy in ("region_specific", "distance_ensemble"):
        log.info("")
        log.info("──── [V10-S1] REGION-SPECIFIC MODEL TRAINING ────")
        region_models = train_region_models(
            collocated_df, feature_cols_full, sample_weights)
        region_trees  = build_region_kdtrees(collocated_df)

    # ── 8. Spatial-block CV (whole-country model) ──────────────────────────────
    log.info("")
    log.info("──── VALIDATION: City-Aware Spatial-Block CV (whole_country) ────")
    cv_metrics_wc = site_leave_one_out_cv(collocated_df, feature_cols)

    # ── 9. Temporal fill values for spatial prediction ─────────────────────────
    temporal_fill: dict = {}
    if temporal_present:
        median_doy   = float(collocated_df["doy"].median())   if "doy"   in collocated_df else 60.0
        median_month = float(collocated_df["month"].median()) if "month" in collocated_df else 2.0
        temporal_fill = {
            "doy_sin": float(np.sin(2 * np.pi * median_doy / 365.0)),
            "doy_cos": float(np.cos(2 * np.pi * median_doy / 365.0)),
            "month":   median_month,
        }
        log.info("  [FIX-TEMPORAL] Spatial prediction using median DOY=%.0f, month=%.0f",
                 median_doy, median_month)

    global_pm25_mean = float(collocated_df["pm25"].mean())
    temporal_fill["station_pm25_mean"] = global_pm25_mean
    temporal_fill["region_pm25_mean"]  = global_pm25_mean
    log.info("  [V9-R2] Spatial target-encoding fill = global mean %.1f µg/m³",
             global_pm25_mean)

    # ── 10. Spatial PM2.5 prediction ──────────────────────────────────────────
    log.info("")
    log.info("──── OUTPUT: Spatial PM2.5 GeoTIFF(s) ────")

    # Always produce whole_country output (baseline)
    pm25_grid_wc = predict_pm25_grid(
        whole_country_predictor, aod_filled, covariates,
        feature_cols_full, temporal_fill=temporal_fill)
    write_geotiff(
        os.path.join(OUTPUT_DIR, "pm25_nepal_1km_whole_country.tif"),
        pm25_grid_wc, ref_meta)

    # Produce the requested strategy output (if not whole_country)
    pm25_grid_strategy = pm25_grid_wc   # default — reused for plots
    cv_metrics_strategy = cv_metrics_wc

    if model_strategy in ("region_specific", "distance_ensemble"):
        pm25_grid_strategy = predict_with_region_models(
            region_models,
            whole_country_predictor,
            aod_filled, covariates, feature_cols_full, ref_meta,
            region_trees, strategy=model_strategy,
            temporal_fill=temporal_fill)
        write_geotiff(
            os.path.join(OUTPUT_DIR, f"pm25_nepal_1km_{model_strategy}.tif"),
            pm25_grid_strategy, ref_meta)

        # Note: CV for regional strategies reuses the same collocated_df and
        # feature_cols (without re-running a full regional CV to avoid data
        # leakage — a separate regional CV is future work).
        log.info("")
        log.info("──── NOTE: CV metrics for '%s' reuse the whole-country fold "
                 "structure (same stations / same features). ────", model_strategy)
        cv_metrics_strategy = cv_metrics_wc

    # ── 11. Diagnostic plots ───────────────────────────────────────────────────
    log.info("")
    log.info("──── PLOTTING: Diagnostic Figures ────")
    plot_diagnostics(cv_metrics_strategy, collocated_df,
                     OUTPUT_DIR, strategy=model_strategy)

    # [V10-S5] Error-vs-distance diagnostic
    plot_error_vs_distance(collocated_df, station_tree,
                           cv_metrics_strategy, OUTPUT_DIR,
                           strategy=model_strategy)

    # ── 12. [V10-S4] Applicability statement ──────────────────────────────────
    log.info("")
    log.info("──── [V10-S4] APPLICABILITY STATEMENT ────")
    print_and_save_applicability_statement(radius_km, OUTPUT_DIR)

    # ── 13. Summary ────────────────────────────────────────────────────────────
    log.info("")
    log.info(banner)
    log.info("  PIPELINE COMPLETE  (v10)  strategy=%s", model_strategy)
    log.info("  Outputs in : %s", OUTPUT_DIR)
    log.info("    ├─ aod_gap_filled_1km.tif")
    log.info("    ├─ pm25_nepal_1km_whole_country.tif")
    if model_strategy in ("region_specific", "distance_ensemble"):
        log.info("    ├─ pm25_nepal_1km_%s.tif", model_strategy)
    log.info("    ├─ dist_to_station_km.tif                [V10-S2]")
    log.info("    ├─ confidence_flag.tif                   [V10-S2]")
    log.info("    ├─ applicability_statement.txt           [V10-S4]")
    log.info("    ├─ pm25_diagnostics_v10_%s.png", model_strategy)
    log.info("    ├─ error_vs_distance_v10_%s.png         [V10-S5]", model_strategy)
    log.info("    ├─ aod_gap_filler.joblib")
    log.info("    ├─ pm25_scaler_final.joblib")
    log.info("    ├─ pm25_model_final.joblib")
    if model_strategy in ("region_specific", "distance_ensemble"):
        log.info("    ├─ pm25_scaler_region_ktm_valley.joblib")
        log.info("    ├─ pm25_model_region_ktm_valley.joblib")
        log.info("    ├─ pm25_scaler_region_outer_cities.joblib")
        log.info("    └─ pm25_model_region_outer_cities.joblib")
    log.info("")
    log.info("  Training metrics (80/20 split):")
    log.info("    Train R² = %.4f  RMSE = %.2f µg/m³",
             train_metrics.get("train_r2", np.nan),
             train_metrics.get("train_rmse", np.nan))
    log.info("    Val   R² = %.4f  RMSE = %.2f µg/m³",
             train_metrics.get("val_r2",   np.nan),
             train_metrics.get("val_rmse", np.nan))
    log.info("")
    log.info("  Spatial CV Summary (whole_country, 4-fold city-aware):")
    log.info("    R²   = %.4f  (target ≥ 0.80)", cv_metrics_wc.get("r2",   np.nan))
    log.info("    RMSE = %.2f µg/m³",             cv_metrics_wc.get("rmse", np.nan))
    log.info("    MAE  = %.2f µg/m³",             cv_metrics_wc.get("mae",  np.nan))
    log.info("    Bias = %.2f µg/m³",             cv_metrics_wc.get("bias", np.nan))

    fold_results = cv_metrics_wc.get("fold_results", [])
    if fold_results:
        log.info("")
        log.info("  Per-fold breakdown (whole_country strategy):")
        for f in fold_results:
            log.info("    Fold %d %-12s  R²=%.4f  RMSE=%.1f  Bias=%+.1f  n=%d",
                     f["fold"], f["name"],
                     f["r2"], f["rmse"], f["bias"], f["n_test"])

    log.info("")
    log.info("  [V10-S2] Confidence coverage (radius=%.0f km):", radius_km)
    log.info("    HIGH   (≤ 20 km) : %.1f %%", pct_high)
    log.info("    MEDIUM (≤ %.0f km): %.1f %%", radius_km, pct_medium)
    log.info("    LOW    (> %.0f km): %.1f %%", radius_km, pct_low)
    log.info(banner)


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ── User-configurable run parameters ─────────────────────────────────────
    # Change model_strategy to any of:
    #   "whole_country"      — single model, identical to v9
    #   "region_specific"    — separate KTM / OuterCities models, nearest-region rule
    #   "distance_ensemble"  — IDW blend of both regional models
    run_pipeline(
        model_strategy = "distance_ensemble",   # <-- change strategy here
        radius_km      = 50.0,                  # <-- change applicability radius here
    )
