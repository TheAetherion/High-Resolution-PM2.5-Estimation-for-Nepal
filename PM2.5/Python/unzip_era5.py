"""
unzip_era5.py  (v2 - uses netCDF4 engine via temp files)
---------------------------------------------------------
Each era5_YYYYMMDD.nc is a ZIP containing:
  - data_stream-oper_stepType-instant.nc  (t2m, d2m, u10, v10, sp, blh)
  - data_stream-oper_stepType-accum.nc    (tp)

Extracts both to a temp folder, merges, writes proper NetCDF4 in place.
Safe to re-run if interrupted -- skips files already converted.
"""

import glob
import logging
import os
import shutil
import sys
import tempfile
import time
import zipfile

ERA5_DIR    = r"C:\Users\samue\Documents\Conferences\Space\PM2.5\ERA5_Data"
KEEP_BACKUP = True   # saves .zip_orig backup of each original

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)-7s]  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            os.path.join(ERA5_DIR, "unzip_log.txt"),
            mode="w",
            encoding="utf-8",  # prevents cp1252 errors on Windows
        ),
    ],
)
log = logging.getLogger(__name__)

try:
    import xarray as xr
except ImportError:
    sys.exit("ERROR: xarray not installed.  pip install xarray")


def is_zip(path):
    try:
        with open(path, "rb") as f:
            return f.read(2) == b"PK"
    except OSError:
        return False


def process_file(path: str) -> bool:
    fname   = os.path.basename(path)
    tmp_dir = None
    tmp_out = path + ".tmp_nc"
    try:
        # 1. Extract all inner files to a temp directory on disk
        tmp_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(path, "r") as zf:
            zf.extractall(tmp_dir)
            names = zf.namelist()

        log.info("  inner files: %s", names)

        # 2. Open each with the netcdf4 engine (handles HDF5-based NetCDF4)
        datasets = []
        for name in names:
            inner = os.path.join(tmp_dir, name)
            if os.path.isfile(inner):
                datasets.append(xr.open_dataset(inner, engine="netcdf4"))

        if not datasets:
            log.error("  %s -- no usable files in ZIP", fname)
            return False

        # 3. Merge instant + accum into one dataset
        merged = (xr.merge(datasets, compat="override", join="outer")
                  if len(datasets) > 1 else datasets[0])

        # 4. Write to temp file, then atomically replace the original ZIP
        merged.to_netcdf(tmp_out, format="NETCDF4")
        for ds in datasets:
            ds.close()
        merged.close()

        if KEEP_BACKUP:
            shutil.copy2(path, path + ".zip_orig")

        os.replace(tmp_out, path)
        return True

    except Exception as exc:
        log.error("  FAILED %s: %s", fname, exc)
        return False

    finally:
        if tmp_dir and os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)
        if os.path.exists(tmp_out):
            try:
                os.remove(tmp_out)
            except OSError:
                pass


def main():
    files = sorted(glob.glob(os.path.join(ERA5_DIR, "era5_????????.nc")))
    if not files:
        log.error("No era5_YYYYMMDD.nc files found in: %s", ERA5_DIR)
        sys.exit(1)

    to_process = [f for f in files if is_zip(f)]
    log.info("Total: %d  |  Need unzip: %d  |  Already OK: %d",
             len(files), len(to_process), len(files) - len(to_process))

    if not to_process:
        log.info("Nothing to do -- all files already proper NetCDF.")
        return

    ok, failed = 0, 0
    t0 = time.time()

    for i, path in enumerate(to_process, 1):
        elapsed = time.time() - t0
        eta_s   = (elapsed / i) * (len(to_process) - i) if i > 1 else 0
        log.info("[%d/%d | ETA %ds]  %s",
                 i, len(to_process), int(eta_s), os.path.basename(path))

        if process_file(path):
            ok += 1
        else:
            failed += 1

    total = time.time() - t0
    log.info("Done -- OK: %d  Failed: %d  Time: %.1fs (%.1f min)",
             ok, failed, total, total / 60)

    if not failed:
        log.info("All done! Re-run your PM2.5 pipeline now.")
    else:
        log.warning("%d files failed -- check unzip_log.txt", failed)

    if KEEP_BACKUP:
        log.info("Originals saved as *.zip_orig -- delete once pipeline runs OK.")


if __name__ == "__main__":
    main()
