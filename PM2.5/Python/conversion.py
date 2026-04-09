"""
convert_era5_grib_to_netcdf.py
──────────────────────────────
Converts ERA5 files that are actually GRIB format (but named .nc)
into proper NetCDF files, in-place or to a separate output folder.

Usage:
    python convert_era5_grib_to_netcdf.py

Requirements:
    pip install cfgrib eccodes xarray netCDF4
    (eccodes also needs the binary library — on Windows this comes bundled with
     the pip package, so the pip install above should be sufficient)
"""

import glob
import os
import logging
import shutil
import sys
import time

# ── Configuration ─────────────────────────────────────────────────────────────

# Folder containing your era5_YYYYMMDD.nc files
ERA5_DIR = r"Your_Path"

# Where to write converted files.
# Set to ERA5_DIR to overwrite in-place (originals are backed up first).
# Set to a different path to write alongside originals.
OUTPUT_DIR = ERA5_DIR   # <-- change if you want a separate output folder

# Whether to keep a .grib_orig backup of each original file (recommended!)
KEEP_BACKUP = True

# Variables your pipeline needs — conversion will warn if any are missing
REQUIRED_VARS = {"t2m", "d2m", "blh", "u10", "v10", "sp", "tp"}

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)-7s]  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(ERA5_DIR, "conversion_log.txt"), mode="w"),
    ],
)
log = logging.getLogger(__name__)

# ── Check dependencies ────────────────────────────────────────────────────────

try:
    import xarray as xr
except ImportError:
    sys.exit("ERROR: xarray not installed. Run: pip install xarray")

try:
    import cfgrib  # noqa: F401 — just checking it's importable
except ImportError:
    sys.exit("ERROR: cfgrib not installed. Run: pip install cfgrib eccodes")


# ── Helpers ───────────────────────────────────────────────────────────────────

def is_grib(path: str) -> bool:
    """Check magic bytes — GRIB files start with b'GRIB'."""
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"GRIB"
    except OSError:
        return False


def is_netcdf(path: str) -> bool:
    """Check magic bytes — NetCDF3 starts with CDF, HDF5/NetCDF4 with \\x89HDF."""
    try:
        with open(path, "rb") as f:
            magic = f.read(4)
            return magic[:3] == b"CDF" or magic[:4] == b"\x89HDF"
    except OSError:
        return False


def merge_grib_datasets(path: str) -> "xr.Dataset":
    """
    GRIB files often split variables across multiple 'messages' that cfgrib
    reads as separate datasets. This merges them all into one Dataset.
    """
    import cfgrib
    datasets = cfgrib.open_datasets(path)
    if len(datasets) == 1:
        return datasets[0]

    # Merge — drop conflicting coords that differ between messages (e.g. step)
    merged = xr.merge(datasets, compat="override", join="outer")
    return merged


def convert_file(src: str, dst: str) -> bool:
    """
    Convert a single GRIB file at `src` to NetCDF at `dst`.
    Returns True on success, False on failure.
    """
    fname = os.path.basename(src)
    try:
        ds = merge_grib_datasets(src)

        # Report which variables are present / missing
        present = set(ds.data_vars)
        missing = REQUIRED_VARS - present
        if missing:
            log.warning("  %s — missing vars: %s  (present: %s)",
                        fname, sorted(missing), sorted(present))
        else:
            log.info("  %s — all required vars present: %s", fname, sorted(present))

        # Drop GRIB-specific metadata variables that don't belong in NetCDF
        grib_junk = [v for v in ds.data_vars
                     if v.startswith("unknown") or v in ("surface",)]
        if grib_junk:
            ds = ds.drop_vars(grib_junk, errors="ignore")

        # Write NetCDF4
        ds.to_netcdf(dst, format="NETCDF4")
        ds.close()
        return True

    except Exception as exc:
        log.error("  FAILED %s: %s", fname, exc)
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_files = sorted(glob.glob(os.path.join(ERA5_DIR, "era5_????????.nc")))
    if not all_files:
        log.error("No files matching era5_YYYYMMDD.nc found in: %s", ERA5_DIR)
        sys.exit(1)

    log.info("Found %d files in %s", len(all_files), ERA5_DIR)

    # Categorise
    already_netcdf = [f for f in all_files if is_netcdf(f)]
    need_convert   = [f for f in all_files if is_grib(f)]
    unknown        = [f for f in all_files
                      if not is_netcdf(f) and not is_grib(f)]

    log.info("  Already NetCDF : %d", len(already_netcdf))
    log.info("  Need conversion: %d (GRIB)", len(need_convert))
    log.info("  Unknown format : %d", len(unknown))

    if unknown:
        for u in unknown:
            log.warning("  Unknown format, skipping: %s", os.path.basename(u))

    if not need_convert:
        log.info("Nothing to convert — all files are already valid NetCDF.")
        return

    # ── Convert ───────────────────────────────────────────────────────────────
    ok, failed = 0, 0
    t0 = time.time()

    for i, src in enumerate(need_convert, 1):
        fname = os.path.basename(src)
        elapsed = time.time() - t0
        eta_s = (elapsed / i) * (len(need_convert) - i) if i > 1 else 0
        log.info("[%d/%d | ETA %.0fs]  Converting %s …",
                 i, len(need_convert), eta_s, fname)

        in_place = (os.path.abspath(OUTPUT_DIR) == os.path.abspath(ERA5_DIR))

        if in_place:
            # Write to a temp file first, then swap
            tmp_dst = src + ".tmp_nc"
            success = convert_file(src, tmp_dst)
            if success:
                if KEEP_BACKUP:
                    bak = src + ".grib_orig"
                    shutil.move(src, bak)
                else:
                    os.remove(src)
                shutil.move(tmp_dst, src)
                ok += 1
            else:
                # Clean up temp file if it was created
                if os.path.exists(tmp_dst):
                    os.remove(tmp_dst)
                failed += 1
        else:
            dst = os.path.join(OUTPUT_DIR, fname)
            success = convert_file(src, dst)
            if success:
                ok += 1
            else:
                failed += 1

    total_time = time.time() - t0
    log.info("── Done ──────────────────────────────────────────")
    log.info("  Converted OK : %d", ok)
    log.info("  Failed       : %d", failed)
    log.info("  Total time   : %.1f s (%.1f min)", total_time, total_time / 60)

    if failed:
        log.warning("Some files failed — check conversion_log.txt for details.")
        log.warning("Your pipeline will use zeros for those dates.")
    else:
        log.info("All done! Re-run your PM2.5 pipeline now.")

    if KEEP_BACKUP and in_place:
        log.info("Originals backed up as *.grib_orig — delete once pipeline runs OK.")


if __name__ == "__main__":
    main()
