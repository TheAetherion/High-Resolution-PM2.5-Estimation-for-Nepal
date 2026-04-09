import os
import requests
import logging
import time
from pathlib import Path
from datetime import datetime, timedelta
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- CONFIGURATION ---
# Your Earthdata Login credentials (https://urs.earthdata.nasa.gov)
# The Bearer token is used only for the JSON directory listing.
# The actual HDF file downloads require Basic auth (username + password).
USERNAME = "Your_Username"        # <-- your Earthdata username
PASSWORD = "Your_Password"   # <-- your Earthdata password
TOKEN    = "Your_Token"  # <-- paste fresh token from urs.earthdata.nasa.gov
SAVE_DIR = Path(r"C:\Users\samue\Documents\Conferences\Space\PM2.5\AOD_Data")
TILES = ["h25v05", "h25v06", "h26v06"]

START_DATE = datetime(2025, 1, 1)
END_DATE   = datetime(2026, 3, 31)

# Timeouts: (connect_seconds, read_seconds)
# MODIS HDF files are large — give up to 20 minutes for the read
TIMEOUT = (30, 1200)

MAX_RETRIES = 5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)


class AuthPreservingSession(requests.Session):
    """
    requests strips the Authorization header on cross-domain redirects by default.
    NASA Earthdata redirects from ladsweb.modaps.eosdis.nasa.gov
    to urs.earthdata.nasa.gov and back, so we must keep the header on every hop.
    """
    def rebuild_auth(self, prepared_request, response):
        # Intentionally do nothing — do NOT call super(), which strips the header.
        pass


def get_session(username: str, password: str) -> requests.Session:
    """Session with Basic auth that survives cross-domain redirects."""
    session = AuthPreservingSession()
    session.auth = (username, password)   # Basic auth for all requests
    retry = Retry(
        total=3,
        backoff_factor=2,
        status_forcelist=[500, 502, 503, 504],
        raise_on_status=False,
        allowed_methods=["GET"],
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.max_redirects = 10
    return session


def download_with_resume(session: requests.Session, url: str, dest: Path,
                         max_retries: int = MAX_RETRIES) -> bool:
    """
    Download `url` to `dest`, resuming from where a partial file left off.
    Returns True on success, False on permanent failure.
    """
    for attempt in range(1, max_retries + 1):
        existing = dest.stat().st_size if dest.exists() else 0

        req_headers = {}
        if existing:
            req_headers["Range"] = f"bytes={existing}-"
            log.info(f"  [Resume] {dest.name} — continuing from byte {existing:,}")

        try:
            with session.get(url, headers=req_headers, stream=True,
                             timeout=TIMEOUT) as r:

                # 206 = partial content (resume), 200 = full file
                if r.status_code == 416:
                    # Server says we already have everything
                    log.info(f"  [Complete] Server reports {dest.name} fully downloaded.")
                    return True

                if r.status_code not in (200, 206):
                    log.error(f"  [HTTP {r.status_code}] {dest.name} — skipping.")
                    return False

                # Detect if server returned an HTML error/redirect page
                # instead of the actual HDF binary
                content_type = r.headers.get("Content-Type", "")
                if "text/html" in content_type:
                    body = b""
                    for chunk in r.iter_content(chunk_size=4096):
                        body += chunk
                        if len(body) > 2000:
                            break
                    if attempt == 1:
                        log.error(
                            f"  [HTML body preview]:\n"
                            f"  Final URL : {r.url}\n"
                            f"  Status    : {r.status_code}\n"
                            f"  Headers   : {dict(r.headers)}\n"
                            f"  Body (2KB): {body.decode('utf-8', errors='replace')[:1000]}"
                        )
                    raise IOError(
                        f"Server returned HTML instead of HDF "
                        f"(likely an auth redirect not followed). "
                        f"Content-Type: {content_type}"
                    )

                total_reported = r.headers.get("Content-Length")
                mode = "ab" if r.status_code == 206 else "wb"
                if r.status_code == 200:
                    existing = 0  # server didn't honour Range; restart

                downloaded = 0
                with open(dest, mode) as fh:
                    for chunk in r.iter_content(chunk_size=256 * 1024):
                        if chunk:
                            fh.write(chunk)
                            downloaded += len(chunk)

                final_size = existing + downloaded

                # Sanity check: real HDF files are at minimum several MB
                MIN_VALID_SIZE = 1 * 1024 * 1024  # 1 MB
                if final_size < MIN_VALID_SIZE:
                    # Read what we saved to help diagnose
                    with open(dest, "rb") as fh:
                        preview = fh.read(500)
                    dest.unlink()  # delete the bogus file
                    raise IOError(
                        f"File too small ({final_size:,} bytes) — likely an "
                        f"error page. Preview: {preview[:200]}"
                    )

                if total_reported:
                    expected = existing + int(total_reported)
                    if final_size != expected:
                        raise IOError(
                            f"Size mismatch: got {final_size:,} bytes, "
                            f"expected {expected:,}"
                        )

                log.info(f"  [OK] {dest.name} ({final_size / 1e6:.1f} MB)")
                return True

        except Exception as exc:
            wait = 2 ** attempt
            log.warning(
                f"  [Attempt {attempt}/{max_retries}] {dest.name} failed: {exc}. "
                f"Retrying in {wait}s…"
            )
            time.sleep(wait)

    log.error(f"  [Giving up] {dest.name} after {max_retries} attempts.")
    return False


def download_aod():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    # Bearer token for the JSON directory listing endpoint
    listing_headers = {"Authorization": f"Bearer {TOKEN}"}
    # Basic auth (username/password) is set on the session for file downloads
    session = get_session(USERNAME, PASSWORD)

    failed_files: list[str] = []

    curr = START_DATE
    while curr <= END_DATE:
        year = curr.year
        doy  = curr.timetuple().tm_yday
        dated_dir = SAVE_DIR / str(curr.date())
        dated_dir.mkdir(parents=True, exist_ok=True)

        log.info(f"--- {curr.date()} (DOY {doy:03d}) ---")

        listing_url = (
            f"https://ladsweb.modaps.eosdis.nasa.gov/archive/allData"
            f"/61/MCD19A2/{year}/{doy:03d}.json"
        )

        try:
            resp = session.get(listing_url, headers=listing_headers, timeout=(30, 60))
        except Exception as exc:
            log.error(f"  [Listing error] {curr.date()}: {exc}")
            curr += timedelta(days=1)
            continue

        if resp.status_code == 401:
            log.error("  [Unauthorized] Token rejected — stopping.")
            break
        if resp.status_code != 200:
            log.warning(f"  [Listing HTTP {resp.status_code}] Skipping {curr.date()}")
            curr += timedelta(days=1)
            continue

        file_list = resp.json().get("content", [])
        matches = [f for f in file_list
                   if any(t in f.get("name", "") for t in TILES)]

        if not matches:
            log.info(f"  No Nepal tiles for {curr.date()}")
            curr += timedelta(days=1)
            continue

        for f in matches:
            filename = f.get("name", "")
            dest = dated_dir / filename

            # Skip only if file looks complete (real HDF files are several MB)
            MIN_VALID_SIZE = 1 * 1024 * 1024  # 1 MB
            if dest.exists():
                size = dest.stat().st_size
                if size >= MIN_VALID_SIZE:
                    log.info(f"  [Exists] {filename} ({size / 1e6:.1f} MB)")
                    continue
                else:
                    log.warning(
                        f"  [Stale] {filename} is only {size:,} bytes — "
                        f"deleting and re-downloading."
                    )
                    dest.unlink()

            # Prefer the explicit fileURL path over downloadsLink —
            # downloadsLink sometimes points to a web page rather than the binary.
            file_path = f.get("fileURL") or f.get("downloadsLink", "")
            if file_path.startswith("http"):
                file_url = file_path
            else:
                file_url = f"https://ladsweb.modaps.eosdis.nasa.gov{file_path}"

            # Last resort: construct the URL directly from known path structure
            if not file_path:
                file_url = (
                    f"https://ladsweb.modaps.eosdis.nasa.gov/archive/allData"
                    f"/61/MCD19A2/{year}/{doy:03d}/{filename}"
                )

            log.info(f"  [Downloading] {filename}")
            log.info(f"  [URL] {file_url}")
            ok = download_with_resume(session, file_url, dest)
            if not ok:
                failed_files.append(f"{curr.date()}/{filename}")

            time.sleep(2)   # be polite between files

        curr += timedelta(days=1)

    # ---- Summary ----
    if failed_files:
        log.warning(f"\n=== {len(failed_files)} file(s) failed permanently ===")
        for name in failed_files:
            log.warning(f"  FAILED: {name}")
    else:
        log.info("\n=== All files downloaded successfully ===")


if __name__ == "__main__":
    download_aod()
