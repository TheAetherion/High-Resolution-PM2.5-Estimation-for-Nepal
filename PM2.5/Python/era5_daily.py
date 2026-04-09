# era5_daily.py
import cdsapi
from pathlib import Path
from datetime import datetime, timedelta

SAVE_DIR = Path(r"C:\Users\samue\Documents\Conferences\Space\PM2.5\ERA5_Data")
START    = datetime(2025, 1, 1)
END      = datetime(2026, 3, 31)
BBOX     = [30.5, 79.5, 26.0, 88.3]   # N, W, S, E — Nepal + buffer

c = cdsapi.Client()

curr = START
while curr <= END:
    out_path = SAVE_DIR / f"era5_{curr.strftime('%Y%m%d')}.nc"
    if out_path.exists():
        print(f"[SKIP] {out_path.name}")
        curr += timedelta(days=1)
        continue

    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[Downloading] ERA5 for {curr.date()}")

    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'variable': [
                '2m_temperature', '2m_dewpoint_temperature',
                'surface_pressure', '10m_u_component_of_wind',
                '10m_v_component_of_wind', 'boundary_layer_height',
                'total_precipitation',
            ],
            'year':  str(curr.year),
            'month': f"{curr.month:02d}",
            'day':   f"{curr.day:02d}",
            'time':  ['06:00', '09:00', '12:00'],   # morning average ≈ MODIS overpass
            'area':  BBOX,
            'format': 'netcdf',
        },
        str(out_path)
    )
    curr += timedelta(days=1)

print("ERA5 download complete.")
