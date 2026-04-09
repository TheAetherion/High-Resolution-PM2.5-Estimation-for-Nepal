# High-Resolution PM₂.₅ Estimation for Nepal
### A Two-Stage Machine Learning Framework Using Satellite Remote Sensing

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Data: Zenodo](https://img.shields.io/badge/Data-Zenodo-blue.svg)](https://zenodo.org)
[![Paper: In Review](https://img.shields.io/badge/Paper-In%20Review-orange.svg)]()

> **Samarpan Mani Gautam, Udhyan Shah, Suresh Acharaya, Dilip Rajak, Jigyashu Ghimire, Liza Dev Pradhan Shrestha**  
> Kathmandu University, Nepal  
> Manuscript under review — *Atmospheric Pollution Research / Air Quality, Atmosphere & Health*

---

## Overview

Nepal is among the world's most polluted countries, yet 92% of its land area has no nearby air quality monitoring station. This project addresses that gap by producing **1-km resolution PM₂.₅ estimates** for Nepal covering **January 2025 to March 2026**, using only freely available satellite data and a reproducible machine learning pipeline.

The framework does two things that most similar studies do not:

1. It is **honest about where it works and where it does not** — every prediction pixel is tagged with a confidence flag based on its distance from the nearest training station.
2. It uses those confidence flags to make a concrete, quantified argument for **regulatory reform**: Nepal needs an annual PM₂.₅ standard, more monitoring stations, and faster EV adoption.

**Key results at a glance:**

| Metric | Value |
|--------|-------|
| Spatial resolution | 1 km |
| Study period | Jan 2025 – Mar 2026 |
| Spatial CV R² | 0.40 |
| RMSE | 32.9 µg/m³ |
| Predicted PM₂.₅ range | 33.9 – 134.4 µg/m³ |
| WHO annual guideline (5 µg/m³) exceeded by | 7× – 27× |
| Land area in LOW-confidence zone | **92%** |

---

## Repository Structure

```
High-Resolution-PM2.5-Estimation-for-Nepal/
│
├── project report.pdf              # Final project report
├── README.md
│
├── AEROSOL_Index/
│   ├── AEROSOL_Index_Data.nc
│   ├── AEROSOL_Index_Plot.png
│   └── AEROSOL_Index_Plot.py
│
├── CH4/
│   ├── CH4_Data.nc
│   ├── CH4_Plot.png
│   └── CH4_Plot.py
│
├── CO/
│   ├── CO_Data.nc
│   ├── CO_Plot.png
│   └── CO_Plot.py
│
├── HCHO/
│   ├── HCHO_Data.nc
│   ├── HCHO_Plot.png
│   └── HCHO_Plot.py
│
├── NO2/
│   ├── NO2_Data.nc
│   ├── NO2_Plot.png
│   └── NO2_Plot.py
│
├── O3/
│   ├── O3_Data.nc
│   ├── O3_Plot.png
│   └── O3_Plot.py
│
├── SO2/
│   ├── SO2_Data.nc
│   ├── SO2_Plot.png
│   └── SO2_Plot.py
│
├── npl_admin_boundaries_shp/       # Nepal administrative boundary shapefiles
│   ├── npl_admin0.*                # National boundary (admin level 0)
│   ├── npl_admin1.*                # Province boundary (admin level 1)
│   ├── npl_admin2.*                # District boundary (admin level 2)
│   ├── npl_admin3.*                # Municipality boundary (admin level 3)
│   ├── npl_admincentroids.*        # Administrative centroids
│   └── npl_adminlines.*            # Administrative boundary lines
│   (each layer has .cpg, .dbf, .prj, .shp, .shx; *_em variants included)
│
├── PM2.5/
│   ├── AOD_Data/                   # MODIS MAIAC MCD19A2 HDF tiles
│   │   ├── 2025-01-01/             # Daily subfolders (Jan 2025 – Mar 2026)
│   │   │   ├── MCD19A2.A2025001.h25v05.061.*.hdf
│   │   │   ├── MCD19A2.A2025001.h25v06.061.*.hdf
│   │   │   └── MCD19A2.A2025001.h26v06.061.*.hdf
│   │   └── ...                     # One subfolder per day
│   │
│   ├── ERA5_Data/                  # ERA5 reanalysis daily NetCDF files
│   │   ├── era5_20250101.nc
│   │   ├── era5_20250101.nc.zip_orig
│   │   ├── ...                     # One .nc + .zip_orig per day
│   │   └── unzip_log.txt
│   │
│   ├── Outputs/                    # Model outputs and diagnostics
│   │   ├── aod_gap_filled_1km.tif              # Gap-filled AOD raster
│   │   ├── aod_gap_filler.joblib               # Trained RF gap-filler model
│   │   ├── applicability_statement.txt
│   │   ├── confidence_flag.tif                 # Confidence-flag raster (0/1/2)
│   │   ├── dist_to_station_km.tif              # Distance-to-nearest-station raster
│   │   ├── pm25_nepal_1km_distance_ensemble.tif  # Final PM₂.₅ raster (ensemble)
│   │   ├── pm25_nepal_1km_whole_country.tif      # Full-country PM₂.₅ raster
│   │   ├── pm25_model_final.joblib             # Final HistGBR model
│   │   ├── pm25_model_region_ktm_valley.joblib
│   │   ├── pm25_model_region_outer_cities.joblib
│   │   ├── pm25_calibrator_final.joblib        # Isotonic calibrators
│   │   ├── pm25_calibrator_region_ktm_valley.joblib
│   │   ├── pm25_calibrator_region_outer_cities.joblib
│   │   ├── pm25_scaler_final.joblib            # Feature scalers
│   │   ├── pm25_scaler_region_ktm_valley.joblib
│   │   ├── pm25_scaler_region_outer_cities.joblib
│   │   ├── diag_A_aod_vs_pm25_distance_ensemble.png
│   │   ├── diag_B_cv_scatter_distance_ensemble.png
│   │   ├── diag_C_fold_r2_distance_ensemble.png
│   │   ├── diag_D_timeseries_distance_ensemble.png
│   │   ├── error_vs_distance_v10_distance_ensemble.png
│   │   ├── pm25_diagnostics_v10_distance_ensemble.png
│   │   └── _tmp_ndvi_*.tif                     # Temporary NDVI intermediates
│   │
│   ├── PM2.5_Data/                 # Ground-truth sensor CSVs (OpenAQ / GD Labs)
│   │   ├── Nepal_Balaju_(SC-26)-_GD_Labs_*.csv
│   │   ├── Nepal_Balkumari(SC-28)-_GD_Labs_*.csv
│   │   ├── ...                     # One CSV per sensor station
│   │   └── Nepal_Tyanglaphat_(SC_-_21)-_GD_Labs_*.csv
│   │
│   ├── Population__Data/
│   │   └── npl_pd_2020_1km_UNadj.tif           # WorldPop 2020 population density
│   │
│   ├── Python/                     # All modelling scripts
│   │   ├── AOD_data.py             # MODIS AOD download / preprocessing
│   │   ├── conversion.py           # Unit / projection conversions
│   │   ├── era5_daily.py           # ERA5 download and daily aggregation
│   │   ├── ground_data.py          # Ground sensor ingestion and QC
│   │   ├── unzip_era5.py           # ERA5 archive extraction
│   │   ├── pm25_nepal_framework.py         # Main framework 
│   │
│   └── Vegetation_Data/            # MODIS NDVI monthly composites
│       ├── MOD13A3.A2025001.h25v05.061.*.hdf
│       ├── MOD13A3.A2025001.h25v06.061.*.hdf
│       ├── MOD13A3.A2025032.h25v05.061.*.hdf
│       └── MOD13A3.A2025032.h25v06.061.*.hdf
│
└── Report/
    ├── nepal_pm25_paper.tex        # LaTeX manuscript source
    ├── nepal_pm25_paper.pdf        # Compiled manuscript
    ├── nepal_pm25_refs.bib         # Bibliography
    ├── nepal_pm25_paper.*          # LaTeX auxiliary files (.aux, .bbl, .blg, etc.)
    ├── AEROSOL_Index_Plot.png      # Figures used in manuscript
    ├── CO_Plot.png
    ├── HCHO_Plot.png
    ├── NO2_Plot.png
    ├── O3_Plot.png
    ├── SO2_Plot.png
    ├── diag_A_aod_vs_pm25_distance_ensemble.png
    ├── diag_B_cv_scatter_distance_ensemble.png
    ├── diag_C_fold_r2_distance_ensemble.png
    ├── diag_D_timeseries_distance_ensemble.png
    └── Papers/                     # Reference literature PDFs
```

---

## Methodology

The pipeline runs in two sequential stages.

### Stage 1 — AOD Gap-Filling (Random Forest)

MODIS MAIAC AOD at 1 km has 3.2% missing pixels over Nepal due to cloud cover and snow. A Random Forest regressor fills those gaps using engineered covariates derived from the surrounding clear-sky field (AOD × RH, AOD × T₂ₘ, AOD/BLH, elevation × latitude, inter-day AOD delta, and others). Out-of-bag R² = 1.00, reflecting the spatial smoothness of AOD fields — gap pixels are interpolated from their neighbours, not extrapolated from independent covariates.

### Stage 2 — PM₂.₅ Prediction (HistGBR + Isotonic Calibration)

A Histogram Gradient Boosting Regressor (Poisson loss) is fitted on 6,226 station-day observations using the full covariate stack:

- Gap-filled MAIAC AOD (550 nm)
- ERA5: T₂ₘ, T_dew, BLH, u₁₀, v₁₀, surface pressure, precipitation, RH
- TROPOMI: NO₂, CO, HCHO, O₃, SO₂, Aerosol Index
- Static: DEM (SRTM 1 km), NDVI (MODIS), population density (WorldPop 2020)
- Temporal: day-of-year sine/cosine, month, station/region target encoding

An isotonic regression calibrator is fitted on a held-out 20% validation split to remove residual bias.

### Distance-Ensemble Strategy

Because 96.2% of training observations are from Kathmandu Valley, two region-specific models are trained:

- **M_KTM**: trained on Valley stations (folds 0–2; 5,991 samples)
- **M_outer**: trained on outer-city stations (fold 3; 235 samples from Hetauda)

Predictions are blended using inverse-distance weighting (IDW):

$$\hat{y}_\text{ens}(\mathbf{x}) = \frac{w_\text{KTM} \cdot \hat{y}_\text{KTM} + w_\text{outer} \cdot \hat{y}_\text{outer}}{w_\text{KTM} + w_\text{outer}}, \quad w_r = \frac{1}{d_r + 1}$$

### Applicability-Domain Confidence Flags

A KD-tree over the 56 unique training-station locations assigns every 1-km pixel one of three confidence flags:

| Flag | Level | Criterion |
|------|-------|-----------|
| 2 | HIGH | ≤ 20 km from nearest station |
| 1 | MEDIUM | 20–50 km |
| 0 | LOW | > 50 km |

These flags are exported as `PM2.5/Outputs/confidence_flag.tif` and should always be used alongside the PM₂.₅ raster when interpreting results.

---

## Data Sources

All input data are freely available from public sources.

| Dataset | Variable | Source | Resolution |
|---------|----------|--------|------------|
| MODIS MAIAC (MCD19A2) | AOD 550 nm | [NASA LAADS DAAC](https://ladsweb.modaps.eosdis.nasa.gov/) | 1 km, daily |
| ERA5 | T₂ₘ, BLH, wind, precip | [Copernicus CDS](https://cds.climate.copernicus.eu/) | 0.25°, daily |
| Sentinel-5P TROPOMI | NO₂, CO, HCHO, O₃, SO₂, AI | [Copernicus CDSE](https://dataspace.copernicus.eu/) / [Google Earth Engine](https://earthengine.google.com/) | ~5.5 km, daily |
| SRTM DEM | Elevation | [NASA/USGS](https://earthexplorer.usgs.gov/) | 1 km |
| WorldPop | Population density | [WorldPop](https://www.worldpop.org/) | 1 km |
| MODIS NDVI (MOD13A3) | Vegetation index | [NASA LAADS DAAC](https://ladsweb.modaps.eosdis.nasa.gov/) | 1 km, monthly |
| GD Labs / OpenAQ | Ground PM₂.₅ | [OpenAQ](https://openaq.org/) | Station, hourly |
| Nepal Admin Boundaries | Shapefiles | [HDX / OCHA](https://data.humdata.org/) | Vector |

> **Note:** Raw satellite tiles are large and not uploaded to this repository. ERA5 daily NetCDF files are stored under `PM2.5/ERA5_Data/` as `.nc` + `.zip_orig` pairs; MODIS AOD tiles are stored under `PM2.5/AOD_Data/<YYYY-MM-DD>/`. Download and unzip scripts are provided in `PM2.5/Python/era5_daily.py`, `PM2.5/Python/unzip_era5.py`, and `PM2.5/Python/AOD_data.py`.

---

## Installation

### Using Conda (recommended)

```bash
git clone https://github.com/TheAetherion/High-Resolution-PM2.5-Estimation-for-Nepal-.git
cd High-Resolution-PM2.5-Estimation-for-Nepal-
conda env create -f environment.yml
conda activate nepal-pm25
```

### Using pip

```bash
git clone https://github.com/TheAetherion/High-Resolution-PM2.5-Estimation-for-Nepal-.git
cd High-Resolution-PM2.5-Estimation-for-Nepal-
pip install -r requirements.txt
```

**Core dependencies:** `scikit-learn`, `numpy`, `pandas`, `rasterio`, `geopandas`, `xarray`, `scipy`, `matplotlib`, `earthengine-api`

---

## Reproducing the Results

Run the scripts in `PM2.5/Python/` in the following order:

```bash
# 1. Download and prepare input data
python PM2.5/Python/AOD_data.py
python PM2.5/Python/era5_daily.py
python PM2.5/Python/unzip_era5.py
python PM2.5/Python/ground_data.py

# 2. Run conversions / preprocessing
python PM2.5/Python/conversion.py

# 3. Run the full modelling framework (latest version)
python PM2.5/Python/pm25_nepal_framework_v11.py

# 4. Diagnostics
python PM2.5/Python/diagnose.py
python PM2.5/Python/diagnose_era5.py
```

Pre-processed outputs (PM₂.₅ GeoTIFFs + confidence-flag raster) are available directly from Zenodo if you want to skip the computationally intensive steps.

---

## Output Files

All final outputs are written to `PM2.5/Outputs/`.

| File | Description |
|------|-------------|
| `pm25_nepal_1km_distance_ensemble.tif` | 1-km PM₂.₅ raster produced by the distance-weighted ensemble |
| `pm25_nepal_1km_whole_country.tif` | Full-country 1-km PM₂.₅ raster |
| `confidence_flag.tif` | Confidence-flag raster (0 = LOW, 1 = MEDIUM, 2 = HIGH) |
| `aod_gap_filled_1km.tif` | Gap-filled MAIAC AOD raster (Stage 1 output) |
| `dist_to_station_km.tif` | Distance-to-nearest-training-station raster |
| `aod_gap_filler.joblib` | Serialised Stage 1 Random Forest model |
| `pm25_model_*.joblib` | Serialised Stage 2 HistGBR models (final + regional) |
| `pm25_calibrator_*.joblib` | Serialised isotonic calibrators |
| `pm25_scaler_*.joblib` | Serialised feature scalers |
| `diag_*.png` | Diagnostic figures (AOD vs PM₂.₅, CV scatter, fold R², time series) |

> **Important:** Always load `confidence_flag.tif` alongside any PM₂.₅ raster. Predictions flagged as LOW (> 50 km from any training station) cover 92% of Nepal and should be treated as spatial extrapolations, not locally-constrained estimates.

---

## Citation

If you use this code or the prediction outputs in your work, please cite:

```bibtex
@article{gautam2025nepal_pm25,
  author  = {Gautam, Samarpan Mani and Shah, Udhyan and Acharaya, Suresh and Rajak, Dilip
             and Ghimire, Jigyashu and Shrestha, Liza Dev Pradhan},
  title   = {High-Resolution {PM}$_{2.5}$ Mapping for {Nepal} Reveals Severe
             Monitoring Gaps and the Urgent Need for Regulatory Reform},
  journal = {Atmospheric Pollution Research},
  year    = {2025},
  note    = {Under review}
}
```

---

## License

Code: [MIT License](LICENSE)  
Prediction data and GeoTIFFs: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

---

## Contact

**Samarpan Mani Gautam**  
Department of Chemical Science and Engineering, Kathmandu University  
✉ [sg03246622@student.ku.edu.np](mailto:sg03246622@student.ku.edu.np)  
🐙 [@TheAetherion](https://github.com/TheAetherion)
