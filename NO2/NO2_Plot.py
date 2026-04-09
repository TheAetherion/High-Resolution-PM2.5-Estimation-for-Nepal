import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
from scipy.interpolate import griddata
from rasterio import features
from affine import Affine
import warnings

warnings.filterwarnings("ignore")

# ── File Paths ────────────────────────────────────────────────────────
nc_file        = r"C:\Users\samue\Documents\Conferences\Space\NO2\NO2_Data.nc"
shapefile_path = r"C:\Users\samue\Documents\Conferences\Space\npl_admin_boundaries_shp\npl_admin0.shp"

# ── Read Data ─────────────────────────────────────────────────────────
ds   = nc.Dataset(nc_file)
prod = ds.groups['PRODUCT']

lon      = prod.variables['longitude'][:].flatten()
lat      = prod.variables['latitude'][:].flatten()
# Standard TROPOMI NO2 variable (tropospheric vertical column)
data_var = prod.variables['nitrogendioxide_tropospheric_column'][0, :, :].flatten()
qa       = prod.variables['qa_value'][0, :, :].flatten()

# QA > 0.75 is recommended for NO2 pollution studies (stricter than 0.5)
mask       = (qa > 0.75) & (~data_var.mask)
lon_clean  = lon[mask]
lat_clean  = lat[mask]
data_clean = data_var[mask]
ds.close()

# ── Geography ─────────────────────────────────────────────────────────
nepal  = gpd.read_file(shapefile_path)
if nepal.crs.to_epsg() != 4326:
    nepal = nepal.to_crs('EPSG:4326')
bounds = nepal.total_bounds   # [W, S, E, N]

# ── Grid & Interpolate ────────────────────────────────────────────────
RES    = 400
grid_x = np.linspace(bounds[0], bounds[2], RES)
grid_y = np.linspace(bounds[3], bounds[1], RES)   # N → S (required by rasterio)
grid_lon, grid_lat = np.meshgrid(grid_x, grid_y)

grid_data = griddata((lon_clean, lat_clean), data_clean,
                     (grid_lon, grid_lat), method='nearest')

# ── Mask Outside Nepal (dy NEGATIVE = north-to-south) ────────────────
dx        = (bounds[2] - bounds[0]) / RES
dy        = (bounds[3] - bounds[1]) / RES
transform = Affine.translation(bounds[0], bounds[3]) * Affine.scale(dx, -dy)

mask_arr = features.geometry_mask(
    nepal.geometry, out_shape=grid_data.shape,
    transform=transform, invert=False
)
grid_data[mask_arr] = np.nan

# ── Plot ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(10, 6), dpi=300)
ax  = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

ax.set_extent([bounds[0] - 0.15, bounds[2] + 0.15,
               bounds[1] - 0.15, bounds[3] + 0.15],
              crs=ccrs.PlateCarree())

ax.add_feature(cfeature.LAND,  facecolor='#ececec', zorder=0)
ax.add_feature(cfeature.OCEAN, facecolor='#cce5f0', zorder=0)

vmin = np.nanpercentile(grid_data, 2)
vmax = np.nanpercentile(grid_data, 98)

# 'RdPu' is widely used for NO2 in atmospheric science publications
im = ax.pcolormesh(grid_lon, grid_lat, grid_data,
                   cmap='RdPu', shading='auto',
                   vmin=vmin, vmax=vmax,
                   transform=ccrs.PlateCarree(), zorder=1)

nepal.boundary.plot(ax=ax, color='black', linewidth=1.2, zorder=2)

gl = ax.gridlines(draw_labels=True, linestyle='--',
                  alpha=0.4, color='gray', linewidth=0.5)
gl.top_labels   = False
gl.right_labels = False
gl.xlabel_style = {'size': 8}
gl.ylabel_style = {'size': 8}

cbar = fig.colorbar(im, ax=ax, orientation='vertical',
                    pad=0.02, shrink=0.80, aspect=28)
cbar.set_label(r'NO$_2$ Column (mol m$^{-2}$)', fontsize=9, labelpad=8)
cbar.ax.tick_params(labelsize=8)

ax.set_title(r'Nitrogen Dioxide (NO$_2$) Concentration', fontsize=12,
             fontweight='bold', pad=10)

plt.savefig('NO2_Plot.png', dpi=300, bbox_inches='tight', pad_inches=0.15)
plt.show()
print("Saved → NO2_Plot.png")
