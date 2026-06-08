"""
Region mask construction and visualization.

The mask-building logic mirrors custom_mask.ipynb exactly:
- ERA5 grid is 721 x 1440 (0.25 deg).
- Row (latitude) index:  (90 - lat) * 4
- Col (longitude) index:  lon * 4
A grid point inside the region bounding box is set to 1 (kept), else 0 (ignored).

This is the geo-spatial mask beta_{i,j} from the paper's VAAWM adaptive loss:
    L_adaptive = sum_v  alpha_v  sum_{(i,j) in G}  beta_{i,j} (Y_{i,j,v} - Yhat_{i,j,v})^2
"""

import numpy as np

GRID_LAT = 721
GRID_LON = 1440


def build_region_mask(lat_min, lat_max, lon_min, lon_max):
    """Build a (721, 1440) float32 mask, 1 inside the bounding box, 0 outside.

    Identical index math to custom_mask.ipynb.
    """
    mask = np.zeros((GRID_LAT, GRID_LON), dtype=np.float32)
    lat_idx_min = int((90 - lat_max) * 4)
    lat_idx_max = int((90 - lat_min) * 4)
    lon_idx_min = int(lon_min * 4)
    lon_idx_max = int(lon_max * 4)
    mask[lat_idx_min:lat_idx_max + 1, lon_idx_min:lon_idx_max + 1] = 1
    return mask


def mask_stats(mask):
    """Return (kept_points, total_points, fraction_pct)."""
    kept = int(mask.sum())
    total = mask.size
    return kept, total, kept / total * 100.0


def visualize_mask(mask, ascii_title="Region mask (beta_i,j)"):
    """Render the region mask over a China-centered map. Returns a matplotlib Figure.

    `ascii_title` must be ASCII because no CJK font is installed for matplotlib.
    """
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    fig = plt.figure(figsize=(9, 5.5))
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

    # Focus the view on China / the region of interest.
    ax.set_extent([70, 135, 15, 55], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=":")
    ax.add_feature(cfeature.LAND, facecolor="#f2efe9")
    ax.add_feature(cfeature.OCEAN, facecolor="#dbeafe")
    ax.gridlines(draw_labels=True, linewidth=0.4, color="gray",
                 alpha=0.5, linestyle="--")

    # Build lat/lon coordinates for the full grid.
    lats = np.linspace(90, -90, GRID_LAT)
    lons = np.linspace(0, 360 - 0.25, GRID_LON)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # Overlay only the kept region (mask == 1) as a semi-transparent patch.
    masked = np.ma.masked_where(mask == 0, mask)
    ax.pcolormesh(lon_grid, lat_grid, masked, transform=ccrs.PlateCarree(),
                  cmap="autumn", alpha=0.55, shading="auto")

    ax.set_title(ascii_title, fontsize=11)
    return fig


def plot_field(field, bbox=None, title="", cmap="viridis", vmin=None, vmax=None):
    """Plot a (721,1440) global field, optionally zoomed to bbox=(lon0,lon1,lat0,lat1).

    `title` must be ASCII (no CJK font installed for matplotlib).
    Returns a matplotlib Figure.
    """
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    fig = plt.figure(figsize=(7.5, 4.6))
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    if bbox is not None:
        ax.set_extent(list(bbox), crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=":")
    ax.gridlines(draw_labels=True, linewidth=0.4, color="gray", alpha=0.5, linestyle="--")

    lats = np.linspace(90, -90, GRID_LAT)
    lons = np.linspace(0, 360 - 0.25, GRID_LON)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    if vmin is None:
        vmin = float(np.nanpercentile(field, 2))
    if vmax is None:
        vmax = float(np.nanpercentile(field, 98))
    im = ax.pcolormesh(lon_grid, lat_grid, field, transform=ccrs.PlateCarree(),
                       cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
    cbar = plt.colorbar(im, ax=ax, orientation="vertical", pad=0.03, shrink=0.85)
    cbar.ax.tick_params(labelsize=8)
    ax.set_title(title, fontsize=10)
    return fig
