"""
Wind-energy pipeline stages 2 & 3:

  Stage 2 - 降尺度 (downscaling):  coarse 0.25 deg wind  ->  finer regional grid.
            Currently a runnable INTERPOLATION baseline (bilinear/bicubic),
            exposed behind a clean `downscale()` interface so a trained CorrDiff
            model can be dropped in later (same in/out contract).
            *** This is NOT CorrDiff. It adds resolution, not new fine-scale
                physics. Clearly a placeholder for a trained diffusion model. ***

  Stage 3 - 功率曲线转换 (power-curve conversion):  hub-height wind extrapolation
            + a standard IEC-style turbine power curve -> capacity factor (0..1).
            This is real, physically-standard wind-energy math.
"""
import numpy as np

# --- Stage 3 parameters (representative multi-MW onshore turbine) -------------
HUB_HEIGHT_M = 100.0      # hub height
REF_HEIGHT_M = 10.0       # ERA5 10m wind reference
SHEAR_ALPHA = 0.143       # power-law shear exponent (~1/7, neutral stability)
CUT_IN = 3.0              # m/s
RATED = 12.0              # m/s
CUT_OUT = 25.0            # m/s


def extrapolate_to_hub(ws10, hub=HUB_HEIGHT_M, ref=REF_HEIGHT_M, alpha=SHEAR_ALPHA):
    """Power-law vertical extrapolation of 10 m wind speed to hub height."""
    return ws10 * (hub / ref) ** alpha


def capacity_factor(ws_hub, cut_in=CUT_IN, rated=RATED, cut_out=CUT_OUT):
    """IEC-style power curve -> capacity factor in [0,1].

    0 below cut-in or above cut-out, cubic ramp between cut-in and rated,
    flat at 1 between rated and cut-out.
    """
    ws = np.asarray(ws_hub, dtype=np.float32)
    cf = np.zeros_like(ws)
    ramp = (ws >= cut_in) & (ws < rated)
    cf[ramp] = (ws[ramp] ** 3 - cut_in ** 3) / (rated ** 3 - cut_in ** 3)
    cf[(ws >= rated) & (ws <= cut_out)] = 1.0
    return np.clip(cf, 0.0, 1.0)


def downscale(field, factor=5, method="bilinear"):
    """Stage-2 downscaling interface (placeholder for a trained CorrDiff).

    field  : 2D numpy array (coarse regional wind field)
    factor : integer upsampling factor (e.g. 0.25deg -> ~0.05deg with factor 5)
    Returns the finer-resolution 2D field.
    """
    import torch
    import torch.nn.functional as F
    t = torch.from_numpy(np.ascontiguousarray(field)).float()[None, None]
    h, w = field.shape
    mode = "bicubic" if method == "bicubic" else "bilinear"
    out = F.interpolate(t, size=(h * factor, w * factor), mode=mode,
                        align_corners=False)
    return out[0, 0].numpy()


def crop_region(field2d, region, pad=0.0):
    """Crop a global (721,1440) field to a region bbox. Returns (sub, extent).

    region: dict with lat_min/lat_max/lon_min/lon_max (degrees).
    extent: (lon0, lon1, lat0, lat1) for plotting.
    """
    lat0 = region["lat_min"] - pad
    lat1 = region["lat_max"] + pad
    lon0 = region["lon_min"] - pad
    lon1 = region["lon_max"] + pad
    # grid: lat 90..-90 (721), lon 0..359.75 (1440)
    r_min = int(round((90 - lat1) * 4))
    r_max = int(round((90 - lat0) * 4))
    c_min = int(round(lon0 * 4))
    c_max = int(round(lon1 * 4))
    sub = field2d[r_min:r_max + 1, c_min:c_max + 1]
    return sub, (lon0, lon1, lat0, lat1)


def run_power_chain(ws10_global, region, factor=5, method="bilinear"):
    """Full stage 2+3 over a region.

    ws10_global : (721,1440) 10m wind speed (m/s), physical units.
    Returns dict with coarse/fine wind fields, hub wind, capacity-factor map,
    mean capacity factor, and the plotting extent.
    """
    coarse, extent = crop_region(ws10_global, region)
    fine = downscale(coarse, factor=factor, method=method)
    hub = extrapolate_to_hub(fine)
    cf = capacity_factor(hub)
    return {
        "coarse_ws": coarse,
        "fine_ws": fine,
        "hub_ws": hub,
        "cf_map": cf,
        "mean_cf": float(cf.mean()),
        "extent": extent,
        "coarse_shape": coarse.shape,
        "fine_shape": fine.shape,
    }


def plot_regional(field, extent, title="", cmap="viridis", cbar_label=""):
    """Plot a regional 2D field (any resolution) with lon/lat extent + coastlines.

    `title`/`cbar_label` must be ASCII (no CJK font for matplotlib).
    """
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    lon0, lon1, lat0, lat1 = extent
    fig = plt.figure(figsize=(6.2, 4.6))
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    ax.set_extent([lon0, lon1, lat0, lat1], crs=ccrs.PlateCarree())
    im = ax.imshow(field, origin="upper", extent=[lon0, lon1, lat0, lat1],
                   transform=ccrs.PlateCarree(), cmap=cmap, aspect="auto")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=":")
    ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5, linestyle="--")
    cb = plt.colorbar(im, ax=ax, orientation="vertical", pad=0.03, shrink=0.85)
    cb.ax.tick_params(labelsize=8)
    if cbar_label:
        cb.set_label(cbar_label, fontsize=8)
    ax.set_title(title, fontsize=10)
    return fig


# --- Altitude-aware level selection (the ▲ in the refined pipeline) -----------
# Standard-atmosphere geopotential height (m, above sea level) for pressure levels.
PRESSURE_HEIGHT_ASL = {"1000": 110.0, "925": 762.0, "850": 1457.0}
# Height-above-ground levels (m AGL).
AGL_LEVELS = {"10m": 10.0, "100m": 100.0}


def level_height_asl(level_name, farm_elevation_m):
    """Approximate height (m, above sea level) of a wind field 'level'.

    Pressure levels use standard-atmosphere geopotential height (independent of
    terrain). AGL levels (10m/100m) are referenced to the wind-farm elevation.
    `level_name` like 'U850'/'V10m' or '850'/'10m'.
    """
    lv = level_name.lstrip("UV")  # 'U850' -> '850', 'V10m' -> '10m'
    if lv in AGL_LEVELS:
        return farm_elevation_m + AGL_LEVELS[lv]
    if lv in PRESSURE_HEIGHT_ASL:
        return PRESSURE_HEIGHT_ASL[lv]
    raise ValueError(f"unknown level {level_name}")


def select_wind_level(farm_elevation_m, hub_height_m, available_levels):
    """Pick the available wind level closest to the turbine height.

    target = farm_elevation + hub_height (m ASL). Returns (best_level, details)
    where details maps level -> (height_asl, |diff|).
    """
    target = farm_elevation_m + hub_height_m
    details = {}
    best, best_diff = None, float("inf")
    for lv in available_levels:
        h = level_height_asl(lv, farm_elevation_m)
        d = abs(h - target)
        details[lv] = (h, d)
        if d < best_diff:
            best, best_diff = lv, d
    return best, details


def wind_speed_at_level(result_fields, level):
    """Compute wind speed sqrt(U^2+V^2) for a given level from a dict of fields.

    result_fields: dict with keys like 'U850','V850',... (2D arrays).
    level: '850'/'10m'/... -> uses 'U'+level and 'V'+level.
    """
    u = result_fields["U" + level]
    v = result_fields["V" + level]
    return np.sqrt(u ** 2 + v ** 2)


def run_power_chain_at_height(wind_global, region, src_level, hub_height_m,
                              farm_elevation_m, factor=5, method="bilinear"):
    """Stage 2+3 starting from a wind-speed field already taken at `src_level`.

    src_level: '10m' | '1000' | '925' | '850' (the level chosen by altitude).
    - AGL levels ('10m'): power-law extrapolate from that AGL height to hub.
    - Pressure levels: assumed already near hub height -> used directly
      (we selected them because they are closest to the turbine).
    Returns same dict as run_power_chain plus the chosen level info.
    """
    coarse, extent = crop_region(wind_global, region)
    fine = downscale(coarse, factor=factor, method=method)

    lv = src_level.lstrip("UV")
    if lv in AGL_LEVELS:
        src_h = AGL_LEVELS[lv]
        hub_ws = extrapolate_to_hub(fine, hub=hub_height_m, ref=src_h)
        note = f"AGL {lv}: power-law {lv}->{hub_height_m:.0f}m"
    else:
        hub_ws = fine  # pressure level already ~hub height
        note = f"pressure {lv}hPa: used directly (~hub height)"

    cf = capacity_factor(hub_ws)
    return {
        "coarse_ws": coarse, "fine_ws": fine, "hub_ws": hub_ws, "cf_map": cf,
        "mean_cf": float(cf.mean()), "extent": extent,
        "coarse_shape": coarse.shape, "fine_shape": fine.shape,
        "src_level": src_level, "note": note,
    }
