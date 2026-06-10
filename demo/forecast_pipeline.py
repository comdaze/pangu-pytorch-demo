"""
End-to-end wind-power forecast pipeline for the chat demo.

  ERA5 初始场 -> Pangu (autoregressive, 逐24h步进) -> CorrDiff 降尺度(占位)
  -> 按海拔选气压层 U/V -> 功率曲线 -> 出力

Runs entirely on the device inference_engine selects. The chat app launches with
CUDA_VISIBLE_DEVICES="" so this never competes with the pre2019 GPU training.

Produces a structured result plus a set of professional matplotlib figures.
"""
import os
import sys
import datetime as dt

import numpy as np
import pandas as pd
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import inference_engine as eng
import wind_power as wp
try:
    import corrdiff_infer as cdi
except Exception:
    cdi = None
from era5_data.config import cfg
from era5_data import utils_data

GRID_LAT, GRID_LON = 721, 1440


def latlon_to_idx(lat, lon):
    r = int(round((90 - lat) * 4))
    c = int(round((lon % 360) * 4))
    return max(0, min(r, GRID_LAT - 1)), max(0, min(c, GRID_LON - 1))


def crop_bbox(field, lat, lon, dlat=2.5, dlon=3.0):
    r0 = int(round((90 - (lat + dlat)) * 4)); r1 = int(round((90 - (lat - dlat)) * 4))
    c0 = int(round((lon - dlon) * 4)); c1 = int(round((lon + dlon) * 4))
    r0, r1 = max(0, r0), min(GRID_LAT - 1, r1)
    c0, c1 = max(0, c0), min(GRID_LON - 1, c1)
    return field[r0:r1 + 1, c0:c1 + 1], (lon - dlon, lon + dlon, lat - dlat, lat + dlat)


def pick_init_date():
    """Default forecast t0: prefer an active-weather date, else latest available."""
    avail = set(eng._all_upper_dates())
    for pref in ["20180401", "20180701"]:
        if pref in avail:
            return pref
    dates = sorted(avail)
    return dates[-1] if dates else "20190701"


def run_forecast(farm, horizon_days=7, init_date=None, factor=5, progress=None):
    """Run the full pipeline. Returns a dict of time series + metadata.

    progress: optional callable(stage_str, frac) for UI streaming.
    """
    device = eng.get_device()  # 'cpu' when CUDA hidden
    init_date = init_date or pick_init_date()

    # altitude-aware level selection (Pangu levels)
    level, level_details = wp.select_wind_level(
        farm["elevation_m"], farm["hub_height_m"], eng.PANGU_WIND_LEVELS)

    # Use our Pre-2019 VAAWM fine-tuned Pangu (trained 2016-2017) when present;
    # otherwise fall back to the official pretrained (zero-shot) weights.
    if eng.pre2019_available():
        variant = "vaawm_pre2019"
        model_label = "Pre-2019 VAAWM 微调 (2016-2017训练)"
    else:
        variant = "zeroshot"
        model_label = "官方预训练 (zero-shot)"

    if progress:
        progress(f"加载 Pangu 模型：{model_label}（{device}）", 0.02)
    model = eng.load_model(device, variant)
    aux = eng.load_aux(device)

    cur_u, cur_s = eng._state_at(init_date, device)
    t0 = pd.to_datetime(init_date, format="%Y%m%d")

    fr, fc = latlon_to_idx(farm["lat"], farm["lon"])
    times, hub_ws_pt, cf_pt = [], [], []
    field_snaps = {}  # lead_day -> (downscaled_field, extent)
    snap_days = sorted(set([1, max(1, horizon_days // 2), horizon_days]))
    use_corrdiff = bool(cdi) and cdi.available()
    downscale_method = "CorrDiff (regression+diffusion)" if use_corrdiff else "bilinear (placeholder)"

    for k in range(1, horizon_days + 1):
        if progress:
            progress(f"Pangu 第 {k}/{horizon_days} 步（+{k*24}h）自回归预报", 0.05 + 0.7 * k / horizon_days)
        with torch.no_grad():
            out, outs = model(cur_u, cur_s, aux["weather_statistics"],
                              aux["constant_maps"], aux["const_h"])
            out, outs = utils_data.normBackData(out, outs, aux["weather_statistics_last"])

        # wind components + speed field at the selected level
        if level == "10m":
            ufield = outs[0, 1].cpu().numpy()
            vfield = outs[0, 2].cpu().numpy()
        else:
            li = eng.PRESSURE_LEVELS.index(int(level))
            ufield = out[0, 3, li].cpu().numpy()
            vfield = out[0, 4, li].cpu().numpy()
        wsfield = np.sqrt(ufield ** 2 + vfield ** 2)

        # farm-area wind = 3x3 cell neighbourhood mean (~75 km, robust to single-cell noise)
        pt = float(wsfield[max(0, fr - 1):fr + 2, max(0, fc - 1):fc + 2].mean())
        lv = level.lstrip("UV")
        if lv in wp.AGL_LEVELS:
            pt_hub = wp.extrapolate_to_hub(np.array([pt]), hub=farm["hub_height_m"],
                                           ref=wp.AGL_LEVELS[lv])[0]
        else:
            pt_hub = pt
        cf = float(wp.capacity_factor(np.array([pt_hub]),
                                      cut_in=farm["cut_in"], rated=farm["rated"],
                                      cut_out=farm["cut_out"])[0])
        times.append(t0 + pd.Timedelta(hours=24 * k))
        hub_ws_pt.append(float(pt_hub))
        cf_pt.append(cf)

        # store downscaled regional snapshot at selected leads
        if k in snap_days:
            su, extent = crop_bbox(ufield, farm["lat"], farm["lon"])
            sv, _ = crop_bbox(vfield, farm["lat"], farm["lon"])
            fine = None
            if use_corrdiff:
                if progress:
                    progress(f"CorrDiff 降尺度（regression+diffusion，18步采样，+{k*24}h，25km→5km）",
                             0.05 + 0.7 * k / horizon_days)
                try:
                    fine = cdi.downscale_speed(su, sv, device=device)
                except Exception as e:
                    print(f"[forecast] CorrDiff downscale failed ({e}); using bilinear", flush=True)
                    fine = None
            if fine is None:
                if progress:
                    progress(f"双线性插值降尺度（占位，+{k*24}h）", 0.05 + 0.7 * k / horizon_days)
                sub = np.sqrt(su ** 2 + sv ** 2)
                fine = wp.downscale(sub, factor=factor, method="bilinear")
            field_snaps[k] = (fine, extent)

        cur_u, cur_s = out, outs  # feed back

    if progress:
        progress("功率曲线换算与出力统计", 0.92)

    cap = farm["capacity_mw"]
    power_mw = [c * cap for c in cf_pt]
    daily_energy = [p * 24 for p in power_mw]  # MWh/day
    total_energy = float(np.sum(daily_energy))
    mean_cf = float(np.mean(cf_pt))

    return {
        "farm": farm,
        "init_date": init_date,
        "level": level,
        "level_details": level_details,
        "device": device,
        "times": times,
        "hub_ws": hub_ws_pt,
        "cf": cf_pt,
        "power_mw": power_mw,
        "daily_energy_mwh": daily_energy,
        "total_energy_mwh": total_energy,
        "mean_cf": mean_cf,
        "field_snaps": field_snaps,
        "horizon_days": horizon_days,
        "downscale_method": downscale_method,
        "pangu_model": model_label,
    }


# --------------------------- professional figures ---------------------------
def _lname(level):
    return {"10m": "10 m", "1000": "1000 hPa", "925": "925 hPa", "850": "850 hPa"}[level]


def fig_region_map(farm):
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    fig = plt.figure(figsize=(7, 4.6))
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    ax.set_extent([farm["lon"] - 6, farm["lon"] + 6, farm["lat"] - 5, farm["lat"] + 5],
                  crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor="#f2efe9")
    ax.add_feature(cfeature.OCEAN, facecolor="#dbeafe")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=":")
    ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5, linestyle="--")
    ax.plot(farm["lon"], farm["lat"], marker="*", markersize=20, color="crimson",
            transform=ccrs.PlateCarree(), zorder=5)
    ax.annotate(farm["id"], (farm["lon"], farm["lat"]), xytext=(6, 6),
                textcoords="offset points", fontsize=9, color="crimson", weight="bold")
    ax.set_title(f"Site location  {farm['id']}  "
                 f"({farm['lat']:.2f}N, {farm['lon']:.2f}E, {farm['elevation_m']} m)",
                 fontsize=10)
    return fig


def fig_wind_field(snap, farm, lead_day, level):
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    field, extent = snap
    fig = plt.figure(figsize=(6, 4.4))
    ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
    ax.set_extent(list(extent), crs=ccrs.PlateCarree())
    im = ax.imshow(field, origin="upper", extent=list(extent), transform=ccrs.PlateCarree(),
                   cmap="YlGnBu", aspect="auto")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=":")
    ax.plot(farm["lon"], farm["lat"], marker="*", markersize=16, color="crimson",
            transform=ccrs.PlateCarree(), zorder=5)
    ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.4, linestyle="--")
    cb = plt.colorbar(im, ax=ax, orientation="vertical", pad=0.03, shrink=0.85)
    cb.set_label("wind speed (m/s)", fontsize=8)
    ax.set_title(f"Downscaled {_lname(level)} wind  ·  lead +{lead_day*24}h", fontsize=10)
    return fig


def fig_timeseries(result):
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    t = result["times"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True)
    ax1.plot(t, result["hub_ws"], "-o", color="#1f77b4", lw=1.8, ms=4)
    ax1.axhline(result["farm"]["rated"], color="green", ls="--", lw=1, label="rated")
    ax1.axhline(result["farm"]["cut_in"], color="orange", ls="--", lw=1, label="cut-in")
    ax1.axhline(result["farm"]["cut_out"], color="red", ls="--", lw=1, label="cut-out")
    ax1.set_ylabel("hub wind speed (m/s)"); ax1.grid(alpha=0.3); ax1.legend(fontsize=8, ncol=3)
    ax1.set_title(f"{result['farm']['id']}  ·  {result['horizon_days']}-day forecast "
                  f"(from {result['init_date']})", fontsize=11)
    ax2.bar(t, result["power_mw"], width=0.6, color="#2ca02c", alpha=0.8)
    ax2.set_ylabel("power (MW)"); ax2.grid(alpha=0.3)
    ax2.set_ylim(0, result["farm"]["capacity_mw"] * 1.05)
    ax2.axhline(result["farm"]["capacity_mw"], color="gray", ls=":", lw=1, label="installed")
    ax2.legend(fontsize=8)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    fig.autofmt_xdate()
    return fig


def fig_power_curve(result):
    import matplotlib.pyplot as plt
    f = result["farm"]
    v = np.linspace(0, 30, 300)
    cf = wp.capacity_factor(v, cut_in=f["cut_in"], rated=f["rated"], cut_out=f["cut_out"])
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(v, cf * f["capacity_mw"], color="#444", lw=2, label="power curve")
    ax.scatter(result["hub_ws"], result["power_mw"], color="crimson", zorder=5,
               s=30, label="forecast points")
    ax.axvline(f["cut_in"], color="orange", ls="--", lw=1)
    ax.axvline(f["rated"], color="green", ls="--", lw=1)
    ax.axvline(f["cut_out"], color="red", ls="--", lw=1)
    ax.set_xlabel("hub wind speed (m/s)"); ax.set_ylabel("power (MW)")
    ax.set_title(f"Power curve  ·  {f['turbine_model']}  ·  {f['capacity_mw']:.0f} MW",
                 fontsize=10)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    return fig
