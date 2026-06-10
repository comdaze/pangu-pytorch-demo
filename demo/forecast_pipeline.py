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


def _available_init_dates():
    """Dates for which we have BOTH input and upper ERA5 initial fields."""
    return sorted(set(eng.list_input_dates()) & set(eng._all_upper_dates()))


def pick_init_date(as_of=None):
    """Pick the analysis date closest in season to `as_of` (default: today).

    The demo only has historical ERA5 initial fields (2016-2019); a live NWP
    feed is not connected. So to honour "today's forecast" we initialise from
    the available reanalysis field whose month/day is closest to today's,
    preferring the most recent year. This keeps the seasonal regime correct.
    """
    avail = _available_init_dates()
    if not avail:
        return "20190701"
    today = as_of or dt.date.today()
    # exact month-day match -> most recent year
    mmdd = today.strftime("%m%d")
    same = [d for d in avail if d[4:] == mmdd]
    if same:
        return sorted(same)[-1]
    # otherwise nearest by day-of-year (season), preferring the most recent year
    best, best_key = None, None
    for d in avail:
        dd = dt.datetime.strptime(d, "%Y%m%d").date()
        diff = abs((dd.replace(year=2000) - today.replace(year=2000)).days)
        diff = min(diff, 365 - diff)
        key = (-diff, d)  # smaller diff first, then most recent date
        if best_key is None or key > best_key:
            best, best_key = d, key
    return best


def run_forecast(farm, horizon_days=7, init_date=None, factor=5, progress=None,
                 step_hours=24):
    """Run the full pipeline. Returns a dict of time series + metadata.

    step_hours: rollout granularity.
        24 -> daily, uses the paper's *hybrid inference* (target wind from the
              VAAWM fine-tune, the rest from the zero-shot base).
        1/3/6 -> finer, uses the official *zero-shot* Pangu model of that horizon
              (we only have a 24h fine-tune, so sub-daily steps are zero-shot).
    progress: optional callable(stage_str, frac) for UI streaming.
    """
    device = eng.get_device()  # 'cpu' when CUDA hidden
    as_of = dt.date.today()
    if init_date is None:
        init_date = pick_init_date(as_of)
        init_note = (f"最接近当前日期 {as_of.strftime('%m-%d')} 的可用 ERA5 再分析场"
                     f"（历史档案 2016–2019；实时 NWP 数据源未接入）")
    else:
        init_note = "用户指定的初始场日期"

    step_hours = step_hours if step_hours in (1, 3, 6, 24) else 24
    total_hours = max(1, int(horizon_days)) * 24
    n_steps = max(1, total_hours // step_hours)
    MAX_STEPS = 56  # bound runtime for the demo (~1s/step/model on GPU)
    n_steps = min(n_steps, MAX_STEPS)

    # altitude-aware level selection (Pangu levels)
    level, level_details = wp.select_wind_level(
        farm["elevation_m"], farm["hub_height_m"], eng.PANGU_WIND_LEVELS)

    # Daily (24h) -> hybrid inference (paper §3.4): target wind channels from the
    # VAAWM fine-tune, all other variables from the zero-shot base, to keep
    # localized wind accuracy while preventing drift. Sub-daily -> zero-shot.
    zs_horizons = eng.zeroshot_horizons_available()
    use_hybrid = (step_hours == 24) and (eng.pre2019_available() or eng.finetuned_available())
    if use_hybrid:
        base = eng.load_model(device, "zeroshot")
        if eng.pre2019_available():
            ft = eng.load_model(device, "vaawm_pre2019"); ft_label = "Pre-2019 VAAWM"
        else:
            ft = eng.load_model(device, "vaawm"); ft_label = "VAAWM 微调"
        aux = eng.load_aux(device)
        model_label = (f"混合推理（base zero-shot Pangu-24h + {ft_label}，"
                       f"目标风场用微调，余用基座；逐24h）")
    else:
        zs_h = step_hours if step_hours in zs_horizons else 24
        zs = eng.load_model(device, f"zs{zs_h}")
        aux = eng.load_aux_h(device, zs_h)
        ft = None
        model_label = f"zero-shot Pangu-{zs_h}h 自回归（逐{zs_h}h）"

    if progress:
        progress(f"加载 Pangu 模型：{model_label}（{device}）", 0.02)

    cur_u, cur_s = eng._state_at(init_date, device)
    t0 = pd.to_datetime(init_date, format="%Y%m%d")

    fr, fc = latlon_to_idx(farm["lat"], farm["lon"])
    times, hub_ws_pt, cf_pt = [], [], []
    field_snaps = {}  # lead_hours -> (downscaled_field, extent)
    snap_steps = sorted(set([1, max(1, n_steps // 2), n_steps]))
    use_corrdiff = bool(cdi) and cdi.available()
    downscale_method = "CorrDiff (regression+diffusion)" if use_corrdiff else "bilinear (placeholder)"

    for k in range(1, n_steps + 1):
        lead_h = step_hours * k
        if progress:
            if use_hybrid:
                step_desc = (
                    f"混合推理 第 {k}/{n_steps} 步 (+{lead_h}h)："
                    f"① 基座 Pangu-24h(zero-shot) 预报全场(z/q/t/msl/t2m)；"
                    f"② {ft_label} 预报目标风场(u/v·u10/v10)；"
                    f"③ 融合(风场取微调，余取基座)→喂回下一步"
                )
            else:
                step_desc = (f"zero-shot Pangu-{step_hours}h 第 {k}/{n_steps} 步 "
                             f"(+{lead_h}h) 自回归预报")
            progress(step_desc, 0.05 + 0.7 * k / n_steps)
        with torch.no_grad():
            if use_hybrid:
                ob, osb = base(cur_u, cur_s, aux["weather_statistics"],
                               aux["constant_maps"], aux["const_h"])
                ob, osb = utils_data.normBackData(ob, osb, aux["weather_statistics_last"])
                ov, osv = ft(cur_u, cur_s, aux["weather_statistics"],
                             aux["constant_maps"], aux["const_h"])
                ov, osv = utils_data.normBackData(ov, osv, aux["weather_statistics_last"])
                out, outs = ob.clone(), osb.clone()
                for ch in eng.TARGET_SURFACE_CH:
                    outs[:, ch] = osv[:, ch]
                for ch in eng.TARGET_UPPER_CH:
                    out[:, ch] = ov[:, ch]
            else:
                out, outs = zs(cur_u, cur_s, aux["weather_statistics"],
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

        # farm-area wind = 3x3 cell neighbourhood mean (~75 km, robust to noise)
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
        times.append(t0 + pd.Timedelta(hours=lead_h))
        hub_ws_pt.append(float(pt_hub))
        cf_pt.append(cf)

        # store downscaled regional snapshot at selected leads
        if k in snap_steps:
            su, extent = crop_bbox(ufield, farm["lat"], farm["lon"])
            sv, _ = crop_bbox(vfield, farm["lat"], farm["lon"])
            fine = None
            if use_corrdiff:
                if progress:
                    progress(f"CorrDiff 降尺度（regression+diffusion，18步采样，+{lead_h}h，25km→5km）",
                             0.05 + 0.7 * k / n_steps)
                try:
                    fine = cdi.downscale_speed(su, sv, device=device)
                except Exception as e:
                    print(f"[forecast] CorrDiff downscale failed ({e}); using bilinear", flush=True)
                    fine = None
            if fine is None:
                if progress:
                    progress(f"双线性插值降尺度（占位，+{lead_h}h）", 0.05 + 0.7 * k / n_steps)
                sub = np.sqrt(su ** 2 + sv ** 2)
                fine = wp.downscale(sub, factor=factor, method="bilinear")
            field_snaps[lead_h] = (fine, extent)

        cur_u, cur_s = out, outs  # feed back

    if progress:
        progress("功率曲线换算与出力统计", 0.92)

    cap = farm["capacity_mw"]
    power_mw = [c * cap for c in cf_pt]
    step_energy = [p * step_hours for p in power_mw]  # MWh per step
    total_energy = float(np.sum(step_energy))
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
        "daily_energy_mwh": step_energy,
        "total_energy_mwh": total_energy,
        "mean_cf": mean_cf,
        "field_snaps": field_snaps,
        "horizon_days": horizon_days,
        "step_hours": step_hours,
        "n_steps": n_steps,
        "init_note": init_note,
        "as_of": as_of.strftime("%Y-%m-%d"),
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


def fig_wind_field(snap, farm, lead_h, level):
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
    ax.set_title(f"Downscaled {_lname(level)} wind  ·  lead +{lead_h}h", fontsize=10)
    return fig


def fig_timeseries(result):
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    t = result["times"]
    step_h = result.get("step_hours", 24)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True)
    ax1.plot(t, result["hub_ws"], "-o", color="#1f77b4", lw=1.8, ms=4)
    ax1.axhline(result["farm"]["rated"], color="green", ls="--", lw=1, label="rated")
    ax1.axhline(result["farm"]["cut_in"], color="orange", ls="--", lw=1, label="cut-in")
    ax1.axhline(result["farm"]["cut_out"], color="red", ls="--", lw=1, label="cut-out")
    ax1.set_ylabel("hub wind speed (m/s)"); ax1.grid(alpha=0.3); ax1.legend(fontsize=8, ncol=3)
    res_txt = "daily" if step_h == 24 else f"{step_h}-hourly"
    ax1.set_title(f"{result['farm']['id']}  ·  {result['horizon_days']}-day forecast "
                  f"({res_txt}, from {result['init_date']})", fontsize=11)
    bar_w = max(0.02, step_h / 24.0 * 0.6)
    ax2.bar(t, result["power_mw"], width=bar_w, color="#2ca02c", alpha=0.8)
    ax2.set_ylabel("power (MW)"); ax2.grid(alpha=0.3)
    ax2.set_ylim(0, result["farm"]["capacity_mw"] * 1.05)
    ax2.axhline(result["farm"]["capacity_mw"], color="gray", ls=":", lw=1, label="installed")
    ax2.legend(fontsize=8)
    fmt = "%m-%d" if step_h == 24 else "%m-%d %Hh"
    ax2.xaxis.set_major_formatter(mdates.DateFormatter(fmt))
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
