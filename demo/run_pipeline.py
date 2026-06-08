"""
Unattended pipeline: download multi-month ERA5 (2019), then fine-tune VAAWM,
then evaluate on a held-out test split. Designed to run for a few hours in the
background. Robust to transient download failures (per-file retries, resumable
because existing files are skipped).

Stays FAITHFUL to the repo's VAAWM objective (normalized wind-speed L1 + Xinjiang
mask), same as demo/finetune_vaawm.py. Only adds a proper temporal train/val/test
split with validation early stopping (the repo uses early stopping too).

Data layout produced under cfg.PG_INPUT_PATH (/opt/dlami/nvme):
  upper/upper_YYYYMMDD.nc   (00:00 UTC, single step, 13 levels, z/q/t/u/v)
  surface/surface_YYYYMM.nc (00:00 UTC daily steps, msl/u10/v10/t2m)

Split (temporal, paper-style):
  TRAIN 2019-01..2019-09   VAL 2019-10   TEST 2019-11..2019-12
"""
import os
import sys
import time
import json
import copy
import traceback

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import xarray as xr
import torch
from torch import nn

from era5_data.config import cfg
from era5_data import utils_data, score
from models.pangu_model import PanguModel
from models.pangu_sample import get_wind_speed

ARCO = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
UP_RENAME = {"geopotential": "z", "specific_humidity": "q", "temperature": "t",
             "u_component_of_wind": "u", "v_component_of_wind": "v"}
SF_RENAME = {"mean_sea_level_pressure": "msl", "10m_u_component_of_wind": "u10",
             "10m_v_component_of_wind": "v10", "2m_temperature": "t2m"}

UP_DIR = os.path.join(cfg.PG_INPUT_PATH, "upper")
SF_DIR = os.path.join(cfg.PG_INPUT_PATH, "surface")
OUT_DIR = os.path.join(cfg.PG_OUT_PATH, "finetune_vaawm", str(cfg.PG.HORIZON))
BEST_PATH = os.path.join(OUT_DIR, "vaawm_multimonth.pth")
STATUS = os.path.join(OUT_DIR, "pipeline_status.json")

YEAR = 2019
TRAIN_MONTHS = list(range(1, 10))   # Jan..Sep
VAL_MONTHS = [10]                   # Oct
TEST_MONTHS = [11, 12]              # Nov..Dec


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def set_status(**kw):
    os.makedirs(OUT_DIR, exist_ok=True)
    st = {}
    if os.path.exists(STATUS):
        try:
            st = json.load(open(STATUS))
        except Exception:
            st = {}
    st.update(kw)
    st["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(st, open(STATUS, "w"), indent=2)


def open_arco():
    return xr.open_zarr(ARCO, chunks=None, storage_options={"token": "anon"})


def fetch_all():
    os.makedirs(UP_DIR, exist_ok=True)
    os.makedirs(SF_DIR, exist_ok=True)
    days = pd.date_range(f"{YEAR}-01-01", f"{YEAR}-12-31", freq="1D")
    months = sorted(set(d.strftime("%Y%m") for d in days))

    ds = open_arco()
    ds_up = ds[list(UP_RENAME.keys())].sel(level=LEVELS).rename(UP_RENAME)
    ds_sf = ds[list(SF_RENAME.keys())].rename(SF_RENAME)

    # ---- upper: one file per day at 00:00 ----
    total = len(days)
    for i, d in enumerate(days):
        ds_str = d.strftime("%Y%m%d")
        out = os.path.join(UP_DIR, f"upper_{ds_str}.nc")
        if os.path.exists(out):
            try:
                x = xr.open_dataset(out); ok = x.time.size >= 1; x.close()
                if ok:
                    continue
            except Exception:
                pass
        ok = False
        for attempt in range(4):
            try:
                sub = ds_up.sel(time=[pd.Timestamp(d.year, d.month, d.day, 0)])
                sub = sub.sortby("level")
                if sub.latitude.values[0] < sub.latitude.values[-1]:
                    sub = sub.sortby("latitude", ascending=False)
                for v in ["z", "q", "t", "u", "v"]:
                    sub[v] = sub[v].astype("float32")
                sub.to_netcdf(out + ".tmp"); os.replace(out + ".tmp", out)
                ok = True
                break
            except Exception as e:
                log(f"upper {ds_str} attempt {attempt+1} failed: {e}")
                time.sleep(5 * (attempt + 1))
                try:
                    ds = open_arco()
                    ds_up = ds[list(UP_RENAME.keys())].sel(level=LEVELS).rename(UP_RENAME)
                    ds_sf = ds[list(SF_RENAME.keys())].rename(SF_RENAME)
                except Exception:
                    pass
        if (i + 1) % 20 == 0 or not ok:
            set_status(stage="download_upper", upper_done=i + 1, upper_total=total)
            log(f"upper progress {i+1}/{total}")

    # ---- surface: one file per month with daily 00:00 steps ----
    for mi, m in enumerate(months):
        out = os.path.join(SF_DIR, f"surface_{m}.nc")
        if os.path.exists(out):
            # keep any existing file (e.g. the original hourly 201907)
            continue
        mdays = [d for d in days if d.strftime("%Y%m") == m]
        times = [pd.Timestamp(d.year, d.month, d.day, 0) for d in mdays]
        for attempt in range(4):
            try:
                sub = ds_sf.sel(time=times)
                if sub.latitude.values[0] < sub.latitude.values[-1]:
                    sub = sub.sortby("latitude", ascending=False)
                for v in ["msl", "u10", "v10", "t2m"]:
                    sub[v] = sub[v].astype("float32")
                sub.to_netcdf(out + ".tmp"); os.replace(out + ".tmp", out)
                break
            except Exception as e:
                log(f"surface {m} attempt {attempt+1} failed: {e}")
                time.sleep(5 * (attempt + 1))
                try:
                    ds = open_arco()
                    ds_sf = ds[list(SF_RENAME.keys())].rename(SF_RENAME)
                except Exception:
                    pass
        set_status(stage="download_surface", surface_done=mi + 1, surface_total=len(months))
        log(f"surface {m} done ({mi+1}/{len(months)})")
    log("fetch_all done")


def split_inputs():
    """Input dates (YYYYMMDD 00:00) whose target (+24h, 00:00) also exists."""
    avail = set()
    for f in os.listdir(UP_DIR):
        if f.startswith("upper_") and f.endswith(".nc") and len(f) == 17:
            avail.add(f[6:14])

    def collect(months):
        out = []
        for d in pd.date_range(f"{YEAR}-01-01", f"{YEAR}-12-31", freq="1D"):
            if d.month not in months:
                continue
            ds_str = d.strftime("%Y%m%d")
            tg = (d + pd.Timedelta(days=1)).strftime("%Y%m%d")
            sf_in = os.path.join(SF_DIR, f"surface_{ds_str[:6]}.nc")
            sf_tg = os.path.join(SF_DIR, f"surface_{tg[:6]}.nc")
            if ds_str in avail and tg in avail and os.path.exists(sf_in) and os.path.exists(sf_tg):
                out.append(ds_str)
        return out

    return collect(TRAIN_MONTHS), collect(VAL_MONTHS), collect(TEST_MONTHS)


def load_pair(s, device):
    t = pd.to_datetime(s + "00", format="%Y%m%d%H")
    end = t + pd.Timedelta(hours=cfg.PG.HORIZON)
    def state(date, tt):
        su = xr.open_dataset(os.path.join(SF_DIR, f"surface_{date[:6]}.nc")).sel(time=tt)
        up = xr.open_dataset(os.path.join(UP_DIR, f"upper_{date}.nc")).sel(time=tt)
        upper = np.concatenate([up[v].values.astype(np.float32)[None] for v in ["z","q","t","u","v"]], 0)
        upper = upper[:, ::-1, :, :].copy()
        surf = np.concatenate([su[v].values.astype(np.float32)[None] for v in ["msl","u10","v10","t2m"]], 0)
        return upper, surf
    inp, inps = state(s, t)
    tgt, tgts = state(end.strftime("%Y%m%d"), end)
    g = lambda a: torch.from_numpy(a).unsqueeze(0)
    return g(inp).to(device), g(inps).to(device), g(tgt).to(device), g(tgts).to(device)


def vaawm_loss(crit, out, outs, tn, tsn, mb, vp):
    a, b, c, d = get_wind_speed(outs, tsn, out, tn)
    return (crit(a, b) * (~mb)).sum() / vp + (crit(c, d) * (~mb)).sum() / vp


def physical_region_rmse(outs_phys, tgts, mt):
    pws = torch.sqrt(outs_phys[0, 1] ** 2 + outs_phys[0, 2] ** 2)
    tws = torch.sqrt(tgts[0, 1] ** 2 + tgts[0, 2] ** 2)
    return (score.weighted_rmse_torch_channels(pws, tws, mt).item(),
            score.weighted_rmse_torch_channels(pws, tws).item())


def train(epochs=10, lr=5e-6, patience=3):
    import random
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    aux = utils_data.loadAllConstants(device=device)
    mask = aux["custom_mask"]; mb = mask == 0; vp = mask.sum(); mt = mask
    crit = nn.L1Loss(reduction="none")

    train_in, val_in, test_in = split_inputs()
    log(f"split: train={len(train_in)} val={len(val_in)} test={len(test_in)}")
    set_status(stage="train", train_n=len(train_in), val_n=len(val_in), test_n=len(test_in))

    model = PanguModel(device=device).to(device)
    model.load_state_dict(torch.load(cfg.PG.BENCHMARK.PRETRAIN_24_torch,
                                     weights_only=True, map_location=device)["model"])
    for p in model.parameters():
        p.requires_grad = True
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=cfg.PG.TRAIN.WEIGHT_DECAY)

    best_val = float("inf"); best_state = None; since = 0
    for ep in range(1, epochs + 1):
        model.train(); random.shuffle(train_in); tr = 0.0; t0 = time.time()
        for s in train_in:
            inp, inps, tgt, tgts = load_pair(s, device)
            opt.zero_grad()
            out, outs = model(inp, inps, aux["weather_statistics"], aux["constant_maps"], aux["const_h"])
            tn, tsn = utils_data.normData(tgt, tgts, aux["weather_statistics_last"])
            loss = vaawm_loss(crit, out, outs, tn, tsn, mb, vp)
            loss.backward(); opt.step(); tr += loss.item()
        tr /= max(len(train_in), 1)

        model.eval(); vl = 0.0
        with torch.no_grad():
            for s in val_in:
                inp, inps, tgt, tgts = load_pair(s, device)
                out, outs = model(inp, inps, aux["weather_statistics"], aux["constant_maps"], aux["const_h"])
                tn, tsn = utils_data.normData(tgt, tgts, aux["weather_statistics_last"])
                vl += vaawm_loss(crit, out, outs, tn, tsn, mb, vp).item()
        vl /= max(len(val_in), 1)
        log(f"epoch {ep}: train={tr:.5f} val={vl:.5f} time={time.time()-t0:.0f}s")
        set_status(stage="train", epoch=ep, train_loss=tr, val_loss=vl, best_val=best_val)
        if vl < best_val - 1e-6:
            best_val = vl; best_state = copy.deepcopy(model.state_dict()); since = 0
            torch.save({"model": best_state, "val_loss": best_val}, BEST_PATH)
            log(f"  -> new best val={vl:.5f} saved")
        else:
            since += 1
            if since >= patience:
                log(f"early stop at epoch {ep}"); break

    # ---- held-out test eval (faithful: zero-shot vs VAAWM, physical 10m wind RMSE) ----
    log("evaluating on held-out test set ...")
    base = PanguModel(device=device).to(device)
    base.load_state_dict(torch.load(cfg.PG.BENCHMARK.PRETRAIN_24_torch,
                                    weights_only=True, map_location=device)["model"]); base.eval()
    model.load_state_dict(best_state if best_state is not None else model.state_dict()); model.eval()
    rz, rv, gz, gv = [], [], [], []
    with torch.no_grad():
        for s in test_in:
            inp, inps, tgt, tgts = load_pair(s, device)
            ob, osb = base(inp, inps, aux["weather_statistics"], aux["constant_maps"], aux["const_h"])
            ov, osv = model(inp, inps, aux["weather_statistics"], aux["constant_maps"], aux["const_h"])
            ob, osb = utils_data.normBackData(ob, osb, aux["weather_statistics_last"])
            ov, osv = utils_data.normBackData(ov, osv, aux["weather_statistics_last"])
            r0, g0 = physical_region_rmse(osb, tgts, mt)
            r1, g1 = physical_region_rmse(osv, tgts, mt)
            rz.append(r0); rv.append(r1); gz.append(g0); gv.append(g1)
    res = {
        "test_samples": len(test_in),
        "zeroshot_region_rmse": float(np.mean(rz)) if rz else None,
        "vaawm_region_rmse": float(np.mean(rv)) if rv else None,
        "zeroshot_global_rmse": float(np.mean(gz)) if gz else None,
        "vaawm_global_rmse": float(np.mean(gv)) if gv else None,
        "best_val_loss": best_val,
    }
    if rz:
        res["region_improvement_pct"] = (np.mean(rz) - np.mean(rv)) / np.mean(rz) * 100
    log("TEST RESULT: " + json.dumps(res))
    set_status(stage="done", result=res, checkpoint=BEST_PATH)
    json.dump(res, open(os.path.join(OUT_DIR, "multimonth_test_result.json"), "w"), indent=2)


def main():
    try:
        set_status(stage="start")
        log("=== pipeline start ===")
        fetch_all()
        train(epochs=10, lr=5e-6, patience=3)
        log("=== pipeline done ===")
    except Exception as e:
        log("PIPELINE ERROR: " + str(e))
        log(traceback.format_exc())
        set_status(stage="error", error=str(e))
        raise


if __name__ == "__main__":
    main()
