"""
Unattended Pre-2019 reproduction pipeline (paper protocol):
  TRAIN 2016-2017   VAL 2019   TEST 2018   (cross-year, as in the paper)

Steps:
  1. Download ERA5 upper (00:00 daily) for 2016, 2017, 2018 from ARCO-ERA5
     (2019 already present). Build surface monthly files for those years.
  2. Fine-tune Pangu-Weather with a FAITHFUL VAAWM loss:
       - FIXED wind-emphasis alpha (softmax of wind=2/others=0; learnable alpha
         was observed to "game" and abandon wind, so we fix it = user priority),
       - soft region weight beta (Xinjiang=1, outside=0.05; nonzero -> keep global),
       - full-variable MSE (normalized space) -> avoids catastrophic forgetting.
  3. Model selection / early stopping on VALIDATION wind-speed RMSE (paper Sec 4.1).
  4. Evaluate on 2018 test: zero-shot vs VAAWM vs hybrid (region + global wind, t2m).

Robust to transient failures (per-file retries, atomic writes, resumable).
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

from era5_data.config import cfg
from era5_data import utils_data, score
from models.pangu_model import PanguModel

import run_pipeline as rp  # reuse ARCO consts, load_pair, dirs

OUT_DIR = os.path.join(cfg.PG_OUT_PATH, "finetune_vaawm", str(cfg.PG.HORIZON))
BEST_PATH = os.path.join(OUT_DIR, "vaawm_pre2019.pth")
STATUS = os.path.join(OUT_DIR, "pre2019_status.json")
RESULT = os.path.join(OUT_DIR, "pre2019_test_result.json")

TRAIN_YEARS = [2016, 2017]
VAL_YEAR = 2019
TEST_YEAR = 2018
L850 = 2


def log(m):
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] {m}", flush=True)


def set_status(**kw):
    os.makedirs(OUT_DIR, exist_ok=True)
    st = {}
    if os.path.exists(STATUS):
        try: st = json.load(open(STATUS))
        except Exception: st = {}
    st.update(kw); st["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(st, open(STATUS, "w"), indent=2)


# ----------------------------------------------------------------- fetch ----
def fetch_year(ds_up, ds_sf, year):
    days = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="1D")
    # upper: one file/day at 00:00
    for i, d in enumerate(days):
        dstr = d.strftime("%Y%m%d")
        out = os.path.join(rp.UP_DIR, f"upper_{dstr}.nc")
        if os.path.exists(out):
            try:
                x = xr.open_dataset(out); ok = x.time.size >= 1; x.close()
                if ok: continue
            except Exception: pass
        for attempt in range(4):
            try:
                sub = ds_up.sel(time=[pd.Timestamp(d.year, d.month, d.day, 0)]).sortby("level")
                if sub.latitude.values[0] < sub.latitude.values[-1]:
                    sub = sub.sortby("latitude", ascending=False)
                for v in ["z", "q", "t", "u", "v"]:
                    sub[v] = sub[v].astype("float32")
                sub.to_netcdf(out + ".tmp"); os.replace(out + ".tmp", out)
                break
            except Exception as e:
                log(f"upper {dstr} retry {attempt+1}: {e}"); time.sleep(5 * (attempt + 1))
                ds_up, ds_sf = open_arco_vars()
        if (i + 1) % 30 == 0:
            set_status(stage=f"download_upper_{year}", done=i + 1, total=len(days))
            log(f"upper {year}: {i+1}/{len(days)}")
    # surface: one file/month with daily 00:00 steps
    months = sorted(set(d.strftime("%Y%m") for d in days))
    for m in months:
        out = os.path.join(rp.SF_DIR, f"surface_{m}.nc")
        if os.path.exists(out):
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
                log(f"surface {m} retry {attempt+1}: {e}"); time.sleep(5 * (attempt + 1))
                ds_up2, ds_sf = open_arco_vars()
        log(f"surface {m} done")


def open_arco_vars():
    ds = rp.open_arco()
    up = ds[list(rp.UP_RENAME.keys())].sel(level=rp.LEVELS).rename(rp.UP_RENAME)
    sf = ds[list(rp.SF_RENAME.keys())].rename(rp.SF_RENAME)
    return up, sf


def fetch_all():
    up, sf = open_arco_vars()
    for y in TRAIN_YEARS + [TEST_YEAR]:
        log(f"=== fetch year {y} ===")
        fetch_year(up, sf, y)
    log("fetch_all done")


# ----------------------------------------------------------------- data -----
def dates_for_year(year):
    out = []
    for d in pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="1D"):
        ds_str = d.strftime("%Y%m%d")
        tg = (d + pd.Timedelta(days=1)).strftime("%Y%m%d")
        if (os.path.exists(os.path.join(rp.UP_DIR, f"upper_{ds_str}.nc")) and
                os.path.exists(os.path.join(rp.UP_DIR, f"upper_{tg}.nc")) and
                os.path.exists(os.path.join(rp.SF_DIR, f"surface_{ds_str[:6]}.nc")) and
                os.path.exists(os.path.join(rp.SF_DIR, f"surface_{tg[:6]}.nc"))):
            out.append(ds_str)
    return out


def build_beta(mask, outside=0.05):
    return mask * (1.0 - outside) + outside


def region_wind_rmse(osurf, oup, tgts, tgt, mt):
    w10p = torch.sqrt(osurf[0, 1] ** 2 + osurf[0, 2] ** 2)
    w10t = torch.sqrt(tgts[0, 1] ** 2 + tgts[0, 2] ** 2)
    w8p = torch.sqrt(oup[0, 3, L850] ** 2 + oup[0, 4, L850] ** 2)
    w8t = torch.sqrt(tgt[0, 3, L850] ** 2 + tgt[0, 4, L850] ** 2)
    return 0.5 * (score.weighted_rmse_torch_channels(w10p, w10t, mt).item() +
                  score.weighted_rmse_torch_channels(w8p, w8t, mt).item())


# ---------------------------------------------------------------- train -----
def train(epochs=10, lr=5e-6, patience=3, beta_outside=0.05):
    import random
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    aux = utils_data.loadAllConstants(device=device)
    mask = aux["custom_mask"]; mt = mask
    beta = build_beta(mask, beta_outside).to(device); bsum = beta.sum()

    # FIXED wind-emphasis alpha (softmax of wind=2 / others=0)
    logits = torch.zeros(9, device=device)
    for i in [3, 4, 6, 7]:  # upper u,v ; surface u10,v10
        logits[i] = 2.0
    alpha = torch.softmax(logits, dim=0)
    log(f"fixed alpha[z,q,t,u,v,msl,u10,v10,t2m]={np.round(alpha.cpu().numpy(),3).tolist()}")

    train_in = []
    for y in TRAIN_YEARS:
        train_in += dates_for_year(y)
    val_in = dates_for_year(VAL_YEAR)
    test_in = dates_for_year(TEST_YEAR)
    # subsample val/test for speed
    val_sub = val_in[::10]
    test_sub = test_in[::6]
    log(f"split: train={len(train_in)} val={len(val_in)}(use {len(val_sub)}) "
        f"test={len(test_in)}(use {len(test_sub)})")
    set_status(stage="train", train_n=len(train_in), val_n=len(val_sub), test_n=len(test_sub))

    model = PanguModel(device=device).to(device)
    model.load_state_dict(torch.load(cfg.PG.BENCHMARK.PRETRAIN_24_torch,
                                     weights_only=True, map_location=device)["model"])
    for p in model.parameters():
        p.requires_grad = True
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=cfg.PG.TRAIN.WEIGHT_DECAY)

    def var_loss(out, outs, tn, tsn):
        ls = []
        for k in range(5):
            se = (out[0, k] - tn[0, k]) ** 2
            ls.append((se * beta).sum() / (bsum * se.shape[0]))
        for k in range(4):
            se = (outs[0, k] - tsn[0, k]) ** 2
            ls.append((se * beta).sum() / bsum)
        return (alpha * torch.stack(ls)).sum()

    best = float("inf"); best_state = None; since = 0
    for ep in range(1, epochs + 1):
        model.train(); random.shuffle(train_in); tr = 0.0; t0 = time.time()
        for s in train_in:
            inp, inps, tgt, tgts = rp.load_pair(s, device)
            opt.zero_grad()
            out, outs = model(inp, inps, aux["weather_statistics"], aux["constant_maps"], aux["const_h"])
            tn, tsn = utils_data.normData(tgt, tgts, aux["weather_statistics_last"])
            loss = var_loss(out, outs, tn, tsn)
            loss.backward(); opt.step(); tr += loss.item()
        tr /= max(len(train_in), 1)

        model.eval(); vr = 0.0
        with torch.no_grad():
            for s in val_sub:
                inp, inps, tgt, tgts = rp.load_pair(s, device)
                out, outs = model(inp, inps, aux["weather_statistics"], aux["constant_maps"], aux["const_h"])
                out, outs = utils_data.normBackData(out, outs, aux["weather_statistics_last"])
                vr += region_wind_rmse(outs, out, tgts, tgt, mt)
        vr /= max(len(val_sub), 1)
        log(f"epoch {ep}: train_loss={tr:.6f} val_windRMSE={vr:.5f} time={time.time()-t0:.0f}s")
        set_status(stage="train", epoch=ep, train_loss=tr, val_wind_rmse=vr, best_val=best)
        if vr < best - 1e-6:
            best = vr; best_state = copy.deepcopy(model.state_dict()); since = 0
            torch.save({"model": best_state, "val_wind_rmse": vr,
                        "alpha": alpha.cpu().numpy().tolist()}, BEST_PATH)
            log(f"  -> new best {vr:.5f} saved")
        else:
            since += 1
            if since >= patience:
                log(f"early stop epoch {ep}"); break

    # ---- test eval on 2018 ----
    log("=== test eval on 2018 ===")
    base = PanguModel(device=device).to(device)
    base.load_state_dict(torch.load(cfg.PG.BENCHMARK.PRETRAIN_24_torch,
                                    weights_only=True, map_location=device)["model"]); base.eval()
    model.load_state_dict(best_state if best_state is not None else model.state_dict()); model.eval()

    def w10(x): return torch.sqrt(x[0, 1] ** 2 + x[0, 2] ** 2)
    A = {k: [] for k in ["zs_r", "vw_r", "zs_g", "vw_g", "zs_t2m_g", "vw_t2m_g"]}
    with torch.no_grad():
        for s in test_sub:
            inp, inps, tgt, tgts = rp.load_pair(s, device)
            ob, osb = base(inp, inps, aux["weather_statistics"], aux["constant_maps"], aux["const_h"])
            ov, osv = model(inp, inps, aux["weather_statistics"], aux["constant_maps"], aux["const_h"])
            ob, osb = utils_data.normBackData(ob, osb, aux["weather_statistics_last"])
            ov, osv = utils_data.normBackData(ov, osv, aux["weather_statistics_last"])
            tw = w10(tgts)
            A["zs_r"].append(score.weighted_rmse_torch_channels(w10(osb), tw, mt).item())
            A["vw_r"].append(score.weighted_rmse_torch_channels(w10(osv), tw, mt).item())
            A["zs_g"].append(score.weighted_rmse_torch_channels(w10(osb), tw).item())
            A["vw_g"].append(score.weighted_rmse_torch_channels(w10(osv), tw).item())
            A["zs_t2m_g"].append(score.weighted_rmse_torch_channels(osb[0, 3], tgts[0, 3]).item())
            A["vw_t2m_g"].append(score.weighted_rmse_torch_channels(osv[0, 3], tgts[0, 3]).item())
    res = {k: float(np.mean(v)) for k, v in A.items() if v}
    if res.get("zs_r"):
        res["region_wind_improvement_pct"] = (res["zs_r"] - res["vw_r"]) / res["zs_r"] * 100
    res["best_val_wind_rmse"] = best
    res["hybrid_t2m_g"] = res.get("zs_t2m_g")  # hybrid non-target = base
    log("TEST RESULT: " + json.dumps(res))
    set_status(stage="done", result=res, checkpoint=BEST_PATH)
    json.dump(res, open(RESULT, "w"), indent=2)


def main():
    try:
        set_status(stage="start")
        log("=== Pre-2019 pipeline start ===")
        fetch_all()
        train(epochs=10, lr=5e-6, patience=3)
        log("=== Pre-2019 pipeline done ===")
    except Exception as e:
        log("PIPELINE ERROR: " + str(e)); log(traceback.format_exc())
        set_status(stage="error", error=str(e)); raise


if __name__ == "__main__":
    main()
