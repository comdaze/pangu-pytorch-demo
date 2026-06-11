"""
Real Pangu-Weather inference engine for the demo.

Loads the pretrained torch checkpoint (pangu_weather_24_torch.pth), the aux
constants, and runs a genuine forward pass on real ERA5 sample data
(NetCDF files shipped in the HuggingFace dataset). No mock numbers here.

Region-masked, latitude-weighted RMSE is computed with the same
era5_data.score function used by the training/eval code.
"""

import os
import sys
import glob
import time

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from era5_data.config import cfg
from era5_data import utils_data, score
from models.pangu_model import PanguModel
import paper_data as pdata
import regions as reg

# ERA5 pressure levels in descending order (matches nctonumpy output ordering).
PRESSURE_LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
SURFACE_IDX = {"msl": 0, "u10": 1, "v10": 2, "t2m": 3}

# Target (wind) variables that the hybrid strategy takes from the fine-tuned model.
# Surface u10/v10 (channels 1,2) and upper-air u/v (channels 3,4).
TARGET_SURFACE_CH = [1, 2]
TARGET_UPPER_CH = [3, 4]


def get_device():
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def gpu_name():
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "CPU"


def data_ready():
    """Check that weights + aux + at least one sample pair are present."""
    if not os.path.exists(cfg.PG.BENCHMARK.PRETRAIN_24_torch):
        return False, "缺少预训练权重 pangu_weather_24_torch.pth"
    aux = os.path.join(cfg.PG_INPUT_PATH, "aux_data")
    if not os.path.exists(os.path.join(aux, "custom_mask.npy")):
        return False, "缺少 aux_data/custom_mask.npy"
    if not list_input_dates():
        return False, "缺少可用的 ERA5 样本（surface/upper 的 .nc 文件）"
    return True, "ok"


def list_input_dates():
    """Return input dates (YYYYMMDD) for which both input and target (+horizon) exist."""
    upper_dir = os.path.join(cfg.PG_INPUT_PATH, "upper")
    surface_dir = os.path.join(cfg.PG_INPUT_PATH, "surface")
    if not os.path.isdir(upper_dir):
        return []
    upper_dates = set()
    for p in glob.glob(os.path.join(upper_dir, "upper_*.nc")):
        d = os.path.basename(p).replace("upper_", "").replace(".nc", "")
        if len(d) == 8:
            upper_dates.add(d)
    surface_months = set()
    for p in glob.glob(os.path.join(surface_dir, "surface_*.nc")):
        m = os.path.basename(p).replace("surface_", "").replace(".nc", "")
        surface_months.add(m)

    import pandas as pd
    valid = []
    for d in sorted(upper_dates):
        dt = pd.to_datetime(d, format="%Y%m%d")
        tgt = dt + pd.Timedelta(hours=cfg.PG.HORIZON)
        tgt_d = tgt.strftime("%Y%m%d")
        if tgt_d in upper_dates and dt.strftime("%Y%m") in surface_months \
                and tgt.strftime("%Y%m") in surface_months:
            valid.append(d)
    return valid


# ---- cached heavy objects (loaded lazily on first use) -------------------
_MODELS = {}
_AUX = None

FINETUNED_PATH = os.path.join(cfg.PG_OUT_PATH, "finetune_vaawm",
                              str(cfg.PG.HORIZON), "vaawm_finetuned.pth")
# Pre-2019 protocol VAAWM model (trained 2016-2017, validated on 2019, held-out
# test 2018). This is the multi-month model that took ~8h to train.
PRE2019_PATH = os.path.join(cfg.PG_OUT_PATH, "finetune_vaawm",
                            str(cfg.PG.HORIZON), "vaawm_pre2019.pth")

_VARIANT_CKPT = {
    "vaawm": FINETUNED_PATH,
    "vaawm_pre2019": PRE2019_PATH,
    # zero-shot official pretrained models per forecast horizon (hours)
    "zs1": cfg.PG.BENCHMARK.PRETRAIN_1_torch,
    "zs3": cfg.PG.BENCHMARK.PRETRAIN_3_torch,
    "zs6": cfg.PG.BENCHMARK.PRETRAIN_6_torch,
    "zs24": cfg.PG.BENCHMARK.PRETRAIN_24_torch,
}

# Map forecast horizon (h) -> zero-shot torch checkpoint path.
ZS_HORIZON_CKPT = {
    1: cfg.PG.BENCHMARK.PRETRAIN_1_torch,
    3: cfg.PG.BENCHMARK.PRETRAIN_3_torch,
    6: cfg.PG.BENCHMARK.PRETRAIN_6_torch,
    24: cfg.PG.BENCHMARK.PRETRAIN_24_torch,
}


def finetuned_available():
    return os.path.exists(FINETUNED_PATH)


def pre2019_available():
    return os.path.exists(PRE2019_PATH)


def zeroshot_horizons_available():
    """Return the sorted list of horizons (h) whose zero-shot torch model exists."""
    return sorted(h for h, p in ZS_HORIZON_CKPT.items() if os.path.exists(p))


def load_model(device, variant="zeroshot"):
    """variant: 'zeroshot' (official pretrained), 'vaawm' (small fine-tune) or
    'vaawm_pre2019' (the multi-month Pre-2019 VAAWM model)."""
    if variant not in _MODELS:
        m = PanguModel(device=device).to(device)
        ckpt_path = _VARIANT_CKPT.get(variant)
        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path, weights_only=True, map_location=device)
        else:
            ckpt = torch.load(cfg.PG.BENCHMARK.PRETRAIN_24_torch,
                              weights_only=True, map_location=device)
        m.load_state_dict(ckpt["model"])
        m.eval()
        _MODELS[variant] = m
    return _MODELS[variant]


def load_aux(device):
    global _AUX
    if _AUX is None:
        _AUX = utils_data.loadAllConstants(device=device)
    return _AUX


_AUX_H = {}


def load_aux_h(device, horizon):
    """Aux constants with the constant_maps mask for a specific horizon (h).

    All other constants are horizon-independent; only constantMask{h}.npy differs.
    """
    if horizon not in _AUX_H:
        import numpy as _np
        base = dict(load_aux(device))  # shallow copy
        mp = os.path.join(cfg.PG_INPUT_PATH, "aux_data", f"constantMask{horizon}.npy")
        base["constant_maps"] = torch.from_numpy(
            _np.load(mp).astype("float32")).to(device)
        _AUX_H[horizon] = base
    return _AUX_H[horizon]


def run_inference(input_date, device=None, variant="zeroshot"):
    """Run a real forward pass for the given input date (YYYYMMDD, 00:00 UTC).

    Returns a dict with denormalized prediction/target fields (numpy) and timing.
    """
    device = device or get_device()
    model = load_model(device, variant)
    aux = load_aux(device)

    ds = utils_data.NetCDFDataset(
        nc_path=cfg.PG_INPUT_PATH, training=False, validation=False,
        startDate=f"{input_date} 00:00:00", endDate=f"{input_date} 00:00:00",
        freq="24h", horizon=cfg.PG.HORIZON)
    import pandas as pd
    key = pd.to_datetime(input_date, format="%Y%m%d")
    inp, inps, tgt, tgts, periods = ds.LoadData(key)

    inp_t = torch.from_numpy(inp).unsqueeze(0).to(device)
    inps_t = torch.from_numpy(inps).unsqueeze(0).to(device)

    t0 = time.time()
    with torch.no_grad():
        out, outs = model(inp_t, inps_t, aux["weather_statistics"],
                          aux["constant_maps"], aux["const_h"])
        out, outs = utils_data.normBackData(out, outs, aux["weather_statistics_last"])
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    return {
        "periods": periods,
        "elapsed": elapsed,
        "pred_surface": outs[0].detach().cpu().numpy(),   # (4, 721, 1440)
        "pred_upper": out[0].detach().cpu().numpy(),       # (5, 13, 721, 1440)
        "tgt_surface": tgts,                                # (4, 721, 1440) numpy
        "tgt_upper": tgt,                                   # (5, 13, 721, 1440) numpy
        "input_surface": inps,
    }


def _wind(u, v):
    return np.sqrt(u ** 2 + v ** 2)


def build_hybrid(base_result, vaawm_result):
    """Hybrid inference (paper Sec 3.4): target wind variables come from the
    fine-tuned model, all other variables come from the base model.

    Returns a result dict in the same format as run_inference.
    """
    hs = base_result["pred_surface"].copy()
    hu = base_result["pred_upper"].copy()
    for ch in TARGET_SURFACE_CH:
        hs[ch] = vaawm_result["pred_surface"][ch]
    for ch in TARGET_UPPER_CH:
        hu[ch] = vaawm_result["pred_upper"][ch]
    return {
        "periods": base_result["periods"],
        "elapsed": base_result["elapsed"] + vaawm_result["elapsed"],
        "pred_surface": hs,
        "pred_upper": hu,
        "tgt_surface": base_result["tgt_surface"],
        "tgt_upper": base_result["tgt_upper"],
        "input_surface": base_result["input_surface"],
    }


def get_field(result, kind):
    """Return (pred_field, target_field, unit) for a named output."""
    ps, ts = result["pred_surface"], result["tgt_surface"]
    pu, tu = result["pred_upper"], result["tgt_upper"]
    if kind == "10米风速":
        return _wind(ps[1], ps[2]), _wind(ts[1], ts[2]), "m/s"
    if kind == "850hPa风速":
        li = PRESSURE_LEVELS.index(850)  # channel 3=u, 4=v
        return _wind(pu[3, li], pu[4, li]), _wind(tu[3, li], tu[4, li]), "m/s"
    if kind == "2米温度 t2m":
        return ps[3], ts[3], "K"
    if kind == "海平面气压 msl":
        return ps[0] / 100.0, ts[0] / 100.0, "hPa"
    raise ValueError(kind)


# Pangu output levels available for altitude-aware selection.
PANGU_WIND_LEVELS = ["10m", "1000", "925", "850"]


def wind_field_at_level(result, level, which="pred"):
    """Global (721,1440) wind-speed field at a named level from a Pangu result.

    level in {'10m','1000','925','850'}. '10m' uses surface u10/v10; pressure
    levels use upper u(ch3)/v(ch4) at the matching level index.
    which: 'pred' or 'tgt'.
    """
    surf = result["pred_surface"] if which == "pred" else result["tgt_surface"]
    up = result["pred_upper"] if which == "pred" else result["tgt_upper"]
    if level == "10m":
        return _wind(surf[1], surf[2])
    li = PRESSURE_LEVELS.index(int(level))
    return _wind(up[3, li], up[4, li])


def masked_rmse(pred_field, target_field, mask=None, device=None):
    device = device or get_device()
    p = torch.from_numpy(np.ascontiguousarray(pred_field)).float().to(device)
    t = torch.from_numpy(np.ascontiguousarray(target_field)).float().to(device)
    m = None
    if mask is not None:
        m = torch.from_numpy(np.ascontiguousarray(mask)).float().to(device)
    return score.weighted_rmse_torch_channels(p, t, m).item()


def _state_at(date_str, device):
    """Load the real ERA5 input state (upper, surface) at date_str 00:00 UTC.

    Loads ONLY the input state (no +horizon target), mirroring NetCDFDataset.
    nctonumpy: concat vars, reverse levels to descending. Returns physical tensors
    shaped (1,5,13,721,1440) and (1,4,721,1440).
    """
    import xarray as xr
    import pandas as pd
    t = pd.to_datetime(f"{date_str} 00:00:00")
    su = xr.open_dataset(os.path.join(cfg.PG_INPUT_PATH, "surface",
                                      f"surface_{date_str[:6]}.nc"))
    su = su.sel(time=t, expver=5) if "expver" in su.keys() else su.sel(time=t)
    up = xr.open_dataset(os.path.join(cfg.PG_INPUT_PATH, "upper",
                                      f"upper_{date_str}.nc"))
    up = up.sel(time=t, expver=5) if "expver" in up.keys() else up.sel(time=t)

    upper = np.concatenate([up[v].values.astype(np.float32)[np.newaxis, ...]
                            for v in ["z", "q", "t", "u", "v"]], axis=0)
    upper = upper[:, ::-1, :, :].copy()  # levels descending (matches nctonumpy)
    surface = np.concatenate([su[v].values.astype(np.float32)[np.newaxis, ...]
                              for v in ["msl", "u10", "v10", "t2m"]], axis=0)
    u = torch.from_numpy(upper).unsqueeze(0).to(device)
    s = torch.from_numpy(surface).unsqueeze(0).to(device)
    return u, s


def run_rollout(start_date, n_steps, strategy, device=None):
    """Autoregressive multi-step rollout (24h per step).

    strategy in {'zeroshot', 'vaawm', 'hybrid'}.
    At each step the model(s) predict the next state which is fed back as input.
    Each prediction is scored against the real ERA5 truth at that lead time.

    Returns dict: leads (list of 'T+k'), rmse_global, rmse_region (10m wind speed).
    """
    import pandas as pd
    device = device or get_device()
    aux = load_aux(device)
    base = load_model(device, "zeroshot")
    ft = load_model(device, "vaawm") if strategy in ("vaawm", "hybrid") else None

    # region mask (Xinjiang) for masked RMSE
    r = pdata.REGIONS["新疆 (Xinjiang)"]
    region_mask = torch.from_numpy(
        reg.build_region_mask(r["lat_min"], r["lat_max"], r["lon_min"], r["lon_max"])
    ).float().to(device)

    cur_u, cur_s = _state_at(start_date, device)
    start = pd.to_datetime(start_date, format="%Y%m%d")

    leads, rmse_g, rmse_r = [], [], []
    avail = set(list_input_dates() + _all_upper_dates())

    for k in range(1, n_steps + 1):
        with torch.no_grad():
            ob, osb = base(cur_u, cur_s, aux["weather_statistics"],
                           aux["constant_maps"], aux["const_h"])
            ob, osb = utils_data.normBackData(ob, osb, aux["weather_statistics_last"])
            if strategy == "zeroshot":
                nu, ns = ob, osb
            else:
                ov, osv = ft(cur_u, cur_s, aux["weather_statistics"],
                             aux["constant_maps"], aux["const_h"])
                ov, osv = utils_data.normBackData(ov, osv, aux["weather_statistics_last"])
                if strategy == "vaawm":
                    nu, ns = ov, osv
                else:  # hybrid: target wind channels from fine-tuned, rest from base
                    nu, ns = ob.clone(), osb.clone()
                    for ch in TARGET_SURFACE_CH:
                        ns[:, ch] = osv[:, ch]
                    for ch in TARGET_UPPER_CH:
                        nu[:, ch] = ov[:, ch]

        # score against truth at this lead (if available)
        tgt_date = (start + pd.Timedelta(hours=cfg.PG.HORIZON * k)).strftime("%Y%m%d")
        if tgt_date in avail:
            tu, ts = _state_at(tgt_date, device)
            pws = torch.sqrt(ns[0, 1] ** 2 + ns[0, 2] ** 2)
            tws = torch.sqrt(ts[0, 1] ** 2 + ts[0, 2] ** 2)
            leads.append(f"T+{k}")
            rmse_g.append(score.weighted_rmse_torch_channels(pws, tws).item())
            rmse_r.append(score.weighted_rmse_torch_channels(pws, tws, region_mask).item())

        cur_u, cur_s = nu, ns  # feed prediction back

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return {"leads": leads, "rmse_global": rmse_g, "rmse_region": rmse_r}


def _all_upper_dates():
    upper_dir = os.path.join(cfg.PG_INPUT_PATH, "upper")
    out = []
    if os.path.isdir(upper_dir):
        for p in glob.glob(os.path.join(upper_dir, "upper_*.nc")):
            d = os.path.basename(p).replace("upper_", "").replace(".nc", "")
            if len(d) == 8:
                out.append(d)
    return out


def max_rollout_steps(start_date):
    """How many 24h steps have ground truth available from start_date."""
    import pandas as pd
    avail = set(_all_upper_dates())
    start = pd.to_datetime(start_date, format="%Y%m%d")
    k = 0
    while True:
        nxt = (start + pd.Timedelta(hours=cfg.PG.HORIZON * (k + 1))).strftime("%Y%m%d")
        if nxt in avail:
            k += 1
        else:
            break
    return k


def eval_heldout(dates, device=None):
    """Single-step (24h) generalization eval over held-out input dates.

    For each strategy, returns mean 10m-wind-speed RMSE (global & Xinjiang) across
    the given input dates. Used to test whether the fine-tune generalizes to days
    it never saw during training.
    """
    device = device or get_device()
    r = pdata.REGIONS["新疆 (Xinjiang)"]
    rmask = reg.build_region_mask(r["lat_min"], r["lat_max"], r["lon_min"], r["lon_max"])
    acc = {s: {"g": [], "r": []} for s in ("zeroshot", "vaawm", "hybrid")}
    for d in dates:
        base = run_inference(d, device, "zeroshot")
        vw = run_inference(d, device, "vaawm")
        hy = build_hybrid(base, vw)
        for name, res in (("zeroshot", base), ("vaawm", vw), ("hybrid", hy)):
            p, t, _ = get_field(res, "10米风速")
            acc[name]["g"].append(masked_rmse(p, t, None, device))
            acc[name]["r"].append(masked_rmse(p, t, rmask, device))
    return {s: {"global": float(np.mean(v["g"])), "region": float(np.mean(v["r"]))}
            for s, v in acc.items()}
