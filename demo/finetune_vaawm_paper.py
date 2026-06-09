"""
Faithful VAAWM fine-tuning (closer to the paper than the repo's crude
--only_use_wind_speed_loss + hard mask).

Paper's adaptive loss (Sec 3.3):
    L = sum_v  alpha_v  sum_{(i,j) in G}  beta_{i,j} (Y_{i,j,v} - Yhat_{i,j,v})^2
  - alpha_v   : per-variable weights, softmax-normalized, LEARNABLE, wind >> others
  - beta_{i,j}: geo-spatial importance, high in target region, NONZERO elsewhere
  - applied over ALL variables (squared error / MSE)

Why this matters: the repo's only_use_wind_speed_loss puts ZERO weight on every
non-wind variable, which catastrophically destroys t2m/msl/etc. The paper keeps
all variables with a small nonzero weight (softmax) and uses a soft region mask,
so the model specializes on regional wind WITHOUT forgetting everything else.

Model selection / early stopping is on VALIDATION RMSE of the target wind speeds
(10m + 850hPa), as in the paper (Sec 4.1).

Variables order:
  upper out channels   : [z, q, t, u, v]   (each 13 levels)
  surface out channels : [msl, u10, v10, t2m]
Target (wind): upper u,v (idx 3,4) and surface u10,v10 (idx 1,2).
"""
import os
import sys
import time
import copy
import json
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch import nn

from era5_data.config import cfg
from era5_data import utils_data, score
from models.pangu_model import PanguModel

import run_pipeline as rp  # reuse load_pair / split_inputs / data layout

OUT_DIR = os.path.join(cfg.PG_OUT_PATH, "finetune_vaawm", str(cfg.PG.HORIZON))
BEST_PATH = os.path.join(OUT_DIR, "vaawm_paper.pth")
STATUS = os.path.join(OUT_DIR, "vaawm_paper_status.json")

UPPER_WIND = [3, 4]      # u, v
SURFACE_WIND = [1, 2]    # u10, v10
L850 = 2                 # 850hPa index in descending levels [1000,925,850,...]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def set_status(**kw):
    os.makedirs(OUT_DIR, exist_ok=True)
    st = {}
    if os.path.exists(STATUS):
        try: st = json.load(open(STATUS))
        except Exception: st = {}
    st.update(kw); st["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    json.dump(st, open(STATUS, "w"), indent=2)


def build_soft_beta(mask, outside=0.05):
    """beta: 1.0 inside region, `outside` elsewhere (nonzero -> global skill kept)."""
    return mask * (1.0 - outside) + outside


def region_wind_rmse(outs_phys, out_phys, tgts, tgt, mt):
    """Physical RMSE of 10m + 850hPa wind speed in the region (mean of the two)."""
    w10p = torch.sqrt(outs_phys[0, 1] ** 2 + outs_phys[0, 2] ** 2)
    w10t = torch.sqrt(tgts[0, 1] ** 2 + tgts[0, 2] ** 2)
    w850p = torch.sqrt(out_phys[0, 3, L850] ** 2 + out_phys[0, 4, L850] ** 2)
    w850t = torch.sqrt(tgt[0, 3, L850] ** 2 + tgt[0, 4, L850] ** 2)
    r10 = score.weighted_rmse_torch_channels(w10p, w10t, mt).item()
    r850 = score.weighted_rmse_torch_channels(w850p, w850t, mt).item()
    return 0.5 * (r10 + r850)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--alpha_lr", type=float, default=1e-2)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--beta_outside", type=float, default=0.05)
    args = ap.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    aux = utils_data.loadAllConstants(device=device)
    mask = aux["custom_mask"]
    mt = mask  # for RMSE masking
    beta = build_soft_beta(mask, args.beta_outside).to(device)  # (721,1440)
    beta_sum = beta.sum()

    # learnable alpha over 9 variables: [z,q,t,u,v, msl,u10,v10,t2m]
    # init wind-biased (logit 2 for wind, 0 otherwise) => alpha_wind >> others
    init = torch.zeros(9, device=device)
    for i in [3, 4, 5 + 1, 5 + 2]:  # upper u,v ; surface u10,v10
        init[i] = 2.0
    alpha_logits = nn.Parameter(init.clone())

    model = PanguModel(device=device).to(device)
    model.load_state_dict(torch.load(cfg.PG.BENCHMARK.PRETRAIN_24_torch,
                                     weights_only=True, map_location=device)["model"])
    for p in model.parameters():
        p.requires_grad = True

    opt = torch.optim.Adam([
        {"params": model.parameters(), "lr": args.lr,
         "weight_decay": cfg.PG.TRAIN.WEIGHT_DECAY},
        {"params": [alpha_logits], "lr": args.alpha_lr},
    ])

    train_in, val_in, test_in = rp.split_inputs()
    log(f"split train={len(train_in)} val={len(val_in)} test={len(test_in)}")
    set_status(stage="train", train_n=len(train_in), val_n=len(val_in), test_n=len(test_in))

    def variable_losses(out, outs, tn, tsn):
        """Per-variable beta-weighted MSE (normalized space). Returns tensor[9]."""
        losses = []
        # upper: mean over levels of spatial beta-weighted MSE
        for k in range(5):
            se = (out[0, k] - tn[0, k]) ** 2                      # (13,721,1440)
            l = (se * beta).sum() / (beta_sum * se.shape[0])
            losses.append(l)
        for k in range(4):
            se = (outs[0, k] - tsn[0, k]) ** 2                    # (721,1440)
            l = (se * beta).sum() / beta_sum
            losses.append(l)
        return torch.stack(losses)

    import random
    best_val = float("inf"); best_state = None; since = 0
    for ep in range(1, args.epochs + 1):
        model.train(); random.shuffle(train_in); tr = 0.0; t0 = time.time()
        for s in train_in:
            inp, inps, tgt, tgts = rp.load_pair(s, device)
            opt.zero_grad()
            out, outs = model(inp, inps, aux["weather_statistics"],
                              aux["constant_maps"], aux["const_h"])
            tn, tsn = utils_data.normData(tgt, tgts, aux["weather_statistics_last"])
            lv = variable_losses(out, outs, tn, tsn)
            alpha = torch.softmax(alpha_logits, dim=0)
            loss = (alpha * lv).sum()
            loss.backward(); opt.step(); tr += loss.item()
        tr /= max(len(train_in), 1)

        # validation: physical region wind RMSE (paper's selection metric)
        model.eval(); vr = 0.0
        with torch.no_grad():
            for s in val_in:
                inp, inps, tgt, tgts = rp.load_pair(s, device)
                out, outs = model(inp, inps, aux["weather_statistics"],
                                  aux["constant_maps"], aux["const_h"])
                out, outs = utils_data.normBackData(out, outs, aux["weather_statistics_last"])
                vr += region_wind_rmse(outs, out, tgts, tgt, mt)
        vr /= max(len(val_in), 1)
        a = torch.softmax(alpha_logits, dim=0).detach().cpu().numpy()
        log(f"epoch {ep}: train_loss={tr:.5f} val_windRMSE={vr:.5f} time={time.time()-t0:.0f}s")
        log(f"  alpha[z,q,t,u,v,msl,u10,v10,t2m]={np.round(a,3).tolist()}")
        set_status(stage="train", epoch=ep, train_loss=tr, val_wind_rmse=vr,
                   best_val=best_val, alpha=a.tolist())
        if vr < best_val - 1e-6:
            best_val = vr; best_state = copy.deepcopy(model.state_dict()); since = 0
            torch.save({"model": best_state, "alpha": a.tolist(), "val_wind_rmse": vr}, BEST_PATH)
            log(f"  -> new best val_windRMSE={vr:.5f} saved")
        else:
            since += 1
            if since >= args.patience:
                log(f"early stop at epoch {ep}"); break

    # ---- held-out test: zero-shot vs VAAWM(paper) vs hybrid, region + global ----
    log("eval on held-out test ...")
    base = PanguModel(device=device).to(device)
    base.load_state_dict(torch.load(cfg.PG.BENCHMARK.PRETRAIN_24_torch,
                                    weights_only=True, map_location=device)["model"]); base.eval()
    model.load_state_dict(best_state if best_state is not None else model.state_dict()); model.eval()

    def wind_rmse(osurf, oup, tgts, tgt, masked):
        m = mt if masked else None
        w10 = torch.sqrt(osurf[0, 1] ** 2 + osurf[0, 2] ** 2)
        t10 = torch.sqrt(tgts[0, 1] ** 2 + tgts[0, 2] ** 2)
        return score.weighted_rmse_torch_channels(w10, t10, m).item()

    def t2m_rmse(osurf, tgts, masked):
        m = mt if masked else None
        return score.weighted_rmse_torch_channels(osurf[0, 3], tgts[0, 3], m).item()

    agg = {k: [] for k in ["zs_r", "vw_r", "zs_g", "vw_g", "hy_t2m_g", "vw_t2m_g", "zs_t2m_g"]}
    with torch.no_grad():
        for s in test_in[::2]:  # subsample test for speed
            inp, inps, tgt, tgts = rp.load_pair(s, device)
            ob, osb = base(inp, inps, aux["weather_statistics"], aux["constant_maps"], aux["const_h"])
            ov, osv = model(inp, inps, aux["weather_statistics"], aux["constant_maps"], aux["const_h"])
            ob, osb = utils_data.normBackData(ob, osb, aux["weather_statistics_last"])
            ov, osv = utils_data.normBackData(ov, osv, aux["weather_statistics_last"])
            agg["zs_r"].append(wind_rmse(osb, ob, tgts, tgt, True))
            agg["vw_r"].append(wind_rmse(osv, ov, tgts, tgt, True))
            agg["zs_g"].append(wind_rmse(osb, ob, tgts, tgt, False))
            agg["vw_g"].append(wind_rmse(osv, ov, tgts, tgt, False))
            agg["zs_t2m_g"].append(t2m_rmse(osb, tgts, False))
            agg["vw_t2m_g"].append(t2m_rmse(osv, tgts, False))
            agg["hy_t2m_g"].append(t2m_rmse(osb, tgts, False))  # hybrid t2m = base
    res = {k: float(np.mean(v)) for k, v in agg.items() if v}
    if res.get("zs_r"):
        res["region_wind_improvement_pct"] = (res["zs_r"] - res["vw_r"]) / res["zs_r"] * 100
    res["best_val_wind_rmse"] = best_val
    res["alpha"] = torch.softmax(alpha_logits, dim=0).detach().cpu().numpy().round(3).tolist()
    log("TEST: " + json.dumps(res))
    set_status(stage="done", result=res, checkpoint=BEST_PATH)
    json.dump(res, open(os.path.join(OUT_DIR, "vaawm_paper_test_result.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
