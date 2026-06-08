"""
Small-scale VAAWM fine-tuning of Pangu-Weather with a proper train/val/test split.

FAITHFUL to the repository's VAAWM objective (models/pangu_sample.train with
--only_use_wind_speed_loss + --use_custom_mask):
  - loss = L1 on wind speed (sqrt(u^2+v^2)) of surface (u10,v10) and upper (u,v),
    computed in NORMALIZED space (target is normData'd, output is normalized),
  - restricted to the Xinjiang region mask beta, averaged over valid points.
The only additions are standard, repo-consistent training hygiene: an explicit
train/val split with validation-loss early stopping (the repo uses EARLY_STOP too).

We do NOT change the loss to physical space or add tricks the repo/paper lack.

Data (2019-07, surface monthly + upper from ARCO-ERA5):
  TRAIN  : 07-01 (24h) -> 07-02 ; 07-02 -> 07-03 ; 07-03 -> 07-04  (00/06/12/18)
  VAL    : 07-04 -> 07-05 ; 07-05 -> 07-06                          (early stopping)
  TEST   : 07-06 -> 07-07 ; 07-07 -> 07-08 ; 07-08 -> 07-09         (held out, eval only)
"""
import os
import sys
import time
import copy
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch
from torch import nn

from era5_data.config import cfg
from era5_data import utils_data
from models.pangu_model import PanguModel
from models.pangu_sample import get_wind_speed

OUT_DIR = os.path.join(cfg.PG_OUT_PATH, "finetune_vaawm", str(cfg.PG.HORIZON))
BEST_PATH = os.path.join(OUT_DIR, "vaawm_finetuned.pth")

# Explicit input timestamps (YYYYMMDDHH). target = input + 24h.
TRAIN_INPUTS = (
    [f"2019070{1}{h:02d}" for h in range(24)]                 # 07-01 all 24h -> 07-02
    + [f"201907020{0}", "2019070206", "2019070212", "2019070218"]   # 07-02 -> 07-03
    + ["2019070300", "2019070306", "2019070312", "2019070318"]      # 07-03 -> 07-04
)
VAL_INPUTS = ["2019070400", "2019070406", "2019070412", "2019070418",  # 07-04 -> 07-05
              "2019070500"]                                            # 07-05 -> 07-06


def make_loader_obj():
    return utils_data.NetCDFDataset(
        nc_path=cfg.PG_INPUT_PATH, training=False, validation=False,
        startDate="20190701 00:00:00", endDate="20190701 00:00:00",
        freq="24h", horizon=cfg.PG.HORIZON)


def valid_inputs(ds, inputs):
    keep = []
    for s in inputs:
        ts = pd.to_datetime(s, format="%Y%m%d%H")
        end = ts + pd.Timedelta(hours=cfg.PG.HORIZON)
        up_in = os.path.join(cfg.PG_INPUT_PATH, "upper", f"upper_{ts.strftime('%Y%m%d')}.nc")
        up_tg = os.path.join(cfg.PG_INPUT_PATH, "upper", f"upper_{end.strftime('%Y%m%d')}.nc")
        if not (os.path.exists(up_in) and os.path.exists(up_tg)):
            continue
        try:
            import xarray as xr
            for f, t in [(up_in, ts), (up_tg, end)]:
                d = xr.open_dataset(f)
                if t not in pd.to_datetime(d.time.values):
                    raise KeyError
                d.close()
            keep.append(s)
        except Exception:
            continue
    return keep


def load_pair(ds, s, device):
    ts = pd.to_datetime(s, format="%Y%m%d%H")
    inp, inps, tgt, tgts, _ = ds.LoadData(ts)
    return (torch.from_numpy(inp).unsqueeze(0).to(device),
            torch.from_numpy(inps).unsqueeze(0).to(device),
            torch.from_numpy(tgt).unsqueeze(0).to(device),
            torch.from_numpy(tgts).unsqueeze(0).to(device))


def vaawm_loss(criterion, out, outs, tgt_n, tgts_n, mask_bool, valid_points):
    o_sws, t_sws, o_ws, t_ws = get_wind_speed(outs, tgts_n, out, tgt_n)
    sl = (criterion(o_sws, t_sws) * (~mask_bool)).sum() / valid_points
    ul = (criterion(o_ws, t_ws) * (~mask_bool)).sum() / valid_points
    return sl + ul


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--patience", type=int, default=3)
    args = ap.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[VAAWM] device={device} epochs={args.epochs} lr={args.lr} patience={args.patience}", flush=True)

    model = PanguModel(device=device).to(device)
    ckpt = torch.load(cfg.PG.BENCHMARK.PRETRAIN_24_torch, weights_only=True, map_location=device)
    model.load_state_dict(ckpt["model"])
    for p in model.parameters():
        p.requires_grad = True

    aux = utils_data.loadAllConstants(device=device)
    mask = aux["custom_mask"]; mask_bool = mask == 0; vp = mask.sum()
    criterion = nn.L1Loss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=cfg.PG.TRAIN.WEIGHT_DECAY)

    ds = make_loader_obj()
    train_in = valid_inputs(ds, TRAIN_INPUTS)
    val_in = valid_inputs(ds, VAL_INPUTS)
    print(f"[VAAWM] train pairs={len(train_in)}  val pairs={len(val_in)}", flush=True)

    import random
    best_val = float("inf"); best_state = None; since = 0
    for epoch in range(1, args.epochs + 1):
        model.train(); random.shuffle(train_in)
        tr = 0.0; t0 = time.time()
        for s in train_in:
            inp, inps, tgt, tgts = load_pair(ds, s, device)
            optimizer.zero_grad()
            out, outs = model(inp, inps, aux["weather_statistics"],
                              aux["constant_maps"], aux["const_h"])
            tn, tsn = utils_data.normData(tgt, tgts, aux["weather_statistics_last"])
            loss = vaawm_loss(criterion, out, outs, tn, tsn, mask_bool, vp)
            loss.backward(); optimizer.step()
            tr += loss.item()
        tr /= max(len(train_in), 1)

        # validation (eval mode, repo's faithful loss)
        model.eval(); vl = 0.0
        with torch.no_grad():
            for s in val_in:
                inp, inps, tgt, tgts = load_pair(ds, s, device)
                out, outs = model(inp, inps, aux["weather_statistics"],
                                  aux["constant_maps"], aux["const_h"])
                tn, tsn = utils_data.normData(tgt, tgts, aux["weather_statistics_last"])
                vl += vaawm_loss(criterion, out, outs, tn, tsn, mask_bool, vp).item()
        vl /= max(len(val_in), 1)
        print(f"[VAAWM] epoch {epoch}: train={tr:.5f} val={vl:.5f} time={time.time()-t0:.1f}s", flush=True)

        if vl < best_val:
            best_val = vl; best_state = copy.deepcopy(model.state_dict()); since = 0
            print(f"  -> new best val={vl:.5f}", flush=True)
        else:
            since += 1
            if since >= args.patience:
                print(f"[VAAWM] early stop at epoch {epoch} (no val improvement for {since})", flush=True)
                break

    torch.save({"model": best_state if best_state is not None else model.state_dict(),
                "val_loss": best_val}, BEST_PATH)
    print(f"[VAAWM] saved best (val={best_val:.5f}) -> {BEST_PATH}", flush=True)


if __name__ == "__main__":
    main()
