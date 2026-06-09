"""
Real CorrDiff downscaling for the chat demo (Stage-2 of the wind pipeline).

Loads the regression + diffusion checkpoints trained on SageMaker (WRF
25km->5km wind) and exposes a single `downscale_uv()` call that the forecast
pipeline uses in place of the bilinear placeholder.

Honesty notes
-------------
* The diffusion model was trained on a *specific* WRF domain with its own
  normalization statistics. Applying it to Pangu output over arbitrary wind
  farms (Xinjiang / Zhejiang) is out-of-distribution: it demonstrates the real
  two-stage CorrDiff *mechanism* (regression mean + diffusion residual), not a
  domain-calibrated downscaling for those exact sites.
* The training data only contained 2 distinct fields (one U, one V) replicated
  across the 4 nominal levels, so the model effectively maps (U,V)->(U,V).
* If PhysicsNeMo or the checkpoints are unavailable, callers fall back to the
  bilinear baseline in wind_power.downscale().

The whole thing runs on whatever device get_device() returns (CPU when the
chat app hides CUDA), so it never competes with GPU training.
"""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

# WRF channel order used during training.
WIND_VARS = ["U1000", "U850", "U10m", "U100m", "V1000", "V850", "V10m", "V100m"]
_U10_IDX = WIND_VARS.index("U10m")
_V10_IDX = WIND_VARS.index("V10m")
_GRID = 64  # model conditioning / output spatial size

# Default checkpoint + stats locations (produced by the SageMaker jobs).
REG_CKPT = os.environ.get(
    "CORRDIFF_REG_CKPT",
    "/opt/dlami/nvme/corrdiff_reg/regckpt/CorrDiffRegressionUNet.mdlus",
)
RES_CKPT = os.environ.get(
    "CORRDIFF_RES_CKPT",
    "/opt/dlami/nvme/corrdiff_diff/diffckpt/EDMPrecondSuperResolution.mdlus",
)
STATS_PATH = os.environ.get(
    "CORRDIFF_STATS",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "corrdiff", "data", "wrf", "stats_25km_5km.json"),
)

_CACHE = {}


def available():
    """True if PhysicsNeMo and both checkpoints are present."""
    if not (os.path.exists(REG_CKPT) and os.path.exists(RES_CKPT)
            and os.path.exists(STATS_PATH)):
        return False
    try:
        import physicsnemo  # noqa: F401
        return True
    except Exception:
        return False


def _load(device):
    key = str(device)
    if key in _CACHE:
        return _CACHE[key]

    from functools import partial
    from physicsnemo import Module
    from physicsnemo.diffusion.samplers import stochastic_sampler
    from physicsnemo.diffusion.generate import diffusion_step, regression_step

    net_reg = Module.from_checkpoint(REG_CKPT, override_args={"use_apex_gn": False})
    net_res = Module.from_checkpoint(RES_CKPT, override_args={"use_apex_gn": False})
    for n in (net_reg, net_res):
        n.use_fp16 = False
        if hasattr(n, "amp_mode"):
            n.amp_mode = False
    net_reg = net_reg.eval().to(device).to(memory_format=torch.channels_last)
    net_res = net_res.eval().to(device).to(memory_format=torch.channels_last)

    stats = json.load(open(STATS_PATH))
    in_mean = np.array([stats["input"][v]["mean"] for v in WIND_VARS], np.float32)
    in_std = np.array([stats["input"][v]["std"] for v in WIND_VARS], np.float32)
    out_mean = np.array([stats["output"][v]["mean"] for v in WIND_VARS], np.float32)
    out_std = np.array([stats["output"][v]["std"] for v in WIND_VARS], np.float32)

    sampler_fn = partial(stochastic_sampler, patching=None, num_steps=18)
    bundle = dict(
        net_reg=net_reg, net_res=net_res, sampler_fn=sampler_fn,
        diffusion_step=diffusion_step, regression_step=regression_step,
        in_mean=in_mean[:, None, None], in_std=in_std[:, None, None],
        out_mean=out_mean[:, None, None], out_std=out_std[:, None, None],
        device=device,
    )
    _CACHE[key] = bundle
    return bundle


def _resize(a, size=_GRID):
    t = torch.from_numpy(np.ascontiguousarray(a)).float()[None, None]
    return F.interpolate(t, size=(size, size), mode="bilinear",
                         align_corners=False)[0, 0].numpy()


@torch.no_grad()
def downscale_uv(u_coarse, v_coarse, device=None):
    """Two-stage CorrDiff downscaling of coarse U/V wind components.

    u_coarse, v_coarse : 2D numpy arrays (coarse regional wind components).
    Returns (u_fine, v_fine) as 64x64 arrays in physical units (m/s).
    Raises if the model/stats are unavailable (callers handle fallback).
    """
    b = _load(device or torch.device("cpu"))
    # Build the 8-channel conditioning input (U replicated x4, V replicated x4),
    # matching how the WRF training data was structured.
    u = _resize(u_coarse)
    v = _resize(v_coarse)
    chans = np.stack([u, u, u, u, v, v, v, v], axis=0)  # (8, 64, 64)
    norm = (chans - b["in_mean"]) / b["in_std"]
    img_lr = torch.from_numpy(norm[None]).float().to(b["device"]).to(
        memory_format=torch.channels_last)

    image_reg = b["regression_step"](
        net=b["net_reg"], img_lr=img_lr,
        latents_shape=torch.Size([1, len(WIND_VARS), _GRID, _GRID]),
    )
    image_res = b["diffusion_step"](
        net=b["net_res"], sampler_fn=b["sampler_fn"],
        img_shape=(_GRID, _GRID), img_out_channels=len(WIND_VARS),
        rank_batches=[torch.as_tensor([0])],
        img_lr=img_lr.to(memory_format=torch.channels_last),
        rank=0, device=b["device"], mean_hr=image_reg[0:1],
    )
    out = (image_reg + image_res)[0].float().cpu().numpy()  # (8, 64, 64) normalized
    out = out * b["out_std"] + b["out_mean"]
    return out[_U10_IDX], out[_V10_IDX]


@torch.no_grad()
def downscale_speed(u_coarse, v_coarse, device=None):
    """Convenience: returns the 64x64 downscaled 10 m wind-speed field."""
    u_f, v_f = downscale_uv(u_coarse, v_coarse, device=device)
    return np.sqrt(u_f ** 2 + v_f ** 2)
