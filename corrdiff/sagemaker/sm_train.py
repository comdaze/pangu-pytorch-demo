"""
SageMaker entry point for CorrDiff (WRF 25km->5km) training.

Lives in corrdiff/sagemaker/ ; the Hydra app train.py and conf/ are one level up
(the corrdiff/ source dir, copied to /opt/ml/code on SageMaker).

Supports two stages:
  * regression : trains the deterministic regression UNet
                 (config_training_wrf_regression).
  * diffusion  : trains the residual diffusion UNet
                 (config_training_wrf_diffusion); requires a regression
                 checkpoint supplied via the SageMaker 'regression' channel.

Installs the PhysicsNeMo stack at runtime, runs train.py on the WRF dataset
provided via the SageMaker 'train' channel, then copies checkpoints to the
model dir for upload to S3.
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys


def pip(*pkgs, no_deps=False):
    cmd = [sys.executable, "-m", "pip", "install", "-q"]
    if no_deps:
        cmd.append("--no-deps")
    cmd += list(pkgs)
    print("[pip]", " ".join(pkgs), flush=True)
    subprocess.check_call(cmd)


def install_deps():
    pip("timm", "s3fs", "hydra-core>=1.2", "omegaconf>=2.3", "einops", "jaxtyping",
        "tensordict", "termcolor", "treelib", "nvtx", "warp-lang", "wandb",
        "tensorboard", "cftime", "dask", "xskillscore", "scipy", "netCDF4", "matplotlib")
    pip("nvidia-physicsnemo==2.1.1", no_deps=True)


def find_regression_checkpoint(reg_dir):
    """Locate a .mdlus regression checkpoint in the regression channel."""
    cands = glob.glob(os.path.join(reg_dir, "**", "*.mdlus"), recursive=True)
    if not cands:
        raise FileNotFoundError(f"no .mdlus checkpoint found under {reg_dir}")
    # prefer the highest training-step checkpoint if multiple are present
    def step(p):
        try:
            return int(os.path.basename(p).split(".")[-2])
        except (IndexError, ValueError):
            return -1
    cands.sort(key=step)
    return cands[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["regression", "diffusion"], default="regression")
    ap.add_argument("--training_duration", type=int, default=100000)
    ap.add_argument("--total_batch_size", type=int, default=32)
    ap.add_argument("--batch_size_per_gpu", type=str, default="16")
    ap.add_argument("--lr", type=float, default=2e-4)
    args = ap.parse_args()

    install_deps()

    # corrdiff source root = parent of this sagemaker/ dir (==/opt/ml/code)
    corrdiff_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    nc = os.path.join(data_dir, "wrf_25km_5km_train.nc")
    stats = os.path.join(data_dir, "stats_25km_5km.json")

    print(f"[sm] stage={args.stage} corrdiff_root={corrdiff_root} "
          f"data={data_dir} files={os.listdir(data_dir)}", flush=True)

    env = dict(os.environ)
    env["WANDB_MODE"] = "offline"
    env["PYTHONUNBUFFERED"] = "1"

    config = (
        "config_training_wrf_regression"
        if args.stage == "regression"
        else "config_training_wrf_diffusion"
    )
    cmd = [
        sys.executable, os.path.join(corrdiff_root, "train.py"),
        f"--config-name={config}",
        f"dataset.data_path={nc}",
        f"dataset.stats_path={stats}",
        f"training.hp.training_duration={args.training_duration}",
        f"training.hp.total_batch_size={args.total_batch_size}",
        f"training.hp.batch_size_per_gpu={args.batch_size_per_gpu}",
        f"training.hp.lr={args.lr}",
        "training.io.print_progress_freq=500",
        "training.io.save_checkpoint_freq=5000",
        "training.io.validation_freq=5000",
        "wandb.mode=offline",
    ]

    if args.stage == "diffusion":
        reg_dir = os.environ.get("SM_CHANNEL_REGRESSION",
                                 "/opt/ml/input/data/regression")
        reg_ckpt = find_regression_checkpoint(reg_dir)
        print(f"[sm] regression checkpoint: {reg_ckpt}", flush=True)
        cmd.append(f"training.io.regression_checkpoint_path={reg_ckpt}")

    print("[sm] running:", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=corrdiff_root, env=env)

    for sub in ["checkpoints_regression", "checkpoints_diffusion", "output", "wandb"]:
        src = os.path.join(corrdiff_root, sub)
        if os.path.isdir(src):
            shutil.copytree(src, os.path.join(model_dir, sub), dirs_exist_ok=True)
    print("[sm] done; model_dir contents:", os.listdir(model_dir), flush=True)


if __name__ == "__main__":
    main()
