#!/usr/bin/env bash
# Restore demo assets from S3 onto /opt/dlami/nvme after an instance stop/start
# (instance-store is wiped on stop). Pulls VAAWM models, aux_data, Pangu torch
# checkpoints, an ERA5 initial-field subset, and the CorrDiff checkpoints.
#
# Usage:  bash demo/ops/restore_assets.sh
set -euo pipefail

export PATH="$PATH:/usr/local/bin"
export AWS_SHARED_CREDENTIALS_FILE="${AWS_SHARED_CREDENTIALS_FILE:-/home/ubuntu/pangu-pytorch-demo/.sm_aws_credentials}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"

NVME=/opt/dlami/nvme
B=s3://sagemaker-us-east-1-383570952416/pangu-demo-backup
REG_JOB=corrdiff-wrf-reg-2026-06-09-05-47-08-678
DIFF_JOB=corrdiff-wrf-diff-2026-06-09-06-22-54-469
SM=s3://sagemaker-us-east-1-383570952416

mkdir -p "$NVME"/{model/finetune_vaawm/24,aux_data,pretrained_model,upper,surface,corrdiff_reg/regckpt,corrdiff_diff/diffckpt}

echo "[1/5] VAAWM models"
aws s3 cp "$B/finetune_vaawm/" "$NVME/model/finetune_vaawm/" --recursive

echo "[2/5] aux_data"
aws s3 cp "$B/aux_data/" "$NVME/aux_data/" --recursive

echo "[3/5] Pangu torch checkpoints"
aws s3 cp "$B/pretrained_model/" "$NVME/pretrained_model/" --recursive

echo "[4/5] ERA5 initial-field subset"
aws s3 cp "$B/era5/upper/" "$NVME/upper/" --recursive
aws s3 cp "$B/era5/surface/" "$NVME/surface/" --recursive

echo "[5/5] CorrDiff checkpoints (from SageMaker job outputs)"
tmp=$(mktemp -d)
aws s3 cp "$SM/$REG_JOB/output/model.tar.gz" "$tmp/reg.tar.gz"
tar -xzf "$tmp/reg.tar.gz" -C "$tmp" 2>/dev/null || true
reg=$(find "$tmp" -name "CorrDiffRegressionUNet*.mdlus" | sort -t. -k3 -n | tail -1)
cp "$reg" "$NVME/corrdiff_reg/regckpt/CorrDiffRegressionUNet.mdlus"
aws s3 cp "$SM/$DIFF_JOB/output/model.tar.gz" "$tmp/diff.tar.gz"
tar -xzf "$tmp/diff.tar.gz" -C "$tmp" 2>/dev/null || true
diff=$(find "$tmp" -name "EDMPrecondSuperResolution*.mdlus" | sort -t. -k3 -n | tail -1)
cp "$diff" "$NVME/corrdiff_diff/diffckpt/EDMPrecondSuperResolution.mdlus"
rm -rf "$tmp"

echo "done. assets restored to $NVME"
