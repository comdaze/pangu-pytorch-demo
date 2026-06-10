#!/usr/bin/env bash
# Back up the irreplaceable / slow-to-rebuild demo assets to S3.
# Ephemeral assets live on /opt/dlami/nvme (instance store) and are lost on
# stop/terminate. CorrDiff checkpoints already live in their SageMaker outputs.
#
# Usage:  bash demo/ops/backup_assets.sh
set -euo pipefail

export PATH="$PATH:/usr/local/bin"
export AWS_SHARED_CREDENTIALS_FILE="${AWS_SHARED_CREDENTIALS_FILE:-/home/ubuntu/pangu-pytorch-demo/.sm_aws_credentials}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"

NVME=/opt/dlami/nvme
B=s3://sagemaker-us-east-1-383570952416/pangu-demo-backup

echo "[1/4] VAAWM fine-tuned models -> S3"
aws s3 cp "$NVME/model/finetune_vaawm/" "$B/finetune_vaawm/" --recursive \
    --exclude "*" --include "*.pth"

echo "[2/4] aux_data constants -> S3"
aws s3 cp "$NVME/aux_data/" "$B/aux_data/" --recursive

echo "[3/4] Pangu torch checkpoints (1/3/6/24h) -> S3"
aws s3 cp "$NVME/pretrained_model/" "$B/pretrained_model/" --recursive \
    --exclude "*" --include "*_torch.pth"

echo "[4/4] ERA5 initial-field subset (1st & 15th of each 2019 month + monthly surface) -> S3"
for m in 01 02 03 04 05 06 07 08 09 10 11 12; do
    for d in 01 15; do
        f="$NVME/upper/upper_2019${m}${d}.nc"
        [ -f "$f" ] && aws s3 cp "$f" "$B/era5/upper/" --only-show-errors
    done
    sf="$NVME/surface/surface_2019${m}.nc"
    [ -f "$sf" ] && aws s3 cp "$sf" "$B/era5/surface/" --only-show-errors
done

echo "done. backup prefix: $B"
