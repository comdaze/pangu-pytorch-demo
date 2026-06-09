"""
Launch CorrDiff (WRF 25km->5km) training on SageMaker.

Two stages:
  * regression (default): trains the deterministic regression UNet.
  * diffusion: trains the residual diffusion UNet; needs a regression
    checkpoint in S3, supplied to the job via a 'regression' input channel.

Uploads the WRF dataset to the default SageMaker bucket and submits a PyTorch
training job using corrdiff/ as the source dir and sagemaker/sm_train.py as entry.

Run from this directory so the repo-root `sagemaker/` package does not shadow
the installed SageMaker SDK:

    AWS_SHARED_CREDENTIALS_FILE=.../.sm_aws_credentials AWS_DEFAULT_REGION=us-east-1 \
        STAGE=diffusion \
        REGRESSION_S3=s3://.../corrdiff-wrf/regression/ \
        python launch_sm_corrdiff.py
"""
import os
import shutil
import tempfile

import sagemaker
from sagemaker.pytorch import PyTorch

ROLE = "arn:aws:iam::383570952416:role/service-role/AmazonSageMaker-ExecutionRole-20231222T090399"
# corrdiff/  (source dir) = parent of this sagemaker/ folder
CORRDIFF = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_LOCAL = os.path.join(CORRDIFF, "data", "wrf")
INSTANCE = os.environ.get("SM_INSTANCE", "ml.g6.2xlarge")
STAGE = os.environ.get("STAGE", "regression")
# S3 location (prefix or object) of the regression .mdlus checkpoint; required
# for the diffusion stage.
REGRESSION_S3 = os.environ.get(
    "REGRESSION_S3",
    "s3://sagemaker-us-east-1-383570952416/corrdiff-wrf/regression/",
)


def main():
    sess = sagemaker.Session()
    bucket = sess.default_bucket()
    print(f"stage={STAGE} instance={INSTANCE} bucket={bucket} "
          f"region={sess.boto_region_name}", flush=True)

    staging = tempfile.mkdtemp()
    for f in ["wrf_25km_5km_train.nc", "stats_25km_5km.json"]:
        shutil.copy(os.path.join(DATA_LOCAL, f), os.path.join(staging, f))
    data_s3 = sess.upload_data(path=staging, bucket=bucket, key_prefix="corrdiff-wrf/data")
    print("data uploaded to:", data_s3, flush=True)

    if STAGE == "diffusion":
        base_job_name = "corrdiff-wrf-diff"
        hp = {
            "stage": "diffusion",
            "training_duration": 200000,
            "total_batch_size": 16,
            "batch_size_per_gpu": 2,
            "lr": 2e-4,
        }
        inputs = {"train": data_s3, "regression": REGRESSION_S3}
    else:
        base_job_name = "corrdiff-wrf-reg"
        hp = {
            "stage": "regression",
            "training_duration": 100000,
            "total_batch_size": 32,
            "batch_size_per_gpu": 16,
            "lr": 2e-4,
        }
        inputs = {"train": data_s3}

    est = PyTorch(
        entry_point="sagemaker/sm_train.py",   # relative to source_dir
        source_dir=CORRDIFF,
        role=ROLE,
        framework_version="2.5.1",
        py_version="py311",
        instance_type=INSTANCE,
        instance_count=1,
        base_job_name=base_job_name,
        max_run=8 * 3600,
        hyperparameters=hp,
        environment={"WANDB_MODE": "offline"},
    )
    est.fit(inputs, wait=False)
    print("JOB_NAME:", est.latest_training_job.name, flush=True)


if __name__ == "__main__":
    main()
