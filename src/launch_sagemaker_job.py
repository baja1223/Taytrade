#!/usr/bin/env python3
"""
Launch a SageMaker PyTorch training job with ONLY supported hyperparameters,
then optionally deploy the trained model to a real-time endpoint.

Enhancements:
- Loads defaults from .env (via python-dotenv)
- Ensures local training data directory is uploaded to S3 if not present
- Uses absolute source_dir for entry point so "No file named aws_train_model.py" error is avoided

Note:
- The training entrypoint (train_model.py) now reads model_meta.json if present in the
  output directory to lock features/window/hidden sizes to that file. You can still
  override via CLI flags; the trainer will merge sensibly.
"""

import argparse
import os
from datetime import datetime
from pathlib import Path
import mimetypes

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from sagemaker.pytorch import PyTorch, PyTorchModel
from sagemaker.inputs import TrainingInput

# Load environment variables from .env if present
load_dotenv()

def _bool_env(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "on"}

def _s3_uri(bucket: str, key_prefix: str) -> str:
    key_prefix = key_prefix.lstrip("/")
    return f"s3://{bucket}/{key_prefix}"

def ensure_data_in_s3(local_dir: Path, s3_client, bucket: str, prefix: str) -> str:
    train_prefix = f"{prefix.rstrip('/')}/train/"
    try:
        resp = s3_client.list_objects_v2(Bucket=bucket, Prefix=train_prefix, MaxKeys=1)
        already_there = ("Contents" in resp and len(resp["Contents"]) > 0)
    except ClientError as e:
        raise RuntimeError(f"Failed to list s3://{bucket}/{train_prefix} - {e}")

    if already_there:
        print(f"\nFound existing training data under s3://{bucket}/{train_prefix} — will reuse.")
        return _s3_uri(bucket, train_prefix)

    if not local_dir.exists() or not local_dir.is_dir():
        raise FileNotFoundError(f"Local training data directory not found: {local_dir}")

    print(f"\nUploading local training data from {local_dir} to s3://{bucket}/{train_prefix} ...")
    uploaded = 0
    for p in local_dir.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(local_dir).as_posix()
        key = f"{train_prefix}{rel}"
        extra = {}
        ctype, _ = mimetypes.guess_type(p.name)
        if ctype:
            extra["ContentType"] = ctype
        try:
            s3_client.upload_file(str(p), bucket, key, ExtraArgs=extra if extra else None)
            uploaded += 1
        except ClientError as e:
            raise RuntimeError(f"Failed to upload {p} to s3://{bucket}/{key}: {e}")

    print(f"Uploaded {uploaded} files to s3://{bucket}/{train_prefix}")
    return _s3_uri(bucket, train_prefix)

def build_parser():
    p = argparse.ArgumentParser(description="Launch SageMaker training with supported flags only")
    # Infra / job settings (defaults from .env)
    p.add_argument("--role-arn", default=os.getenv("SAGEMAKER_ROLE_ARN"), required=False)
    p.add_argument("--region", default=os.getenv("AWS_REGION", "us-east-1"))
    p.add_argument("--bucket", default=os.getenv("S3_BUCKET_NAME"), required=False)
    p.add_argument("--prefix", default=os.getenv("S3_PREFIX", "eth-1m"))
    p.add_argument("--instance-type", default=os.getenv("TRAIN_INSTANCE_TYPE", "ml.g5.2xlarge"))
    p.add_argument("--instance-count", type=int, default=int(os.getenv("TRAIN_INSTANCE_COUNT", 1)))
    p.add_argument("--py-version", default=os.getenv("PY_VERSION", "py311"))
    p.add_argument("--framework-version", default=os.getenv("FRAMEWORK_VERSION", "2.3"))
    p.add_argument("--image-uri", default=None, help="Optional custom image")
    p.add_argument("--job-name", default=None)

    # Data location
    p.add_argument("--local-train-dir", default=os.getenv("DATA_DIR", "eth_1m_data"))
    p.add_argument("--train-s3-uri", default=os.getenv("TRAIN_S3_URI"), required=False)

    # Trainer flags (SUPPORTED ONLY; optimistic defaults)
    p.add_argument("--window-size", type=int, default=192)
    p.add_argument("--hidden-size", type=int, default=768)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--bidirectional", type=bool, default=True)
    p.add_argument("--disable-scaling", type=bool, default=False)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--learning-rate", type=float, default=3e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--accumulate", type=int, default=2)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--price-col", default="close")

    # Deployment controls (defaults from .env)
    p.add_argument("--deploy", action="store_true", default=True)
    p.add_argument("--endpoint-name", default=os.getenv("ENDPOINT_NAME"))
    p.add_argument("--endpoint-instance-type", default=os.getenv("ENDPOINT_INSTANCE_TYPE", "ml.m5.large"))
    p.add_argument("--endpoint-initial-count", type=int, default=int(os.getenv("ENDPOINT_INSTANCES", 1)))
    p.add_argument("--deploy-entry-point", default="inference.py")
    return p

def main():
    args = build_parser().parse_args()

    if not args.role_arn:
        raise ValueError("Missing --role-arn (or SAGEMAKER_ROLE_ARN in .env)")
    if not args.bucket:
        raise ValueError("Missing --bucket (or S3_BUCKET_NAME in .env)")

    script_dir = Path(__file__).resolve().parent
    entry_point_path = script_dir / "aws_train_model.py"
    if not entry_point_path.exists():
        raise ValueError(
            f'No file named "aws_train_model.py" was found in directory "{script_dir}". '
            "Ensure aws_train_model.py is alongside launch_sagemaker_job.py."
        )

    if args.train_s3_uri:
        train_s3_uri = args.train_s3_uri
        print(f"\nUsing provided training S3 URI: {train_s3_uri}")
    else:
        s3 = boto3.client("s3", region_name=args.region)
        local_dir = (Path(args.local_train_dir)
                     if Path(args.local_train_dir).is_absolute()
                     else (script_dir / args.local_train_dir).resolve())
        train_s3_uri = ensure_data_in_s3(local_dir, s3, args.bucket, args.prefix)

    job_name = args.job_name or f"eth-lstm-opt-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    print("\n=== SageMaker Training Job Configuration ===")
    print(f"Region:             {args.region}")
    print(f"Role ARN:           {args.role_arn}")
    print(f"Instance Type:      {args.instance_type}")
    print(f"Instance Count:     {args.instance_count}")
    print(f"Framework Version:  {args.framework_version}")
    print(f"Python Version:     {args.py_version}")
    print(f"Image URI:          {args.image_uri or '[default for framework]'}")
    print(f"Train S3 URI:       {train_s3_uri}")
    print(f"Bucket/Prefix:      {args.bucket} / {args.prefix}")
    print(f"Job Name:           {job_name}")

    hyperparameters = {
        "window-size": args.window_size,
        "hidden-size": args.hidden_size,
        "num-layers": args.num_layers,
        "dropout": args.dropout,
        "bidirectional": True if args.bidirectional else False,
        "epochs": args.epochs,
        "batch-size": args.batch_size,
        "learning-rate": args.learning_rate,
        "weight-decay": args.weight_decay,
        "val-frac": args.val_frac,
        "accumulate": args.accumulate,
        "seed": args.seed,
        "price-col": args.price_col,
    }
    if args.disable_scaling:
        hyperparameters["disable-scaling"] = True

    print("\n--- Trainer Hyperparameters (supported only) ---")
    for k in sorted(hyperparameters.keys()):
        print(f"{k:16s}: {hyperparameters[k]}")

    estimator_kwargs = dict(
        entry_point=entry_point_path.name,
        source_dir=str(script_dir),
        role=args.role_arn,
        instance_type=args.instance_type,
        instance_count=args.instance_count,
        framework_version=args.framework_version,
        py_version=args.py_version,
        image_uri=args.image_uri,
        hyperparameters=hyperparameters,
    )
    est = PyTorch(**{k: v for k, v in estimator_kwargs.items() if v is not None})

    inputs = {"train": TrainingInput(s3_data=train_s3_uri, content_type="text/csv")}
    print("\nStarting SageMaker training job...")
    est.fit(inputs=inputs, job_name=job_name)
    print("\nTraining job submitted. Check SageMaker Console for live logs and metrics.")
    print(f"Job Name: {job_name}")

    if args.deploy:
        endpoint_name = args.endpoint_name or f"{job_name}-endpoint"
        print("\n=== Deployment Configuration ===")
        print(f"Endpoint Name:           {endpoint_name}")
        print(f"Endpoint Instance Type:  {args.endpoint_instance_type}")
        print(f"Endpoint Initial Count:  {args.endpoint_initial_count}")
        print(f"Serving Entry Point:     {args.deploy_entry_point}")

        model_data = est.latest_training_job.model_artifacts
        sm_model = PyTorchModel(
            model_data=model_data,
            role=args.role_arn,
            entry_point=args.deploy_entry_point,
            source_dir=str(script_dir),
            framework_version=args.framework_version,
            py_version=args.py_version,
            image_uri=args.image_uri,
        )
        print("\nDeploying model to endpoint...")
        predictor = sm_model.deploy(
            initial_instance_count=args.endpoint_initial_count,
            instance_type=args.endpoint_instance_type,
            endpoint_name=endpoint_name,
        )
        print(f"Endpoint deployed: {predictor.endpoint_name}")
        print("\n=== Deployment Complete ===")
    else:
        print("\n--deploy was disabled; skipping endpoint creation.")

if __name__ == "__main__":
    main()