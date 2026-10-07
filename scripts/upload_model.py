"""Publish an approved model and its report to the configured S3 bucket."""

import argparse
import json
import os
from pathlib import Path
import sys

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from src.quality import check_quality


def publish_model(bucket: str, model_path: Path, report_path: Path):
    if not bucket or bucket.startswith("s3://"):
        raise ValueError("ARTIFACT_BUCKET must be a bucket name without s3://")
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    check_quality(report["f1_score"])
    region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    client = boto3.client("s3", region_name=region)
    client.upload_file(str(model_path), bucket, "artifacts/current/model.joblib")
    client.upload_file(
        str(report_path), bucket, "artifacts/current/report.json",
        ExtraArgs={"ContentType": "application/json"},
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("models/model.joblib"))
    parser.add_argument("--report", type=Path, default=Path("outputs/report.json"))
    args = parser.parse_args()
    try:
        publish_model(os.environ.get("ARTIFACT_BUCKET", ""), args.model, args.report)
    except (BotoCoreError, ClientError, OSError, ValueError, KeyError) as error:
        print(f"Publication failed: {error}", file=sys.stderr)
        return 1
    print("Approved model and report published to S3.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
