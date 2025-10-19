import io
import os
from typing import Optional

import pandas as pd

try:
    import boto3
    from botocore.exceptions import BotoCoreError, NoCredentialsError
except Exception:  # boto3 optional at import-time for local-only use
    boto3 = None
    BotoCoreError = Exception
    NoCredentialsError = Exception


def read_csv(path: str, boto3_session: Optional[object] = None, **pd_kwargs) -> pd.DataFrame:
    """Read a CSV from local filesystem or S3.

    Supports paths like:
      - local: /app/data/file.csv or relative paths
      - s3: s3://bucket/key.csv

    Credentials resolution strategy (no `aws configure` needed inside container):
      - If running in AWS (EC2/ECS/EKS/CodeBuild), the IAM role attached to the instance/task is used automatically.
      - If AWS_ACCESS_KEY_ID/SECRET/SESSION are provided as environment variables, they will be used.
      - If a boto3_session is passed, that session is used.
    """
    if path.startswith("s3://"):
        if boto3 is None:
            raise RuntimeError("boto3 is required to read from S3. Please ensure it is installed in the image.")

        # Build/resolve session
        session = boto3_session
        if session is None:
            # If env vars exist, let boto3 pick them up automatically; otherwise it will use instance metadata/IRSA
            session = boto3.Session()

        s3 = session.client("s3")
        # Parse bucket/key
        no_scheme = path[len("s3://") :]
        bucket, key = no_scheme.split("/", 1)
        try:
            obj = s3.get_object(Bucket=bucket, Key=key)
            body = obj["Body"].read()
        except (BotoCoreError, NoCredentialsError) as e:
            raise RuntimeError(f"Failed to read {path}: {e}")
        return pd.read_csv(io.BytesIO(body), **pd_kwargs)
    else:
        return pd.read_csv(path, **pd_kwargs)


__all__ = ["read_csv"]
