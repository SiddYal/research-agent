"""Report artifacts in S3-compatible object storage (Neon Object Storage, S3, R2...).

Postgres keeps the object keys; readers get short-lived presigned links, so the
bucket stays private.
"""

from __future__ import annotations

from typing import Protocol

from research_agent.config import Settings
from research_agent.logging import get_logger

log = get_logger(__name__)


class ArtifactStore(Protocol):
    def put(self, key: str, body: bytes, content_type: str) -> None: ...
    def url(self, key: str, expires_s: int = 3600) -> str: ...


class S3ArtifactStore:
    def __init__(self, settings: Settings, client=None):
        if client is None:
            import boto3
            from botocore.config import Config

            client = boto3.client(
                "s3",
                endpoint_url=settings.aws_endpoint_url_s3,
                region_name=settings.aws_region,
                aws_access_key_id=settings.aws_access_key_id,
                aws_secret_access_key=settings.aws_secret_access_key,
                # Neon requires path-style addressing and SigV4.
                config=Config(s3={"addressing_style": "path"}, signature_version="s3v4",
                              retries={"max_attempts": 4, "mode": "standard"},
                              connect_timeout=10, read_timeout=30),
            )
        self.client = client
        self.bucket = settings.reports_bucket

    def put(self, key: str, body: bytes, content_type: str) -> None:
        self.client.put_object(Bucket=self.bucket, Key=key, Body=body, ContentType=content_type)

    def url(self, key: str, expires_s: int = 3600) -> str:
        return self.client.generate_presigned_url(
            "get_object", Params={"Bucket": self.bucket, "Key": key}, ExpiresIn=expires_s
        )


def make_artifact_store(settings: Settings) -> ArtifactStore | None:
    """None (local files only) unless a bucket and an S3 endpoint are configured."""
    if not (settings.reports_bucket and settings.aws_endpoint_url_s3):
        return None
    log.info("artifact_store", bucket=settings.reports_bucket, endpoint=settings.aws_endpoint_url_s3)
    return S3ArtifactStore(settings)
