import boto3

import logging


s3client = None
logger = logging.getLogger(__file__)


def init_s3_client(access_key, secret_key, endpoint_url, region_name="us"):
    global s3client
    s3client = boto3.client(
        "s3",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        endpoint_url=endpoint_url,
        region_name=region_name,
    )


def get_s3_url(bucket_name, k):
    global s3client
    if not s3client:
        return None
    try:
        return s3client.generate_presigned_url(
            "get_object", Params={"Key": k, "Bucket": bucket_name}
        )
    except Exception:
        logger.warning(f"Cannot get URL for {k} in {bucket_name}.")
        return None
