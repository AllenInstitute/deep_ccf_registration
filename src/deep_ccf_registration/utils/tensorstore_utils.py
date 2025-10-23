from urllib.parse import urlparse


def create_kvstore(path: str, aws_credentials_method: str = "default"):
    """
    Create tensorstore kvstore

    Parameters
    ----------
    path
    aws_credentials_method

    Returns
    -------

    """

    def parse_s3_uri(s3_uri):
        parsed = urlparse(s3_uri)
        bucket = parsed.netloc
        key = parsed.path.lstrip('/')
        return bucket, key

    if path.startswith("s3://"):
        bucket, key = parse_s3_uri(s3_uri=path)
        kvstore = {
            "driver": "s3",
            "bucket": bucket,
            "path": key,
            "aws_credentials": {"type": aws_credentials_method},
        }
    else:
        kvstore = {"driver": "file", "path": path}
    return kvstore
