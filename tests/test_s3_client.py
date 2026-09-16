from __future__ import annotations

from unittest.mock import MagicMock, patch

from botocore.exceptions import ClientError

from src.core.config import S3Config
from src.core.s3_client import S3Client


def _not_found() -> ClientError:
    return ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject")


def test_generate_student_photo_url_returns_presigned_url_for_first_existing_extension():
    with patch("src.core.s3_client.boto3.client") as make_client:
        boto_client = MagicMock()
        make_client.return_value = boto_client
        boto_client.head_object.side_effect = [_not_found(), None]
        boto_client.generate_presigned_url.return_value = "https://signed.example/felipehl.jpg"

        s3 = S3Client(S3Config(bucket="bucket-test"))
        url = s3.generate_student_photo_url("ES2025-T1", "felipehl")

    assert url == "https://signed.example/felipehl.jpg"
    boto_client.generate_presigned_url.assert_called_once_with(
        "get_object",
        Params={"Bucket": "bucket-test", "Key": "fotos/ES2025-T1/felipehl.jpg"},
        ExpiresIn=3600,
    )


def test_generate_student_photo_url_returns_none_when_no_extension_matches():
    with patch("src.core.s3_client.boto3.client") as make_client:
        boto_client = MagicMock()
        make_client.return_value = boto_client
        boto_client.head_object.side_effect = _not_found()

        s3 = S3Client(S3Config(bucket="bucket-test"))
        url = s3.generate_student_photo_url("ES2025-T1", "ghost")

    assert url is None
    boto_client.generate_presigned_url.assert_not_called()
