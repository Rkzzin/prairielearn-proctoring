from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.core.config import S3Config
from src.core.s3_client import S3Client


def test_generate_student_photo_url_candidates_signs_all_extensions_without_network_calls():
    with patch("src.core.s3_client.boto3.client") as make_client:
        boto_client = MagicMock()
        make_client.return_value = boto_client
        boto_client.generate_presigned_url.side_effect = [
            "https://signed.example/felipehl.jpg",
            "https://signed.example/felipehl.png",
            "https://signed.example/felipehl.jpeg",
        ]

        s3 = S3Client(S3Config(bucket="bucket-test"))
        urls = s3.generate_student_photo_url_candidates("ES2025-T1", "felipehl")

    assert urls == [
        "https://signed.example/felipehl.jpg",
        "https://signed.example/felipehl.png",
        "https://signed.example/felipehl.jpeg",
    ]
    # Presign é local — não pode bater na rede (head_object) pra montar os candidatos,
    # senão a rota /sessions/{id} trava esperando o S3 (era exatamente o bug: 3
    # round-trips síncronos deixavam a página lenta pra abrir).
    boto_client.head_object.assert_not_called()
    assert boto_client.generate_presigned_url.call_args_list[0].kwargs["Params"] == {
        "Bucket": "bucket-test",
        "Key": "fotos/ES2025-T1/felipehl.jpg",
    }
    assert boto_client.generate_presigned_url.call_args_list[1].kwargs["Params"]["Key"] == (
        "fotos/ES2025-T1/felipehl.png"
    )
    assert boto_client.generate_presigned_url.call_args_list[2].kwargs["Params"]["Key"] == (
        "fotos/ES2025-T1/felipehl.jpeg"
    )
