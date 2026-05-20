from __future__ import annotations

import json

import pytest

from src.kiosk.allowlist import (
    build_allowlist_config,
    build_chromium_policies,
    normalize_host,
    write_extension_config,
)
from src.kiosk.profile import ChromiumProfileCleaner, ProfileCleanupError


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("prairielearn.org", "prairielearn.org"),
        ("https://PrairieLearn.org/pl/course", "prairielearn.org"),
        ("*.example.edu.br", "example.edu.br"),
        ("127.0.0.1:8000/status", "127.0.0.1"),
        ("", None),
        ("https://*.bad", None),
    ],
)
def test_normalize_host(raw, expected):
    assert normalize_host(raw) == expected


def test_build_allowlist_config_includes_start_url_host_and_deduplicates():
    config = build_allowlist_config(
        start_url="https://prairielearn.org/pl/course",
        allowlist=["example.edu", "https://example.edu/path", "cdn.example.edu"],
    )

    assert [site.host for site in config.sites] == [
        "cdn.example.edu",
        "example.edu",
        "login.microsoftonline.com",
        "prairielearn.org",
        "us.prairielearn.com",
    ]


def test_write_extension_config_writes_small_runtime_file(tmp_path):
    config = build_allowlist_config(
        start_url="https://prairielearn.org/pl",
        allowlist=["example.edu"],
    )

    path = write_extension_config(config, extension_dir=tmp_path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["startUrl"] == "https://prairielearn.org/pl"
    assert {site["host"] for site in payload["sites"]} == {
        "prairielearn.org",
        "login.microsoftonline.com",
        "us.prairielearn.com",
        "example.edu",
    }


def test_build_chromium_policies_blocks_by_default_and_allows_sites():
    config = build_allowlist_config(
        start_url="https://prairielearn.org/pl",
        allowlist=["example.edu"],
    )

    policies = build_chromium_policies(config, extension_id="abcdefghijklmnop")

    assert policies["URLBlocklist"] == ["*"]
    assert "https://prairielearn.org/*" in policies["URLAllowlist"]
    assert "https://login.microsoftonline.com/*" in policies["URLAllowlist"]
    assert "https://us.prairielearn.com/*" in policies["URLAllowlist"]
    assert "https://*.example.edu/*" in policies["URLAllowlist"]
    assert policies["DeveloperToolsAvailability"] == 2
    assert policies["IncognitoModeAvailability"] == 1
    assert policies["ExtensionInstallAllowlist"] == ["abcdefghijklmnop"]


def test_profile_cleaner_removes_dedicated_profile(tmp_path):
    profile = tmp_path / "proctor-chromium-profile"
    (profile / "Default" / "Local Storage").mkdir(parents=True)
    (profile / "Default" / "Cookies").write_text("cookie", encoding="utf-8")

    ChromiumProfileCleaner(profile).cleanup()

    assert not profile.exists()


def test_profile_cleaner_rejects_unsafe_target(tmp_path):
    profile = tmp_path / "regular-profile"
    profile.mkdir()

    with pytest.raises(ProfileCleanupError):
        ChromiumProfileCleaner(profile).cleanup()
