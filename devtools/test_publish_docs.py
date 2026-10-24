"""Check which documentation publications are allowed to move the stable alias."""

import json
from unittest.mock import patch

import pytest
from publish_docs import publication, publish


@pytest.mark.parametrize(
    "version,prerelease,expected",
    [
        ("dev", False, ("dev", [])),
        ("v3.5.0", False, ("3.5.0", ["stable"])),
        ("v3.5.1", False, ("3.5.1", ["stable"])),
        ("v3.5.0rc1", False, ("3.5.0rc1", [])),
        ("v3.5.0.dev1", False, ("3.5.0.dev1", [])),
        ("v3.5.0", True, ("3.5.0", [])),
    ],
)
def test_publication(version, prerelease, expected):
    assert publication(version, prerelease) == expected


@pytest.mark.parametrize("has_stable,default", [(False, "dev"), (True, "stable")])
def test_development_default(has_stable, default):
    versions = [{"version": "dev", "aliases": []}]
    if has_stable:
        versions.append({"version": "3.5.0", "aliases": ["stable"]})
    with (
        patch("publish_docs.subprocess.run") as run,
        patch("publish_docs.subprocess.check_output", return_value=json.dumps(versions)),
    ):
        publish("dev", push=True)
    assert run.call_args_list[0].args[0] == ["mike", "deploy", "--update-aliases", "dev"]
    assert run.call_args_list[-1].args[0] == ["mike", "set-default", default, "--push"]
