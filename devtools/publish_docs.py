"""Publish MkDocs versions without promoting development or prerelease docs to stable."""

import argparse
import json
import subprocess

from packaging.version import Version


def publication(version: str, prerelease: bool = False) -> tuple[str, list[str]]:
    """Return the version directory and any aliases to update."""
    if version == "dev":
        return "dev", []
    release = Version(version)
    aliases = [] if prerelease or release.is_prerelease or release.is_devrelease else ["stable"]
    return str(release), aliases


def publish(version: str, prerelease: bool = False, push: bool = False) -> None:
    """Build a version and select an existing default before optionally pushing."""
    version, aliases = publication(version, prerelease)
    subprocess.run(["mike", "deploy", "--update-aliases", version, *aliases], check=True)
    versions = json.loads(subprocess.check_output(["mike", "list", "--json"], text=True))
    identifiers = {identifier for item in versions for identifier in [item["version"], *item["aliases"]]}
    default = "stable" if "stable" in identifiers else "dev" if "dev" in identifiers else version
    command = ["mike", "set-default", default]
    if push:
        command.append("--push")
    subprocess.run(command, check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, help="Release tag or 'dev' for main.")
    parser.add_argument("--prerelease", action="store_true", help="Do not move the stable alias.")
    parser.add_argument("--push", action="store_true", help="Push the generated gh-pages commits.")
    args = parser.parse_args()
    publish(args.version, args.prerelease, args.push)
