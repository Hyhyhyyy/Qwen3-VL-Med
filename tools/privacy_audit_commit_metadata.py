#!/usr/bin/env python3
"""Scan public commit metadata without preventing outside contributions."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

from privacy_audit_history import patterns


def git_text(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    ).stdout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-name")
    parser.add_argument("--expected-email")
    parser.add_argument(
        "--require-single-contributor",
        action="store_true",
        help="Release-maintainer audit only; do not use this mode for pull-request CI.",
    )
    args = parser.parse_args()
    if args.require_single_contributor and not (args.expected_name and args.expected_email):
        parser.error("strict mode requires --expected-name and --expected-email")

    repo = Path(__file__).resolve().parent.parent
    os.chdir(repo)

    commits = [line for line in git_text("rev-list", "--all").splitlines() if line]
    findings: list[str] = []
    contributors: set[tuple[str, str]] = set()
    message_patterns = patterns()
    for commit in commits:
        raw = git_text("show", "-s", "--format=%an%x00%ae%x00%cn%x00%ce%x00%B", commit)
        author_name, author_email, committer_name, committer_email, message = raw.split("\0", 4)
        contributors.add((author_name, author_email))
        short = commit[:12]
        if args.require_single_contributor:
            if author_name != args.expected_name or author_email != args.expected_email:
                findings.append(f"unexpected author identity: {short}")
            if committer_name != args.expected_name or committer_email != args.expected_email:
                findings.append(f"unexpected committer identity: {short}")
        for rule, regex in message_patterns:
            for field_name, value in (
                ("author metadata", f"{author_name} {author_email}"),
                ("committer metadata", f"{committer_name} {committer_email}"),
                ("commit message", message),
            ):
                if regex.search(value):
                    findings.append(f"{rule} in {field_name}: {short}")

    if findings:
        for finding in sorted(set(findings)):
            print(f"COMMIT_METADATA_PRIVACY_ERROR {finding}", file=sys.stderr)
        return 1
    print(
        "COMMIT_METADATA_PRIVACY_OK "
        f"commits={len(commits)} contributors={len(contributors)} "
        f"strict={args.require_single_contributor}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
