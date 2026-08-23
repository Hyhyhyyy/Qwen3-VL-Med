#!/usr/bin/env python3
"""Require one public contributor identity and scan commit messages safely."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys

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
    parser.add_argument("--expected-name", required=True)
    parser.add_argument("--expected-email", required=True)
    args = parser.parse_args()

    commits = [line for line in git_text("rev-list", "--all").splitlines() if line]
    findings: list[str] = []
    message_patterns = patterns()
    for commit in commits:
        raw = git_text("show", "-s", "--format=%an%x00%ae%x00%cn%x00%ce%x00%B", commit)
        author_name, author_email, committer_name, committer_email, message = raw.split("\0", 4)
        short = commit[:12]
        if author_name != args.expected_name or author_email != args.expected_email:
            findings.append(f"unexpected author identity: {short}")
        if committer_name != args.expected_name or committer_email != args.expected_email:
            findings.append(f"unexpected committer identity: {short}")
        if re.search(r"(?im)^\s*co-authored-by\s*:", message):
            findings.append(f"co-author trailer is not permitted: {short}")
        for rule, regex in message_patterns:
            if regex.search(message):
                findings.append(f"{rule} in commit message: {short}")

    if findings:
        for finding in sorted(set(findings)):
            print(f"COMMIT_METADATA_PRIVACY_ERROR {finding}", file=sys.stderr)
        return 1
    print(f"COMMIT_METADATA_PRIVACY_OK commits={len(commits)} contributors=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
