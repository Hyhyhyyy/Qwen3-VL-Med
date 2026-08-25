#!/usr/bin/env python3
"""Scan every blob reachable from Git refs without printing matched content."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath


MAX_BLOB_BYTES = 5 * 1024 * 1024
BLOCKED_SUFFIXES = {
    ".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".onnx", ".gguf",
    ".h5", ".hdf5", ".npz", ".npy", ".dcm", ".dicom", ".svs",
    ".ndpi", ".mrxs", ".tif", ".tiff", ".jpg", ".jpeg", ".png",
    ".parquet", ".zip", ".7z", ".rar", ".tar", ".gz", ".zst", ".xz",
    ".bz2", ".gpg", ".age", ".pem", ".key",
}
ALLOWED_HISTORICAL_DATA_FILES = {
    "docs/environment_config.csv",
    "docs/metrics/metric_dictionary.csv",
    "docs/metrics/r04_r07_pareto_template.csv",
    "docs/metrics/r04_r07_split_template.csv",
}


def git(*args: str) -> bytes:
    return subprocess.run(
        ["git", *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    ).stdout


def patterns() -> list[tuple[str, re.Pattern[str]]]:
    key_head = "BEGIN" + r"[ A-Z]*PRIVATE KEY"
    openai = "s" + r"k-(?:proj-|svcacct-)?[A-Za-z0-9_-]{16,}"
    private_mount = "/course" + r"[0-9]{2,}"
    cloud_host = "px-" + r"cloud[0-9]*"
    provider = "mat" + "pool"
    root_endpoint = "root" + "@"
    legacy_private_counts = r"\b(?:19" + "082|18" + "426|6" + "56|3" + "27)\b"
    disclosed_data_scale = r"(?:共有|剩余|训练集|测试集)[^\r\n]{0,40}\d+\s*(?:条|例|张)"
    return [
        ("private-key-header", re.compile(key_head)),
        ("github-token", re.compile(r"gh[opusr]_[A-Za-z0-9_]{20,}")),
        ("aws-access-key", re.compile(r"AKIA[0-9A-Z]{16}")),
        ("openai-api-key", re.compile(openai)),
        ("credential-in-url", re.compile(r"https?://[^\s/:]+:[^\s/@]+@")),
        ("private-mount", re.compile(private_mount, re.IGNORECASE)),
        ("private-cloud-host", re.compile(cloud_host, re.IGNORECASE)),
        ("private-provider", re.compile(provider, re.IGNORECASE)),
        ("privileged-ssh-endpoint", re.compile(root_endpoint, re.IGNORECASE)),
        ("windows-user-path", re.compile(r"[A-Za-z]:\\Users\\", re.IGNORECASE)),
        ("legacy-private-count", re.compile(legacy_private_counts)),
        ("disclosed-data-scale", re.compile(disclosed_data_scale)),
        ("git-lfs-pointer", re.compile(r"version\s+https://git-lfs\.github\.com/spec/v1")),
        ("phone-number", re.compile(r"\b1[3-9][0-9]{9}\b")),
        ("national-id", re.compile(r"\b[0-9]{17}[0-9Xx]\b")),
        (
            "medical-record-identifier",
            re.compile(
                r"(?:MRN|medical_record_number|accession_number)[\s\"']*[:=][\s\"']*[A-Za-z0-9-]{4,}",
                re.IGNORECASE,
            ),
        ),
    ]


def suspicious_filename(path: str) -> bool:
    name = PurePosixPath(path.replace("\\", "/")).name.lower()
    return name in {
        ".env",
        "id_rsa",
        "id_ed25519",
        "authorized_keys",
        "known_hosts",
        "credentials.json",
    } or name.endswith((".pem", ".key"))


def main() -> int:
    repo = Path(__file__).resolve().parent.parent
    os.chdir(repo)
    objects = git("rev-list", "--objects", "--all").decode("utf-8", "replace").splitlines()
    paths_by_oid: dict[str, set[str]] = {}
    for line in objects:
        oid, _, path = line.partition(" ")
        if path:
            paths_by_oid.setdefault(oid, set()).add(path)

    findings: list[str] = []
    scanned = 0
    compiled = patterns()
    for oid, paths in paths_by_oid.items():
        if git("cat-file", "-t", oid).decode().strip() != "blob":
            continue
        size = int(git("cat-file", "-s", oid).decode())
        label = sorted(paths)[0]
        normalized_paths = {path.replace("\\", "/") for path in paths}
        for path in normalized_paths:
            suffix = PurePosixPath(path).suffix.lower()
            if suffix in BLOCKED_SUFFIXES and path not in ALLOWED_HISTORICAL_DATA_FILES:
                findings.append(f"blocked historical file type: {path} ({oid[:12]})")
        if size > MAX_BLOB_BYTES:
            findings.append(f"oversized historical blob: {label} ({size} bytes, {oid[:12]})")
            continue
        if any(suspicious_filename(path) for path in paths):
            findings.append(f"sensitive filename in history: {label} ({oid[:12]})")
        data = git("cat-file", "-p", oid)
        if b"\x00" in data[:8192]:
            scanned += 1
            continue
        text = data.decode("utf-8", "replace")
        for rule, regex in compiled:
            if regex.search(text):
                findings.append(f"{rule}: {label} ({oid[:12]})")
        scanned += 1

    if findings:
        for finding in sorted(set(findings)):
            print(f"HISTORY_PRIVACY_ERROR {finding}", file=sys.stderr)
        return 1
    print(f"HISTORY_PRIVACY_AUDIT_OK blobs={scanned} max_blob_bytes={MAX_BLOB_BYTES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
