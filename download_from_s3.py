#!/usr/bin/env python3
"""
Download one file from S3 with a terminal progress bar.

Usage:
  python download_from_s3.py
  python download_from_s3.py --bucket hsihsak --key database/embedded_output.jsonl --out-dir .
"""

from __future__ import annotations

import argparse
import threading
from pathlib import Path

import boto3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download one file from S3.")
    parser.add_argument("--bucket", default="hsihsak", help="S3 bucket name")
    parser.add_argument(
        "--key",
        default="database/embedded_output.jsonl",
        help="S3 object key for the file to download",
    )
    parser.add_argument(
        "--out-dir",
        default=".",
        help="Local output directory for the downloaded file",
    )
    parser.add_argument("--region", default=None, help="Optional AWS region")
    parser.add_argument("--profile", default=None, help="Optional AWS profile name")
    return parser.parse_args()


def _format_size(size_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f}{unit}"
        value /= 1024
    return f"{size_bytes}B"


class ProgressBar:
    def __init__(self, total_bytes: int) -> None:
        self.total = max(total_bytes, 1)
        self.downloaded = 0
        self.bar_width = 32
        self._lock = threading.Lock()

    def __call__(self, bytes_amount: int) -> None:
        with self._lock:
            self.downloaded += bytes_amount
            if self.downloaded > self.total:
                self.downloaded = self.total
            self._render()

    def _render(self) -> None:
        fraction = self.downloaded / self.total
        filled = int(self.bar_width * fraction)
        bar = "#" * filled + "-" * (self.bar_width - filled)
        percent = fraction * 100
        done = _format_size(self.downloaded)
        total = _format_size(self.total)
        remaining = _format_size(self.total - self.downloaded)
        print(
            f"\r[{bar}] {percent:6.2f}%  {done}/{total}  remaining: {remaining}",
            end="",
            flush=True,
        )

    def finish(self) -> None:
        with self._lock:
            self.downloaded = self.total
            self._render()
            print()


def main() -> None:
    args = parse_args()

    session_kwargs = {}
    if args.profile:
        session_kwargs["profile_name"] = args.profile

    client_kwargs = {}
    if args.region:
        client_kwargs["region_name"] = args.region

    session = boto3.Session(**session_kwargs)
    s3 = session.client("s3", **client_kwargs)

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    key = args.key.lstrip("/")
    local_path = out_dir / Path(key).name

    metadata = s3.head_object(Bucket=args.bucket, Key=key)
    total_size = int(metadata.get("ContentLength", 0))
    progress = ProgressBar(total_size)

    print(f"Downloading s3://{args.bucket}/{key} -> {local_path}")
    s3.download_file(args.bucket, key, str(local_path), Callback=progress)
    progress.finish()
    print("Download complete.")


if __name__ == "__main__":
    main()