#!/usr/bin/env python
"""Download and unpack the PushT zarr replay buffer."""

from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path


DEFAULT_URL = "https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/pusht")
    parser.add_argument("--url", default=DEFAULT_URL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / "pusht.zip"
    if not zip_path.exists():
        print(f"Downloading {args.url} -> {zip_path}")
        urllib.request.urlretrieve(args.url, zip_path)
    else:
        print(f"Using existing {zip_path}")
    print(f"Unpacking {zip_path}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)

    zarr_dirs = sorted(out_dir.glob("*.zarr"))
    if not zarr_dirs:
        nested = sorted(out_dir.rglob("*.zarr"))
        zarr_dirs = nested
    if not zarr_dirs:
        raise FileNotFoundError(f"No .zarr directory found after unpacking {zip_path}")

    canonical = out_dir / "pusht_cchi_v7_replay.zarr"
    if zarr_dirs[0] != canonical and not canonical.exists():
        if zarr_dirs[0].is_dir():
            shutil.move(str(zarr_dirs[0]), canonical)
        else:
            raise FileNotFoundError(f"Unexpected zarr path: {zarr_dirs[0]}")
    print(f"PushT zarr ready: {canonical}")


if __name__ == "__main__":
    main()
