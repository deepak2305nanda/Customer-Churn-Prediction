from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import requests


DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"


@dataclass(frozen=True)
class DownloadResult:
    path: Path
    sha256: str
    bytes: int


def _sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def download(out_path: Path | None = None, url: str = DATA_URL, timeout_s: int = 60) -> DownloadResult:
    project_root = Path(__file__).resolve().parents[2]
    if out_path is None:
        out_path = project_root / "data" / "raw" / "telco_churn.csv"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    resp = requests.get(url, timeout=timeout_s)
    resp.raise_for_status()
    content = resp.content

    sha = _sha256_bytes(content)
    out_path.write_bytes(content)

    return DownloadResult(path=out_path, sha256=sha, bytes=len(content))


def main() -> None:
    # Allow overriding the URL for experimentation.
    url = os.environ.get("CHURN_DATA_URL", DATA_URL)
    result = download(url=url)
    print("download OK", {"path": str(result.path), "bytes": result.bytes, "sha256": result.sha256[:12] + "..."})


if __name__ == "__main__":
    main()

