"""Download the real-image evaluation sources (TUM RGB-D Pioneer sequences, GND bags).

Run on the login node (thor shares the NFS):

    uv run scripts/download_real_eval.py --tum                   # 4 fr2 pioneer tgz + extract
    uv run scripts/download_real_eval.py --tum pioneer_slam      # one sequence
    uv run scripts/download_real_eval.py --gnd-list              # list GND Dataverse files
    uv run scripts/download_real_eval.py --gnd AU_chunk01.bag    # one ~3 GB bag chunk

TUM  -> data/real/tum/rgbd_dataset_freiburg2_<seq>/   (rgb/, depth/, groundtruth.txt, ...)
GND  -> data/real/gnd/<file>

Idempotent: a file whose size matches the server's Content-Length is skipped; a partial
file is resumed (HTTP Range). Extraction is skipped when the sequence dir exists.
"""

import argparse
import json
import sys
import tarfile
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TUM_DIR = REPO_ROOT / "data" / "real" / "tum"
GND_DIR = REPO_ROOT / "data" / "real" / "gnd"

TUM_BASE = "https://cvg.cit.tum.de/rgbd/dataset/freiburg2/"
TUM_SEQUENCES = ["pioneer_360", "pioneer_slam", "pioneer_slam2", "pioneer_slam3"]

GND_DOI = "doi:10.13021/orc2020/JUIW5F"
GND_API = "https://dataverse.orc.gmu.edu/api"


def _remote_size(url: str) -> int | None:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req) as r:
        cl = r.headers.get("Content-Length")
        return int(cl) if cl else None


def fetch(url: str, dest: Path, chunk: int = 1 << 20) -> Path:
    """Stream `url` to `dest`, resuming a partial file. Prints progress every ~100 MB."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    total = _remote_size(url)
    have = dest.stat().st_size if dest.exists() else 0
    if total is not None and have == total:
        print(f"  skip (complete): {dest.name} ({have / 1e9:.2f} GB)")
        return dest
    headers = {"Range": f"bytes={have}-"} if have else {}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as r, open(dest, "ab" if have else "wb") as f:
        if have and r.status != 206:  # server ignored Range: start over
            f.seek(0)
            f.truncate()
            have = 0
        done, mark = have, have
        while True:
            buf = r.read(chunk)
            if not buf:
                break
            f.write(buf)
            done += len(buf)
            if done - mark >= 100 << 20:
                mark = done
                pct = f" ({100 * done / total:.0f}%)" if total else ""
                print(f"  {dest.name}: {done / 1e9:.2f} GB{pct}", flush=True)
    print(f"  done: {dest.name} ({done / 1e9:.2f} GB)")
    return dest


def download_tum(sequences: list[str]) -> None:
    for seq in sequences:
        name = f"rgbd_dataset_freiburg2_{seq}"
        out_dir = TUM_DIR / name
        if (out_dir / "groundtruth.txt").exists():
            print(f"[tum] {name}: already extracted")
            continue
        tgz = fetch(f"{TUM_BASE}{name}.tgz", TUM_DIR / f"{name}.tgz")
        print(f"[tum] extracting {tgz.name} ...", flush=True)
        with tarfile.open(tgz) as tf:
            tf.extractall(TUM_DIR, filter="data")
        assert (out_dir / "groundtruth.txt").exists(), f"unexpected archive layout in {tgz}"
        print(f"[tum] {name}: ok -> {out_dir}")


def gnd_files() -> list[dict]:
    url = f"{GND_API}/datasets/:persistentId/?persistentId={GND_DOI}"
    with urllib.request.urlopen(url) as r:
        data = json.load(r)
    files = data["data"]["latestVersion"]["files"]
    return [
        {"name": f["dataFile"]["filename"], "id": f["dataFile"]["id"],
         "size": f["dataFile"]["filesize"]}
        for f in files
    ]


def download_gnd(names: list[str]) -> None:
    index = {f["name"]: f for f in gnd_files()}
    for name in names:
        if name not in index:
            sys.exit(f"[gnd] no such file: {name} (use --gnd-list)")
        f = index[name]
        print(f"[gnd] {name} ({f['size'] / 1e9:.2f} GB)")
        fetch(f"{GND_API}/access/datafile/{f['id']}", GND_DIR / name)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tum", nargs="*", metavar="SEQ",
                   help=f"TUM fr2 sequences to fetch (default all: {' '.join(TUM_SEQUENCES)})")
    p.add_argument("--gnd", nargs="*", metavar="FILE", help="GND Dataverse file names to fetch")
    p.add_argument("--gnd-list", action="store_true", help="list GND files and exit")
    args = p.parse_args()
    if args.gnd_list:
        for f in sorted(gnd_files(), key=lambda f: f["name"]):
            print(f"{f['size'] / 1e9:7.2f} GB  {f['name']}")
        return
    if args.tum is None and not args.gnd:
        p.error("nothing to do: pass --tum [SEQ ...], --gnd FILE ..., or --gnd-list")
    if args.tum is not None:
        download_tum(args.tum or TUM_SEQUENCES)
    if args.gnd:
        download_gnd(args.gnd)


if __name__ == "__main__":
    main()
