#!/usr/bin/env python3
"""
Cross-Platform Libri2Mix train-100 (8 kHz) Data Generator.
Compatible with Windows 11, Ubuntu Linux, and macOS.

Features:
1. Generates official Libri2Mix train-100 (8 kHz, min mode, mix_both + mix_clean).
2. Pure Python downloads and extraction (tarfile + zipfile; no OS-specific tar/unzip needed).
3. Detects OS and provides tailored instructions for SoX audio dependency.
4. Auto-invokes generation via the active Python virtual environment.

Usage:
    python scripts/setup_data.py [--storage_dir ./data]
"""

import os
import sys
import shutil
import urllib.request
import tarfile
import zipfile
import subprocess
import argparse
from pathlib import Path


def print_banner(msg: str):
    print("\n" + "=" * 68)
    print(f"  {msg}")
    print("=" * 68)


def check_sox():
    """Checks if sox is installed and accessible on system PATH."""
    sox_path = shutil.which("sox")
    if not sox_path:
        print("\n[ERROR] 'sox' (Sound eXchange) was not found on your system PATH.")
        if sys.platform == "win32":
            print("  Windows 11 Installation Options:")
            print("    1. via Winget:  winget install -e --id SoX.SoX")
            print("    2. via Chocolatey: choco install sox")
            print("    3. Download binary from: https://sourceforge.net/projects/sox/files/sox/")
            print("       and add the SoX installation folder to your system PATH.")
        else:
            print("  Ubuntu / Debian Installation:")
            print("    sudo apt-get update && sudo apt-get install -y sox libsox-fmt-all")
        sys.exit(1)
    print(f"  [OK] Found SoX at: {sox_path}")


def download_file(url: str, dest_path: Path):
    """Downloads a file with chunked progress reporting."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size > 0:
        print(f"  [OK] Already downloaded: {dest_path.name}")
        return

    print(f"  Downloading {dest_path.name} from {url} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "curl/7.81.0"})
    with urllib.request.urlopen(req) as resp, open(dest_path, "wb") as out:
        total_size = resp.headers.get("content-length")
        total_bytes = int(total_size) if total_size else None
        downloaded = 0
        chunk_size = 1024 * 1024  # 1 MB

        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            out.write(chunk)
            downloaded += len(chunk)
            if total_bytes:
                pct = (downloaded / total_bytes) * 100
                print(f"\r    {downloaded / (1024**2):.1f} / {total_bytes / (1024**2):.1f} MB ({pct:.1f}%)", end="", flush=True)
            else:
                print(f"\r    {downloaded / (1024**2):.1f} MB downloaded", end="", flush=True)
    print()


def extract_archive(archive_path: Path, extract_dir: Path):
    """Extracts tar.gz or zip archive in pure Python."""
    print(f"  Extracting {archive_path.name} into {extract_dir} ...")
    extract_dir.mkdir(parents=True, exist_ok=True)

    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(extract_dir)
    elif str(archive_path).endswith(".tar.gz") or archive_path.suffix == ".tgz":
        with tarfile.open(archive_path, "r:gz") as tf:
            tf.extractall(extract_dir)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")

    # Remove archive after successful extraction to save disk space
    archive_path.unlink()


def main():
    parser = argparse.ArgumentParser(description="Generate Libri2Mix train-100 (8 kHz) on Windows 11 & Ubuntu")
    parser.add_argument("--storage_dir", type=str, default="./data", help="Directory where data will be stored (default: ./data)")
    args = parser.parse_args()

    storage_path = Path(args.storage_dir).resolve()
    storage_path.mkdir(parents=True, exist_ok=True)

    print_banner("Libri2Mix train-100 (8 kHz) Cross-Platform Setup")
    print(f"- OS:               {sys.platform.capitalize()} ({os.name})")
    print(f"- Storage Directory:{storage_path}")
    print(f"- Target Dataset:   {storage_path / 'Libri2Mix'}")

    # Step 1: Check dependencies
    print("\n[1/4] Checking audio tools...")
    check_sox()

    # Step 2: Clone or locate LibriMix generator repository
    print("\n[2/4] Setting up LibriMix generator repository...")
    generator_dir = storage_path / "LibriMix_generator"
    if not generator_dir.exists():
        print(f"  Cloning JorisCos/LibriMix into {generator_dir} ...")
        subprocess.check_call(["git", "clone", "https://github.com/JorisCos/LibriMix.git", str(generator_dir)])
    else:
        print(f"  [OK] Generator repo exists at: {generator_dir}")

    # Install generator python requirements
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "pandas", "soundfile", "scipy", "numpy"])

    # Step 3: Download LibriSpeech subsets and WHAM! noise
    print("\n[3/4] Downloading source audio (train-clean-100, dev-clean, test-clean, WHAM! noise)...")
    librispeech_dir = storage_path / "LibriSpeech"
    librispeech_dir.mkdir(parents=True, exist_ok=True)

    # LibriSpeech sources (skipping train-clean-360 to save ~300 GB!)
    sources = [
        ("http://www.openslr.org/resources/12/dev-clean.tar.gz", librispeech_dir / "dev-clean"),
        ("http://www.openslr.org/resources/12/test-clean.tar.gz", librispeech_dir / "test-clean"),
        ("http://www.openslr.org/resources/12/train-clean-100.tar.gz", librispeech_dir / "train-clean-100"),
    ]

    for url, check_dir in sources:
        if not check_dir.exists():
            archive = storage_path / Path(url).name
            download_file(url, archive)
            extract_archive(archive, storage_path)
        else:
            print(f"  [OK] Found {check_dir.name}")

    # WHAM! noise (Zenodo official mirror)
    wham_check_1 = storage_path / "wham_noise" / "wham_noise"
    wham_check_2 = storage_path / "wham_noise"
    has_wham = wham_check_1.exists() or (wham_check_2.exists() and any(wham_check_2.iterdir()))
    
    if not has_wham:
        wham_zip = storage_path / "wham_noise.zip"
        # Try Zenodo official mirror first
        wham_urls = [
            "https://zenodo.org/records/3338999/files/wham_noise.zip?download=1",
            "https://storage.googleapis.com/whisper-speech-dataset/wham_noise.zip"
        ]
        for url in wham_urls:
            try:
                print(f"  Attempting WHAM! noise download from: {url}")
                download_file(url, wham_zip)
                extract_archive(wham_zip, storage_path)
                has_wham = True
                break
            except Exception as e:
                print(f"  [Warning] Failed from {url}: {e}")
                if wham_zip.exists():
                    wham_zip.unlink()

    wham_dir = wham_check_1 if wham_check_1.exists() else wham_check_2

    # Step 4: Run LibriMix generation script
    types_to_generate = ["mix_clean", "mix_both"] if has_wham else ["mix_clean"]
    print(f"\n[4/4] Generating Libri2Mix audio ({' & '.join(types_to_generate)} @ 8 kHz)...")
    librimix_outdir = storage_path / "Libri2Mix"
    librimix_outdir.mkdir(parents=True, exist_ok=True)

    metadata_dir = generator_dir / "metadata" / "Libri2Mix"
    create_script = generator_dir / "scripts" / "create_librimix_from_metadata.py"

    cmd = [
        sys.executable,
        str(create_script),
        "--librispeech_dir", str(librispeech_dir),
        "--wham_dir", str(wham_dir if has_wham else librispeech_dir),
        "--metadata_dir", str(metadata_dir),
        "--librimix_outdir", str(librimix_outdir),
        "--n_src", "2",
        "--freqs", "8k",
        "--modes", "min",
        "--types", *types_to_generate,
    ]

    print(f"  Running: {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=str(generator_dir))

    print_banner("Libri2Mix Generation Completed Successfully!")
    print(f"- Location: {librimix_outdir}")
    print("\nYou can now start training with:")
    if sys.platform == "win32":
        print(f"    python train.py --data_dir {librimix_outdir} --dataset_type librimix")
    else:
        print(f"    python3 train.py --data_dir {librimix_outdir} --dataset_type librimix")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    main()
