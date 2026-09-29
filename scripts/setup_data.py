#!/usr/bin/env python3
"""
Cross-Platform Libri2Mix train-100 (8 kHz) Data Generator.
Compatible with Windows 11, Ubuntu Linux, and macOS.

Features:
1. Generates official Libri2Mix train-100 (8 kHz, min mode).
2. Interactive prompt: choose clean-only (~6.3 GB) OR clean + noisy (~23 GB).
3. Pure Python downloads and extraction (tarfile + zipfile; no OS-specific tools).
4. Detects OS and provides tailored SoX installation instructions.
5. Auto-invokes generation via the active Python virtual environment.

Usage:
    # Interactive (recommended):
    python scripts/setup_data.py

    # Non-interactive flags:
    python scripts/setup_data.py --with_noise      # clean + noisy mixtures
    python scripts/setup_data.py --clean_only      # clean mixtures, skip WHAM! download
    python scripts/setup_data.py --storage_dir /mnt/data  # custom storage location
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
        # Check standard Windows paths
        candidates = [
            Path(sys.prefix) / "Scripts",
            Path(r"C:\Program Files (x86)\sox-14-4-2"),
            Path(r"C:\Program Files\sox-14-4-2"),
        ]
        for c in candidates:
            if (c / "sox.exe").exists():
                os.environ["PATH"] = str(c) + os.pathsep + os.environ.get("PATH", "")
                sox_path = shutil.which("sox")
                if sox_path:
                    break

    if not sox_path:
        print("\n[ERROR] 'sox' (Sound eXchange) was not found on your system PATH.")
        print("  SoX is required to resample and mix audio at 8 kHz.")
        if sys.platform == "win32":
            print("  Windows 11 Installation Options:")
            print("    1. via Winget:      winget install -e --id SoX.SoX")
            print("    2. via Chocolatey:  choco install sox")
            print("    3. Download binary: https://sourceforge.net/projects/sox/files/sox/")
            print("       then add the SoX folder to your system PATH.")
        elif sys.platform == "darwin":
            print("  macOS Installation:")
            print("    brew install sox")
        else:
            print("  Ubuntu / Debian Installation:")
            print("    sudo apt-get update && sudo apt-get install -y sox libsox-fmt-all")
        sys.exit(1)
    print(f"  [OK] Found SoX at: {sox_path}")


def download_file(url: str, dest_path: Path):
    """Downloads a file with a live progress bar."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size > 0:
        print(f"  [OK] Already downloaded: {dest_path.name}")
        return

    print(f"  Downloading {dest_path.name} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "curl/7.81.0"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(dest_path, "wb") as out:
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
                bar_len = 38
                filled = int(bar_len * downloaded / total_bytes)
                bar = "#" * filled + "-" * (bar_len - filled)
                print(
                    f"\r    [{bar}] {downloaded/(1024**2):.1f}/{total_bytes/(1024**2):.1f} MB ({pct:.1f}%)",
                    end="",
                    flush=True,
                )
            else:
                print(f"\r    {downloaded/(1024**2):.1f} MB downloaded", end="", flush=True)
    print()  # newline after progress bar


def extract_archive(archive_path: Path, extract_dir: Path):
    """Extracts tar.gz or zip archive in pure Python, then removes the archive."""
    print(f"  Extracting {archive_path.name} ...")
    extract_dir.mkdir(parents=True, exist_ok=True)

    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(extract_dir)
    elif str(archive_path).endswith(".tar.gz") or archive_path.suffix == ".tgz":
        with tarfile.open(archive_path, "r:gz") as tf:
            tf.extractall(extract_dir)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")

    archive_path.unlink()  # free disk space after extraction


def prompt_noise_choice(storage_path: Path) -> bool:
    """
    Interactively asks whether to include WHAM! background noise in the mixtures.

    Returns:
        True  → download WHAM! noise and generate both mix_clean + mix_both
        False → skip WHAM! and generate mix_clean only
    """
    # Check if WHAM! is already present (previous partial run)
    wham_check_1 = storage_path / "wham_noise" / "wham_noise"
    wham_check_2 = storage_path / "wham_noise"
    already_have_wham = wham_check_1.exists() or (
        wham_check_2.exists() and any(wham_check_2.iterdir())
    )
    if already_have_wham:
        print("  [OK] WHAM! noise already downloaded — will generate both mix_clean & mix_both.")
        return True

    print()
    print("=" * 68)
    print("  Mixture Type — choose what to generate:")
    print("=" * 68)
    print()
    print("  [1] Clean only  (mix_clean)                  ~6.3 GB total")
    print("      Two speakers mixed together, no background noise.")
    print("      Faster download. Good for first experiments.")
    print()
    print("  [2] Clean + Noisy  (mix_clean & mix_both)    ~23 GB total")
    print("      Same as above PLUS a noisy version with real background")
    print("      noise from the WHAM! dataset (adds ~17 GB WHAM! download).")
    print("      Needed for the default --mixture_type mix_both training.")
    print()

    # Non-interactive environments default to clean-only (safe, smaller)
    if not sys.stdin.isatty():
        print("  [Non-interactive mode] Defaulting to option 1 (clean only).")
        return False

    try:
        choice = input("  Enter choice [1/2] (default 1 – clean only): ").strip() or "1"
    except (EOFError, KeyboardInterrupt):
        print()
        choice = "1"

    return choice.strip() == "2"


def main():
    parser = argparse.ArgumentParser(
        description="Generate Libri2Mix train-100 (8 kHz) — cross-platform setup",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/setup_data.py                    # interactive prompt
  python scripts/setup_data.py --clean_only       # no WHAM! noise (~6.3 GB)
  python scripts/setup_data.py --with_noise       # include WHAM! noise (~23 GB)
  python scripts/setup_data.py --storage_dir /mnt/data --clean_only
        """,
    )
    parser.add_argument(
        "--storage_dir",
        type=str,
        default="./data",
        help="Directory where all data will be stored (default: ./data)",
    )

    # Noise mode — mutually exclusive flags
    noise_group = parser.add_mutually_exclusive_group()
    noise_group.add_argument(
        "--clean_only",
        action="store_true",
        help="Generate clean mixtures only (mix_clean). Skips 17 GB WHAM! download.",
    )
    noise_group.add_argument(
        "--with_noise",
        action="store_true",
        help="Download WHAM! noise (~17 GB) and generate both mix_clean and mix_both.",
    )
    # Keep --skip_wham as a hidden alias for --clean_only (backwards compat)
    noise_group.add_argument("--skip_wham", action="store_true", help=argparse.SUPPRESS)

    args = parser.parse_args()

    storage_path = Path(args.storage_dir).resolve()
    storage_path.mkdir(parents=True, exist_ok=True)

    print_banner("Libri2Mix train-100 (8 kHz) — Cross-Platform Setup")
    print(f"  OS:               {sys.platform.capitalize()} ({os.name})")
    print(f"  Storage:          {storage_path}")
    print(f"  Output dataset:   {storage_path / 'Libri2Mix'}")

    # ── Step 1: SoX ─────────────────────────────────────────────────────────
    print("\n[1/4] Checking audio tools...")
    check_sox()

    # ── Step 2: LibriMix generator repo ─────────────────────────────────────
    print("\n[2/4] Setting up LibriMix generator...")
    generator_dir = storage_path / "LibriMix_generator"
    if not generator_dir.exists():
        print(f"  Cloning JorisCos/LibriMix into {generator_dir} ...")
        subprocess.check_call(
            ["git", "clone", "https://github.com/JorisCos/LibriMix.git", str(generator_dir)]
        )
    else:
        print(f"  [OK] Generator repo exists at: {generator_dir}")

    # Install generator dependencies into current environment
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet", "pandas", "soundfile", "scipy", "numpy"]
    )

    # ── Step 3: Resolve noise preference ────────────────────────────────────
    # Priority: --with_noise > --clean_only / --skip_wham > interactive prompt
    if args.with_noise:
        want_noise = True
        print("\n  [--with_noise] Will download WHAM! noise and generate mix_clean + mix_both.")
    elif args.clean_only or args.skip_wham:
        want_noise = False
        print("\n  [--clean_only] Skipping WHAM! noise. Will generate mix_clean only.")
    else:
        want_noise = prompt_noise_choice(storage_path)

    # ── Step 4: Download LibriSpeech ────────────────────────────────────────
    print("\n[3/4] Downloading LibriSpeech source audio...")
    librispeech_dir = storage_path / "LibriSpeech"
    librispeech_dir.mkdir(parents=True, exist_ok=True)

    # train-clean-100 (~6.3 GB), dev-clean (~337 MB), test-clean (~346 MB)
    sources = [
        ("http://www.openslr.org/resources/12/dev-clean.tar.gz",        librispeech_dir / "dev-clean"),
        ("http://www.openslr.org/resources/12/test-clean.tar.gz",       librispeech_dir / "test-clean"),
        ("http://www.openslr.org/resources/12/train-clean-100.tar.gz",  librispeech_dir / "train-clean-100"),
    ]

    for url, check_dir in sources:
        if not check_dir.exists():
            archive = storage_path / Path(url).name
            download_file(url, archive)
            extract_archive(archive, storage_path)
        else:
            print(f"  [OK] Found {check_dir.name}")

    # ── Step 5: Optionally download WHAM! noise ──────────────────────────────
    wham_check_1 = storage_path / "wham_noise" / "wham_noise"
    wham_check_2 = storage_path / "wham_noise"
    has_wham = wham_check_1.exists() or (wham_check_2.exists() and any(wham_check_2.iterdir()))

    if want_noise and not has_wham:
        wham_zip = storage_path / "wham_noise.zip"
        wham_url = "https://zenodo.org/records/3338999/files/wham_noise.zip?download=1"
        print(f"\n  Downloading WHAM! noise from Zenodo (~17 GB)...")
        print(f"  URL: {wham_url}")
        try:
            download_file(wham_url, wham_zip)
            extract_archive(wham_zip, storage_path)
            has_wham = True
        except Exception as exc:
            print(f"\n  [Warning] WHAM! noise download failed: {exc}")
            print("  Falling back to clean-only mixtures.")
            if wham_zip.exists():
                wham_zip.unlink()

    wham_dir = wham_check_1 if wham_check_1.exists() else wham_check_2

    # ── Step 6: Generate Libri2Mix ───────────────────────────────────────────
    types_to_generate = ["mix_clean", "mix_both"] if has_wham else ["mix_clean"]
    print(f"\n[4/4] Generating Libri2Mix ({' & '.join(types_to_generate)} @ 8 kHz)...")
    librimix_outdir = storage_path / "Libri2Mix"

    metadata_dir  = generator_dir / "metadata" / "Libri2Mix"
    create_script = generator_dir / "scripts" / "create_librimix_from_metadata.py"

    cmd = [
        sys.executable,
        str(create_script),
        "--librispeech_dir", str(librispeech_dir),
        "--wham_dir",        str(wham_dir if has_wham else librispeech_dir),
        "--metadata_dir",    str(metadata_dir),
        "--librimix_outdir", str(storage_path),
        "--n_src",  "2",
        "--freqs",  "8k",
        "--modes",  "min",
        "--types",  *types_to_generate,
    ]

    print(f"  Running: {' '.join(cmd)}\n")
    subprocess.check_call(cmd, cwd=str(generator_dir), env=os.environ.copy())

    # ── Done ─────────────────────────────────────────────────────────────────
    print_banner("Libri2Mix Generation Completed Successfully!")
    print(f"  Location: {librimix_outdir}")
    print(f"  Types:    {', '.join(types_to_generate)}")

    py = "python" if sys.platform == "win32" else "python3"
    mixture_flag = "--mixture_type mix_both" if has_wham else "--mixture_type mix_clean"
    print(f"\n  Start training with:")
    print(f"    {py} train.py --data_dir {librimix_outdir} --dataset_type librimix {mixture_flag}")
    if not has_wham:
        print()
        print("  Note: only mix_clean was generated. To add noisy mixtures later, re-run:")
        print(f"    python scripts/setup_data.py --storage_dir {args.storage_dir} --with_noise")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    main()
