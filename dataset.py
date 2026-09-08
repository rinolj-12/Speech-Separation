"""
Datasets and Audio Mixture Loaders for Speech Separation.

Provides:
1. MiniLibriMixDataset: Real 2-speaker speech separation dataset from LibriMix.
2. WavFolderDataset: Dataset loader for custom audio/speech WAV files.
3. download_mini_librimix: Automated downloader and extractor for MiniLibriMix.
4. get_dataloaders: High-level dataloader factory for training and validation.
"""

import os
import urllib.request
import zipfile
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, List


class MiniLibriMixDataset(Dataset):
    """
    Dataset loader for Libri2Mix / MiniLibriMix real speech separation dataset.

    Expects standard LibriMix structure:
        data_dir/
            train/ (or val/ or dev/)
                mix_clean/ (or mix_both/ or mix/)
                s1/
                s2/
    """

    def __init__(
        self,
        data_dir: str = "./data/MiniLibriMix",
        sample_rate: int = 8000,
        segment_length: float = 2.0,
        subset: str = "train",  # 'train' or 'val' or 'dev' or 'test'
        mixture_type: str = "mix_clean",  # 'mix_clean' or 'mix_both'
        cache_in_memory: bool = True,
        epoch_samples: Optional[int] = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.segment_samples = int(sample_rate * segment_length)
        self.pairs = []

        target_subsets = ["val", "dev", "test"] if subset in ["val", "dev", "test"] else ["train"]

        if os.path.isdir(data_dir):
            for root, _, _ in os.walk(data_dir):
                path_parts = [p.lower() for p in os.path.normpath(root).split(os.sep)]
                if any(s in path_parts for s in target_subsets) or subset == "all":
                    cand_s1 = os.path.join(root, "s1")
                    cand_s2 = os.path.join(root, "s2")
                    cand_mix = os.path.join(root, mixture_type)
                    if not os.path.isdir(cand_mix):
                        cand_mix = os.path.join(root, "mix_both")
                    if not os.path.isdir(cand_mix):
                        cand_mix = os.path.join(root, "mix")

                    if os.path.isdir(cand_s1) and os.path.isdir(cand_s2):
                        for f in sorted(os.listdir(cand_s1)):
                            if f.endswith((".wav", ".flac")):
                                p_s1 = os.path.join(cand_s1, f)
                                p_s2 = os.path.join(cand_s2, f)
                                p_mix = os.path.join(cand_mix, f) if os.path.isdir(cand_mix) else None
                                if os.path.exists(p_s1) and os.path.exists(p_s2):
                                    self.pairs.append((p_mix if (p_mix and os.path.exists(p_mix)) else None, p_s1, p_s2))
                        if len(self.pairs) > 0 and subset != "all":
                            break

            # Fallback if no specific subset folder found
            if len(self.pairs) == 0:
                for root, _, _ in os.walk(data_dir):
                    cand_s1 = os.path.join(root, "s1")
                    cand_s2 = os.path.join(root, "s2")
                    cand_mix = os.path.join(root, mixture_type)
                    if os.path.isdir(cand_s1) and os.path.isdir(cand_s2):
                        for f in sorted(os.listdir(cand_s1)):
                            if f.endswith((".wav", ".flac")):
                                p_s1 = os.path.join(cand_s1, f)
                                p_s2 = os.path.join(cand_s2, f)
                                p_mix = os.path.join(cand_mix, f) if os.path.isdir(cand_mix) else None
                                if os.path.exists(p_s1) and os.path.exists(p_s2):
                                    self.pairs.append((p_mix if (p_mix and os.path.exists(p_mix)) else None, p_s1, p_s2))
                        if len(self.pairs) > 0:
                            break

        self.subset = subset
        self.cache_in_memory = cache_in_memory
        self._raw_cache = {}
        self.epoch_samples = epoch_samples

    def __len__(self) -> int:
        if self.epoch_samples is not None and self.epoch_samples > 0:
            return self.epoch_samples
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.pairs) == 0:
            raise RuntimeError(f"MiniLibriMixDataset found 0 audio pairs in {self.data_dir}. Please ensure dataset is downloaded.")

        item_idx = idx % len(self.pairs)

        if self.cache_in_memory and item_idx in self._raw_cache:
            s1_full, s2_full, mix_full = self._raw_cache[item_idx]
        else:
            p_mix, p_s1, p_s2 = self.pairs[item_idx]
            s1_full, _ = sf.read(p_s1)
            s2_full, _ = sf.read(p_s2)

            if s1_full.ndim > 1:
                s1_full = s1_full.mean(axis=-1)
            if s2_full.ndim > 1:
                s2_full = s2_full.mean(axis=-1)

            if p_mix and os.path.exists(p_mix):
                mix_full, _ = sf.read(p_mix)
                if mix_full.ndim > 1:
                    mix_full = mix_full.mean(axis=-1)
            else:
                min_len = min(len(s1_full), len(s2_full))
                mix_full = s1_full[:min_len] + s2_full[:min_len]

            # Ensure all match min length
            common_len = min(len(s1_full), len(s2_full), len(mix_full))
            s1_full = s1_full[:common_len]
            s2_full = s2_full[:common_len]
            mix_full = mix_full[:common_len]

            if self.cache_in_memory:
                self._raw_cache[item_idx] = (s1_full, s2_full, mix_full)

        total_len = len(mix_full)
        target_len = self.segment_samples

        if total_len > target_len and self.subset == "train":
            # Dynamic random cropping during training to explore full multi-second utterance
            start_offset = np.random.randint(0, total_len - target_len + 1)
            s1 = s1_full[start_offset : start_offset + target_len]
            s2 = s2_full[start_offset : start_offset + target_len]
            mix = mix_full[start_offset : start_offset + target_len]
        else:
            # Deterministic start for validation / evaluation or short clips
            if total_len < target_len:
                pad_len = target_len - total_len
                s1 = np.pad(s1_full, (0, pad_len))
                s2 = np.pad(s2_full, (0, pad_len))
                mix = np.pad(mix_full, (0, pad_len))
            else:
                s1 = s1_full[:target_len]
                s2 = s2_full[:target_len]
                mix = mix_full[:target_len]

        # Normalization
        max_val = max(np.max(np.abs(mix)), 1e-7)
        mix, s1, s2 = mix / max_val, s1 / max_val, s2 / max_val

        mix_tensor = torch.from_numpy(mix.copy()).unsqueeze(0).float()  # [1, Time]
        targets_tensor = torch.from_numpy(np.stack([s1.copy(), s2.copy()], axis=0)).float()  # [2, Time]

        return mix_tensor, targets_tensor


class WavFolderDataset(Dataset):
    """
    Dataset loader for paired audio WAV files in a custom directory.
    """

    def __init__(
        self,
        folder_path: str,
        sample_rate: int = 8000,
        segment_length: float = 2.0,
    ):
        super().__init__()
        self.folder_path = folder_path
        self.sample_rate = sample_rate
        self.segment_samples = int(sample_rate * segment_length)
        self.wav_files = []

        if os.path.isdir(folder_path):
            for root, _, files in os.walk(folder_path):
                for f in sorted(files):
                    if f.endswith((".wav", ".flac", ".ogg")):
                        self.wav_files.append(os.path.join(root, f))

    def __len__(self) -> int:
        return max(len(self.wav_files) // 2, 0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.wav_files) < 2:
            raise RuntimeError(f"WavFolderDataset requires at least 2 WAV files in {self.folder_path}, found {len(self.wav_files)}.")

        idx1 = (idx * 2) % len(self.wav_files)
        idx2 = (idx * 2 + 1) % len(self.wav_files)

        data1, _ = sf.read(self.wav_files[idx1])
        data2, _ = sf.read(self.wav_files[idx2])

        if data1.ndim > 1:
            data1 = data1.mean(axis=-1)
        if data2.ndim > 1:
            data2 = data2.mean(axis=-1)

        def pad_crop(arr: np.ndarray, target_len: int) -> np.ndarray:
            if len(arr) < target_len:
                return np.pad(arr, (0, target_len - len(arr)))
            return arr[:target_len]

        s1 = pad_crop(data1, self.segment_samples)
        s2 = pad_crop(data2, self.segment_samples)

        mixture = s1 + s2
        max_val = max(np.max(np.abs(mixture)), 1e-7)
        mixture, s1, s2 = mixture / max_val, s1 / max_val, s2 / max_val

        mix_tensor = torch.from_numpy(mixture).unsqueeze(0).float()
        targets_tensor = torch.from_numpy(np.stack([s1, s2], axis=0)).float()
        return mix_tensor, targets_tensor


def _has_audio_files(directory: str) -> bool:
    """Checks if directory contains extracted .wav or .flac audio files."""
    if not os.path.exists(directory):
        return False
    for _, _, files in os.walk(directory):
        if any(f.endswith((".wav", ".flac")) for f in files):
            return True
    return False


def download_mini_librimix(target_dir: str = "./data/MiniLibriMix") -> str:
    """
    Downloads and extracts MiniLibriMix dataset (~580 MB) from Zenodo if not present.
    """
    os.makedirs(target_dir, exist_ok=True)
    zip_path = os.path.join(target_dir, "MiniLibriMix.zip")

    if not _has_audio_files(target_dir):
        url = "https://zenodo.org/records/3871592/files/MiniLibriMix.zip?download=1"
        print(f"[Dataset] Downloading MiniLibriMix (~580MB) from Zenodo...", flush=True)

        downloaded = False
        # Try curl / wget first with cross-platform quoting
        if os.system(f'curl -L --fail -o "{zip_path}" "{url}"') == 0 and os.path.exists(zip_path) and os.path.getsize(zip_path) > 100000000:
            downloaded = True
        elif os.system(f'wget -O "{zip_path}" "{url}"') == 0 and os.path.exists(zip_path) and os.path.getsize(zip_path) > 100000000:
            downloaded = True

        if not downloaded:
            req = urllib.request.Request(url, headers={"User-Agent": "curl/7.81.0"})
            try:
                with urllib.request.urlopen(req) as response, open(zip_path, "wb") as out_file:
                    chunk_size = 1024 * 1024  # 1MB
                    bytes_dl = 0
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        out_file.write(chunk)
                        bytes_dl += len(chunk)
                        if bytes_dl % (50 * 1024 * 1024) == 0:
                            print(f"  Downloaded {bytes_dl / (1024 * 1024):.0f} MB ...", flush=True)
                downloaded = True
            except Exception as e:
                print(f"[Dataset Warning] Download failed: {e}", flush=True)

        if downloaded or (os.path.exists(zip_path) and os.path.getsize(zip_path) > 500000000):
            print("[Dataset] Extracting MiniLibriMix.zip ...", flush=True)
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(target_dir)
            print(f"[Dataset] MiniLibriMix extracted successfully to {target_dir}!", flush=True)
        else:
            print(f"[Dataset] Downloading in progress or please ensure MiniLibriMix.zip is in {target_dir}", flush=True)
    else:
        print(f"[Dataset] MiniLibriMix directory ready at {target_dir}", flush=True)

    return target_dir


def get_dataloaders(
    dataset_type: str = "mini_librimix",
    sample_rate: int = 8000,
    segment_length: float = 2.0,
    batch_size: int = 4,
    data_dir: str = "./data/MiniLibriMix",
    wav_folder: Optional[str] = None,
    num_workers: int = 2,
    pin_memory: Optional[bool] = None,
    train_samples_per_epoch: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Factory creating training and validation DataLoaders for MiniLibriMix or custom WAV folders.
    """
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    if dataset_type in ["mini_librimix", "librimix"]:
        folder = data_dir if data_dir else "./data/MiniLibriMix"
        if not _has_audio_files(folder):
            download_mini_librimix(folder)

        train_dataset = MiniLibriMixDataset(
            folder,
            sample_rate=sample_rate,
            segment_length=segment_length,
            subset="train",
            epoch_samples=train_samples_per_epoch,
        )
        val_dataset = MiniLibriMixDataset(folder, sample_rate=sample_rate, segment_length=segment_length, subset="val")
        if len(val_dataset) == 0:
            val_dataset = MiniLibriMixDataset(folder, sample_rate=sample_rate, segment_length=segment_length, subset="dev")
        if len(val_dataset) == 0:
            val_dataset = train_dataset

        if len(train_dataset) == 0:
            raise RuntimeError(
                f"No audio samples found in '{folder}'. MiniLibriMix is still downloading or not extracted. "
                f"Please wait for the background download to finish, or check './data/MiniLibriMix'."
            )

    elif dataset_type == "wav_folder" and wav_folder:
        train_dataset = WavFolderDataset(wav_folder, sample_rate=sample_rate, segment_length=segment_length)
        val_dataset = train_dataset
    else:
        raise ValueError(f"Unknown dataset_type '{dataset_type}'. Choose 'mini_librimix' or 'wav_folder'.")

    persistent = (num_workers > 0)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True if len(train_dataset) >= batch_size else False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )

    return train_loader, val_loader


if __name__ == "__main__":
    print("=" * 60)
    print("  Testing MiniLibriMix Dataset & DataLoaders")
    print("=" * 60)
    
    data_directory = "./data/MiniLibriMix"
    sample_rate = 8000
    segment_sec = 2.0
    batch_size = 4
    
    print(f"Loading datasets from: {data_directory}")
    train_loader, val_loader = get_dataloaders(
        dataset_type="mini_librimix",
        sample_rate=sample_rate,
        segment_length=segment_sec,
        batch_size=batch_size,
        data_dir=data_directory,
        num_workers=0,
    )
    
    print(f"[OK] Train samples: {len(train_loader.dataset)} ({len(train_loader)} batches of {batch_size})")
    print(f"[OK] Val samples:   {len(val_loader.dataset)} ({len(val_loader)} batches of {batch_size})")
    
    # Fetch a sample batch
    mix_batch, target_batch = next(iter(train_loader))
    print("\n--- Sample Batch Verification ---")
    print(f"Mixture tensor shape:  {mix_batch.shape} (Expected: [{batch_size}, 1, {int(sample_rate * segment_sec)}])")
    print(f"Target tensor shape:   {target_batch.shape} (Expected: [{batch_size}, 2, {int(sample_rate * segment_sec)}])")
    print(f"Mixture value range:   [{mix_batch.min().item():.3f}, {mix_batch.max().item():.3f}]")
    print(f"Target value range:    [{target_batch.min().item():.3f}, {target_batch.max().item():.3f}]")
    
    print("\n[OK] MiniLibriMix Dataset verification completed successfully!")

