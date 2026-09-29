# 🧠 Spiking Conv-TasNet — Energy-Efficient Speech Separation

> **TL;DR** — A speech separation model that splits a recording of two people
> talking simultaneously into two clean audio streams, using **Spiking Neural
> Networks (SNNs)** for ultra-low power consumption.

---

## 📖 What Is This?

**Spiking Conv-TasNet** combines two powerful ideas:

| Concept | What it does |
|---|---|
| **Conv-TasNet** | A state-of-the-art time-domain architecture that separates overlapping speech signals |
| **Spiking Neural Networks (SNNs)** | Brain-inspired neurons that fire discrete spikes instead of continuous values — dramatically reducing energy use |

The result is an end-to-end model that can separate two speakers from a mixed
audio recording while being far more energy-efficient than standard deep learning
approaches.

---

## ⚡ Quick Start

### Prerequisites

| Tool | Required? | Notes |
|---|---|---|
| **Python 3.9+** | ✅ Always | `python --version` to check |
| **Git** | ✅ Always | For cloning the repo |
| **SoX** | ⚠️ Full dataset only | Only needed for `scripts/setup_data.py` (Libri2Mix) |
| **NVIDIA GPU** | ❌ Optional | CPU works; GPU trains ~20–50× faster |

> [!NOTE]
> **SoX is NOT needed** if you use MiniLibriMix (the recommended first-run
> option). It is only required for generating the full Libri2Mix dataset.

---

### Step 1 — Clone & Create Environment

```bash
git clone <your-repo-url>
cd Speech
python -m venv .venv
```

Activate the virtual environment:

| Platform | Command |
|---|---|
| **Linux / macOS** | `source .venv/bin/activate` |
| **Windows CMD** | `.venv\Scripts\activate.bat` |
| **Windows PowerShell** | `.\venv\Scripts\Activate.ps1` |

---

### Step 2 — Install Dependencies

**🔵 CPU only (any laptop, no GPU needed):**
```bash
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

**🟢 NVIDIA GPU (CUDA 12.8):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

> [!TIP]
> Not sure which CUDA version you have? Run `nvidia-smi` and check the top-right
> corner. Visit [pytorch.org/get-started](https://pytorch.org/get-started/locally/)
> to find the right install command for your CUDA version.

---

### Step 3 — Get the Dataset

**Which dataset should I use?**

```
Are you on a laptop / CPU only, or just want a quick test?
  → Use MiniLibriMix  (~580 MB, auto-downloaded, no SoX needed)

Do you want to train seriously with full data (~13,900 pairs)?
  → Use Libri2Mix train-100  (~6.3–23 GB, requires SoX)
```

#### Option A — MiniLibriMix (recommended for first run)

Just run `train.py` — on first launch it detects no dataset and asks:

```
  [1] Download MiniLibriMix (~580 MB, ~1 000 pairs, no SoX needed)   ← press Enter
  [2] Generate full Libri2Mix train-100 via setup script
  [3] Abort
```

Choose **1** and it downloads + extracts everything automatically.

#### Option B — Full Libri2Mix train-100

Requires **SoX** installed first:

| OS | Install command |
|---|---|
| Ubuntu / Debian | `sudo apt-get install -y sox libsox-fmt-all` |
| macOS | `brew install sox` |
| Windows | `winget install -e --id SoX.SoX` |

Then run the setup script (interactive — it asks clean or noisy):

```bash
python scripts/setup_data.py
```

```
  [1] Clean only  (mix_clean)              ~6.3 GB    ← no noise, faster
  [2] Clean + Noisy  (mix_clean & mix_both)  ~23 GB   ← adds 17 GB WHAM! noise
```

Or skip the prompt with flags:
```bash
python scripts/setup_data.py --clean_only    # ~6.3 GB,  generates mix_clean
python scripts/setup_data.py --with_noise    # ~23 GB,   generates mix_clean + mix_both
```

---

### Step 4 — Train

**🔵 CPU only (MiniLibriMix — will auto-download on first run):**
```bash
python train.py --device cpu --no_amp --batch_size 2
```

**🟢 GPU — MiniLibriMix (quick test, auto-downloaded):**
```bash
python train.py --device cuda --dataset_type mini_librimix --data_dir ./data/MiniLibriMix
```

**🟢 GPU — Full Libri2Mix (after running setup_data.py):**
```bash
# Clean mixtures only:
python train.py --device cuda --mixture_type mix_clean --data_dir ./data/Libri2Mix

# Clean + noisy (default, requires --with_noise setup):
python train.py --device cuda --mixture_type mix_both --data_dir ./data/Libri2Mix
```

> [!TIP]
> Training was designed for a GPU. On CPU expect ~10–50× slower speed.
> Use `--batch_size 2 --no_amp` on CPU to reduce memory usage.
>
> If `./data/Libri2Mix` is missing, training automatically falls back to
> `./data/MiniLibriMix` if it exists — so you can run immediately after install.

---

## 📊 Dataset Details

| Property | MiniLibriMix | Libri2Mix train-100 |
|---|---|---|
| Training pairs | ~1,000 | ~13,900 |
| Audio duration | ~3.5 hours | ~58 hours |
| Download size | ~580 MB | ~6.3 GB (clean) / ~23 GB (with noise) |
| Requires SoX | ❌ No | ✅ Yes |
| Best for | First run, CPU, quick iteration | Benchmark results, serious training |
| Sample rate | 8 kHz | 8 kHz |
| Segment length | 2.0 s (16,000 samples) | 2.0 s (16,000 samples) |

---

## 🏗️ Architecture at a Glance

```
Input: Mixed Audio x(t)   [Batch, 1, Time]
            │
            ▼
┌──────────────────────────────────────────────────────┐
│                    ENCODER                           │
│  Option A: 1-D Conv filterbank  →  SpikeEncoder     │
│  Option B: STFT spectrogram frontend                 │
└──────────────────────────────────────────────────────┘
            │
            ▼   Spike Tensor  [Timesteps, Batch, N, L]
┌──────────────────────────────────────────────────────┐
│              SNN SEPARATOR (TCN)                     │
│  • Bottleneck 1×1 Conv + LIF neuron dynamics         │
│  • Dilated Depthwise Conv1D  (d = 1, 2, 4, …)       │
│  • Surrogate gradient learning (FastSigmoid / ATan)  │
│  • Generates two multiplicative masks M₁, M₂        │
└──────────────────────────────────────────────────────┘
            │
            ▼   Masks  [Batch, 2, N, L]
┌──────────────────────────────────────────────────────┐
│                    DECODER                           │
│  • Apply masks: Wₖ = Mₖ ⊙ W                         │
│  • Transposed 1-D Conv  OR  iSTFT synthesis          │
└──────────────────────────────────────────────────────┘
            │
            ▼
Output: ŝ₁(t), ŝ₂(t)  [Batch, 2, Time]
```

### Encoder Variants

| # | Encoder | How it works | Checkpoint |
|---|---|---|---|
| A | **Standard 1-D Conv** | Learned filterbank + SpikeEncoder (bit-plane / rate / population coding) | `checkpoints/best_snn_standard_plif.pt` |
| B | **STFT Spectrogram** | STFT magnitude features with mixture-phase preservation; reconstructed via iSTFT | `checkpoints/best_snn_spectrogram_plif.pt` |

> [!NOTE]
> Checkpoint filenames follow the pattern: `best_snn_{encoder_type}_{neuron_type}.pt`

---

## 📁 Project Structure

```
Speech/
├── train.py              # Main training script
├── test_model.py         # Evaluation & encoder comparison
│
├── model.py              # SpikingConvTasNet — top-level model
├── encoder.py            # Learned 1-D Conv & STFT encoders
├── decoder.py            # Transposed Conv & iSTFT decoders
├── separator.py          # Spiking TCN mask estimator
├── snn.py                # SNN neurons (PLIF/ALIF/LIF), surrogates, tdBN
├── spike_encoder.py      # Spike encoding schemes (Bit-Plane, Population, …)
│
├── losses.py             # SI-SDR uPIT loss + Multi-Resolution STFT loss
├── dataset.py            # Data loaders + MiniLibriMix auto-downloader
├── config.py             # ModelConfig & TrainConfig dataclasses
│
├── visualize.py          # Waveform, spectrogram, mask & spike-raster plots
│
├── scripts/
│   └── setup_data.py     # Libri2Mix train-100 generator (interactive)
│
├── checkpoints/          # Saved model weights (.pt files)
├── test_outputs/         # Audio & visualization outputs (auto-generated)
├── requirements.txt      # Python dependencies (PyTorch installed separately)
└── .gitattributes        # CRLF/LF line-ending rules
```

---

## 🚀 Training Reference

<details>
<summary><b>All training flags for <code>train.py</code> — click to expand</b></summary>

### Model Architecture

| Flag | Default | Description |
|---|---|---|
| `--encoder_type` | `standard` | `standard` (1-D Conv) or `spectrogram`/`stft` (STFT frontend) |
| `--neuron_type` | `plif` | SNN neuron model: `plif`, `alif`, `aplif`, `fs_neuron`, `lif` |
| `--spike_encoding` | `direct_current` | Spike encoding: `direct_current`, `bit_plane`, `rate`, `population`, `threshold`, `learnable_plif` |
| `--snn_readout` | `membrane` | Mask readout: `membrane`, `weighted_bit`, `rate` |
| `--use_residual_bridge` | `True` | Residual skip to mask head (`--no_residual_bridge` to disable) |
| `--snn_timesteps` | `6` | Number of SNN simulation timesteps $S$ |
| `--snn_beta` | `0.9` | LIF membrane decay factor $\beta$ |
| `--surrogate` | `fast_sigmoid` | Surrogate gradient: `fast_sigmoid`, `atan`, `piecewise` |
| `--num_repeats` | `2` | Dilation stack repeats $R$ |
| `--bottleneck_channels` | `128` | Bottleneck channels $B$ |
| `--hidden_channels` | `256` | Depthwise conv channels $H$ |
| `--encoder_channels` | auto (64 or 129) | Latent channels $N$ (auto-set based on encoder type) |
| `--mask_activation` | `sigmoid` | Mask activation: `sigmoid`, `softmax`, `relu` |
| `--stft_n_fft` | `256` | FFT size (256 = 32 ms frame @ 8 kHz → 129 bins) |
| `--stft_hop_length` | `64` | Hop length (64 = 8 ms @ 8 kHz) |
| `--stft_use_projection` | auto | Apply 1×1 Conv between STFT bins and latent channels |
| `--population_factor` | `4` | Gaussian population size factor (population coding only) |

### Training Schedule

| Flag | Default | Description |
|---|---|---|
| `--epochs` | `70` | Total training epochs |
| `--batch_size` | `4` | Mini-batch size |
| `--grad_accum_steps` | `4` | Gradient accumulation steps (effective batch = 4×4 = 16) |
| `--lr` | `1e-3` | Adam initial learning rate |
| `--lr_scheduler` | `cosine` | Scheduler: `cosine` (CosineAnnealingLR) or `plateau` (ReduceLROnPlateau) |
| `--min_lr` | `1e-5` | Minimum learning rate bound |
| `--patience` | `10` | Plateau patience before LR decay |
| `--weight_decay` | `1e-5` | L2 regularization |
| `--mr_stft_weight` | `0.0` | Multi-Resolution STFT loss weight (0 = disabled) |
| `--segment_length` | `2.0` | Audio crop duration in seconds |
| `--mixture_type` | `mix_both` | `mix_both` (speech + noise) or `mix_clean` |
| `--train_samples_per_epoch` | `0` | Virtual epoch size (`0` = full dataset) |

### Hardware & Performance

| Flag | Default | Description |
|---|---|---|
| `--device` | auto | `auto`, `cuda`, `cuda:0`, or `cpu` |
| `--amp` | `True` | FP16 Automatic Mixed Precision (`--no_amp` to disable on CPU) |
| `--checkpointing` | `True` | Gradient checkpointing (~80% VRAM savings) |
| `--multi_gpu` | `False` | `DataParallel` across multiple GPUs (experimental) |
| `--num_workers` | `0` | DataLoader workers (`0` = safe default) |
| `--pin_memory` | auto | Async host→device memory pinning (auto-enabled on CUDA) |
| `--num_threads` | `6` | CPU threads used in CPU-only mode |
| `--max_train_batches` | `None` | Limit train batches per epoch (debugging) |
| `--max_val_batches` | `None` | Limit validation batches |

### Paths

| Flag | Default | Description |
|---|---|---|
| `--dataset_type` | `librimix` | `librimix`, `mini_librimix`, or `wav_folder` |
| `--data_dir` | `./data/Libri2Mix` | Dataset root (auto-prompts download if missing) |
| `--checkpoint_dir` | `./checkpoints` | Where to save best model weights |
| `--resume` | — | Resume from checkpoint: `--resume auto` or `--resume path/to/ckpt.pt` |

### Required Files

| File | Purpose |
|---|---|
| [`config.py`](file:///home/rinolj/Music/Speech/config.py) | `ModelConfig` & `TrainConfig` dataclasses |
| [`model.py`](file:///home/rinolj/Music/Speech/model.py) | `SpikingConvTasNet` — assembles all modules |
| [`encoder.py`](file:///home/rinolj/Music/Speech/encoder.py) + [`decoder.py`](file:///home/rinolj/Music/Speech/decoder.py) | 1-D Conv / STFT frontends |
| [`snn.py`](file:///home/rinolj/Music/Speech/snn.py) + [`spike_encoder.py`](file:///home/rinolj/Music/Speech/spike_encoder.py) | SNN neuron dynamics & spike encoding |
| [`separator.py`](file:///home/rinolj/Music/Speech/separator.py) | Spiking TCN mask estimator |
| [`losses.py`](file:///home/rinolj/Music/Speech/losses.py) | uPIT + SI-SDR + MR-STFT losses |
| [`dataset.py`](file:///home/rinolj/Music/Speech/dataset.py) | Data loaders for Libri2Mix / MiniLibriMix + auto-downloader |
| `data/Libri2Mix/` or `data/MiniLibriMix/` | Audio data (auto-downloaded on first run, or via `scripts/setup_data.py`) |

</details>

---

## 🧪 Testing & Evaluation

### Evaluate a Single Checkpoint

```bash
python test_model.py --device cpu --snn_checkpoint checkpoints/best_snn_standard_plif.pt
```

### Compare Both Encoders Side-by-Side

```bash
python test_model.py --compare_encoders --device cpu
```

---

## 📂 Test Outputs (`test_outputs/`)

Running `test_model.py` generates files organized by encoder:

```
test_outputs/
├── standard_encoder/
│   ├── audio/              mixture.wav, target_speaker1/2.wav, separated_speaker1/2.wav
│   ├── waveforms/          waveform_*.png
│   ├── spectrograms/       spectrogram_*.png
│   ├── masks/              mask_*.png
│   └── snn/                snn_*.png  (spike rasters & firing rate charts)
├── spectrogram/
│   └── (same structure)
└── comparisons/
    └── comparison_*.png    (cross-encoder SI-SDR, latency & waveform charts)
```

---

## 📐 Model Size

| Encoder | Config | Trainable Parameters |
|---|---|---|
| `standard` | N=64, B=128, H=256, R=2 | ~1.6 M |
| `spectrogram` | N=64 (proj from 129 bins), B=128, H=256, R=2 | ~1.6 M |
