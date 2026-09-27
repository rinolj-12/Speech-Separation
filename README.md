# 🧠 Spiking Conv-TasNet — Energy-Efficient Speech Separation

> **TL;DR** — A speech separation model that splits a recording of two people talking at the same time into two clean audio streams, using **Spiking Neural Networks (SNNs)** for ultra-low power consumption.

---

## 📖 What Is This?

**Spiking Conv-TasNet** combines two powerful ideas:

| Concept | What it does |
|---|---|
| **Conv-TasNet** | A state-of-the-art time-domain architecture that separates overlapping speech signals |
| **Spiking Neural Networks (SNNs)** | Brain-inspired neurons that fire spikes instead of using continuous activations — dramatically reducing energy use |

The result is an end-to-end model that can separate two speakers from a mixed audio recording while being far more energy-efficient than standard deep learning approaches.

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
├── test_pipeline.py      # Automated unit tests & shape checks
│
├── model.py              # SpikingConvTasNet — top-level model
├── encoder.py            # Learned 1-D Conv & STFT encoders
├── decoder.py            # Transposed Conv & iSTFT decoders
├── separator.py          # Spiking TCN mask estimator
├── snn.py                # SNN neurons (PLIF/ALIF/LIF), surrogates, tdBN
├── spike_encoder.py      # Spike encoding schemes (Bit-Plane, Population, …)
│
├── losses.py             # SI-SDR uPIT loss + Multi-Resolution STFT loss
├── dataset.py            # Libri2Mix & MiniLibriMix data loaders
├── config.py             # ModelConfig & TrainConfig dataclasses
├── compat.py             # PyTorch AMP / GradScaler compatibility
│
├── visualize.py          # Waveform, spectrogram, mask & spike-raster plots
├── verify_cuda.py        # CUDA diagnostic & benchmark
│
├── scripts/
│   └── setup_data.py     # Cross-platform Libri2Mix train-100 generator
│
├── checkpoints/          # Saved model weights (.pt files)
├── test_outputs/         # Audio & visualization outputs (auto-generated)
├── requirements.txt      # Python dependencies
└── .gitattributes        # CRLF/LF line-ending rules
```

---

## ⚡ Quick Start

### Step 1 — Clone & Create Environment

```bash
git clone <your-repo-url>
cd Speech
python -m venv .venv
```

Activate the virtual environment:

| Platform | Command |
|---|---|
| Linux / macOS | `source .venv/bin/activate` |
| Windows CMD | `.venv\Scripts\activate.bat` |
| Windows PowerShell | `.\venv\Scripts\Activate.ps1` |

### Step 2 — Install PyTorch

**🟢 With NVIDIA GPU (CUDA 12.8):**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

**🔵 CPU only (no GPU):**
```bash
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

### Step 3 — Get the Dataset

| Dataset | Size | How to get it |
|---|---|---|
| **MiniLibriMix** *(recommended for first run)* | ~580 MB | **Auto-downloaded** — just run `train.py --dataset_type mini_librimix` |
| **Libri2Mix train-100** *(full GPU training)* | ~15–20 GB | Run `python scripts/setup_data.py --storage_dir ./data` once (requires SoX) |

> [!IMPORTANT]
> **MiniLibriMix** is fetched automatically from Zenodo the first time you train. No manual download needed.
>
> **Libri2Mix train-100** requires SoX audio tools. Use the setup script:
> ```bash
> python scripts/setup_data.py --storage_dir ./data
> ```

### Step 4 — Train

**🟢 GPU (CUDA) — recommended:**
```bash
# Standard 1-D Conv encoder on full Libri2Mix
python train.py --device cuda --encoder_type standard

# STFT Spectrogram encoder variant
python train.py --device cuda --encoder_type spectrogram --mr_stft_weight 0.0

# Quick test with MiniLibriMix (~580 MB, auto-downloaded)
python train.py --device cuda --dataset_type mini_librimix --data_dir ./data/MiniLibriMix
```

**🔵 CPU only:**
```bash
python train.py --device cpu --dataset_type mini_librimix \
    --data_dir ./data/MiniLibriMix --batch_size 2 --no_amp --no_checkpointing
```

> [!TIP]
> If `./data/Libri2Mix` is missing, training automatically falls back to `./data/MiniLibriMix` — so you can run immediately after install.

---

## 📊 Dataset Details

The project targets **Libri2Mix `train-100`** — the standard academic speech separation benchmark.

| Property | Value |
|---|---|
| Sample rate | 8 kHz |
| Segment length | 2.0 s (16,000 samples) |
| Mixture type | `mix_both` — 2 overlapping speakers + background noise |
| Training pairs | ~13,900 (~58 hours) |
| Validation pairs | ~3,000 (~11 hours) |

---

## 🚀 Training Reference

<details open>
<summary><b>All Training Parameters for <code>train.py</code></b></summary>

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
| `--device` | `cuda` | `auto`, `cuda`, `cuda:0`, or `cpu` |
| `--amp` | `True` | FP16 Automatic Mixed Precision (`--no_amp` to disable) |
| `--checkpointing` | `True` | Gradient checkpointing (~80% VRAM savings; `--no_checkpointing` to disable) |
| `--multi_gpu` | `False` | `DataParallel` across multiple GPUs (experimental) |
| `--num_workers` | `0` | DataLoader workers (`0` = single process, avoids IPC/OOM issues) |
| `--pin_memory` | auto | Async host→device memory pinning (auto-enabled on CUDA) |
| `--num_threads` | `6` | CPU threads used when running on CPU |
| `--max_train_batches` | `None` | Limit train batches per epoch (debugging) |
| `--max_val_batches` | `None` | Limit validation batches |

### Paths

| Flag | Default | Description |
|---|---|---|
| `--dataset_type` | `librimix` | `librimix`, `mini_librimix`, or `wav_folder` |
| `--data_dir` | `./data/Libri2Mix` | Dataset root (auto-fallback to `./data/MiniLibriMix`) |
| `--checkpoint_dir` | `./checkpoints` | Where to save best model weights |

### Required Files

| File | Purpose |
|---|---|
| [`config.py`](file:///home/rinolj/Music/Speech/config.py) | `ModelConfig` & `TrainConfig` dataclasses |
| [`model.py`](file:///home/rinolj/Music/Speech/model.py) | `SpikingConvTasNet` — assembles all modules |
| [`encoder.py`](file:///home/rinolj/Music/Speech/encoder.py) + [`decoder.py`](file:///home/rinolj/Music/Speech/decoder.py) | 1-D Conv / STFT frontends |
| [`snn.py`](file:///home/rinolj/Music/Speech/snn.py) + [`spike_encoder.py`](file:///home/rinolj/Music/Speech/spike_encoder.py) | SNN neuron dynamics & spike encoding |
| [`separator.py`](file:///home/rinolj/Music/Speech/separator.py) | Spiking TCN mask estimator |
| [`losses.py`](file:///home/rinolj/Music/Speech/losses.py) | uPIT + SI-SDR + MR-STFT losses |
| [`dataset.py`](file:///home/rinolj/Music/Speech/dataset.py) | Data loaders for Libri2Mix / MiniLibriMix |
| `data/Libri2Mix/` or `data/MiniLibriMix/` | Audio data (configured via `--data_dir`) |

</details>

---

## 🧪 Testing & Evaluation

### Unit Tests (no GPU required)

```bash
# Linux / macOS
source .venv/bin/activate
python3 test_pipeline.py

# Windows
.\.venv\Scripts\Activate.ps1
python test_pipeline.py
```

### Compare Both Encoders (requires checkpoints)

```bash
# Linux / macOS
python3 test_model.py --compare_encoders --device cpu

# Windows
python test_model.py --compare_encoders --device cpu
```

### Evaluate a Single Checkpoint

```bash
python3 test_model.py --device cpu --snn_checkpoint checkpoints/best_snn_standard_plif.pt
```

---

## 📂 Test Outputs (`test_outputs/`)

Running `test_model.py` generates the following files, organized by encoder:

```
test_outputs/
├── standard_encoder/
│   ├── mixture.wav
│   ├── target_speaker1.wav, target_speaker2.wav
│   ├── separated_speaker1.wav, separated_speaker2.wav
│   ├── waveform_*.png          # Waveform overlays & comparisons
│   ├── spectrogram_*.png       # Per-speaker spectrogram views
│   ├── mask_*.png              # Estimated masks & distributions
│   └── snn_*.png               # Spike rasters & firing rate charts
├── spectrogram/
│   └── (same structure as above)
└── comparison_*.png            # Cross-encoder SI-SDR, latency & waveform charts
```

---

## 📐 Model Size

| Encoder | Trainable Parameters |
|---|---|
| `standard` (64ch, B=64, H=128, R=2) | ~332,160 |
| `spectrogram` (129 STFT bins → 64ch) | ~345,472 |
