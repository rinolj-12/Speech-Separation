# Spiking Conv-TasNet: Energy-Efficient SNN Speech Separation

An end-to-end, ultra-low-power speech separation architecture combining the time-domain convolutional architecture of **Conv-TasNet** with **Spiking Neural Network (SNN)** Leaky Integrate-and-Fire (LIF) dynamics and surrogate gradient backpropagation.

---

## Architecture Overview

```
Input Audio Mixture x(t) [B, 1, T]
                 │
                 ▼
┌────────────────────────────────────────────────────────┐
│                   ENCODER MODULE                       │
│  Option 1: Standard 1-D Conv + SpikeEncoder            │
│  Option 2: STFT Spectrogram Frontend + Phase Bypass    │
└────────────────────────────────────────────────────────┘
                 │
                 ▼  Spike Tensor [S, B, N, L]
┌────────────────────────────────────────────────────────┐
│                SNN SEPARATOR (TCN)                     │
│  - Bottleneck 1x1 Conv + SNN LIF Dynamics              │
│  - Dilated Depthwise Conv1D Blocks (d = 1, 2, 4, ...) │
│  - Surrogate Gradient Learning (FastSigmoid / ATan)    │
│  - Multiplicative Mask Estimation (M_1, M_2)           │
└────────────────────────────────────────────────────────┘
                 │
                 ▼  Estimated Masks [B, 2, N, L]
┌────────────────────────────────────────────────────────┐
│                   DECODER MODULE                       │
│  - Continuous Latent Modulation: W_k = M_k ⊙ W         │
│  - Transposed 1-D Deconvolution / iSTFT Synthesis      │
└────────────────────────────────────────────────────────┘
                 │
                 ▼
Output Separated Sources: ŝ_1(t), ŝ_2(t) [B, 2, T]
```

---

## Encoder Architectures

| Architecture | Description | Checkpoint |
|---|---|---|
| **1. Standard 1-D Conv + Spike** | Learned 1-D Conv filterbank + `SpikeEncoder` (bit-plane / rate / population) | `checkpoints/best_snn_standard_plif.pt` |
| **2. STFT Spectrogram + iSTFT** | Short-Time Fourier Transform (STFT) magnitude feature representation with mixture phase preservation and iSTFT reconstruction | `checkpoints/best_snn_spectrogram_plif.pt` |

> [!NOTE]
> Checkpoint filenames follow the pattern: `best_snn_{encoder_type}_{neuron_type}.pt`

---

## Directory Structure

```
Speech/
├── config.py             # Dataclass configurations (ModelConfig, TrainConfig)
├── encoder.py            # ConvTasNet 1-D learned and STFT encoders
├── decoder.py            # ConvTasNet 1-D transposed convolutional and iSTFT decoders
├── snn.py                # PLIF/ALIF/APLIF/FS-Neuron/LIF, surrogates, tdBN, SpikingConvBlock1d
├── spike_encoder.py      # Bit-Plane, Population, Learnable PLIF, and Direct spike encoders
├── separator.py          # SpikingTCNSeparator with membrane readout & residual bridge
├── model.py              # SpikingConvTasNet model architecture
├── losses.py             # Multi-Resolution STFT loss & Permutation Invariant Training (uPIT)
├── dataset.py            # MiniLibriMix dataset loader & automatic downloader
├── compat.py             # Cross-version PyTorch AMP / GradScaler compatibility helpers
├── visualize.py          # Waveform, Spectrogram, Mask, and SNN Spike raster plotting
├── test_model.py         # Testing & evaluation suite across encoder subfolders
├── train.py              # Training script for SNN on MiniLibriMix
├── test_pipeline.py      # Automated unit tests and shape verification suite
├── test_outputs/         # Test audio waves and separate PNG plots by encoder
│   ├── standard_encoder/ # Standard Conv + Spike audio & separate PNGs
│   └── spectrogram/      # STFT Spectrogram audio & separate PNGs
├── checkpoints/          # Model weights
└── requirements.txt      # Project dependencies
```

---

## Installation

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate   # Linux / macOS
# .venv\Scripts\activate    # Windows

# Install dependencies (CUDA build — requires NVIDIA GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install snntorch numpy matplotlib scipy soundfile

# CPU-only build (for laptops / machines without a GPU)
pip install torch torchvision torchaudio
pip install snntorch numpy matplotlib scipy soundfile
```

---

## Dataset: MiniLibriMix

The project uses **MiniLibriMix** (the standard speech separation benchmark derived from LibriSpeech):
- **Sample Rate**: 8 kHz
- **Audio Segment Length**: 2.0 seconds (16,000 samples)
- **Mixture Type**: `mix_clean` (2 overlapping human speakers)
- **Automatic Download**: Running `train.py` will automatically download and unpack MiniLibriMix (~580 MB from Zenodo) into `./data/MiniLibriMix/` if not present.

---

## 🚀 Training

All optimal hyperparameters are configured by default in `config.py`.

```bash
# 1. Standard 1-D Conv + Spike (GPU Recommended)
python3 train.py --device cuda --encoder_type standard

# 2. Spectrogram Frontend Variant (GPU Recommended)
python3 train.py --device cuda --encoder_type spectrogram --mr_stft_weight 0.3

# 3. CPU / Laptop Training (Low RAM)
python3 train.py --device cpu --batch_size 4 --no_amp
```

<details>
<summary><b>Hyperparameter Reference & CLI Options (Click to expand)</b></summary>

### Key Defaults
| Setting | Default | Description |
|---|---|---|
| `encoder_type` | `standard` | `standard` (1-D Conv), `spectrogram` (STFT) |
| `neuron_type` | `plif` | Parametric LIF with learnable decay $\beta$ and threshold |
| `spike_encoding` | `bit_plane` | Radix-2 bit-plane encoding ($2^4=16$ levels in 4 steps) |
| `snn_readout` | `membrane` | Continuous membrane potential readout |
| `use_residual_bridge` | `True` | Continuous linear skip connection to mask head |
| `snn_timesteps` | `4` | Simulation steps $S=4$ |
| `batch_size` | `16` | Mini-batch size (`4` on CPU) |
| `epochs` | `70` | Training epochs |
| `amp` | `True` | FP16 Automatic Mixed Precision (GPU only) |

</details>

---

## Testing & Output Visualization

### Run Unit Tests (no training, CPU-safe)

```bash
source .venv/bin/activate
python3 test_pipeline.py
```

### Encoder Comparison Benchmark (requires checkpoints)

```bash
python3 test_model.py --compare_encoders --device cpu
```

### Individual Model Evaluation

```bash
python3 test_model.py --device cpu --snn_checkpoint checkpoints/best_snn_standard_plif.pt
```

---

## Generated Files in `test_outputs/`

Each encoder subfolder (`standard_encoder/`, `spectrogram/`) contains:

1. **Audio Waves (`.wav`)**:
   - `mixture.wav`
   - `target_speaker1.wav` & `target_speaker2.wav`
   - `separated_speaker1.wav` & `separated_speaker2.wav`

2. **Separate Visualizations (`.png`)**:
   - **Waveforms**: `waveform_mixture.png`, `waveform_target_speaker1.png`, `waveform_target_speaker2.png`, `waveform_separated_speaker1.png`, `waveform_separated_speaker2.png`, `waveform_speaker1_overlay.png`, `waveform_speaker2_overlay.png`, `waveform_comparison_all.png`
   - **Spectrograms**: `spectrogram_mixture.png`, `spectrogram_target_speaker1.png`, `spectrogram_target_speaker2.png`, `spectrogram_separated_speaker1.png`, `spectrogram_separated_speaker2.png`, `spectrogram_speaker1_comparison.png`, `spectrogram_speaker2_comparison.png`, `spectrogram_overview.png`
   - **Masks**: `mask_speaker1.png`, `mask_speaker2.png`, `mask_contrast.png`, `mask_distribution.png`, `mask_analysis_summary.png`
   - **SNN Spikes**: `snn_spike_raster.png`, `snn_firing_rates.png`, `snn_raster_summary.png`

3. **Cross-Encoder Comparison Charts** (in root `test_outputs/`):
   - `comparison_metrics.png` (SI-SDR and SI-SNRi bar chart)
   - `comparison_latency.png` (Inference latency & model parameters)
   - `comparison_waveforms.png` (Multi-model waveform comparison)
   - `comparison_spectrograms.png` (Multi-model spectrogram comparison)

---

## Model Parameters (Full Configuration)

| Encoder Type | Trainable Parameters |
|---|---|
| `standard` (64ch, B=64, H=128, R=2) | ~332,160 |
| `spectrogram` (129 STFT bins → 64ch) | ~345,472 |
