# Add Spectrogram (STFT / iSTFT) Option to Compare 3 Encoder Pipelines

## Objective
Enable a 3-way architectural comparison for speech separation by introducing a **Spectrogram (STFT / iSTFT)** encoder-decoder option alongside the existing two encoder options:
1. **Option 1 (`standard`)**: Standard 1-D Learned Conv Filterbank Encoder $\to$ `SpikeEncoder` $\to$ Spiking TCN Separator $\to$ 1-D Transposed Conv Decoder.
2. **Option 2 (`spiking`)**: Direct End-to-End Spiking LIF Filterbank Encoder $\to$ Spiking TCN Separator $\to$ 1-D Transposed Conv Decoder.
3. **Option 3 (`spectrogram`)**: STFT Spectrogram Magnitude Encoder + Projection $\to$ `SpikeEncoder` $\to$ Spiking TCN Separator $\to$ iSTFT Spectrogram Decoder with mixture phase.

All 3 pipelines produce time-domain separated waveforms $[B, K, T]$, enabling identical training loss ($\text{SI-SDR}$ with uPIT) and direct benchmark comparison.

---

## Proposed Architectural Design

```
Option 1 (Standard):
Waveform x [B, 1, T] ──> ConvTasNetEncoder ──> W [B, N, L] ──> SpikeEncoder ──> Spiking TCN ──> Masks ──> ConvTasNetDecoder ──> Separated Audio [B, K, T]

Option 2 (Direct Spiking):
Waveform x [B, 1, T] ──> SpikingConvTasNetEncoder ──> Spikes [S, B, N, L] & W ──> Spiking TCN ──> Masks ──> ConvTasNetDecoder ──> Separated Audio [B, K, T]

Option 3 (Spectrogram - NEW):
Waveform x [B, 1, T] ──> SpectrogramEncoder (STFT + Proj) ──> W [B, N, L] & Phase ──> SpikeEncoder ──> Spiking TCN ──> Masks ──> SpectrogramDecoder (iSTFT) ──> Separated Audio [B, K, T]
```

---

## Proposed Changes

### Configuration
#### [MODIFY] [config.py](file:///c:/Users/navne/Music/Speech/config.py)
- Update `ModelConfig.encoder_type` to allow `"spectrogram"` (along with `"spiking"` and `"standard"`).
- Add STFT parameters:
  - `stft_n_fft: int = 256` (filter size, e.g. 32ms @ 8kHz)
  - `stft_hop_length: int = 64` (hop size, e.g. 8ms @ 8kHz)
  - `stft_win_length: Optional[int] = 256`
  - `stft_window: str = "hann"`

---

### Encoder Module
#### [MODIFY] [encoder.py](file:///c:/Users/navne/Music/Speech/encoder.py)
- Implement `SpectrogramEncoder(nn.Module)`:
  - Computes Short-Time Fourier Transform (STFT) of input waveform via `torch.stft`.
  - Extracts magnitude spectrogram $|X| \in [B, F, L]$ and phase $\angle X \in [B, F, L]$ where $F = N_{\text{FFT}} / 2 + 1$.
  - Applies a 1x1 1D Convolution (`nn.Conv1d(F, encoder_channels, 1)` + `nn.ReLU()`) to map frequency bins to `encoder_channels` $N$ (e.g. 129 $\to$ 64), matching the TCN separator width.
  - Returns `(w, phase)`.

---

### Decoder Module
#### [MODIFY] [decoder.py](file:///c:/Users/navne/Music/Speech/decoder.py)
- Implement `SpectrogramDecoder(nn.Module)`:
  - Projects masked latent features $[B, K, N, L]$ back to frequency bins $[B*K, F, L]$ via 1x1 Conv + ReLU.
  - Reconstructs complex spectrogram using the masked magnitude and mixture phase: $X_{\text{recon}} = \text{polar}(|X_{\text{masked}}|, \angle X_{\text{mix}})$.
  - Applies inverse STFT (`torch.istft`) with exact length matching to `target_length`.
  - Returns time-domain separated waveforms $[B, K, T]$.

---

### Model Pipeline
#### [MODIFY] [model.py](file:///c:/Users/navne/Music/Speech/model.py)
- In `SpikingConvTasNet`:
  - Support `encoder_type="spectrogram"` (instantiates `SpectrogramEncoder`, `SpikeEncoder`, `SpikingTCNSeparator`, and `SpectrogramDecoder`).
  - Route forward pass: `w, phase = self.encoder(x)` $\to$ `spike_seq = self.spike_encoder(w)` $\to$ `masks = self.separator(spike_seq)` $\to$ `w_masked = masks * w.unsqueeze(1)` $\to$ `separated = self.decoder(w_masked, phase=phase, target_length=orig_time)`.
- In `ConvTasNetBaseline`:
  - Support `encoder_type="spectrogram"` for ANN benchmarking as well.

---

### Training & Checkpoints
#### [MODIFY] [train.py](file:///c:/Users/navne/Music/Speech/train.py)
- Expand `--encoder_type` choices to include `spectrogram` (or `stft`).
- Add `--stft_n_fft` and `--stft_hop_length` CLI arguments.
- Save checkpoints with distinct naming `checkpoints/best_{model}_{encoder_type}.pt` (e.g., `best_snn_spiking.pt`, `best_snn_standard.pt`, `best_snn_spectrogram.pt`) so all three configurations can coexist and be compared.

---

### Testing, Evaluation & 3-Way Benchmark
#### [MODIFY] [test_model.py](file:///c:/Users/navne/Music/Speech/test_model.py)
- Support loading and evaluating all 3 encoder checkpoints.
- Add multi-model comparison mode `--compare_encoders` / `--compare_all` that loads:
  1. `checkpoints/best_snn_standard.pt`
  2. `checkpoints/best_snn_spiking.pt`
  3. `checkpoints/best_snn_spectrogram.pt`
  and prints a comparison table:
  - Mean Output SI-SDR (dB)
  - Mean SI-SNRi (Improvement)
  - Trainable Parameters
  - Average Latency per Sample (ms)

---

### Automated Unit Testing
#### [MODIFY] [test_pipeline.py](file:///c:/Users/navne/Music/Speech/test_pipeline.py)
- Add unit tests for:
  - `SpectrogramEncoder` shape & phase extraction
  - `SpectrogramDecoder` inverse STFT reconstruction & length matching
  - End-to-end forward pass and surrogate gradient backpropagation for `SpikingConvTasNet(encoder_type="spectrogram")`
  - CUDA and AMP FP16 verification for the spectrogram model

---

## Verification Plan

### Automated Tests
1. Run updated test suite in `test_pipeline.py` with Python 3.11 virtual environment:
   ```powershell
   .\venv311\Scripts\python.exe test_pipeline.py
   ```
2. Verify all 3 configurations can be instantiated, run forward pass, backward pass, and CUDA AMP autocast without NaN or shape mismatch:
   ```powershell
   .\venv311\Scripts\python.exe -c "from model import SpikingConvTasNet; from config import ModelConfig; import torch; [SpikingConvTasNet(ModelConfig(encoder_type=t))(torch.randn(2, 1, 16000)) for t in ['standard', 'spiking', 'spectrogram']]; print('All 3 models initialized and executed successfully!')"
   ```
3. Run a quick multi-batch training sanity check for each of the 3 encoder types:
   ```powershell
   .\venv311\Scripts\python.exe train.py --encoder_type standard --epochs 1 --max_train_batches 2 --max_val_batches 2
   .\venv311\Scripts\python.exe train.py --encoder_type spiking --epochs 1 --max_train_batches 2 --max_val_batches 2
   .\venv311\Scripts\python.exe train.py --encoder_type spectrogram --epochs 1 --max_train_batches 2 --max_val_batches 2
   ```
4. Verify testing/comparison script:
   ```powershell
   .\venv311\Scripts\python.exe test_model.py --help
   ```
