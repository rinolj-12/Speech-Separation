# Training Overview for Speech‑Separation Model

## What the repository uses (industry‑standard)

| Component | Implementation in this repo | Why it’s standard |
|---|---|---|
| **Loss** | `CombinedPITLoss` = **SI‑SDR** (via `NegSISDRLoss`) **+** Multi‑Resolution STFT auxiliary loss (`MultiResolutionSTFTLoss`). | SI‑SDR is the de‑facto objective for source separation. Adding a multi‑resolution STFT term stabilises training and improves perceptual quality – a pattern seen in recent Conv‑TasNet / SNN‑TasNet papers. |
| **Permutation‑Invariant Training (uPIT)** | `PITLossWrapper` / `CombinedPITLoss` automatically finds the best source ordering per utterance. | Removes the need for a fixed source ordering; essential for multi‑speaker separation. |
| **Metrics printed each step** | - `Loss` – combined loss value.  <br>- `SI‑SDR` – raw separation quality (dB). <br>- `SI‑SDRi` – improvement over the mixture (dB). | These are the benchmark metrics used in WSJ0‑2mix, LibriMix, Mini‑LibriMix and reported in papers. |
| **Mixed‑Precision (AMP)** | `torch.amp.autocast` + `torch.cuda.amp.GradScaler`. | Standard for fast GPU training while keeping numerical stability. |
| **Gradient accumulation** | Controlled via `--grad_accum_steps` (default 1). | Allows larger effective batch sizes on limited GPU memory – a common practical tweak. |
| **Learning‑rate scheduling** | `ReduceLROnPlateau` **or** `CosineAnnealingLR` (chosen via CLI). | Both schedulers are widely used; cosine annealing is especially popular for audio models. |
| **Validation loop** | Separate `validate()` returns mean SI‑SDR on a held‑out set. | Gives a reliable model‑selection signal and matches research evaluation protocols. |
| **Checkpointing** | Saves *best‑model* based on validation SI‑SDR (see `train.py`). | Guarantees the model with highest true performance is retained. |

## Typical training log line
```
Epoch [01/70] | Batch [010/500] | Loss:  5.23 | SI‑SDR:  7.65 dB | SI‑SDRi:  2.90 dB | Speed: 120.4 samples/s | | Peak VRAM: 8250/9000 MB
```
- **Loss** → combined uPIT loss (should decrease).  
- **SI‑SDR** → absolute separation quality (6‑9 dB is a good range for Mini‑LibriMix).  
- **SI‑SDRi** → improvement over the mixed input; values ≥ 2 dB are already respectable, state‑of‑the‑art often reaches 3‑4 dB.  
- **VRAM** helps you spot OOM issues.

## Quick sanity‑check checklist before full training
1. **Resolve Git conflict** (`scripts/setup_data.py`).  
   ```bash
   git status
   # edit the conflicted file
   git add scripts/setup_data.py
   git commit -m "Resolve merge conflict"
   ```
2. **Verify data paths** – `get_dataloaders` expects `data/mini_librimix/...`.  Update `--data_root` if needed.
3. **Dry‑run** a short training: 
   ```bash
   python train.py --max_epochs 2 --max_batches 20
   ```
   Ensure loss prints and the script exits cleanly.
4. **Monitor GPU** – `nvidia-smi`.  If memory > 90 %, lower `batch_size` or increase `grad_accum_steps`.
5. **Tune** (optional)  
   - Adjust `mr_stft_weight` (default from `ModelConfig`).  
   - Change learning‑rate scheduler parameters.  
   - Increase `grad_accum_steps` for larger effective batch size.

## Where to find the core code
- **Training loop** – `train.py` (lines 55‑149).  
- **Loss definitions** – `losses.py` (SI‑SDR, MR‑STFT, uPIT wrappers).  
- **Validation** – `train.py` function `validate()` (lines 152‑177).  
- **Model configuration** – `config.py` (`ModelConfig`, `TrainConfig`).

---
*This markdown file gives you a concise reference that you can keep alongside the repository for quick troubleshooting and for comparing your training runs against published baselines.*
