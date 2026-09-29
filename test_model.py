"""
Model Testing and Evaluation Suite for Spiking Conv-TasNet Speech Separation.

Performs:
1. Evaluation on MiniLibriMix validation dataset.
2. Metrics computation: Input SI-SDR, Output SI-SDR, SI-SNRi (Improvement), Latency.
3. Audio generation: Saves separated audio WAV files organized in encoder-specific subfolders.
4. Visualizations: Saves separate individual PNG plots for waveforms, spectrograms, masks, and SNN activity.
"""

import os
import argparse
import time
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from typing import Tuple, List, Optional, Dict

from config import ModelConfig, TrainConfig
from model import SpikingConvTasNet, build_model
from losses import calculate_sisdr, NegSISDRLoss, PITLossWrapper
from dataset import get_dataloaders
from visualize import (
    plot_waveform_comparison,
    plot_spectrograms,
    plot_snn_raster,
    plot_mask_analysis,
    plot_multi_encoder_comparison,
    save_separate_waveforms,
    save_separate_spectrograms,
    save_separate_mask_plots,
    save_separate_snn_raster_plots,
    save_separate_comparison_plots,
)


def get_encoder_folder_info(encoder_type: str, base_output_dir: str = "./test_outputs") -> Tuple[str, str]:
    """
    Resolves the canonical subfolder name and path for a given encoder architecture.
    Subfolders:
      - 'standard_encoder' (Standard 1-D Conv + Spike)
      - 'spectrogram' (STFT Spectrogram + iSTFT)
    """
    enc = str(encoder_type).strip().lower()
    if "stand" in enc:
        folder_name = "standard_encoder"
    elif "spect" in enc or "stft" in enc:
        folder_name = "spectrogram"
    else:
        folder_name = enc

    target_dir = os.path.join(base_output_dir, folder_name)
    os.makedirs(target_dir, exist_ok=True)
    return folder_name, target_dir


def load_model(checkpoint_path: str, device: torch.device = torch.device("cpu")) -> Tuple[nn.Module, ModelConfig, dict]:
    """Loads model from checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at '{checkpoint_path}'.")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    config = checkpoint.get("model_config", ModelConfig())
    model = SpikingConvTasNet(config)

    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except RuntimeError as e:
        print(f"[Warning] Exact state_dict key match failed: {e}\nAttempting load with strict=False...")
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    model.to(device)
    model.eval()

    return model, config, checkpoint


def resolve_device(device_arg: str) -> torch.device:
    """Resolves device string to a valid torch.device with CUDA fallback."""
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        return torch.device("cpu")
    elif device_arg.startswith("cuda"):
        if not torch.cuda.is_available():
            print("[Warning] CUDA requested but not available. Falling back to CPU.")
            return torch.device("cpu")
        return torch.device(device_arg)
    else:
        return torch.device("cpu")


def evaluate_model(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> dict:
    """Evaluates model over validation data loader and computes SI-SDR, SI-SNRi, and accurate CUDA/CPU latency."""
    criterion = PITLossWrapper(loss_fn=NegSISDRLoss(), num_sources=2)
    model.eval()

    total_output_sisdr = []
    total_input_sisdr = []
    latencies = []

    with torch.no_grad():
        for i, (mix_batch, target_batch) in enumerate(val_loader):
            if max_batches is not None and i >= max_batches:
                break

            mix_batch = mix_batch.to(device, non_blocking=True)
            target_batch = target_batch.to(device, non_blocking=True)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            est_sources = model(mix_batch)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0 / mix_batch.size(0))

            # Metric calculation
            loss, best_est, out_sisdr = criterion(est_sources, target_batch)
            _, _, in_sisdr = criterion(mix_batch.repeat(1, 2, 1), target_batch)

            out_val = out_sisdr.cpu().tolist()
            if isinstance(out_val, list):
                total_output_sisdr.extend(out_val)
            else:
                total_output_sisdr.append(out_val)

            in_val = in_sisdr.cpu().tolist()
            if isinstance(in_val, list):
                total_input_sisdr.extend(in_val)
            else:
                total_input_sisdr.append(in_val)

    out_arr = np.array(total_output_sisdr)
    in_arr = np.array(total_input_sisdr)
    sisdri_arr = out_arr - in_arr

    results = {
        "mean_output_sisdr": float(np.mean(out_arr)),
        "std_output_sisdr": float(np.std(out_arr)),
        "median_output_sisdr": float(np.median(out_arr)),
        "min_output_sisdr": float(np.min(out_arr)),
        "max_output_sisdr": float(np.max(out_arr)),
        "mean_input_sisdr": float(np.mean(in_arr)),
        "mean_sisdri": float(np.mean(sisdri_arr)),
        "avg_latency_per_sample_ms": float(np.mean(latencies)),
    }
    return results


def run_sample_separation(
    model: nn.Module,
    config: ModelConfig,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    output_dir: str = "./test_outputs",
):
    """Runs separation on representative test samples and saves audio + separate plots in encoder subfolder."""
    enc_type = getattr(config, "encoder_type", "standard")
    folder_name, enc_output_dir = get_encoder_folder_info(enc_type, output_dir)
    os.makedirs(enc_output_dir, exist_ok=True)

    criterion = PITLossWrapper(loss_fn=NegSISDRLoss(), num_sources=2)

    mix_batch, target_batch = next(iter(val_loader))
    mix_batch = mix_batch.to(device, non_blocking=True)
    target_batch = target_batch.to(device, non_blocking=True)

    with torch.no_grad():
        x = mix_batch[0:1]  # single sample [1, 1, Time]
        targets = target_batch[0]  # [2, Time]

        # Model forward and intermediate extraction based on encoder_type
        if model.encoder_type in ["spectrogram", "stft"]:
            w, phase = model.encoder(x)
        else:
            w = model.encoder(x)
            phase = None

        if isinstance(model, SpikingConvTasNet):
            spike_seq = model.spike_encoder(w)
            masks = model.separator(spike_seq)
        else:
            spike_seq = None
            masks = model.separator(w)

        w_masked = masks * w.unsqueeze(1)
        if model.encoder_type in ["spectrogram", "stft"]:
            separated = model.decoder(w_masked, phase=phase, target_length=x.shape[-1])
        else:
            separated = model.decoder(w_masked, target_length=x.shape[-1])

        # Find best permutation alignment with ground truth
        _, best_separated, sample_sisdr = criterion(separated, targets.unsqueeze(0))
        best_separated = best_separated[0]  # [2, Time]

    # Convert tensors to numpy arrays
    mix_np = x[0, 0].cpu().numpy()
    targets_np = targets.cpu().numpy()
    est_np = best_separated.cpu().numpy()
    masks_np = masks[0].cpu()

    sr = config.sample_rate

    # Define categorized subdirectories inside encoder folder
    audio_dir = os.path.join(enc_output_dir, "audio")
    waveform_dir = os.path.join(enc_output_dir, "waveforms")
    spec_dir = os.path.join(enc_output_dir, "spectrograms")
    mask_dir = os.path.join(enc_output_dir, "masks")
    snn_dir = os.path.join(enc_output_dir, "snn")

    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(waveform_dir, exist_ok=True)
    os.makedirs(spec_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(snn_dir, exist_ok=True)

    # Save audio files inside audio/ subfolder
    audio_mix_path = os.path.join(audio_dir, "mixture.wav")
    audio_tgt1_path = os.path.join(audio_dir, "target_speaker1.wav")
    audio_tgt2_path = os.path.join(audio_dir, "target_speaker2.wav")
    audio_sep1_path = os.path.join(audio_dir, "separated_speaker1.wav")
    audio_sep2_path = os.path.join(audio_dir, "separated_speaker2.wav")

    sf.write(audio_mix_path, mix_np, sr)
    sf.write(audio_tgt1_path, targets_np[0], sr)
    sf.write(audio_tgt2_path, targets_np[1], sr)
    sf.write(audio_sep1_path, est_np[0], sr)
    sf.write(audio_sep2_path, est_np[1], sr)

    print(f"\n[Audio Wave Outputs Saved in '{audio_dir}']")
    print(f"  - Mixture:              {audio_mix_path}")
    print(f"  - Ground Truth Spk 1:   {audio_tgt1_path}")
    print(f"  - Ground Truth Spk 2:   {audio_tgt2_path}")
    print(f"  - Separated Spk 1:      {audio_sep1_path}")
    print(f"  - Separated Spk 2:      {audio_sep2_path}")

    # Generate and save separate visual analysis plots in respective subfolders
    print(f"\n[Generating Categorized PNG Visualizations in '{enc_output_dir}']")
    wav_paths = save_separate_waveforms(
        mix_np, targets_np, est_np, sample_rate=sr,
        output_dir=waveform_dir,
        title_suffix=f"(SI-SDR = {sample_sisdr.item():.2f} dB)"
    )
    for p in wav_paths.values():
        print(f"  - Waveform PNG:         {p}")

    spec_paths = save_separate_spectrograms(
        mix_np, targets_np, est_np, sample_rate=sr,
        output_dir=spec_dir,
    )
    for p in spec_paths.values():
        print(f"  - Spectrogram PNG:      {p}")

    mask_paths = save_separate_mask_plots(
        masks_np,
        output_dir=mask_dir,
        title_suffix=f"({model.encoder_type.capitalize()} Masks)"
    )
    for p in mask_paths.values():
        print(f"  - Mask PNG:             {p}")

    if spike_seq is not None:
        raster_paths = save_separate_snn_raster_plots(spike_seq, output_dir=snn_dir)
        for p in raster_paths.values():
            print(f"  - SNN Spike PNG:        {p}")

    print(f"[Done] All audio waves and separate PNGs organized into subfolders in '{enc_output_dir}'")


def main():
    default_model_cfg = ModelConfig()
    default_train_cfg = TrainConfig()

    parser = argparse.ArgumentParser(description="Test, Evaluate & Compare Speech Separation Models on NVIDIA CUDA / CPU")
    parser.add_argument("--snn_checkpoint", type=str, default=None, help="Path to SNN checkpoint (default: auto-detects in checkpoints/)")
    parser.add_argument("--ann_checkpoint", type=str, default=None, help="Path to ANN model checkpoint (optional)")
    parser.add_argument("--device", type=str, default="auto", help="Device: 'auto', 'cuda', 'cuda:0', or 'cpu'")
    parser.add_argument(
        "--dataset_type",
        type=str,
        default=default_train_cfg.dataset_type,
        choices=["librimix", "mini_librimix", "wav_folder"],
        help="Dataset type: 'librimix' (default), 'mini_librimix', or 'wav_folder'",
    )
    parser.add_argument("--data_dir", type=str, default=default_train_cfg.data_dir, help="Path to Libri2Mix (or MiniLibriMix) dataset")
    parser.add_argument("--segment_length", type=float, default=default_model_cfg.segment_length, help="Audio segment length in seconds for evaluation")
    parser.add_argument("--batch_size", type=int, default=default_train_cfg.batch_size, help="Batch size for validation")
    parser.add_argument("--mixture_type", type=str, default=default_train_cfg.mixture_type, choices=["mix_both", "mix_clean"], help="LibriMix mixture condition: 'mix_both' (noisy) or 'mix_clean' (clean speech only)")
    parser.add_argument("--max_batches", type=int, default=None, help="Limit number of validation batches (optional)")
    parser.add_argument("--output_dir", type=str, default="./test_outputs", help="Directory to save test audio & plots")
    parser.add_argument("--compare_encoders", action="store_true", help="Run benchmark comparing standard and spectrogram encoders")
    args = parser.parse_args()

    device = resolve_device(args.device)
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    
    print("=" * 68)
    print("       Spiking Conv-TasNet Speech Separation Model Testing")
    print("=" * 68)
    print(f"- Device:          {device} ({gpu_name})")
    print(f"- Dataset:         {args.dataset_type} ({args.data_dir})")
    print(f"- Mixture Type:    {args.mixture_type}")

    # Load Validation Dataset
    eval_segment_len = args.segment_length
    print(f"\n[1/3] Preparing Validation Dataset from: {args.data_dir} (Segment length: {eval_segment_len:.1f}s, condition: {args.mixture_type}) ...")
    _, val_loader = get_dataloaders(
        dataset_type=args.dataset_type,
        sample_rate=8000,
        segment_length=eval_segment_len,
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        mixture_type=args.mixture_type,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    print(f"  - Validation pairs: {len(val_loader.dataset)} samples across {len(val_loader)} batches")

    # Encoder Comparison Benchmark Mode
    if args.compare_encoders:
        print("\n" + "=" * 80)
        print("          ENCODER COMPARISON BENCHMARK ON MINILIBRIMIX")
        print("=" * 80)
        encoder_candidates = [
            ("Standard 1-D Conv + Spike", ["checkpoints/best_snn_standard.pt", "checkpoints/best_snn_standard_plif.pt"]),
            ("STFT Spectrogram + iSTFT", ["checkpoints/best_snn_spectrogram.pt", "checkpoints/best_snn_spectrogram_plif.pt"]),
        ]

        benchmark_rows = []
        loaded_models = []
        for name, candidate_paths in encoder_candidates:
            ckpt_path = None
            for p in candidate_paths:
                if os.path.exists(p):
                    ckpt_path = p
                    break

            if ckpt_path is None:
                print(f"  [Skip] Checkpoint not found for '{name}' (checked: {', '.join(candidate_paths)})")
                continue

            print(f"\n---> Evaluating [{name}] ({ckpt_path}) ...")
            model, cfg, meta = load_model(ckpt_path, device=device)
            n_params = sum(p.numel() for p in model.parameters())
            results = evaluate_model(model, val_loader, device=device, max_batches=args.max_batches)
            benchmark_rows.append({
                "name": name,
                "encoder_type": getattr(cfg, "encoder_type", "unknown"),
                "params": n_params,
                "out_sisdr": results["mean_output_sisdr"],
                "sisdri": results["mean_sisdri"],
                "latency": results["avg_latency_per_sample_ms"],
            })
            loaded_models.append((name, model, cfg))

        if benchmark_rows:
            print("\n" + "=" * 88)
            print(f"{'Encoder Architecture':<30} | {'Out SI-SDR':<12} | {'SI-SNRi':<12} | {'Params':<10} | {'Latency':<12}")
            print("-" * 88)
            for r in benchmark_rows:
                print(f"{r['name']:<30} | {r['out_sisdr']:>9.2f} dB | {r['sisdri']:>9.2f} dB | {r['params']:>10,} | {r['latency']:>8.1f} ms")
            print("=" * 88)

            # Generate individual outputs for each encoder in its dedicated subfolder
            print(f"\n[Generating Subfolder Outputs for Each Encoder in: {args.output_dir}] ...")
            os.makedirs(args.output_dir, exist_ok=True)
            criterion = PITLossWrapper(loss_fn=NegSISDRLoss(), num_sources=2)

            mix_batch, target_batch = next(iter(val_loader))
            mix_batch = mix_batch.to(device, non_blocking=True)
            target_batch = target_batch.to(device, non_blocking=True)

            x = mix_batch[0:1]  # [1, 1, Time]
            targets = target_batch[0]  # [2, Time]
            sr = 8000

            mix_np = x[0, 0].cpu().numpy()
            targets_np = targets.cpu().numpy()

            model_preds_dict = {}
            for name, model, cfg in loaded_models:
                enc_type = getattr(cfg, "encoder_type", "standard").lower()
                folder_name, enc_dir = get_encoder_folder_info(enc_type, args.output_dir)
                print(f"\n---> Generating Audio Waves & Separate PNGs for [{name}] in '{enc_dir}' ...")
                
                with torch.no_grad():
                    if enc_type in ["spectrogram", "stft"]:
                        w, phase = model.encoder(x)
                        spike_seq = model.spike_encoder(w) if isinstance(model, SpikingConvTasNet) else None
                        masks = model.separator(spike_seq if spike_seq is not None else w)
                        w_masked = masks * w.unsqueeze(1)
                        separated = model.decoder(w_masked, phase=phase, target_length=x.shape[-1])
                    else:
                        w = model.encoder(x)
                        spike_seq = model.spike_encoder(w) if isinstance(model, SpikingConvTasNet) else None
                        masks = model.separator(spike_seq if spike_seq is not None else w)
                        w_masked = masks * w.unsqueeze(1)
                        separated = model.decoder(w_masked, target_length=x.shape[-1])

                    _, best_separated, s_sisdr = criterion(separated, targets.unsqueeze(0))
                    est_np = best_separated[0].cpu().numpy()
                    masks_np = masks[0].cpu()

                # Baseline mixture input SI-SDR
                _, _, in_sisdr = criterion(x.repeat(1, 2, 1), targets.unsqueeze(0))
                model_preds_dict[name] = {
                    "est": est_np,
                    "sisdr": float(s_sisdr.item()),
                    "sisdri": float(s_sisdr.item() - in_sisdr.item()),
                }

                # Define categorized subdirectories inside encoder folder
                audio_dir = os.path.join(enc_dir, "audio")
                waveform_dir = os.path.join(enc_dir, "waveforms")
                spec_dir = os.path.join(enc_dir, "spectrograms")
                mask_dir = os.path.join(enc_dir, "masks")
                snn_dir = os.path.join(enc_dir, "snn")

                os.makedirs(audio_dir, exist_ok=True)
                os.makedirs(waveform_dir, exist_ok=True)
                os.makedirs(spec_dir, exist_ok=True)
                os.makedirs(mask_dir, exist_ok=True)
                os.makedirs(snn_dir, exist_ok=True)

                # Save audio files inside this encoder's audio/ subfolder
                audio_mix_path = os.path.join(audio_dir, "mixture.wav")
                audio_tgt1_path = os.path.join(audio_dir, "target_speaker1.wav")
                audio_tgt2_path = os.path.join(audio_dir, "target_speaker2.wav")
                audio_sep1_path = os.path.join(audio_dir, "separated_speaker1.wav")
                audio_sep2_path = os.path.join(audio_dir, "separated_speaker2.wav")

                sf.write(audio_mix_path, mix_np, sr)
                sf.write(audio_tgt1_path, targets_np[0], sr)
                sf.write(audio_tgt2_path, targets_np[1], sr)
                sf.write(audio_sep1_path, est_np[0], sr)
                sf.write(audio_sep2_path, est_np[1], sr)

                # Save separate PNGs inside this encoder's categorized subfolders
                save_separate_waveforms(
                    mix_np, targets_np, est_np, sample_rate=sr,
                    output_dir=waveform_dir,
                    title_suffix=f"[{name}] (SI-SDR = {s_sisdr.item():.2f} dB)"
                )
                save_separate_spectrograms(
                    mix_np, targets_np, est_np, sample_rate=sr,
                    output_dir=spec_dir,
                )
                save_separate_mask_plots(
                    masks_np,
                    output_dir=mask_dir,
                    title_suffix=f"({name} Masks)"
                )
                if spike_seq is not None:
                    save_separate_snn_raster_plots(spike_seq, output_dir=snn_dir)

            # Generate cross-encoder comparison figures in comparisons/ subfolder
            comp_dir = os.path.join(args.output_dir, "comparisons")
            os.makedirs(comp_dir, exist_ok=True)
            print(f"\n[Generating Cross-Encoder Comparison PNGs in '{comp_dir}'] ...")
            comp_paths = save_separate_comparison_plots(
                mix_np, targets_np, model_preds_dict,
                benchmark_rows=benchmark_rows,
                sample_rate=sr,
                output_dir=comp_dir,
            )
            for p in comp_paths.values():
                print(f"  - Comparison PNG:       {p}")

        else:
            print("  No encoder checkpoints found in 'checkpoints/' directory.")
            print("  Train the models first using:")
            print("    python train.py --encoder_type standard")
            print("    python train.py --encoder_type spectrogram")
        return

    # Single Model Evaluation Mode
    ckpt_path = args.snn_checkpoint
    if ckpt_path is None:
        candidates = [
            "checkpoints/best_snn_standard_plif.pt",
            "checkpoints/best_snn_spectrogram_plif.pt",
            "checkpoints/best_snn_standard.pt",
            "checkpoints/best_snn_spectrogram.pt",
        ]
        for cand in candidates:
            if os.path.exists(cand):
                ckpt_path = cand
                break
        if ckpt_path is None:
            print("\n[Error] No SNN model checkpoint found in 'checkpoints/' directory.")
            print("Please train a model first using: python train.py")
            print("Or specify a checkpoint path via: python test_model.py --snn_checkpoint <path>")
            return

    print(f"\n[2/3] Loading Model Checkpoint: {ckpt_path} ...")
    snn_model, snn_config, snn_meta = load_model(ckpt_path, device=device)
    total_snn_params = sum(p.numel() for p in snn_model.parameters())
    enc_type = getattr(snn_config, "encoder_type", "standard")
    print(f"  - Model Type:       Spiking Conv-TasNet ({enc_type.upper()})")
    print(f"  - Checkpoint Epoch: {snn_meta.get('epoch', 'N/A')}")
    print(f"  - Checkpoint Val SI-SDR: {snn_meta.get('val_sisdr', 0.0):.2f} dB")
    print(f"  - Parameters:       {total_snn_params:,}")
    print(f"  - Timesteps (S):    {snn_config.snn_timesteps}")
    print(f"  - Membrane Decay:   {snn_config.snn_beta}")
    print(f"  - Surrogate Grad:   {snn_config.surrogate}")

    # Evaluate SNN
    print(f"\n[3/3] Running Comprehensive Evaluation on {device}...")
    snn_results = evaluate_model(snn_model, val_loader, device=device, max_batches=args.max_batches)

    # Print Summary Table
    print("\n" + "=" * 68)
    print("                    EVALUATION BENCHMARK RESULTS")
    print("=" * 68)
    print(f"{'Metric':<36} | {'SNN Conv-TasNet':<20}")
    print("-" * 68)
    print(f"{'Encoder Type':<36} | {enc_type.upper():>20}")
    print(f"{'Input Mixture SI-SDR (dB)':<36} | {snn_results['mean_input_sisdr']:>17.2f} dB")
    print(f"{'Separated Output SI-SDR (dB)':<36} | {snn_results['mean_output_sisdr']:>17.2f} dB")
    print(f"{'SI-SDR Improvement (SI-SNRi)':<36} | {snn_results['mean_sisdri']:>17.2f} dB")
    print(f"{'Std Dev SI-SDR (dB)':<36} | {snn_results['std_output_sisdr']:>17.2f} dB")
    print(f"{'Median SI-SDR (dB)':<36} | {snn_results['median_output_sisdr']:>17.2f} dB")
    print(f"{'Min / Max SI-SDR (dB)':<36} | {snn_results['min_output_sisdr']:.1f} / {snn_results['max_output_sisdr']:.1f} dB")
    print(f"{'Latency per Sample (ms)':<36} | {snn_results['avg_latency_per_sample_ms']:>17.1f} ms")
    print("=" * 68)

    # Generate sample audio files and plots in encoder subfolder
    print(f"\nGenerating Sample Audio and Visual Diagnostic Plots in Encoder Subfolder...")
    run_sample_separation(snn_model, snn_config, val_loader, device=device, output_dir=args.output_dir)

    print("\n[OK] Model Testing and Evaluation Completed Successfully!")


if __name__ == "__main__":
    main()
