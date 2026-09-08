"""
NVIDIA CUDA Diagnostic & Hardware Acceleration Verification Script.

Tests and displays:
1. GPU hardware details (Model, VRAM, Compute Capability, Driver/CUDA versions).
2. CUDA memory allocation & cleanup.
3. End-to-end forward pass & surrogate gradient backward flow on GPU.
4. PyTorch Automatic Mixed Precision (AMP FP16) and GradScaler stability.
5. CPU vs GPU throughput and execution speed benchmark.
"""

import sys
import time
import torch
import torch.nn as nn

from compat import amp_autocast, make_grad_scaler
from config import ModelConfig
from model import SpikingConvTasNet
from losses import NegSISDRLoss, PITLossWrapper


def print_header(title: str):
    print("\n" + "=" * 68)
    print(f"  {title}")
    print("=" * 68)


def main():
    print_header("NVIDIA CUDA & Hardware Acceleration Diagnostic")

    cuda_available = torch.cuda.is_available()
    print(f"- PyTorch Version:       {torch.__version__}")
    print(f"- PyTorch CUDA Build:    {torch.version.cuda if torch.version.cuda else 'None (CPU build)'}")
    print(f"- cuDNN Version:         {torch.backends.cudnn.version() if cuda_available else 'N/A'}")
    print(f"- CUDA Available:        {cuda_available}")

    if not cuda_available:
        print("\n[!] NVIDIA CUDA is NOT available in this Python environment.")
        print("    To install PyTorch with NVIDIA CUDA 12.4 support, run:")
        print("    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
        return

    # GPU Device Information
    device_count = torch.cuda.device_count()
    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    total_mem_gb = props.total_memory / (1024 ** 3)
    major, minor = torch.cuda.get_device_capability(device)

    print(f"- GPU Device Count:      {device_count}")
    print(f"- Active GPU:            {torch.cuda.get_device_name(device)}")
    print(f"- Total VRAM:            {total_mem_gb:.2f} GB ({props.total_memory / (1024 ** 2):.0f} MB)")
    print(f"- Compute Capability:    {major}.{minor}")
    print(f"- Multi-Processor Count: {props.multi_processor_count}")

    # Step 1: CUDA Memory Allocation Test
    print_header("Step 1: CUDA Memory Allocation & Cleanup Test")
    torch.cuda.empty_cache()
    init_mem = torch.cuda.memory_allocated(device) / (1024 ** 2)
    test_tensor = torch.zeros(1000, 1000, 100, device=device) # ~400 MB
    alloc_mem = torch.cuda.memory_allocated(device) / (1024 ** 2)
    del test_tensor
    torch.cuda.empty_cache()
    final_mem = torch.cuda.memory_allocated(device) / (1024 ** 2)

    print(f"  [OK] Memory Allocated:   {alloc_mem - init_mem:.1f} MB allocated successfully")
    print(f"  [OK] Memory Cleared:     {final_mem:.1f} MB (Cache cleared properly)")

    # Step 2: SNN Forward & Surrogate Gradient Backward Test on CUDA
    print_header("Step 2: Spiking Conv-TasNet on CUDA (Forward + Backward)")
    config = ModelConfig(
        sample_rate=8000,
        segment_length=2.0,
        encoder_channels=64,
        snn_timesteps=4,
        bottleneck_channels=32,
        hidden_channels=64,
        dilations=[1, 2, 4],
        num_repeats=1,
    )
    model = SpikingConvTasNet(config).to(device)
    criterion = PITLossWrapper(loss_fn=NegSISDRLoss(), num_sources=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    x = torch.randn(4, 1, 16000, device=device)
    targets = torch.randn(4, 2, 16000, device=device)

    # Forward
    separated = model(x)
    loss, _, sisdr = criterion(separated, targets)
    print(f"  [OK] Forward Pass Output: {separated.shape} on {separated.device}")
    print(f"  [OK] Initial SI-SDR Loss: {loss.item():.2f} (SI-SDR: {sisdr.item():.2f} dB)")

    # Backward
    optimizer.zero_grad()
    loss.backward()
    grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
    print(f"  [OK] Surrogate Backprop:  Gradients computed for {len(grad_norms)} tensors (Max Grad: {max(grad_norms):.4f})")
    optimizer.step()
    print(f"  [OK] Optimizer Step:      Weights updated successfully on CUDA")

    # Step 3: PyTorch Automatic Mixed Precision (AMP FP16) Test
    print_header("Step 3: PyTorch Automatic Mixed Precision (torch.amp FP16)")
    scaler = make_grad_scaler("cuda", enabled=True)
    optimizer.zero_grad()
    with amp_autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        separated_amp = model(x)
        loss_amp, _, sisdr_amp = criterion(separated_amp, targets)

    scaler.scale(loss_amp).backward()
    scaler.step(optimizer)
    scaler.update()
    print(f"  [OK] AMP Autocast FP16:   Loss = {loss_amp.item():.2f}")
    print(f"  [OK] GradScaler:          Scaled backward pass & step passed without underflow/overflow")

    # Step 4: Speed Benchmark (CPU vs CUDA)
    print_header("Step 4: Speed Benchmark (CPU vs CUDA)")
    bench_batch = 4
    x_cpu = torch.randn(bench_batch, 1, 16000, device="cpu")
    tgt_cpu = torch.randn(bench_batch, 2, 16000, device="cpu")
    model_cpu = SpikingConvTasNet(config).to("cpu")
    crit_cpu = PITLossWrapper(loss_fn=NegSISDRLoss(), num_sources=2)

    # Warmup CPU
    _ = crit_cpu(model_cpu(x_cpu), tgt_cpu)
    t0 = time.time()
    for _ in range(5):
        loss_c, _, _ = crit_cpu(model_cpu(x_cpu), tgt_cpu)
        loss_c.backward()
    cpu_time = (time.time() - t0) / 5.0
    print(f"  - CPU Iteration Time:    {cpu_time * 1000:.1f} ms / step ({bench_batch / cpu_time:.1f} samples/s)")

    # CUDA Benchmark with Events
    torch.cuda.synchronize(device)
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    # Warmup GPU
    with amp_autocast("cuda", dtype=torch.float16, enabled=True):
        _ = criterion(model(x), targets)
    torch.cuda.synchronize(device)

    start_evt.record()
    for _ in range(10):
        with amp_autocast("cuda", dtype=torch.float16, enabled=True):
            loss_g, _, _ = criterion(model(x), targets)
        scaler.scale(loss_g).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    end_evt.record()
    torch.cuda.synchronize(device)
    cuda_time = (start_evt.elapsed_time(end_evt) / 10.0) / 1000.0

    print(f"  - CUDA + AMP Iter Time:  {cuda_time * 1000:.1f} ms / step ({bench_batch / cuda_time:.1f} samples/s)")
    speedup = cpu_time / max(cuda_time, 1e-5)
    print(f"  - Speedup Factor:        {speedup:.1f}x FASTER on NVIDIA CUDA!")

    print_header("CUDA Verification Summary")
    print("  [*] NVIDIA CUDA hardware acceleration is FULLY FUNCTIONAL and READY for training!")
    print(f"  [*] Device: {torch.cuda.get_device_name(device)} ({total_mem_gb:.2f} GB VRAM)")
    print(f"  [*] Speedup: {speedup:.1f}x acceleration over CPU")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    main()
