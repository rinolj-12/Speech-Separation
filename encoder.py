"""
ConvTasNet 1-D Learned Encoder.

Transforms a 1-D time-domain audio mixture waveform into a continuous 2-D latent
feature representation W = ReLU(Conv1d(x)).
"""

import torch
import torch.nn as nn
from typing import Optional


class ConvTasNetEncoder(nn.Module):
    """
    Learned 1-D Convolutional Encoder for Conv-TasNet.

    Acts as a learned filterbank that maps time-domain audio to a high-dimensional
    latent space, analogous to a learned STFT representation.

    Args:
        in_channels (int): Input audio channels (1 for single-channel mono). Default: 1.
        encoder_channels (int): N, number of learned filters in filterbank. Default: 64.
        kernel_size (int): L, length of each filter in samples (e.g. 16 samples = 2ms @ 8kHz). Default: 16.
        stride (int): Hop size between filter applications (e.g. 8 samples = 50% overlap). Default: 8.
    """

    def __init__(
        self,
        in_channels: int = 1,
        encoder_channels: int = 64,
        kernel_size: int = 16,
        stride: int = 8,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.encoder_channels = encoder_channels
        self.kernel_size = kernel_size
        self.stride = stride

        # 1-D Convolution filterbank (analogous to learned Fourier basis)
        self.conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=encoder_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=False,
        )
        self.relu = nn.ReLU()

        # Initialize weights with standard normal distribution
        nn.init.kaiming_normal_(self.conv1d.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor, verbose: bool = False) -> torch.Tensor:
        """
        Forward pass for 1-D Encoder.

        Args:
            x (torch.Tensor): Time-domain audio tensor.
                              Shape: [Batch, 1, Time] or [Batch, Time].
            verbose (bool): If True, print tensor shapes.

        Returns:
            torch.Tensor: Non-negative latent representation W.
                          Shape: [Batch, N, Latent_Length]
                          where Latent_Length L = floor((Time - kernel_size) / stride) + 1.
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, T] -> [B, 1, T]

        w = self.conv1d(x)
        w = self.relu(w)

        if verbose:
            print(f"[Encoder] Input waveform: {x.shape} -> Latent representation W: {w.shape}")

        return w


class SpectrogramEncoder(nn.Module):
    """
    Short-Time Fourier Transform (STFT) Spectrogram Encoder.

    Transforms raw 1-D time-domain audio mixture waveform into a time-frequency
    magnitude spectrogram representation W and phase tensor, with optional
    learnable 1x1 1-D convolution projection to match the desired latent channel width N.

    Args:
        n_fft (int): Size of Fourier Transform (default: 256, e.g. 32ms @ 8kHz).
        hop_length (int): Hop length / frame shift (default: 64, e.g. 8ms @ 8kHz).
        win_length (Optional[int]): Window size (default: matches n_fft).
        window (str): Window function ('hann', 'hamming'). Default: 'hann'.
        encoder_channels (int): Target latent dimension N for separator interface (default: 64).
        use_projection (bool): Whether to project STFT frequency bins (n_fft//2 + 1)
                               to encoder_channels via 1x1 Conv + GroupNorm + ReLU. Default: True.
    """

    def __init__(
        self,
        n_fft: int = 256,
        hop_length: int = 64,
        win_length: Optional[int] = None,
        window: str = "hann",
        encoder_channels: int = 64,
        use_projection: bool = True,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length if win_length is not None else n_fft
        self.n_freq = self.n_fft // 2 + 1
        self.encoder_channels = encoder_channels
        self.use_projection = use_projection

        # Create and register STFT analysis window
        if window.lower() == "hamming":
            win_tensor = torch.hamming_window(self.win_length)
        else:
            win_tensor = torch.hann_window(self.win_length)
        self.register_buffer("window", win_tensor)

        # Optional 1x1 Conv projection between STFT frequency bins and latent channels
        if self.use_projection or (self.n_freq != self.encoder_channels):
            self.proj = nn.Conv1d(self.n_freq, self.encoder_channels, kernel_size=1, bias=False)
            self.norm = nn.GroupNorm(1, self.encoder_channels, eps=1e-8)
            self.relu = nn.ReLU()
            nn.init.kaiming_normal_(self.proj.weight, mode="fan_out", nonlinearity="relu")
        else:
            self.proj = None
            self.norm = None
            self.relu = None

    def forward(self, x: torch.Tensor, verbose: bool = False):
        """
        Forward pass computing STFT spectrogram.

        Args:
            x (torch.Tensor): Time-domain audio tensor [Batch, 1, Time] or [Batch, Time].
            verbose (bool): If True, print tensor shapes.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - w (torch.Tensor): Magnitude feature representation [Batch, N, Latent_Length].
                - phase (torch.Tensor): STFT phase angles [Batch, F, Latent_Length] for reconstruction.
        """
        if x.dim() == 3:
            x_flat = x.squeeze(1)  # [B, 1, T] -> [B, T]
        else:
            x_flat = x

        window = self.window.to(dtype=x.dtype, device=x.device)
        
        # Compute Complex STFT: [Batch, F, Frames]
        X = torch.stft(
            x_flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
        )

        mag = torch.abs(X)     # [B, F, L]
        phase = torch.angle(X) # [B, F, L]

        # Project frequency bins F -> N if configured
        if self.proj is not None:
            w = self.relu(self.norm(self.proj(mag)))
        else:
            w = mag

        if verbose:
            print(f"[SpectrogramEncoder] Waveform {x.shape} -> STFT Mag {mag.shape} -> Latent W {w.shape}")

        return w, phase


