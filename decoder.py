"""
ConvTasNet 1-D Learned Decoder.

Transforms the separated latent feature representation back into the time domain
using a 1-D Transposed Convolution (synthesizer filterbank).
"""

import torch
import torch.nn as nn
from typing import Optional


class ConvTasNetDecoder(nn.Module):
    """
    Learned 1-D Transposed Convolutional Decoder for Conv-TasNet.

    Acts as an inverse filterbank that synthesizes separated time-domain audio
    from the masked latent representations.

    Args:
        in_channels (int): N, number of latent channels from encoder. Default: 64.
        out_channels (int): Audio channels per speaker (1 for mono). Default: 1.
        kernel_size (int): L, length of each synthesis filter (matches encoder kernel). Default: 16.
        stride (int): Hop size (matches encoder stride). Default: 8.
    """

    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 1,
        kernel_size: int = 16,
        stride: int = 8,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        # 1-D Transposed Convolution (overlap-add synthesis filterbank)
        self.conv_transpose1d = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=False,
        )

        # Initialize weights
        nn.init.kaiming_normal_(self.conv_transpose1d.weight, mode="fan_in")

    def forward(
        self,
        w_masked: torch.Tensor,
        target_length: Optional[int] = None,
        verbose: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass for 1-D Decoder.

        Args:
            w_masked (torch.Tensor): Masked latent features.
                Can be:
                - Single speaker: [Batch, N, Latent_Length]
                - Multiple speakers: [Batch, Num_Sources, N, Latent_Length]
            target_length (int, optional): Original waveform length T for exact length matching.
            verbose (bool): If True, print tensor shapes.

        Returns:
            torch.Tensor: Reconstructed time-domain waveform(s).
                - For single speaker input: [Batch, 1, Time]
                - For multi-speaker input:  [Batch, Num_Sources, Time]
        """
        is_multi_source = (w_masked.dim() == 4)

        if is_multi_source:
            # Shape: [B, K, N, L] -> flatten batch and speaker dimensions: [B*K, N, L]
            b, k, n, l = w_masked.shape
            w_flat = w_masked.view(b * k, n, l)
            audio_flat = self.conv_transpose1d(w_flat)  # [B*K, 1, T_recon]
            
            # Trim or pad to exact target length if specified
            if target_length is not None:
                if audio_flat.shape[-1] > target_length:
                    audio_flat = audio_flat[..., :target_length]
                elif audio_flat.shape[-1] < target_length:
                    pad_len = target_length - audio_flat.shape[-1]
                    audio_flat = nn.functional.pad(audio_flat, (0, pad_len))
            
            # Reshape back to [B, K, T]
            time_dim = audio_flat.shape[-1]
            audio = audio_flat.view(b, k, time_dim)
        else:
            # Shape: [B, N, L]
            audio = self.conv_transpose1d(w_masked)  # [B, 1, T_recon]
            if target_length is not None:
                if audio.shape[-1] > target_length:
                    audio = audio[..., :target_length]
                elif audio.shape[-1] < target_length:
                    pad_len = target_length - audio.shape[-1]
                    audio = nn.functional.pad(audio, (0, pad_len))

        if verbose:
            print(f"[Decoder] Latent input: {w_masked.shape} -> Synthesized waveform: {audio.shape}")

        return audio


class SpectrogramDecoder(nn.Module):
    """
    Inverse Short-Time Fourier Transform (iSTFT) Spectrogram Decoder.

    Synthesizes time-domain separated waveforms from masked latent features and mixture phase.
    Unprojects latent channels N back to STFT frequency bins F (if projection was used),
    combines estimated magnitudes with mixture phase angles, and applies inverse STFT.

    Args:
        n_fft (int): Size of Fourier Transform (default: 256).
        hop_length (int): Hop length / frame shift (default: 64).
        win_length (Optional[int]): Window size (default: matches n_fft).
        window (str): Window function ('hann', 'hamming'). Default: 'hann'.
        encoder_channels (int): Latent dimension N from separator (default: 64).
        use_projection (bool): Whether 1x1 Conv projection is needed from N -> F. Default: True.
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

        # Create and register STFT synthesis window
        if window.lower() == "hamming":
            win_tensor = torch.hamming_window(self.win_length)
        else:
            win_tensor = torch.hann_window(self.win_length)
        self.register_buffer("window", win_tensor)

        # Unprojection: [encoder_channels N -> n_freq F]
        if self.use_projection or (self.n_freq != self.encoder_channels):
            self.unproj = nn.Conv1d(self.encoder_channels, self.n_freq, kernel_size=1, bias=False)
            self.relu = nn.ReLU()
            nn.init.kaiming_normal_(self.unproj.weight, mode="fan_in")
        else:
            self.unproj = None
            self.relu = None

    def forward(
        self,
        w_masked: torch.Tensor,
        phase: Optional[torch.Tensor] = None,
        target_length: Optional[int] = None,
        verbose: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass for Spectrogram Decoder using iSTFT.

        Args:
            w_masked (torch.Tensor): Masked latent features [Batch, Num_Sources, N, Latent_Length]
                                     or [Batch, N, Latent_Length].
            phase (torch.Tensor, optional): Mixture STFT phase [Batch, F, Latent_Length].
            target_length (int, optional): Original waveform length in samples.
            verbose (bool): If True, print tensor shapes.

        Returns:
            torch.Tensor: Reconstructed time-domain audio [Batch, Num_Sources, Time] or [Batch, 1, Time].
        """
        is_multi_source = (w_masked.dim() == 4)

        if is_multi_source:
            b, k, n, l = w_masked.shape
            w_flat = w_masked.view(b * k, n, l)

            # Map from latent channels N -> frequency bins F
            if self.unproj is not None:
                mag_flat = self.relu(self.unproj(w_flat))
            else:
                mag_flat = w_flat

            # Expand phase across sources: [B, F, L] -> [B*K, F, L]
            if phase is not None:
                phase_flat = phase.unsqueeze(1).repeat(1, k, 1, 1).view(b * k, -1, l)
            else:
                phase_flat = torch.zeros_like(mag_flat)

            # Combine magnitude and phase into complex STFT representation
            complex_spec = torch.polar(mag_flat.float(), phase_flat.float())
            if complex_spec.dtype not in (torch.complex64, torch.complex128):
                complex_spec = complex_spec.cfloat()

            window = self.window.to(dtype=torch.float32, device=complex_spec.device)

            # Synthesize time-domain audio via inverse STFT
            audio_flat = torch.istft(
                complex_spec,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=window,
                length=target_length,
            )

            # Reshape back to [Batch, Num_Sources, Time]
            time_dim = audio_flat.shape[-1]
            audio = audio_flat.view(b, k, time_dim)

            # Exact length trim/pad if necessary
            if target_length is not None:
                if audio.shape[-1] > target_length:
                    audio = audio[..., :target_length]
                elif audio.shape[-1] < target_length:
                    pad_len = target_length - audio.shape[-1]
                    audio = nn.functional.pad(audio, (0, pad_len))
        else:
            b, n, l = w_masked.shape
            if self.unproj is not None:
                mag = self.relu(self.unproj(w_masked))
            else:
                mag = w_masked

            if phase is not None:
                phase_val = phase
            else:
                phase_val = torch.zeros_like(mag)

            complex_spec = torch.polar(mag.float(), phase_val.float())
            if complex_spec.dtype not in (torch.complex64, torch.complex128):
                complex_spec = complex_spec.cfloat()

            window = self.window.to(dtype=torch.float32, device=complex_spec.device)

            audio_flat = torch.istft(
                complex_spec,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=window,
                length=target_length,
            )

            audio = audio_flat.unsqueeze(1)  # [B, 1, Time]
            if target_length is not None:
                if audio.shape[-1] > target_length:
                    audio = audio[..., :target_length]
                elif audio.shape[-1] < target_length:
                    pad_len = target_length - audio.shape[-1]
                    audio = nn.functional.pad(audio, (0, pad_len))

        if verbose:
            print(f"[SpectrogramDecoder] Latent input: {w_masked.shape} -> Synthesized waveform: {audio.shape}")

        return audio

