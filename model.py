"""
Spiking Conv-TasNet Architecture:
SpikingConvTasNet (SNN-based Separator with PLIF/ALIF/FS-Neuron, Bit-Plane/Population Coding, Continuous Membrane Readout, and Continuous Residual Bridge)
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple

from config import ModelConfig
from encoder import ConvTasNetEncoder, SpectrogramEncoder
from decoder import ConvTasNetDecoder, SpectrogramDecoder
from spike_encoder import SpikeEncoder
from separator import SpikingTCNSeparator


# ==============================================================================
# 1. Spiking Conv-TasNet Model
# ==============================================================================

class SpikingConvTasNet(nn.Module):
    """
    SNN-based Conv-TasNet Speech Separation Model.

    Modular Pipeline:
        Mixture Waveform x: [B, 1, T]
            | (Encoder: 1-D Learned Conv or STFT Spectrogram)
        Latent Representation W: [B, N, L]
            | (Spike Encoding: Bit-Plane, Population, Learnable PLIF, or Direct)
        Spike Sequence: [S, B, N, L]
            | (Spiking TCN Separator with PLIF / FS-Neuron / ALIF dynamics & Membrane Readout)
        Speaker Masks M: [B, K, N, L]
            | (Masking: W_k = M_k * W)
        Masked Latent Features W_masked: [B, K, N, L]
            | (Decoder: 1-D Transposed Conv or iSTFT with mixture phase)
        Separated Waveforms s_hat: [B, K, T]
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        if config is None:
            config = ModelConfig()
        self.config = config

        # 1. Encoder and Decoder initialization based on encoder_type
        self.encoder_type = getattr(config, "encoder_type", "standard").lower()

        if self.encoder_type in ["spectrogram", "stft"]:
            self.encoder = SpectrogramEncoder(
                n_fft=config.stft_n_fft,
                hop_length=config.stft_hop_length,
                win_length=config.stft_win_length,
                window=config.stft_window,
                encoder_channels=config.encoder_channels,
                use_projection=config.stft_use_projection,
            )
            self.spike_encoder = SpikeEncoder(
                timesteps=config.snn_timesteps,
                encoding_type=config.spike_encoding,
                channels=config.encoder_channels,
                threshold=config.snn_threshold,
                population_factor=getattr(config, "population_factor", 4),
                surrogate=config.surrogate,
            )
            self.decoder = SpectrogramDecoder(
                n_fft=config.stft_n_fft,
                hop_length=config.stft_hop_length,
                win_length=config.stft_win_length,
                window=config.stft_window,
                encoder_channels=config.encoder_channels,
                use_projection=config.stft_use_projection,
            )

        else:  # Standard 1-D Learned Encoder
            self.encoder = ConvTasNetEncoder(
                in_channels=1,
                encoder_channels=config.encoder_channels,
                kernel_size=config.encoder_kernel,
                stride=config.encoder_stride,
            )
            self.spike_encoder = SpikeEncoder(
                timesteps=config.snn_timesteps,
                encoding_type=config.spike_encoding,
                channels=config.encoder_channels,
                threshold=config.snn_threshold,
                population_factor=getattr(config, "population_factor", 4),
                surrogate=config.surrogate,
            )
            self.decoder = ConvTasNetDecoder(
                in_channels=config.encoder_channels,
                out_channels=1,
                kernel_size=config.encoder_kernel,
                stride=config.encoder_stride,
            )

        # 3. Spiking TCN Separator with selectable neuron dynamics & continuous membrane readout
        self.separator = SpikingTCNSeparator(
            in_channels=config.encoder_channels,
            bottleneck_channels=config.bottleneck_channels,
            hidden_channels=config.hidden_channels,
            kernel_size=config.kernel_size,
            dilations=config.dilations,
            num_repeats=config.num_repeats,
            num_sources=config.num_sources,
            mask_activation=config.mask_activation,
            neuron_type=getattr(config, "neuron_type", "plif"),
            beta=config.snn_beta,
            threshold=config.snn_threshold,
            timesteps=config.snn_timesteps,
            snn_readout=getattr(config, "snn_readout", "membrane"),
            use_residual_bridge=getattr(config, "use_residual_bridge", True),
            surrogate=config.surrogate,
            use_checkpointing=getattr(config, "use_checkpointing", True),
        )

    def forward(
        self,
        x: torch.Tensor,
        verbose: Optional[bool] = None,
    ) -> torch.Tensor:
        """
        Forward pass through complete Spiking Conv-TasNet.

        Args:
            x (torch.Tensor): Mixture audio waveform.
                              Shape: [Batch, 1, Time] or [Batch, Time].
            verbose (bool, optional): If True, prints tensor shapes throughout the pipeline.

        Returns:
            torch.Tensor: Separated audio waveforms.
                          Shape: [Batch, Num_Sources, Time]
        """
        if verbose is None:
            verbose = self.config.verbose_shapes

        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, T] -> [B, 1, T]

        batch, _, orig_time = x.shape

        if verbose:
            print("\n" + "=" * 50)
            print(f" [SpikingConvTasNet ({self.encoder_type.upper()})] Forward Pass Shape Trace")
            print("=" * 50)
            print(f"1. Input Mixture Waveform      : {x.shape} (B={batch}, Time={orig_time})")

        # Step 1 & 2: Encode mixture into Spikes and Latent representation
        if self.encoder_type in ["spectrogram", "stft"]:
            w, phase = self.encoder(x, verbose=verbose)
            spike_seq = self.spike_encoder(w, verbose=verbose)
        else:
            w = self.encoder(x, verbose=verbose)
            spike_seq = self.spike_encoder(w, verbose=verbose)
            phase = None

        if verbose:
            print(f"2. Latent Representation W     : {w.shape} (B={w.shape[0]}, N={w.shape[1]}, L={w.shape[2]})")
            print(f"3. Spiking Sequence (S={self.config.snn_timesteps})     : {spike_seq.shape}")

        # Step 3: Estimate speaker separation masks via Spiking TCN (with continuous bridge if enabled)
        bridge_w = w if getattr(self.config, "use_residual_bridge", True) else None
        masks = self.separator(spike_seq, continuous_w=bridge_w, verbose=verbose)
        if verbose:
            print(f"4. Predicted Speaker Masks M   : {masks.shape} (Sources K={masks.shape[1]})")

        # Step 4: Mask the latent representation W
        w_expanded = w.unsqueeze(1)
        w_masked = masks * w_expanded
        if verbose:
            print(f"5. Masked Latent Representations: {w_masked.shape}")

        # Step 5: Decode masked latent features into separated waveforms
        if self.encoder_type in ["spectrogram", "stft"]:
            separated_waveforms = self.decoder(w_masked, phase=phase, target_length=orig_time, verbose=verbose)
        else:
            separated_waveforms = self.decoder(w_masked, target_length=orig_time, verbose=verbose)

        if verbose:
            print(f"6. Separated Output Waveforms   : {separated_waveforms.shape}")
            print("=" * 50 + "\n")

        return separated_waveforms


def build_model(config: Optional[ModelConfig] = None) -> SpikingConvTasNet:
    """Factory helper to build Spiking Conv-TasNet model."""
    if config is None:
        config = ModelConfig()
    return SpikingConvTasNet(config)

