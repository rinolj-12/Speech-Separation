"""
Speech Separation Losses and Permutation Invariant Training (uPIT).

Implements:
1. Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)
2. Negative SI-SDR Loss (NegSISDRLoss)
3. Multi-Resolution STFT Loss (MultiResolutionSTFTLoss)
4. Spike Activity Regularization Loss (SpikeActivityRegularization)
5. Utterance-level Permutation Invariant Training (uPIT) Combined Loss Wrapper (CombinedPITLoss)
"""

import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Sequence


# ==============================================================================
# 1. SI-SDR Metric and Loss
# ==============================================================================

def calculate_sisdr(
    estimate: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
    zero_mean: bool = True,
) -> torch.Tensor:
    """
    Calculate Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) in decibels (dB).

    Formula:
        s_target = (<estimate, target> / ||target||^2) * target
        e_noise  = estimate - s_target
        SI-SDR   = 10 * log10(||s_target||^2 / ||e_noise||^2)
    """
    estimate = estimate.float()
    target = target.float()

    if zero_mean:
        estimate = estimate - torch.mean(estimate, dim=-1, keepdim=True)
        target = target - torch.mean(target, dim=-1, keepdim=True)

    dot = torch.sum(estimate * target, dim=-1, keepdim=True)
    target_energy = torch.sum(target ** 2, dim=-1, keepdim=True) + eps

    alpha = dot / target_energy
    s_target = alpha * target
    e_noise = estimate - s_target

    s_target_energy = torch.sum(s_target ** 2, dim=-1) + eps
    e_noise_energy = torch.sum(e_noise ** 2, dim=-1) + eps

    ratio = s_target_energy / e_noise_energy  # both already have +eps; no clamp needed
    sisdr = 10.0 * torch.log10(ratio)
    return sisdr


class NegSISDRLoss(nn.Module):
    """Negative SI-SDR Loss function for minimization."""

    def __init__(self, eps: float = 1e-8, zero_mean: bool = True):
        super().__init__()
        self.eps = eps
        self.zero_mean = zero_mean

    def forward(self, estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return -calculate_sisdr(estimate, target, eps=self.eps, zero_mean=self.zero_mean)


# ==============================================================================
# 2. Multi-Resolution STFT Loss
# ==============================================================================

class SingleResolutionSTFTLoss(nn.Module):
    """Computes spectral convergence and log STFT magnitude loss for a single FFT resolution."""

    def __init__(self, n_fft: int = 512, hop_length: int = 128, win_length: int = 512):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer("window", torch.hann_window(win_length))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # x, y: [..., Time]
        orig_shape = x.shape
        x_flat = x.view(-1, orig_shape[-1]).float()
        y_flat = y.view(-1, orig_shape[-1]).float()

        window = self.window.to(dtype=torch.float32, device=x.device)

        X = torch.stft(
            x_flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
        )
        Y = torch.stft(
            y_flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
        )

        mag_x = torch.abs(X) + 1e-7
        mag_y = torch.abs(Y) + 1e-7

        # Spectral convergence loss
        sc_loss = torch.norm(mag_y - mag_x, p="fro", dim=(-2, -1)) / (torch.norm(mag_y, p="fro", dim=(-2, -1)) + 1e-7)
        # Log magnitude L1 loss
        log_loss = F.l1_loss(torch.log(mag_x), torch.log(mag_y), reduction="none").mean(dim=(-2, -1))

        total_single_loss = sc_loss + log_loss
        return total_single_loss.view(orig_shape[:-1])


class MultiResolutionSTFTLoss(nn.Module):
    """
    Multi-Resolution STFT Auxiliary Loss across 3 complementary time-frequency scales:
    (512, 128, 512), (256, 64, 256), and (128, 32, 128).
    Provides smooth, dense gradient flow to stabilize early SNN separation training.
    """

    def __init__(
        self,
        resolutions: Sequence[Tuple[int, int, int]] = (
            (512, 128, 512),
            (256, 64, 256),
            (128, 32, 128),
        ),
    ):
        super().__init__()
        self.loss_layers = nn.ModuleList([
            SingleResolutionSTFTLoss(n_fft, hop, win) for (n_fft, hop, win) in resolutions
        ])

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss = 0.0
        for layer in self.loss_layers:
            loss = loss + layer(x, y)
        return loss / len(self.loss_layers)


# ==============================================================================
# 3. Permutation Invariant Training (uPIT) Loss Wrappers
# ==============================================================================

class PITLossWrapper(nn.Module):
    """
    Utterance-level Permutation Invariant Training (uPIT) Loss Wrapper for single metric.
    """

    def __init__(self, loss_fn: Optional[nn.Module] = None, num_sources: int = 2):
        super().__init__()
        self.loss_fn = loss_fn if loss_fn is not None else NegSISDRLoss()
        self.num_sources = num_sources
        self.permutations = list(itertools.permutations(range(num_sources)))

    def forward(
        self,
        estimates: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_sources, time_len = estimates.shape
        assert num_sources == self.num_sources, f"Expected {self.num_sources} sources, got {num_sources}"

        pair_rows = []
        for i in range(num_sources):
            pair_cols = []
            for j in range(num_sources):
                pair_cols.append(self.loss_fn(estimates[:, i, :], targets[:, j, :]))
            pair_rows.append(torch.stack(pair_cols, dim=1))
        pairwise_losses = torch.stack(pair_rows, dim=1)  # [B, num_sources, num_sources]

        perm_losses = []
        for perm in self.permutations:
            loss_for_perm = torch.stack([pairwise_losses[:, perm[j], j] for j in range(num_sources)], dim=0)
            perm_losses.append(loss_for_perm.mean(dim=0))

        stacked_perm_losses = torch.stack(perm_losses, dim=0)
        min_losses, best_perm_indices = torch.min(stacked_perm_losses, dim=0)

        # Vectorized permutation reordering via torch.gather (no Python batch loop)
        perm_tensor = torch.tensor(
            self.permutations, dtype=torch.long, device=estimates.device
        )  # [P, K]
        best_perm_idx = perm_tensor[best_perm_indices]  # [B, K]
        best_estimates = torch.gather(
            estimates,
            dim=1,
            index=best_perm_idx.unsqueeze(-1).expand(-1, -1, estimates.shape[-1]),
        )

        sisdr_scores = -min_losses.mean()
        total_loss = min_losses.mean()

        return total_loss, best_estimates, sisdr_scores


class CombinedPITLoss(nn.Module):
    """
    Composite uPIT Separation Loss:
        L_total = L_SISDR + mr_stft_weight * L_MR_STFT
    Evaluates permutations jointly to preserve pitch and temporal alignment.
    """

    def __init__(
        self,
        mr_stft_weight: float = 0.5,
        num_sources: int = 2,
    ):
        super().__init__()
        self.mr_stft_weight = mr_stft_weight
        self.num_sources = num_sources
        self.sisdr_loss = NegSISDRLoss()
        self.mr_stft_loss = MultiResolutionSTFTLoss()
        self.permutations = list(itertools.permutations(range(num_sources)))

    def forward(
        self,
        estimates: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            estimates: [Batch, Num_Sources, Time]
            targets:   [Batch, Num_Sources, Time]

        Returns:
            Tuple (total_loss, best_estimates, sisdr_scores)
        """
        batch_size, num_sources, _ = estimates.shape

        # Step 1: Pairwise time-domain SI-SDR computation (fast & near-zero autograd memory)
        pairwise_sisdr = torch.zeros(batch_size, num_sources, num_sources, device=estimates.device)
        for i in range(num_sources):
            for j in range(num_sources):
                pairwise_sisdr[:, i, j] = self.sisdr_loss(estimates[:, i, :], targets[:, j, :])

        # Step 2: Determine optimal permutation per batch item based on SI-SDR
        perm_losses = []
        for perm in self.permutations:
            loss_for_perm = torch.stack([pairwise_sisdr[:, perm[j], j] for j in range(num_sources)], dim=0)
            perm_losses.append(loss_for_perm.mean(dim=0))

        stacked = torch.stack(perm_losses, dim=0)
        min_sisdr_losses, best_perm_indices = torch.min(stacked, dim=0)

        # Vectorized permutation reordering via torch.gather (no Python batch loop)
        perm_tensor = torch.tensor(
            self.permutations, dtype=torch.long, device=estimates.device
        )  # [P, K]
        best_perm_idx = perm_tensor[best_perm_indices]  # [B, K]
        best_estimates = torch.gather(
            estimates,
            dim=1,
            index=best_perm_idx.unsqueeze(-1).expand(-1, -1, estimates.shape[-1]),
        )

        # Step 4: Evaluate expensive Multi-Resolution STFT ONLY on the aligned predictions
        if self.mr_stft_weight > 0:
            stft_loss = torch.zeros(batch_size, device=estimates.device)
            for j in range(num_sources):
                stft_loss = stft_loss + self.mr_stft_loss(best_estimates[:, j, :], targets[:, j, :])
            mean_stft_loss = (stft_loss / num_sources).mean()
            total_loss = min_sisdr_losses.mean() + self.mr_stft_weight * mean_stft_loss
        else:
            total_loss = min_sisdr_losses.mean()

        mean_sisdr = -min_sisdr_losses.mean()

        return total_loss, best_estimates, mean_sisdr
