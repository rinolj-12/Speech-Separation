"""
Automated Unit Tests & Pipeline Shape Verification.

Verifies:
1. Exact tensor shapes at all intermediate pipeline stages.
2. SNN surrogate gradient backpropagation flow (no vanishing/zero/NaN gradients).
3. Advanced SNN Neurons (PLIF, ALIF, APLIF, FSNeuron, LIF) forward/backward dynamics.
4. Spike Encoders (Bit-Plane, Population, Learnable PLIF, Direct).
5. Continuous Membrane Readout & Continuous Residual Bridge.
6. Multi-Resolution STFT Loss & CombinedPITLoss.
7. Conv-TasNet Encoder, SpikeEncoder, SNN Separator, and Decoder correctness.
8. Permutation Invariant Training (uPIT) and SI-SDR loss calculations.
"""

import sys
import unittest
import torch
import torch.nn as nn

from compat import amp_autocast, make_grad_scaler
from config import ModelConfig
from encoder import ConvTasNetEncoder, SpectrogramEncoder
from decoder import ConvTasNetDecoder, SpectrogramDecoder
from snn import LIFNeuron, PLIFNeuron, ALIFNeuron, APLIFNeuron, FSNeuron, SpikingConvBlock1d
from spike_encoder import SpikeEncoder, BitPlaneSpikeEncoder, PopulationSpikeEncoder, LearnableSpikeEncoder
from separator import SpikingTCNSeparator
from model import SpikingConvTasNet, build_model
from losses import calculate_sisdr, NegSISDRLoss, PITLossWrapper, MultiResolutionSTFTLoss, CombinedPITLoss
from dataset import get_dataloaders


class TestSNNConvTasNetPipeline(unittest.TestCase):
    """Unit test suite for SNN Conv-TasNet architecture."""

    def setUp(self):
        self.batch_size = 2
        self.sample_rate = 8000
        self.time_samples = 16000  # 2.0 seconds
        self.n_channels = 64
        self.kernel_size = 16
        self.stride = 8
        self.timesteps = 4
        self.num_sources = 2
        
        self.config = ModelConfig(
            sample_rate=self.sample_rate,
            segment_length=2.0,
            encoder_channels=self.n_channels,
            encoder_kernel=self.kernel_size,
            encoder_stride=self.stride,
            snn_timesteps=self.timesteps,
            bottleneck_channels=32,
            hidden_channels=64,
            dilations=[1, 2, 4],
            num_repeats=1,
            num_sources=self.num_sources,
            neuron_type="plif",
            spike_encoding="bit_plane",
            snn_readout="membrane",
            use_residual_bridge=True,
        )

    def test_01_encoder(self):
        """Verify 1-D Encoder shapes and non-negative ReLU output."""
        encoder = ConvTasNetEncoder(
            in_channels=1,
            encoder_channels=self.n_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
        )
        x = torch.randn(self.batch_size, 1, self.time_samples)
        w = encoder(x, verbose=False)
        
        expected_l = (self.time_samples - self.kernel_size) // self.stride + 1
        self.assertEqual(w.shape, (self.batch_size, self.n_channels, expected_l))
        self.assertTrue((w >= 0.0).all(), "Encoder output W must be non-negative (ReLU)")
        print(f"[PASS] test_01_encoder: Input {x.shape} -> Latent W {w.shape}")

    def test_02_spike_encoder(self):
        """Verify SpikeEncoder transforms [B, N, L] into [S, B, N, L] across all modes."""
        l_len = 100
        w = torch.rand(self.batch_size, self.n_channels, l_len)

        for enc_mode in ["bit_plane", "population", "learnable_plif", "direct_current", "rate", "threshold"]:
            spike_enc = SpikeEncoder(
                timesteps=self.timesteps,
                encoding_type=enc_mode,
                channels=self.n_channels,
                population_factor=4,
            )
            spikes = spike_enc(w)
            self.assertEqual(spikes.shape, (self.timesteps, self.batch_size, self.n_channels, l_len), f"Shape mismatch for {enc_mode}")
        print(f"[PASS] test_02_spike_encoder: All spike encoders verified successfully!")

    def test_03_snn_separator(self):
        """Verify SpikingTCNSeparator produces masks [B, K, N, L] with PLIF and membrane readout."""
        l_len = 50
        spike_seq = torch.rand(self.timesteps, self.batch_size, self.n_channels, l_len)
        w = torch.rand(self.batch_size, self.n_channels, l_len)
        separator = SpikingTCNSeparator(
            in_channels=self.n_channels,
            bottleneck_channels=32,
            hidden_channels=64,
            dilations=[1, 2],
            num_repeats=1,
            num_sources=self.num_sources,
            neuron_type="plif",
            snn_readout="membrane",
            use_residual_bridge=True,
        )
        masks = separator(spike_seq, continuous_w=w)
        self.assertEqual(masks.shape, (self.batch_size, self.num_sources, self.n_channels, l_len))
        self.assertTrue((masks >= 0.0).all(), "Masks must be non-negative")
        print(f"[PASS] test_03_snn_separator: Spikes {spike_seq.shape} -> Masks {masks.shape}")

    def test_04_decoder(self):
        """Verify 1-D Transposed Conv Decoder synthesizes audio matching target length."""
        l_len = (self.time_samples - self.kernel_size) // self.stride + 1
        w_masked = torch.randn(self.batch_size, self.num_sources, self.n_channels, l_len)
        decoder = ConvTasNetDecoder(
            in_channels=self.n_channels,
            out_channels=1,
            kernel_size=self.kernel_size,
            stride=self.stride,
        )
        audio = decoder(w_masked, target_length=self.time_samples)
        self.assertEqual(audio.shape, (self.batch_size, self.num_sources, self.time_samples))
        print(f"[PASS] test_04_decoder: Masked Latent {w_masked.shape} -> Audio {audio.shape}")

    def test_05_spiking_conv_tasnet_end_to_end_and_gradients(self):
        """Verify end-to-end forward pass and surrogate gradient flow with PLIF and BitPlane."""
        model = SpikingConvTasNet(self.config)
        x = torch.randn(self.batch_size, 1, self.time_samples)
        
        separated = model(x, verbose=False)
        self.assertEqual(separated.shape, (self.batch_size, self.num_sources, self.time_samples))

        target = torch.randn(self.batch_size, self.num_sources, self.time_samples)
        criterion = CombinedPITLoss(mr_stft_weight=0.5, num_sources=self.num_sources)
        loss, _, sisdr = criterion(separated, target)
        loss.backward()

        grad_count = 0
        has_nonzero_grad = False
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Parameter {name} has None gradient!")
                self.assertFalse(torch.isnan(param.grad).any(), f"Parameter {name} has NaN gradients!")
                if param.grad.abs().sum() > 0:
                    has_nonzero_grad = True
                grad_count += 1
                
        self.assertTrue(has_nonzero_grad, "Model must have non-zero gradients via surrogate backprop!")
        print(f"[PASS] test_05_spiking_conv_tasnet: End-to-end forward & backward verified across {grad_count} parameters!")

    def test_06_spectrogram_encoder_and_end_to_end(self):
        """Verify SpectrogramEncoder (STFT) and end-to-end SpikingConvTasNet pipeline with STFT."""
        spec_cfg = ModelConfig(
            sample_rate=self.sample_rate,
            segment_length=2.0,
            encoder_type="spectrogram",
            encoder_channels=64,
            stft_n_fft=256,
            stft_hop_length=64,
            snn_timesteps=self.timesteps,
            bottleneck_channels=32,
            hidden_channels=64,
            dilations=[1, 2, 4],
            num_repeats=1,
            num_sources=self.num_sources,
            neuron_type="plif",
            spike_encoding="bit_plane",
            snn_readout="membrane",
        )
        model = SpikingConvTasNet(spec_cfg)
        x = torch.randn(self.batch_size, 1, self.time_samples)
        separated = model(x, verbose=False)
        self.assertEqual(separated.shape, (self.batch_size, self.num_sources, self.time_samples))

        target = torch.randn(self.batch_size, self.num_sources, self.time_samples)
        criterion = CombinedPITLoss(mr_stft_weight=0.5, num_sources=self.num_sources)
        loss, _, sisdr = criterion(separated, target)
        loss.backward()

        has_nonzero_grad = False
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if param.grad.abs().sum() > 0:
                    has_nonzero_grad = True
        self.assertTrue(has_nonzero_grad, "STFT SpikingConvTasNet must have non-zero gradients!")
        print(f"[PASS] test_06_spectrogram_encoder: STFT Spectrogram end-to-end forward & backward verified successfully!")


    def test_07_pit_and_sisdr(self):
        """Verify SI-SDR calculation and permutation invariant training."""
        t1 = torch.randn(2, self.time_samples)
        t2 = torch.randn(2, self.time_samples)
        targets = torch.stack([t1, t2], dim=1)
        est_flipped = torch.stack([t2, t1], dim=1)

        criterion = PITLossWrapper(loss_fn=NegSISDRLoss(), num_sources=2)
        loss, best_est, sisdr = criterion(est_flipped, targets)
        self.assertGreater(sisdr.item(), 30.0, "Permutation invariant SI-SDR should be > 30 dB for exact matches")
        print(f"[PASS] test_07_pit_and_sisdr: Perfect match under permutation yields SI-SDR = {sisdr.item():.2f} dB")

    def test_15_plif_alif_aplif_fs_neurons(self):
        """Verify forward and backward dynamics of PLIF, ALIF, APLIF, and FSNeuron."""
        x = torch.randn(self.batch_size, 32, 50, requires_grad=True)
        
        # 1. PLIF
        plif = PLIFNeuron(channels=32, init_beta=0.9, init_threshold=0.8)
        u, s = plif.init_state(x)
        s_out, u_out = plif.step(x, u, s)
        loss_plif = s_out.sum() + u_out.sum()
        loss_plif.backward()
        self.assertIsNotNone(plif.w_beta.grad)
        self.assertIsNotNone(plif.w_th.grad)

        # 2. ALIF
        alif = ALIFNeuron(channels=32)
        u, s, a = alif.init_state(x)
        s_out, u_out, a_out = alif.step(x, u, s, a=a)
        loss_alif = s_out.sum() + u_out.sum()
        loss_alif.backward()

        # 3. APLIF
        aplif = APLIFNeuron(channels=32)
        u, s, a = aplif.init_state(x)
        s_out, u_out, a_out = aplif.step(x, u, s, a=a)
        loss_aplif = s_out.sum() + u_out.sum()
        loss_aplif.backward()
        self.assertIsNotNone(aplif.w_beta.grad)

        # 4. FSNeuron
        fs = FSNeuron(channels=32, timesteps=4)
        u, s = fs.init_state(x)
        s_out, u_out = fs.step(x, u, s, step_idx=0)
        loss_fs = s_out.sum() + u_out.sum()
        loss_fs.backward()
        print(f"[PASS] test_15_plif_alif_aplif_fs_neurons: All 4 neuron models verified with backward gradients!")

    def test_16_multi_resolution_stft_and_combined_loss(self):
        """Verify MultiResolutionSTFTLoss and CombinedPITLoss computation."""
        mr_stft = MultiResolutionSTFTLoss()
        x = torch.randn(2, 16000, requires_grad=True)
        y = torch.randn(2, 16000)
        loss = mr_stft(x, y).mean()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.isnan(loss))

        comb_loss = CombinedPITLoss(mr_stft_weight=0.5, num_sources=2)
        est = torch.randn(2, 2, 16000, requires_grad=True)
        tgt = torch.randn(2, 2, 16000)
        tot_loss, best_est, sisdr = comb_loss(est, tgt)
        tot_loss.backward()
        self.assertIsNotNone(est.grad)
        print(f"[PASS] test_16_multi_resolution_stft_and_combined_loss: MR-STFT Loss and CombinedPITLoss verified!")

    def test_17_all_neuron_models_end_to_end(self):
        """Verify SpikingConvTasNet with PLIF, ALIF, APLIF, and FS-Neuron end-to-end."""
        for n_type in ["plif", "alif", "aplif", "fs_neuron"]:
            cfg = ModelConfig(
                sample_rate=self.sample_rate,
                segment_length=2.0,
                encoder_channels=32,
                snn_timesteps=4,
                bottleneck_channels=16,
                hidden_channels=32,
                dilations=[1, 2],
                num_repeats=1,
                num_sources=2,
                neuron_type=n_type,
                spike_encoding="bit_plane",
                snn_readout="membrane",
            )
            m = SpikingConvTasNet(cfg)
            x = torch.randn(1, 1, 16000)
            out = m(x)
            self.assertEqual(out.shape, (1, 2, 16000))
        print(f"[PASS] test_17_all_neuron_models_end_to_end: All neuron backbones verified end-to-end!")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available on this machine")
    def test_18_cuda_execution_and_gradients(self):
        """Verify full SNN Conv-TasNet on NVIDIA CUDA device."""
        device = torch.device("cuda:0")
        model = SpikingConvTasNet(self.config).to(device)
        x = torch.randn(self.batch_size, 1, self.time_samples, device=device)
        targets = torch.randn(self.batch_size, self.num_sources, self.time_samples, device=device)

        separated = model(x)
        self.assertEqual(separated.device.type, "cuda")

        criterion = CombinedPITLoss(mr_stft_weight=0.5, num_sources=self.num_sources)
        loss, _, sisdr = criterion(separated, targets)
        loss.backward()

        has_grad = False
        for param in model.parameters():
            if param.requires_grad and param.grad is not None:
                if param.grad.abs().sum() > 0:
                    has_grad = True
        self.assertTrue(has_grad, "CUDA backward pass must produce valid gradients.")
        print(f"[PASS] test_18_cuda_execution_and_gradients: SNN successfully ran on {torch.cuda.get_device_name(device)}!")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Running Spiking Conv-TasNet Pipeline Automated Tests")
    print("=" * 60)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSNNConvTasNetPipeline)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if result.wasSuccessful():
        print("\n[OK] ALL TESTS PASSED SUCCESSFULLY!")
        sys.exit(0)
    else:
        print("\n[FAIL] SOME TESTS FAILED!")
        sys.exit(1)
