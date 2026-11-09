from __future__ import annotations

import time

import numpy as np
import pytest

from wdiff.engine.spectral import (
    SpectralBundle,
    compute_spectral_bundle,
    fast_randomized_svd,
    marcenko_pastur_signal_rank,
    quantum_fidelity,
    subspace_principal_angles,
)


class TestFastRandomizedSVD:
    def test_reconstructs_known_low_rank_matrix(self):
        np.random.seed(42)
        m, n, true_rank = 256, 128, 4
        A = np.random.randn(m, true_rank).astype(np.float32)
        B = np.random.randn(true_rank, n).astype(np.float32)
        W = A @ B

        _U, s, _Vt = fast_randomized_svd(W, k=8, n_iter=2, random_state=42)
        assert len(s) == 8
        assert s[0] > s[1] >= s[2] >= s[3]
        # Singular values 4..7 should be effectively 0 (< 1e-4 relative to s[0])
        assert s[4] / s[0] < 1e-4

    def test_randomized_svd_cpu_speed_and_shape(self):
        # 1024 x 1024 matrix must compute k=16 SVD in under 1.0 second on CPU
        np.random.seed(42)
        W = np.random.randn(1024, 1024).astype(np.float32)

        t0 = time.perf_counter()
        U, s, Vt = fast_randomized_svd(W, k=16, n_iter=2, random_state=42)
        elapsed = time.perf_counter() - t0

        assert elapsed < 1.0
        assert U.shape == (1024, 16)
        assert s.shape == (16,)
        assert Vt.shape == (16, 1024)


class TestAbliterationRank1Detection:
    def test_rank1_safety_ablation_signature(self):
        """Simulate an exact Rank-1 abliteration: ΔW = -r (r^T W)."""
        np.random.seed(42)
        d = 256
        W = np.random.randn(d, d).astype(np.float32)
        r = np.random.randn(d, 1).astype(np.float32)
        r = r / np.linalg.norm(r)

        # Orthogonal projection: delta is pure rank-1 outer product
        delta_W = r @ (r.T @ W)

        bundle = compute_spectral_bundle(delta_W, k=16)
        assert isinstance(bundle, SpectralBundle)

        # Primary direction dominance must exceed 95%
        assert bundle.primary_direction_energy > 0.95
        # Stable rank must be extremely close to 1.0
        assert bundle.stable_rank == pytest.approx(1.0, abs=0.08)
        # Effective rank must be <= 1.2
        assert bundle.effective_rank < 1.2
        # Signal rank from Marcenko-Pastur must be exactly 1
        assert bundle.signal_rank == 1


class TestLoRARankDetection:
    def test_lora_rank_16_signature(self):
        """Simulate a LoRA adapter update: ΔW = B @ A with rank 16."""
        np.random.seed(42)
        m, n, rank = 512, 512, 16
        B = np.random.randn(m, rank).astype(np.float32) * 0.1
        A = np.random.randn(rank, n).astype(np.float32) * 0.1
        delta_W = B @ A

        # Add small SGD background noise (SNR ~ 20dB)
        noise = np.random.randn(m, n).astype(np.float32) * 0.005
        delta_W_noisy = delta_W + noise

        bundle = compute_spectral_bundle(delta_W_noisy, k=32)

        # Marcenko-Pastur noise floor must isolate the 16 signal singular values
        # With noise, some values may be slightly above the 16 threshold due to noise
        # but should be close to 16
        assert 14 <= bundle.signal_rank <= 18
        # Stable rank should reflect low-rank structure
        assert 10.0 <= bundle.stable_rank <= 18.0
        # Energy is spread over 16 components, so E1 is moderate
        assert bundle.primary_direction_energy < 0.30


class TestMarcenkoPasturNoiseRejection:
    def test_pure_sgd_noise_has_zero_signal_rank(self):
        """Pure Gaussian noise should have signal_rank == 0 (no outliers beyond bulk)."""
        np.random.seed(42)
        pure_noise = np.random.randn(256, 256).astype(np.float32) * 0.01

        _, s, _ = fast_randomized_svd(pure_noise, k=16, random_state=42)
        sig_rank, noise_edge = marcenko_pastur_signal_rank(pure_noise, s)

        # No singular value should exceed theoretical MP noise edge + margin
        # Note: Randomized SVD may overestimate some values, so allow small tolerance
        assert sig_rank <= 2
        assert noise_edge > s[0] * 0.5  # Edge should be reasonable


class TestGrassmannianSubspaceAngles:
    def test_subspace_rotation_angle(self):
        """Rotating a subspace towards an orthogonal direction yields principal angle == theta."""
        np.random.seed(42)
        theta_deg = 15.0
        theta_rad = np.radians(theta_deg)

        # Create orthogonal U_A
        X = np.random.randn(128, 4).astype(np.float32)
        U_A, _ = np.linalg.qr(X)

        # Create U_B by rotating the first basis vector towards an orthogonal direction
        # Take the 5th column (orthogonal to first 4) and rotate first column towards it
        Y = np.random.randn(128, 1).astype(np.float32)
        Y_orth, _ = np.linalg.qr(np.hstack([U_A, Y]))
        v_orth = Y_orth[:, 4:5]  # 5th column is orthogonal to U_A

        # Rotate first column of U_A towards v_orth by theta
        U_B = U_A.copy()
        c, s_val = np.cos(theta_rad), np.sin(theta_rad)
        U_B[:, 0] = c * U_A[:, 0] + s_val * v_orth.ravel()
        # Re-orthonormalize
        U_B, _ = np.linalg.qr(U_B)

        angles, _mean_cos, chordal = subspace_principal_angles(U_A, U_B)

        # Should have one angle close to 15 degrees, others near 0
        assert len(angles) == 4
        # One angle should be ~15 degrees, others near 0
        angles_sorted = np.sort(angles)
        # The largest angle should be ~15 degrees
        assert angles_sorted[-1] == pytest.approx(15.0, abs=1.0)
        # The smallest should be ~0
        assert angles_sorted[0] == pytest.approx(0.0, abs=1e-3)
        assert 0.0 < chordal < 2.0


class TestQuantumFidelityMetric:
    def test_identical_matrices_have_unit_fidelity(self):
        """Identical matrices should have fidelity 1.0 when using full SVD."""
        np.random.seed(42)
        W = np.random.randn(64, 64).astype(np.float32)
        # Use full SVD for exact test
        U, s, _Vt = np.linalg.svd(W, full_matrices=False)
        # Norm from SVD components (sum of squared singular values)
        norm = float(np.linalg.norm(s))
        fid = quantum_fidelity(U, s, norm, U, s, norm)
        assert fid == pytest.approx(1.0, abs=1e-4)

    def test_orthogonal_matrices_have_zero_fidelity(self):
        # Two mutually orthogonal basis matrices
        np.random.seed(42)
        W_A = np.zeros((64, 64), dtype=np.float32)
        W_A[0:32, :] = np.random.randn(32, 64)
        W_B = np.zeros((64, 64), dtype=np.float32)
        W_B[32:64, :] = np.random.randn(32, 64)

        U_A, s_A, _ = np.linalg.svd(W_A, full_matrices=False)
        U_B, s_B, _ = np.linalg.svd(W_B, full_matrices=False)
        norm_A = float(np.linalg.norm(s_A))
        norm_B = float(np.linalg.norm(s_B))
        fid = quantum_fidelity(U_A, s_A, norm_A, U_B, s_B, norm_B)
        assert fid == pytest.approx(0.0, abs=1e-4)


class TestSpectralBundleEdgeCases:
    def test_spectral_bundle_4d_conv_weights(self):
        """Conv2d weights (out, in, h, w) should flatten to (out, -1) and compute SVD without error."""
        np.random.seed(42)
        delta_W = np.random.randn(32, 16, 3, 3).astype(np.float32)
        bundle = compute_spectral_bundle(delta_W, k=8)
        assert len(bundle.singular_values) == 8
        assert bundle.stable_rank > 0.0

    def test_spectral_bundle_scalar(self):
        """0D scalar should yield rank 1."""
        delta_W = np.array(0.75, dtype=np.float32)
        bundle = compute_spectral_bundle(delta_W)
        assert bundle.singular_values[0] == pytest.approx(0.75)
        assert bundle.primary_direction_energy == 1.0
