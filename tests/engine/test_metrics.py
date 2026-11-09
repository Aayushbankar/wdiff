import numpy as np
import pytest

from wdiff.engine.metrics import (
    MetricsBundle,
    compute_all_metrics,
    cosine_similarity,
    max_element_delta,
    mean_element_delta,
    relative_frobenius,
)


class TestRelativeFrobenius:
    def test_identical_tensors(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        b = a.copy()
        assert relative_frobenius(a, b) == 0.0

    def test_scaled_tensor(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        b = 2.0 * a
        # ||a - 2a||_F = ||-a||_F = ||a||_F
        # ||a||_F / ||a||_F = 1.0
        result = relative_frobenius(a, b)
        assert np.isclose(result, 1.0, rtol=1e-6)

    def test_zero_tensor(self):
        a = np.zeros((10, 10), dtype=np.float32)
        b = np.ones((10, 10), dtype=np.float32) * 5.0
        # ||0 - 5||_F = sqrt(100 * 25) = 50
        # ||0||_F = 0, so result = 50 / (0 + eps) -> large
        result = relative_frobenius(a, b)
        assert result > 1e6  # effectively infinite due to eps

    def test_large_tensor_stability(self):
        # Test numerical stability with large tensor (14336 x 4096 ~ 58M elements)
        # Use smaller for test speed but still substantial
        rng = np.random.default_rng(42)
        a = rng.normal(size=(1024, 1024)).astype(np.float32)
        b = a + rng.normal(scale=1e-3, size=(1024, 1024)).astype(np.float32)
        result = relative_frobenius(a, b)
        assert np.isfinite(result)
        assert result >= 0.0


class TestCosineSimilarity:
    def test_identical_tensors(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        b = a.copy()
        assert np.isclose(cosine_similarity(a, b), 1.0, rtol=1e-6)

    def test_inverted_tensors(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        b = -a
        assert np.isclose(cosine_similarity(a, b), -1.0, rtol=1e-6)

    def test_orthogonal_vectors(self):
        a = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
        b = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.float32)
        result = cosine_similarity(a, b)
        assert np.isclose(result, 0.0, atol=1e-6)

    def test_orthogonal_matrices(self):
        a = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
        b = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        result = cosine_similarity(a, b)
        assert np.isclose(result, 0.0, atol=1e-6)

    def test_zero_tensor(self):
        a = np.zeros((10, 10), dtype=np.float32)
        b = np.ones((10, 10), dtype=np.float32)
        # 0 dot anything = 0, norms: 0 and >0 -> 0 / eps = 0
        result = cosine_similarity(a, b)
        assert np.isclose(result, 0.0, atol=1e-6)


class TestMaxElementDelta:
    def test_identical_tensors(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        b = a.copy()
        assert max_element_delta(a, b) == 0.0

    def test_known_delta(self):
        a = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
        b = np.array([[1.0, -2.0], [3.0, -4.0]], dtype=np.float32)
        assert max_element_delta(a, b) == 4.0

    def test_negative_delta(self):
        a = np.array([[5.0, 5.0], [5.0, 5.0]], dtype=np.float32)
        b = np.array([[3.0, 7.0], [4.0, 6.0]], dtype=np.float32)
        assert max_element_delta(a, b) == 2.0


class TestMeanElementDelta:
    def test_identical_tensors(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        b = a.copy()
        assert mean_element_delta(a, b) == 0.0

    def test_known_delta(self):
        a = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
        b = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        # |1|+|2|+|3|+|4| / 4 = 10/4 = 2.5
        assert np.isclose(mean_element_delta(a, b), 2.5, rtol=1e-6)


class TestComputeAllMetrics:
    def test_identical_tensors(self):
        a = np.random.default_rng(42).normal(size=(64, 256)).astype(np.float32)
        b = a.copy()
        m = compute_all_metrics(a, b)
        assert isinstance(m, MetricsBundle)
        assert np.isclose(m.relative_frobenius, 0.0, atol=1e-6)
        assert np.isclose(m.cosine_similarity, 1.0, atol=1e-6)
        assert np.isclose(m.max_element_delta, 0.0, atol=1e-6)
        assert np.isclose(m.mean_element_delta, 0.0, atol=1e-6)

    def test_inverted_tensors(self):
        a = np.random.default_rng(42).normal(size=(64, 256)).astype(np.float32)
        b = -a
        m = compute_all_metrics(a, b)
        assert np.isclose(m.cosine_similarity, -1.0, rtol=1e-6)
        assert np.isclose(m.relative_frobenius, 2.0, rtol=1e-6)

    def test_small_perturbation(self):
        a = np.random.default_rng(42).normal(size=(64, 256)).astype(np.float32)
        b = a + np.random.default_rng(123).normal(scale=1e-4, size=a.shape).astype(np.float32)
        m = compute_all_metrics(a, b)
        assert m.relative_frobenius > 0.0
        assert m.relative_frobenius < 1e-3  # small perturbation
        assert m.cosine_similarity > 0.999
        assert m.max_element_delta > 0.0
        assert m.mean_element_delta > 0.0

    def test_large_tensor_no_nan_overflow(self):
        rng = np.random.default_rng(42)
        # 14336 x 4096 would be too slow for tests, use smaller but substantial
        a = rng.normal(size=(2048, 2048)).astype(np.float32)
        b = a + rng.normal(scale=0.01, size=a.shape).astype(np.float32)
        m = compute_all_metrics(a, b)
        assert np.isfinite(m.relative_frobenius)
        assert np.isfinite(m.cosine_similarity)
        assert np.isfinite(m.max_element_delta)
        assert np.isfinite(m.mean_element_delta)
        assert not np.isnan(m.relative_frobenius)
        assert not np.isnan(m.cosine_similarity)


class TestShapeMismatch:
    def test_relative_frobenius_raises(self):
        a = np.zeros((10, 10), dtype=np.float32)
        b = np.zeros((5, 5), dtype=np.float32)
        with pytest.raises(ValueError, match="Shape mismatch"):
            relative_frobenius(a, b)

    def test_cosine_similarity_raises(self):
        a = np.zeros((10, 10), dtype=np.float32)
        b = np.zeros((5, 5), dtype=np.float32)
        with pytest.raises(ValueError, match="Shape mismatch"):
            cosine_similarity(a, b)

    def test_max_element_delta_raises(self):
        a = np.zeros((10, 10), dtype=np.float32)
        b = np.zeros((5, 5), dtype=np.float32)
        with pytest.raises(ValueError, match="Shape mismatch"):
            max_element_delta(a, b)

    def test_mean_element_delta_raises(self):
        a = np.zeros((10, 10), dtype=np.float32)
        b = np.zeros((5, 5), dtype=np.float32)
        with pytest.raises(ValueError, match="Shape mismatch"):
            mean_element_delta(a, b)

    def test_compute_all_metrics_raises(self):
        a = np.zeros((10, 10), dtype=np.float32)
        b = np.zeros((5, 5), dtype=np.float32)
        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_all_metrics(a, b)


class TestHandComputedValues:
    def test_exact_frobenius(self):
        # A = [[3, 4]], B = [[0, 0]]
        # diff = [[3, 4]], ||diff||_F = 5
        # ||A||_F = 5, so 5/5 = 1.0
        a = np.array([[3.0, 4.0]], dtype=np.float32)
        b = np.array([[0.0, 0.0]], dtype=np.float32)
        assert np.isclose(relative_frobenius(a, b), 1.0)

    def test_exact_cosine(self):
        # A = [3, 4], B = [4, 3]
        # dot = 12 + 12 = 24
        # ||A|| = 5, ||B|| = 5, cos = 24/25 = 0.96
        a = np.array([[3.0, 4.0]], dtype=np.float32)
        b = np.array([[4.0, 3.0]], dtype=np.float32)
        assert np.isclose(cosine_similarity(a, b), 0.96)

    def test_exact_max_delta(self):
        a = np.array([[1.0, 5.0], [3.0, 2.0]], dtype=np.float32)
        b = np.array([[2.0, 5.0], [3.0, 8.0]], dtype=np.float32)
        # deltas: |1|, |0|, |0|, |6| -> max = 6
        assert max_element_delta(a, b) == 6.0

    def test_exact_mean_delta(self):
        a = np.array([[1.0, 2.0]], dtype=np.float32)
        b = np.array([[2.0, 4.0]], dtype=np.float32)
        # deltas: 1, 2 -> mean = 1.5
        assert np.isclose(mean_element_delta(a, b), 1.5)
