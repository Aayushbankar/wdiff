from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class SpectralBundle:
    singular_values: tuple[float, ...]
    primary_direction_energy: float
    stable_rank: float
    effective_rank: float
    signal_rank: int
    marcenko_pastur_edge: float
    signal_energy_ratio: float


def fast_randomized_svd(
    A: np.ndarray,
    k: int = 16,
    n_iter: int = 2,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fast randomized SVD using Halko-Martinsson-Tropp algorithm.

    Parameters:
        A: Input matrix of shape (m, n)
        k: Number of singular vectors to compute
        n_iter: Number of power iterations (2-4 typically sufficient)
        random_state: Seed for reproducibility

    Returns:
        U: Left singular vectors, shape (m, k)
        s: Singular values, shape (k,)
        Vt: Right singular vectors, shape (k, n)
    """
    m, n = A.shape
    k = min(k, min(m, n))

    if random_state is not None:
        np.random.seed(random_state)

    # Oversampling parameter
    p = min(10, min(m, n) - k)
    l = k + p

    # Step 1: Generate random test matrix Omega
    Omega = np.random.randn(n, l).astype(A.dtype)

    # Step 2: Build orthonormal basis for range of A
    Y = A @ Omega
    for _ in range(n_iter):
        Y = A @ (A.T @ Y)

    # Orthonormalize
    Q, _ = np.linalg.qr(Y, mode="reduced")

    # Step 3: Project A onto the subspace
    B = Q.T @ A  # shape (l, n)

    # Step 4: SVD on small matrix
    U_tilde, s, Vt = np.linalg.svd(B, full_matrices=False)

    # Step 4: Truncate to k
    U = Q @ U_tilde[:, :k]
    s = s[:k]
    Vt = Vt[:k, :]

    return U, s, Vt


def marcenko_pastur_signal_rank(
    A: np.ndarray,
    s: np.ndarray,
) -> tuple[int, float]:
    """
    Estimate signal rank using Marcenko-Pastur noise floor.

    Parameters:
        A: Original matrix
        s: Singular values from SVD

    Returns:
        signal_rank: Number of singular values beyond noise floor
        noise_edge: Theoretical noise edge s_+
    """
    if A.ndim > 2:
        A = A.reshape(A.shape[0], -1)
    m, n = A.shape

    # Estimate noise variance using robust median absolute deviation
    # Subsample for speed if matrix is large
    max_samples = 100000
    flat_A = A.ravel()
    if len(flat_A) > max_samples:
        idx = np.random.choice(len(flat_A), max_samples, replace=False)
        sample = flat_A[idx]
    else:
        sample = flat_A

    median_A = np.median(sample)
    mad = np.median(np.abs(sample - median_A))
    sigma_hat = mad / 0.67448975  # MAD to std conversion for Gaussian

    # Marcenko-Pastur upper edge for singular values of m×n noise matrix
    # s_+ = sigma * (sqrt(m) + sqrt(n))
    s_plus = sigma_hat * (np.sqrt(m) + np.sqrt(n))

    # Count singular values beyond noise edge (with small margin for numerical stability)
    signal_rank = int(np.sum(s > s_plus * 1.01))

    return signal_rank, float(s_plus)


def subspace_principal_angles(
    U_A: np.ndarray,
    U_B: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """
    Compute principal angles between two subspaces.

    Parameters:
        U_A: Orthonormal basis for subspace A, shape (m, k)
        U_B: Orthonormal basis for subspace B, shape (m, k)

    Returns:
        angles: Principal angles in degrees, sorted ascending
        mean_cos: Mean cosine of principal angles
        chordal_distance: sqrt(k - ||M||_F^2)
    """
    # Compute overlap matrix M = U_A^T U_B
    M = U_A.T @ U_B

    # SVD of M gives cosines of principal angles
    _, s_M, _ = np.linalg.svd(M, full_matrices=False)

    # Clip to [-1, 1] for numerical stability
    s_M = np.clip(s_M, 0.0, 1.0)

    # Principal angles in degrees
    angles = np.degrees(np.arccos(s_M))
    angles = np.sort(angles)

    mean_cos = float(np.mean(s_M))

    # Chordal distance
    chordal = float(np.sqrt(U_A.shape[1] - np.sum(s_M**2)))

    return angles, mean_cos, chordal


def quantum_fidelity(
    U_A: np.ndarray,
    s_A: np.ndarray,
    norm_A: float,
    U_B: np.ndarray,
    s_B: np.ndarray,
    norm_B: float,
) -> float:
    """
    Quantum fidelity between two low-rank matrices represented by SVD.

    F = (sum_i sigma_i(C))^2 where C = diag(s_A/norm_A) @ U_A^T @ U_B @ diag(s_B/norm_B)

    Parameters:
        U_A, U_B: Left singular vectors
        s_A, s_B: Singular values
        norm_A, norm_B: Frobenius norms of original matrices (or SVD-truncated norms)

    Returns:
        Quantum fidelity in [0, 1]
    """
    # Normalize singular values
    s_A_norm = s_A / norm_A if norm_A > 0 else s_A
    s_B_norm = s_B / norm_B if norm_B > 0 else s_B

    # Overlap matrix
    M = U_A.T @ U_B

    # Construct C = diag(s_A_norm) @ M @ diag(s_B_norm)
    C = np.diag(s_A_norm) @ M @ np.diag(s_B_norm)

    # Singular values of C
    _, s_C, _ = np.linalg.svd(C, full_matrices=False)

    fidelity = float(np.sum(s_C) ** 2)
    return np.clip(fidelity, 0.0, 1.0)


def compute_spectral_bundle(
    delta_W: np.ndarray,
    k: int = 16,
) -> SpectralBundle:
    """
    Compute comprehensive spectral bundle from weight delta matrix.

    Parameters:
        delta_W: Weight difference matrix (2D), vector (1D), or tensor (>2D)
        k: Number of singular vectors for randomized SVD

    Returns:
        SpectralBundle with all spectral metrics
    """
    # Handle scalar (0D)
    if delta_W.ndim == 0:
        val = abs(float(delta_W))
        return SpectralBundle(
            singular_values=(val,),
            primary_direction_energy=1.0 if val > 0 else 0.0,
            stable_rank=1.0 if val > 0 else 0.0,
            effective_rank=1.0 if val > 0 else 0.0,
            signal_rank=1 if val > 0 else 0,
            marcenko_pastur_edge=0.0,
            signal_energy_ratio=1.0 if val > 0 else 0.0,
        )

    # Handle 1D vectors (degenerate rank-1 case)
    if delta_W.ndim == 1:
        norm = float(np.linalg.norm(delta_W))
        if norm == 0:
            return SpectralBundle(
                singular_values=(0.0,),
                primary_direction_energy=0.0,
                stable_rank=0.0,
                effective_rank=0.0,
                signal_rank=0,
                marcenko_pastur_edge=0.0,
                signal_energy_ratio=0.0,
            )
        return SpectralBundle(
            singular_values=(norm,),
            primary_direction_energy=1.0,
            stable_rank=1.0,
            effective_rank=1.0,
            signal_rank=1,
            marcenko_pastur_edge=0.0,
            signal_energy_ratio=1.0,
        )

    # Flatten higher-dimensional tensors (e.g. 4D conv weights: (out, in, h, w) -> (out, -1))
    matrix = delta_W.reshape(delta_W.shape[0], -1) if delta_W.ndim > 2 else delta_W

    # Compute randomized SVD
    _U, s, _Vt = fast_randomized_svd(matrix, k=k, n_iter=2, random_state=42)

    # Compute spectral metrics
    s_squared = s**2
    total_energy = float(np.sum(s_squared))

    if total_energy == 0:
        return SpectralBundle(
            singular_values=tuple(float(x) for x in s),
            primary_direction_energy=0.0,
            stable_rank=0.0,
            effective_rank=0.0,
            signal_rank=0,
            marcenko_pastur_edge=0.0,
            signal_energy_ratio=0.0,
        )

    # Primary direction energy (E1 = sigma_1^2 / sum sigma_i^2)
    primary_direction_energy = float(s_squared[0] / total_energy)

    # Stable rank = ||W||_F^2 / sigma_1^2
    stable_rank = float(total_energy / s_squared[0]) if s_squared[0] > 0 else 0.0

    # Effective rank = exp(-sum p_i log p_i) where p_i = sigma_i^2 / sum sigma_j^2
    p = s_squared / total_energy
    p = p[p > 0]  # Avoid log(0)
    effective_rank = float(np.exp(-np.sum(p * np.log(p))))

    # Marcenko-Pastur signal rank
    signal_rank, mp_edge = marcenko_pastur_signal_rank(matrix, s)

    # Signal energy ratio (variance above noise floor / total variance)
    if signal_rank > 0 and signal_rank < len(s):
        signal_energy = float(np.sum(s_squared[:signal_rank]))
        signal_energy_ratio = signal_energy / total_energy
    else:
        signal_energy_ratio = 0.0 if signal_rank == 0 else 1.0

    return SpectralBundle(
        singular_values=tuple(float(x) for x in s),
        primary_direction_energy=primary_direction_energy,
        stable_rank=stable_rank,
        effective_rank=effective_rank,
        signal_rank=signal_rank,
        marcenko_pastur_edge=float(mp_edge),
        signal_energy_ratio=signal_energy_ratio,
    )
