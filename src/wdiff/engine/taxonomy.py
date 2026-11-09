from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from wdiff.engine.metrics import MetricsBundle
from wdiff.engine.spectral import SpectralBundle


class TaxonomyBadge(Enum):
    UNMODIFIED = "Unmodified"
    RANK1_ABLATION = "Surgical Rank-1 Ablation"
    LOW_RANK_ADAPTER = "Low-Rank Adapter (LoRA/DPO)"
    SPARSE_UPDATE = "Sparse Update (TIES/DARE)"
    FULL_RANK_CPT = "Full-Rank Continual Learning"
    STOCHASTIC_NOISE = "Stochastic SGD Noise"


@dataclass(frozen=True, slots=True)
class TaxonomyResult:
    badge: TaxonomyBadge
    confidence: float
    explanation: str
    evidence: dict[str, float | int | str]


def classify_weight_delta(
    delta_W: np.ndarray,
    metrics: MetricsBundle,
    spectral: SpectralBundle,
) -> TaxonomyResult:
    """
    Classify a weight delta matrix into a taxonomy category.

    Parameters:
        delta_W: Weight difference matrix
        metrics: Basic metrics (relative_frobenius, cosine_similarity, etc.)
        spectral: Spectral bundle from compute_spectral_bundle

    Returns:
        TaxonomyResult with badge, confidence, explanation, and evidence
    """
    # Compute sparsity
    sparsity = float(np.mean(np.abs(delta_W) < 1e-7))

    # Extract key metrics
    delta_f = metrics.relative_frobenius
    max_delta = metrics.max_element_delta
    primary_energy = spectral.primary_direction_energy
    stable_rank = spectral.stable_rank
    signal_rank = spectral.signal_rank
    effective_rank = spectral.effective_rank

    # Evidence dictionary
    evidence = {
        "sparsity": sparsity,
        "relative_frobenius": delta_f,
        "max_element_delta": max_delta,
        "primary_direction_energy": primary_energy,
        "stable_rank": stable_rank,
        "signal_rank": signal_rank,
        "effective_rank": effective_rank,
    }

    # Rule 1: Unmodified (zero or near-zero delta)
    if max_delta < 1e-7 or delta_f < 1e-6:
        return TaxonomyResult(
            badge=TaxonomyBadge.UNMODIFIED,
            confidence=0.99,
            explanation="Weights are effectively identical (ΔF < 1e-6, max Δ < 1e-7).",
            evidence=evidence,
        )

    # Rule 2: Sparse update (TIES/DARE style)
    if sparsity >= 0.65:
        confidence = min(0.95, 0.7 + sparsity * 0.3)
        return TaxonomyResult(
            badge=TaxonomyBadge.SPARSE_UPDATE,
            confidence=confidence,
            explanation=f"Sparse update detected ({sparsity:.1%} zero elements). Consistent with TIES/DARE sparse merging.",
            evidence=evidence,
        )

    # Rule 3: Rank-1 abliteration
    if primary_energy > 0.90 and stable_rank <= 1.35:
        confidence = min(0.98, 0.85 + (primary_energy - 0.9) * 2)
        return TaxonomyResult(
            badge=TaxonomyBadge.RANK1_ABLATION,
            confidence=confidence,
            explanation=f"Surgical Rank-1 Ablation: {primary_energy:.1%} energy in primary direction, stable rank {stable_rank:.2f}.",
            evidence=evidence,
        )

    # Rule 4: Pure stochastic noise
    if signal_rank == 0:
        return TaxonomyResult(
            badge=TaxonomyBadge.STOCHASTIC_NOISE,
            confidence=0.90,
            explanation="Pure stochastic SGD noise: no signal singular values beyond Marcenko-Pastur noise floor.",
            evidence=evidence,
        )

    # Rule 5: Low-rank adapter (LoRA/DPO)
    if 2 <= signal_rank <= 64 and stable_rank <= 40:
        confidence = 0.85
        return TaxonomyResult(
            badge=TaxonomyBadge.LOW_RANK_ADAPTER,
            confidence=confidence,
            explanation=f"Low-Rank Adapter (LoRA/DPO): signal rank {signal_rank}, stable rank {stable_rank:.1f}.",
            evidence=evidence,
        )

    # Default: Full-rank continual learning
    return TaxonomyResult(
        badge=TaxonomyBadge.FULL_RANK_CPT,
        confidence=0.75,
        explanation=f"Full-Rank Continual Learning: signal rank {signal_rank}, stable rank {stable_rank:.1f}.",
        evidence=evidence,
    )
