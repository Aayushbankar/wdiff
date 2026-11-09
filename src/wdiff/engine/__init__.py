from __future__ import annotations

from .diff import (
    DiffResult,
    DiffSummary,
    TensorDiffResult,
    diff_models,
)
from .inspect import (
    InspectionSummary,
    LayerInspectionResult,
    ModelInspectionResult,
    inspect_model,
)
from .lens import (
    LayerLensResult,
    TokenConcept,
    Vocabulary,
    decode_layer_concept,
    is_down_proj_layer,
    project_vector_to_logits_chunked,
)
from .lora import (
    LoRAAdapter,
    LoRADiffResult,
    LoRALayer,
    apply_lora,
    apply_lora_adapter,
    diff_with_lora,
    load_lora_adapter,
    merge_lora_into_model,
)
from .metrics import (
    MetricsBundle,
    compute_all_metrics,
    cosine_similarity,
    max_element_delta,
    mean_element_delta,
    relative_frobenius,
)
from .ring_buffer import PipelinedRingBuffer, RingBufferSlot
from .rope import (
    deinterleave_rope,
    interleave_rope,
    is_rope_tensor,
)
from .spectral import (
    SpectralBundle,
    compute_spectral_bundle,
    fast_randomized_svd,
    marcenko_pastur_signal_rank,
    quantum_fidelity,
    subspace_principal_angles,
)
from .taxonomy import (
    TaxonomyBadge,
    TaxonomyResult,
    classify_weight_delta,
)

__all__ = [
    "DiffResult",
    "DiffSummary",
    "InspectionSummary",
    "LayerInspectionResult",
    "LayerLensResult",
    "LoRAAdapter",
    "LoRADiffResult",
    "LoRALayer",
    "MetricsBundle",
    "ModelInspectionResult",
    "PipelinedRingBuffer",
    "RingBufferSlot",
    "SpectralBundle",
    "TaxonomyBadge",
    "TaxonomyResult",
    "TensorDiffResult",
    "TokenConcept",
    "Vocabulary",
    "apply_lora",
    "apply_lora_adapter",
    "classify_weight_delta",
    "compute_all_metrics",
    "compute_spectral_bundle",
    "cosine_similarity",
    "decode_layer_concept",
    "deinterleave_rope",
    "diff_models",
    "diff_with_lora",
    "fast_randomized_svd",
    "inspect_model",
    "interleave_rope",
    "is_down_proj_layer",
    "is_rope_tensor",
    "load_lora_adapter",
    "marcenko_pastur_signal_rank",
    "max_element_delta",
    "mean_element_delta",
    "merge_lora_into_model",
    "project_vector_to_logits_chunked",
    "quantum_fidelity",
    "relative_frobenius",
    "subspace_principal_angles",
]
