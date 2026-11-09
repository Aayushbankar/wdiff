"""Generate synthetic test fixtures for wdiff readers."""

import os

import numpy as np
from gguf import GGMLQuantizationType, GGUFWriter
from gguf.quants import quantize
from safetensors.numpy import save_file

FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
os.makedirs(FIXTURE_DIR, exist_ok=True)

np.random.seed(42)

# Small tensors for fast testing
# Use dimensions compatible with Q4_K/Q8_0 block size (256) for quantized tests
# Last dimension must be multiple of 256 for K-quants
tensors = {
    "embedding.weight": np.random.randn(64, 256).astype(np.float32),
    "layer.0.attn_q.weight": np.random.randn(256, 256).astype(np.float32),
    "layer.0.attn_k.weight": np.random.randn(256, 256).astype(np.float32),
    "layer.0.attn_v.weight": np.random.randn(256, 256).astype(np.float32),
    "layer.0.attn_output.weight": np.random.randn(256, 256).astype(np.float32),
    "layer.0.ffn_gate.weight": np.random.randn(256, 256).astype(np.float32),
    "layer.0.ffn_up.weight": np.random.randn(256, 256).astype(np.float32),
    "layer.0.ffn_down.weight": np.random.randn(256, 256).astype(np.float32),
    "lm_head.weight": np.random.randn(64, 256).astype(np.float32),
}

# Create modified versions for diff testing
tensors_b = {k: v + np.random.randn(*v.shape).astype(np.float32) * 0.01 for k, v in tensors.items()}

# Save safetensors
save_file(tensors, os.path.join(FIXTURE_DIR, "model_a.safetensors"))
save_file(tensors_b, os.path.join(FIXTURE_DIR, "model_b.safetensors"))
print(f"Created safetensors fixtures in {FIXTURE_DIR}")

# Create GGUF fixtures (F32 - no quantization for simplicity)
writer_a = GGUFWriter(os.path.join(FIXTURE_DIR, "model_a.gguf"), "test_model")
for name, tensor in tensors.items():
    writer_a.add_tensor(name, tensor, raw_dtype=GGMLQuantizationType.F32)
writer_a.write_header_to_file()
writer_a.write_kv_data_to_file()
writer_a.write_tensors_to_file()
writer_a.close()

writer_b = GGUFWriter(os.path.join(FIXTURE_DIR, "model_b.gguf"), "test_model")
for name, tensor in tensors_b.items():
    writer_b.add_tensor(name, tensor, raw_dtype=GGMLQuantizationType.F32)
writer_b.write_header_to_file()
writer_b.write_kv_data_to_file()
writer_b.write_tensors_to_file()
writer_b.close()

print(f"Created GGUF fixtures in {FIXTURE_DIR}")

# Also create a Q8_0 quantized version for testing dequantization (clean round-trip)
writer_q = GGUFWriter(os.path.join(FIXTURE_DIR, "model_a_q8_0.gguf"), "test_model")
for name, tensor in tensors.items():
    quantized_data = quantize(tensor, GGMLQuantizationType.Q8_0)
    writer_q.add_tensor(name, quantized_data, raw_dtype=GGMLQuantizationType.Q8_0)
writer_q.write_header_to_file()
writer_q.write_kv_data_to_file()
writer_q.write_tensors_to_file()
writer_q.close()
print(f"Created Q8_0 quantized GGUF fixture in {FIXTURE_DIR}")
