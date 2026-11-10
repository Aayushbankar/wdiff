from __future__ import annotations

import html as html_lib
import json
from pathlib import Path
from typing import Any

from wdiff.engine.diff import DiffResult
from wdiff.engine.inspect import ModelInspectionResult

DiffOrInspection = DiffResult | ModelInspectionResult

# Consistent taxonomy palette across terminal and web dashboards
BADGE_COLORS = {
    "RANK1_ABLATION": "#e879f9",
    "LOW_RANK_ADAPTER": "#60a5fa",
    "SPARSE_UPDATE": "#fbbf24",
    "FULL_RANK_CPT": "#34d399",
    "STOCHASTIC_NOISE": "#f87171",
    "UNMODIFIED": "#6b7280",
}

BADGE_LABELS = {
    "RANK1_ABLATION": "Rank-1 Ablation",
    "LOW_RANK_ADAPTER": "Low-Rank Adapter",
    "SPARSE_UPDATE": "Sparse Update",
    "FULL_RANK_CPT": "Full-Rank CPT",
    "STOCHASTIC_NOISE": "Stochastic SGD Noise",
    "UNMODIFIED": "Unmodified",
}


def generate_html_report(
    result: DiffOrInspection,
    output_path: str | Path,
    title: str | None = None,
) -> Path:
    """Generate a standalone, self-contained interactive HTML report."""
    output_path = Path(output_path)

    if isinstance(result, ModelInspectionResult):
        html = _generate_inspection_html_content(result, title)
    else:
        html = _generate_diff_html_content(result, title)

    output_path.write_text(html, encoding="utf-8")
    return output_path


def _fmt_num(v: Any, digits: int = 4) -> str:
    if v is None or v == "—":
        return "—"
    try:
        return f"{float(v):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def _format_shape(shape: tuple[int, ...]) -> str:
    if not shape:
        return "—"
    return " × ".join(str(d) for d in shape)


def _prepare_table_data(result: DiffResult) -> list[dict]:
    rows = []
    for t in result.tensor_results:
        badge = t.taxonomy.badge.name if t.taxonomy else None

        if t.status == "PARTIAL_MATCH" and t.partial_info:
            pi = t.partial_info
            shape_display = f"{pi.common_shape[0]:,}→{t.shape_b[0]:,} × " + " × ".join(
                str(d) for d in pi.common_shape[1:]
            )
        else:
            shape_display = (
                _format_shape(t.shape_a)
                if t.shape_a == t.shape_b
                else f"{_format_shape(t.shape_a)} → {_format_shape(t.shape_b)}"
            )

        # Spectral payload
        spectral_data = None
        if t.spectral:
            spectral_data = {
                "singular_values": [float(sv) for sv in t.spectral.singular_values],
                "marcenko_pastur_edge": float(t.spectral.marcenko_pastur_edge),
                "signal_rank": int(t.spectral.signal_rank),
                "effective_rank": float(t.spectral.effective_rank),
                "stable_rank": float(t.spectral.stable_rank),
                "primary_direction_energy": float(t.spectral.primary_direction_energy),
                "signal_energy_ratio": float(t.spectral.signal_energy_ratio),
            }

        # Subspace geometry payload
        subspace_data = None
        subspace = getattr(t, "subspace", None)
        if subspace:
            subspace_data = {
                "principal_angles_deg": [float(a) for a in subspace.principal_angles_deg],
                "mean_cosine": float(subspace.mean_cosine),
                "chordal_distance": float(subspace.chordal_distance),
                "geodesic_distance": float(subspace.geodesic_distance),
                "max_principal_angle": float(subspace.max_principal_angle),
                "uhlmann_fidelity": float(subspace.uhlmann_fidelity),
            }

        # Taxonomy payload
        taxonomy_data = None
        if t.taxonomy:
            taxonomy_data = {
                "badge": t.taxonomy.badge.name,
                "badge_value": t.taxonomy.badge.value,
                "confidence": float(t.taxonomy.confidence),
                "explanation": str(t.taxonomy.explanation),
                "evidence": {
                    k: float(v) if isinstance(v, (int, float)) else str(v)
                    for k, v in t.taxonomy.evidence.items()
                },
            }

        # Metrics payload
        metrics_data = None
        if t.metrics:
            metrics_data = {
                "relative_frobenius": float(t.metrics.relative_frobenius),
                "cosine_similarity": float(t.metrics.cosine_similarity),
                "max_element_delta": float(t.metrics.max_element_delta),
                "mean_element_delta": float(t.metrics.mean_element_delta),
            }

        rows.append(
            {
                "name": t.name,
                "shape": shape_display,
                "rel_frobenius": f"{t.metrics.relative_frobenius:.4f}" if t.metrics else "—",
                "cosine_sim": f"{t.metrics.cosine_similarity:.4f}" if t.metrics else "—",
                "max_delta": f"{t.metrics.max_element_delta:.4f}" if t.metrics else "—",
                "signal_rank": t.spectral.signal_rank if t.spectral else None,
                "effective_rank": t.spectral.effective_rank if t.spectral else None,
                "max_principal_angle": (subspace.max_principal_angle if subspace else None),
                "uhlmann_fidelity": (subspace.uhlmann_fidelity if subspace else None),
                "badge": badge,
                "status": t.status,
                "spectral": spectral_data,
                "subspace": subspace_data,
                "taxonomy": taxonomy_data,
                "metrics": metrics_data,
                "partial_info": (
                    {
                        "common_shape": list(t.partial_info.common_shape),
                        "differing_axis": t.partial_info.differing_axis,
                        "extra_count": t.partial_info.extra_count,
                        "extra_source": t.partial_info.extra_source,
                        "extra_norm": t.partial_info.extra_norm,
                        "extra_mean": t.partial_info.extra_mean,
                        "extra_std": t.partial_info.extra_std,
                    }
                    if t.partial_info
                    else None
                ),
                "lens": (
                    {
                        "layer_name": t.lens.layer_name,
                        "canonical_name": t.lens.canonical_name,
                        "positive_tokens": [
                            {"token_id": tc.token_id, "token": tc.token, "logit": tc.logit}
                            for tc in t.lens.positive_tokens
                        ],
                        "negative_tokens": [
                            {"token_id": tc.token_id, "token": tc.token, "logit": tc.logit}
                            for tc in t.lens.negative_tokens
                        ],
                        "singular_value": t.lens.singular_value,
                        "energy_ratio": t.lens.energy_ratio,
                    }
                    if t.lens
                    else None
                ),
            }
        )
    return rows


def _prepare_summary_data(result: DiffResult) -> dict:
    s = result.summary
    return {
        "matched_count": s.matched_count,
        "partial_match_count": s.partial_match_count,
        "mismatched_shape_count": s.mismatched_shape_count,
        "only_in_a_count": s.only_in_a_count,
        "only_in_b_count": s.only_in_b_count,
        "lens_analyzed_count": s.lens_analyzed_count,
        "duration_seconds": result.duration_seconds,
        "mean_frobenius": s.mean_frobenius,
        "max_frobenius": s.max_frobenius_value,
        "max_frobenius_tensor": s.max_frobenius_tensor,
        "min_cosine_value": s.min_cosine_value,
    }


def _prepare_inspection_table_data(result: ModelInspectionResult) -> list[dict]:
    rows = []
    for t in result.layers:
        spectral_data = None
        if t.spectral:
            spectral_data = {
                "singular_values": [float(sv) for sv in t.spectral.singular_values],
                "marcenko_pastur_edge": float(t.spectral.marcenko_pastur_edge),
                "signal_rank": int(t.spectral.signal_rank),
                "effective_rank": float(t.spectral.effective_rank),
                "stable_rank": float(t.spectral.stable_rank),
                "primary_direction_energy": float(t.spectral.primary_direction_energy),
                "signal_energy_ratio": float(t.spectral.signal_energy_ratio),
            }

        rows.append(
            {
                "name": t.name,
                "shape": _format_shape(t.shape),
                "dtype": t.dtype,
                "param_count": t.param_count,
                "norm": t.norm,
                "outlier_score": t.outlier_score,
                "dead_neuron_count": t.dead_neuron_count,
                "is_outlier_risk": t.is_outlier_risk,
                "status": getattr(t, "status", "HEALTHY"),
                "signal_rank": t.spectral.signal_rank if t.spectral else None,
                "effective_rank": t.spectral.effective_rank if t.spectral else None,
                "badge": t.taxonomy.badge.name if getattr(t, "taxonomy", None) else None,
                "spectral": spectral_data,
                "rel_frobenius": f"{t.norm:.2e}" if t.norm else "—",
                "max_delta": f"{t.outlier_score:.2f}" if t.outlier_score else "—",
            }
        )
    return rows


def _prepare_inspection_summary_data(result: ModelInspectionResult) -> dict:
    s = result.summary
    return {
        "total_params": s.total_params,
        "trainable_params_mb": s.trainable_params_mb,
        "architecture": s.architecture,
        "tensor_count": s.tensor_count,
        "dead_neurons_total": s.dead_neurons_total,
        "outlier_risk_layer_count": s.outlier_risk_layer_count,
        "quantization_readiness": s.quantization_readiness,
        "duration_seconds": result.summary.duration_seconds,
        "mean_frobenius": getattr(s, "mean_frobenius", 0.0),
        "min_cosine_value": getattr(s, "min_cosine_value", 1.0),
    }


def _get_embedded_css() -> str:
    # Scientific instrument aesthetics: obsidian deep-space canvas, precision hairline borders,
    # high-contrast typography, interactive drawer and topology components.
    return r"""
:root {
  --bg: #07090e;
  --bg-card: #0c121e;
  --bg-elevated: #111a2d;
  --bg-hover: #16223b;
  --border: #1a2538;
  --border-glow: #24344f;
  --border-focus: #38bdf8;
  --fg: #f1f5f9;
  --fg-muted: #94a3b8;
  --fg-dim: #64748b;
  --cyan: #38bdf8;
  --violet: #c084fc;
  --rose: #f43f5e;
  --emerald: #10b981;
  --amber: #f59e0b;
  --slate: #475569;
  --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
  --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
html, body {
  background: var(--bg);
  color: var(--fg);
  font-family: var(--sans);
  line-height: 1.5;
  -webkit-font-smoothing: antialiased;
}
body {
  background-image:
    radial-gradient(1200px 500px at 20% -10%, #0e1c31 0%, transparent 70%),
    radial-gradient(1000px 400px at 80% 0%, #171633 0%, transparent 65%);
  min-height: 100vh;
}
.shell {
  max-width: 1440px;
  margin: 0 auto;
  padding: 32px 28px 64px;
}
.top-nav {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 24px;
  padding-bottom: 16px;
  border-bottom: 1px solid var(--border);
}
.brand-group {
  display: flex;
  align-items: center;
  gap: 12px;
}
.brand-logo {
  width: 32px;
  height: 32px;
  border-radius: 8px;
  background: linear-gradient(135deg, #0284c7, #9333ea);
  display: flex;
  align-items: center;
  justify-content: center;
  color: #fff;
  font-weight: 800;
  font-size: 15px;
  box-shadow: 0 0 16px rgba(56, 189, 248, 0.3);
}
.brand-title {
  font-size: 18px;
  font-weight: 700;
  letter-spacing: -0.01em;
  color: var(--fg);
}
.brand-tag {
  font-size: 11px;
  padding: 2px 8px;
  border-radius: 999px;
  background: rgba(56, 189, 248, 0.1);
  border: 1px solid rgba(56, 189, 248, 0.25);
  color: var(--cyan);
  font-family: var(--mono);
}
.header-meta {
  display: flex;
  align-items: center;
  gap: 16px;
  font-size: 13px;
  color: var(--fg-muted);
}
.header-pill {
  padding: 4px 10px;
  border-radius: 6px;
  background: var(--bg-card);
  border: 1px solid var(--border);
  font-family: var(--mono);
  font-size: 12px;
  color: var(--fg);
}

.masthead {
  display: grid;
  grid-template-columns: 1fr 340px;
  gap: 20px;
  margin-bottom: 20px;
}
.masthead-info {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 20px 24px;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
}
.masthead-info h1 {
  font-size: 24px;
  font-weight: 700;
  letter-spacing: -0.02em;
  color: #fff;
  margin-bottom: 6px;
  word-break: break-all;
}
.masthead-info .deck {
  color: var(--fg-muted);
  font-size: 14px;
  margin-bottom: 16px;
}
.model-spec-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
  padding-top: 14px;
  border-top: 1px solid var(--border);
}
.model-box {
  display: flex;
  flex-direction: column;
  gap: 2px;
}
.model-box-label {
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--fg-dim);
  font-weight: 600;
}
.model-box-path {
  font-family: var(--mono);
  font-size: 12px;
  color: var(--fg);
  word-break: break-all;
  line-height: 1.4;
}
.format-pill {
  display: inline-block;
  padding: 1px 6px;
  border-radius: 4px;
  font-size: 10px;
  font-family: var(--mono);
  font-weight: 600;
  text-transform: uppercase;
  margin-left: 6px;
  background: rgba(148, 163, 184, 0.1);
  border: 1px solid var(--border);
  color: var(--fg-muted);
}

.verdict-card {
  background: linear-gradient(180deg, var(--bg-card), var(--bg-elevated));
  border: 1px solid var(--border-glow);
  border-radius: 12px;
  padding: 22px;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
  position: relative;
  overflow: hidden;
}
.verdict-card::before {
  content: "";
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 3px;
  background: linear-gradient(90deg, var(--cyan), var(--violet));
}
.verdict-eyebrow {
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--cyan);
}
.verdict-headline {
  font-size: 20px;
  font-weight: 700;
  color: #fff;
  margin: 6px 0;
  letter-spacing: -0.01em;
}
.verdict-detail {
  font-size: 13px;
  color: var(--fg-muted);
  line-height: 1.45;
}

.kpi-row {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
  gap: 12px;
  margin-bottom: 24px;
}
.kpi-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 14px 16px;
  transition: border-color 0.15s ease, transform 0.15s ease;
}
.kpi-card:hover {
  border-color: var(--border-glow);
  transform: translateY(-1px);
}
.kpi-card.hero {
  background: linear-gradient(180deg, rgba(56, 189, 248, 0.06), var(--bg-card));
  border-color: rgba(56, 189, 248, 0.3);
}
.kpi-val {
  font-size: 24px;
  font-weight: 700;
  font-family: var(--mono);
  letter-spacing: -0.02em;
  color: #fff;
}
.kpi-lbl {
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--fg-dim);
  margin-top: 4px;
}
.kpi-sub {
  font-size: 11px;
  font-family: var(--mono);
  color: var(--cyan);
  margin-top: 2px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.panel {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 12px;
  margin-bottom: 24px;
  overflow: hidden;
}
.panel-header {
  padding: 16px 20px;
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  flex-wrap: wrap;
}
.panel-title-group h2 {
  font-size: 16px;
  font-weight: 600;
  color: #fff;
  letter-spacing: -0.01em;
}
.panel-title-group p {
  font-size: 12px;
  color: var(--fg-muted);
  margin-top: 2px;
}

/* Architecture Topology Map */
.topology-container {
  padding: 20px;
  background: var(--bg);
  overflow-x: auto;
}
.topology-flow {
  display: flex;
  align-items: stretch;
  gap: 10px;
  min-width: 800px;
  padding-bottom: 6px;
}
.topo-node {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 12px 14px;
  display: flex;
  flex-direction: column;
  justify-content: center;
  transition: all 0.15s ease;
  cursor: pointer;
  user-select: none;
}
.topo-node:hover {
  border-color: var(--border-focus);
  background: var(--bg-elevated);
}
.topo-node.active {
  border-color: var(--cyan);
  box-shadow: 0 0 12px rgba(56, 189, 248, 0.2);
}
.topo-node-title {
  font-size: 12px;
  font-weight: 600;
  color: #fff;
  margin-bottom: 4px;
}
.topo-node-sub {
  font-size: 10px;
  font-family: var(--mono);
  color: var(--fg-dim);
}
.topo-connector {
  display: flex;
  align-items: center;
  color: var(--slate);
  font-weight: 700;
  font-size: 14px;
}
.topo-block {
  flex: 1;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 10px 12px;
  min-width: 130px;
  cursor: pointer;
  transition: all 0.15s ease;
}
.topo-block:hover {
  border-color: var(--border-glow);
  background: var(--bg-elevated);
}
.topo-block-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 8px;
}
.topo-block-name {
  font-size: 12px;
  font-weight: 700;
  color: #fff;
}
.topo-block-badge {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--slate);
}
.topo-subcells {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 4px;
}
.topo-cell {
  background: rgba(255, 255, 255, 0.03);
  border: 1px solid rgba(255, 255, 255, 0.05);
  border-radius: 4px;
  padding: 3px 6px;
  font-size: 10px;
  font-family: var(--mono);
  color: var(--fg-muted);
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.topo-cell.hot {
  border-color: rgba(244, 63, 94, 0.4);
  color: var(--rose);
  background: rgba(244, 63, 94, 0.08);
}
.topo-cell.lora {
  border-color: rgba(96, 165, 250, 0.4);
  color: var(--cyan);
  background: rgba(96, 165, 250, 0.08);
}

/* Charts & Views */
.chart-tabs {
  display: flex;
  gap: 8px;
}
.tab-btn {
  padding: 6px 12px;
  border-radius: 6px;
  background: var(--bg-card);
  border: 1px solid var(--border);
  color: var(--fg-muted);
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.15s ease;
}
.tab-btn.active, .tab-btn:hover {
  background: var(--bg-elevated);
  border-color: var(--border-glow);
  color: #fff;
}
.chart-body {
  padding: 16px;
  background: var(--bg);
  position: relative;
}
.chart-wrap svg {
  width: 100%;
  height: 260px;
  display: block;
}
.legend-bar {
  display: flex;
  align-items: center;
  gap: 14px;
  flex-wrap: wrap;
}
.leg-item {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 11px;
  color: var(--fg-muted);
}
.leg-dot {
  width: 8px;
  height: 8px;
  border-radius: 2px;
  background: var(--lc);
}

/* Controls */
.controls-bar {
  padding: 12px 20px;
  background: var(--bg-card);
  border-bottom: 1px solid var(--border);
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 10px;
}
.search-box {
  flex: 1 1 240px;
  position: relative;
}
.search-box input {
  width: 100%;
  padding: 7px 12px;
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 6px;
  color: #fff;
  font-family: var(--sans);
  font-size: 13px;
  transition: border-color 0.15s ease;
}
.search-box input:focus {
  outline: none;
  border-color: var(--border-focus);
}
.filter-select {
  padding: 7px 10px;
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 6px;
  color: var(--fg);
  font-family: var(--sans);
  font-size: 13px;
  cursor: pointer;
}
.filter-select:focus {
  outline: none;
  border-color: var(--border-focus);
}

/* Table */
.table-scroll {
  overflow-x: auto;
  max-height: 600px;
}
table.forensic-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
  text-align: left;
}
table.forensic-table thead th {
  position: sticky;
  top: 0;
  background: var(--bg-elevated);
  padding: 10px 14px;
  font-weight: 600;
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--fg-dim);
  border-bottom: 1px solid var(--border);
  white-space: nowrap;
  z-index: 2;
}
table.forensic-table tbody tr {
  border-bottom: 1px solid var(--border);
  cursor: pointer;
  transition: background 0.1s ease;
}
table.forensic-table tbody tr:hover {
  background: var(--bg-hover);
}
table.forensic-table tbody tr.selected {
  background: rgba(56, 189, 248, 0.08);
  border-left: 3px solid var(--cyan);
}
table.forensic-table tbody td {
  padding: 10px 14px;
  vertical-align: middle;
}
.td-name code {
  font-family: var(--mono);
  font-size: 12px;
  color: #e2e8f0;
}
.td-shape {
  font-family: var(--mono);
  font-size: 11px;
  color: var(--fg-dim);
  white-space: nowrap;
}
.td-num {
  font-family: var(--mono);
  font-size: 12px;
  color: var(--fg);
  text-align: right;
  white-space: nowrap;
}
.delta-meter {
  display: inline-flex;
  align-items: center;
  gap: 6px;
}
.mini-bar {
  width: 32px;
  height: 6px;
  background: rgba(255, 255, 255, 0.08);
  border-radius: 3px;
  overflow: hidden;
  position: relative;
}
.mini-bar-fill {
  height: 100%;
  border-radius: 3px;
  background: var(--cyan);
}
.badge {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 999px;
  font-size: 11px;
  font-weight: 600;
  font-family: var(--mono);
  background: var(--bc);
  color: #07090e;
  white-space: nowrap;
}
.status-pill {
  display: inline-block;
  padding: 2px 7px;
  border-radius: 4px;
  font-size: 11px;
  font-weight: 600;
  font-family: var(--mono);
  border: 1px solid var(--border);
  white-space: nowrap;
}
.status-match { color: var(--emerald); border-color: rgba(16, 185, 129, 0.3); background: rgba(16, 185, 129, 0.08); }
.status-partial_match { color: var(--amber); border-color: rgba(245, 158, 11, 0.3); background: rgba(245, 158, 11, 0.08); }
.status-shape_mismatch { color: var(--rose); border-color: rgba(244, 63, 94, 0.3); background: rgba(244, 63, 94, 0.08); }
.status-only_in_a, .status-only_in_b { color: var(--amber); border-color: rgba(245, 158, 11, 0.3); background: rgba(245, 158, 11, 0.08); }

.lens-cell {
  font-size: 11px;
  font-family: var(--mono);
}
.lens-tag {
  display: inline-block;
  padding: 1px 6px;
  border-radius: 3px;
  margin: 1px;
}
.lens-pos { background: rgba(16, 185, 129, 0.15); color: var(--emerald); border: 1px solid rgba(16, 185, 129, 0.3); }
.lens-neg { background: rgba(244, 63, 94, 0.15); color: var(--rose); border: 1px solid rgba(244, 63, 94, 0.3); }

/* Slide-over Detail Drawer */
.drawer-backdrop {
  position: fixed;
  top: 0; left: 0; right: 0; bottom: 0;
  background: rgba(0, 0, 0, 0.6);
  backdrop-filter: blur(4px);
  z-index: 100;
  opacity: 0;
  pointer-events: none;
  transition: opacity 0.2s ease;
}
.drawer-backdrop.open {
  opacity: 1;
  pointer-events: auto;
}
.drawer {
  position: fixed;
  top: 0; right: 0; bottom: 0;
  width: 580px;
  max-width: 90vw;
  background: var(--bg-card);
  border-left: 1px solid var(--border-glow);
  box-shadow: -8px 0 32px rgba(0, 0, 0, 0.5);
  z-index: 101;
  transform: translateX(100%);
  transition: transform 0.25s cubic-bezier(0.16, 1, 0.3, 1);
  display: flex;
  flex-direction: column;
}
.drawer.open {
  transform: translateX(0);
}
.drawer-header {
  padding: 18px 24px;
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  justify-content: space-between;
  background: var(--bg-elevated);
}
.drawer-header h3 {
  font-size: 16px;
  font-weight: 700;
  color: #fff;
  font-family: var(--mono);
  word-break: break-all;
}
.close-btn {
  background: none;
  border: none;
  color: var(--fg-dim);
  font-size: 20px;
  cursor: pointer;
  padding: 4px;
  line-height: 1;
}
.close-btn:hover { color: #fff; }
.drawer-content {
  padding: 24px;
  overflow-y: auto;
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 20px;
}
.drawer-card {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 16px;
}
.drawer-card-title {
  font-size: 12px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--fg-muted);
  margin-bottom: 12px;
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.metric-pair-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px;
}
.metric-pair {
  display: flex;
  flex-direction: column;
}
.metric-pair-k {
  font-size: 11px;
  color: var(--fg-dim);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}
.metric-pair-v {
  font-size: 15px;
  font-family: var(--mono);
  font-weight: 600;
  color: #fff;
}
.spec-plot svg {
  width: 100%;
  height: 160px;
  display: block;
}
.footer-bar {
  margin-top: 32px;
  padding-top: 20px;
  border-top: 1px solid var(--border);
  text-align: center;
  font-size: 12px;
  color: var(--fg-dim);
}

@media (max-width: 1024px) {
  .masthead { grid-template-columns: 1fr; }
  .model-spec-grid { grid-template-columns: 1fr; }
}
"""


def _generate_diff_html_content(result: DiffResult, title: str | None = None) -> str:
    table_data = _prepare_table_data(result)
    summary_data = _prepare_summary_data(result)
    has_deep = any(t.spectral is not None or t.taxonomy is not None for t in result.tensor_results)
    has_lens = any(t.lens is not None for t in result.tensor_results)
    table_data_json = json.dumps(table_data).replace("</", "<\\/").replace("<script", "<\\script")

    if title is None:
        title = f"wdiff · {Path(result.model_a_path).name} ↔ {Path(result.model_b_path).name}"
    safe_title = html_lib.escape(title)

    matched = summary_data["matched_count"]
    peak = getattr(result, "peak_ram_mb", 0.0) or 0.0
    peak_label = f"{peak / 1024:.2f} GB" if peak >= 1024 else (f"{peak:.0f} MB" if peak else "—")
    max_df = summary_data.get("max_frobenius", 0.0) or 0.0
    max_df_tensor = html_lib.escape(str(summary_data.get("max_frobenius_tensor") or "—"))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{safe_title}</title>
<style>
{_get_embedded_css()}
</style>
</head>
<body>
<div class="shell">
  <nav class="top-nav">
    <div class="brand-group">
      <div class="brand-logo">W</div>
      <div>
        <div class="brand-title">wdiff Report</div>
        <div style="font-size:11px;color:var(--fg-dim);">Neural Weight Space Forensics & Spectral Diff</div>
      </div>
      <span class="brand-tag">v0.3.0</span>
    </div>
    <div class="header-meta">
      <span>Duration: <strong style="color:#fff">{summary_data["duration_seconds"]:.2f}s</strong></span>
      <span>Peak RSS: <strong style="color:#fff">{html_lib.escape(peak_label)}</strong></span>
    </div>
  </nav>

  <header class="masthead">
    <div class="masthead-info">
      <div>
        <h1>{safe_title}</h1>
        <p class="deck">wdiff Report · Differential analysis report</p>
      </div>
      <div class="model-spec-grid">
        <div class="model-box">
          <span class="model-box-label">Base Model A</span>
          <span class="model-box-path"><code>{html_lib.escape(str(result.model_a_path))}</code><span class="format-pill">{html_lib.escape(str(result.format_a))}</span></span>
        </div>
        <div class="model-box">
          <span class="model-box-label">Target Model B</span>
          <span class="model-box-path"><code>{html_lib.escape(str(result.model_b_path))}</code><span class="format-pill">{html_lib.escape(str(result.format_b))}</span></span>
        </div>
      </div>
    </div>

    <div class="verdict-card" id="verdictBox">
      <div>
        <div class="verdict-eyebrow">Forensic Verdict</div>
        <div class="verdict-headline" id="verdictHeadline">—</div>
      </div>
      <div class="verdict-detail" id="verdictDetail">—</div>
    </div>
  </header>

  <section class="kpi-row" aria-label="Key Forensic Metrics">
    <div class="kpi-card"><div class="kpi-val">{matched}</div><div class="kpi-lbl">Matched Layers</div></div>
    <div class="kpi-card"><div class="kpi-val">{summary_data.get("partial_match_count", 0)}</div><div class="kpi-lbl">Partial Matches</div></div>
    <div class="kpi-card"><div class="kpi-val">{summary_data.get("lens_analyzed_count", 0)}</div><div class="kpi-lbl">Decoded Concepts</div></div>
    <div class="kpi-card"><div class="kpi-val">{summary_data["mismatched_shape_count"]}</div><div class="kpi-lbl">Shape Mismatches</div></div>
    <div class="kpi-card"><div class="kpi-val">{summary_data["only_in_a_count"]}</div><div class="kpi-lbl">Only in A</div></div>
    <div class="kpi-card"><div class="kpi-val">{summary_data["only_in_b_count"]}</div><div class="kpi-lbl">Only in B</div></div>
    <div class="kpi-card hero"><div class="kpi-val">{summary_data["mean_frobenius"]:.4f}</div><div class="kpi-lbl">Mean Rel. ΔF</div></div>
    <div class="kpi-card hero">
      <div class="kpi-val">{max_df:.4f}</div>
      <div class="kpi-lbl">Max Rel. ΔF</div>
      <div class="kpi-sub" title="{max_df_tensor}">{max_df_tensor}</div>
    </div>
    <div class="kpi-card"><div class="kpi-val">{summary_data["min_cosine_value"]:.4f}</div><div class="kpi-lbl">Min Cosine</div></div>
    <div class="kpi-card"><div class="kpi-val">{summary_data["duration_seconds"]:.2f}s</div><div class="kpi-lbl">Duration</div></div>
  </section>

  <!-- Interactive Model Architecture Topology Pipeline -->
  <section class="panel topology-panel">
    <div class="panel-header">
      <div class="panel-title-group">
        <h2>Model Architecture Topology</h2>
        <p>Interactive transformer execution flow. Click any block or projection cell to filter table and inspect spectrum.</p>
      </div>
      <div class="legend-bar" id="legend"></div>
    </div>
    <div class="topology-container">
      <div class="topology-flow" id="topologyFlow"></div>
    </div>
  </section>

  <!-- Multi-View Landscape SVG Chart -->
  <section class="panel chart-panel">
    <div class="panel-header">
      <div class="panel-title-group">
        <h2 id="chartTitle">Layer Delta Profile</h2>
        <p id="chartDesc">Relative Frobenius delta across network layers, color-coded by spectral taxonomy.</p>
      </div>
      <div class="chart-tabs">
        <button class="tab-btn active" id="btnTabDelta">Relative ΔF</button>
        <button class="tab-btn" id="btnTabCosine">Cosine Sim</button>
        <button class="tab-btn" id="btnTabTheta">Subspace Drift (θmax)</button>
      </div>
    </div>
    <div class="chart-body">
      <div class="chart-wrap">
        <svg id="deltaChart" viewBox="0 0 1000 260" preserveAspectRatio="none" role="img" aria-label="Forensic profile chart"></svg>
      </div>
    </div>
  </section>

  <!-- Forensic Layer Detail Table -->
  <section class="panel table-panel">
    <div class="panel-header">
      <div class="panel-title-group">
        <h2>Layer Forensics Telemetry</h2>
        <p id="tableMeta"></p>
      </div>
    </div>
    <div class="controls-bar">
      <div class="search-box">
        <input type="text" id="searchInput" placeholder="Filter by layer name, block, or type… [/]" aria-label="Search layers">
      </div>
      <select class="filter-select" id="statusFilter" aria-label="Filter by status">
        <option value="">All Status</option>
        <option value="MATCH">MATCH</option>
        <option value="PARTIAL_MATCH">PARTIAL_MATCH</option>
        <option value="SHAPE_MISMATCH">SHAPE_MISMATCH</option>
        <option value="ONLY_IN_A">ONLY_IN_A</option>
        <option value="ONLY_IN_B">ONLY_IN_B</option>
      </select>
      <select class="filter-select" id="badgeFilter" aria-label="Filter by classification">
        <option value="">All Classifications</option>
        <option value="RANK1_ABLATION">Rank-1 Ablation</option>
        <option value="LOW_RANK_ADAPTER">Low-Rank Adapter</option>
        <option value="SPARSE_UPDATE">Sparse Update</option>
        <option value="FULL_RANK_CPT">Full-Rank CPT</option>
        <option value="STOCHASTIC_NOISE">Stochastic SGD Noise</option>
        <option value="UNMODIFIED">Unmodified</option>
      </select>
      <select class="filter-select" id="lensFilter" aria-label="Filter by concept lens">
        <option value="">All Concept Lens</option>
        <option value="has_lens">Has Decoded Concepts</option>
        <option value="no_lens">No Concept Lens</option>
      </select>
      <select class="filter-select" id="sortSelect" aria-label="Sort">
        <option value="delta_desc">Sort: ΔF high → low</option>
        <option value="delta_asc">Sort: ΔF low → high</option>
        <option value="cosine_asc">Sort: Cosine low → high</option>
        <option value="theta_desc">Sort: θmax drift high → low</option>
        <option value="name">Sort: Layer Name</option>
      </select>
    </div>
    <div class="table-scroll">
      <table class="forensic-table" id="dataTable">
        <thead>
          <tr>
            <th>Layer Name</th>
            <th>Shape</th>
            <th>Rel. ΔF</th>
            <th>Cosine Sim</th>
            <th>Max |Δ|</th>
            <th>Signal Rank</th>
            <th>Effective Rank</th>
            <th>θmax°</th>
            <th>Fidelity</th>
            <th>Classification</th>
            <th>Concept Lens</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody id="tableBody"></tbody>
      </table>
    </div>
  </section>

  <!-- Slide-Over Layer Detail Drawer -->
  <div class="drawer-backdrop" id="drawerBackdrop"></div>
  <aside class="drawer" id="layerDrawer" aria-label="Layer Forensic Inspector">
    <div class="drawer-header">
      <div>
        <div style="font-size:11px;color:var(--cyan);text-transform:uppercase;font-weight:600;letter-spacing:0.08em;">Layer Inspector</div>
        <h3 id="drawerLayerName">—</h3>
      </div>
      <button class="close-btn" id="drawerCloseBtn" aria-label="Close drawer">✕</button>
    </div>
    <div class="drawer-content" id="drawerBody">
      <!-- Injected via JavaScript -->
    </div>
  </aside>

  <footer class="footer-bar">
    <p>Generated by <strong>wdiff</strong> — streaming neural weight diffing & subspace forensics. Completely self-contained, air-gapped report.</p>
  </footer>
</div>

<script>
const TABLE_DATA = {table_data_json};
const SUMMARY = {json.dumps(summary_data)};
const HAS_DEEP = {json.dumps(has_deep)};
const HAS_LENS = {json.dumps(has_lens)};
const BADGE_COLORS = {json.dumps(BADGE_COLORS)};
const BADGE_LABELS = {json.dumps(BADGE_LABELS)};

function escapeHtml(str) {{
  if (str === null || str === undefined) return '';
  const p = document.createElement('p');
  p.textContent = String(str);
  return p.innerHTML;
}}

function parseNum(v) {{
  if (v === null || v === undefined || v === '—') return NaN;
  const n = parseFloat(v);
  return Number.isFinite(n) ? n : NaN;
}}

let currentChartMode = 'delta'; // 'delta' | 'cosine' | 'theta'
let selectedLayerName = null;

const tableBody = document.getElementById('tableBody');
const searchInput = document.getElementById('searchInput');
const statusFilter = document.getElementById('statusFilter');
const badgeFilter = document.getElementById('badgeFilter');
const lensFilter = document.getElementById('lensFilter');
const sortSelect = document.getElementById('sortSelect');
const layerDrawer = document.getElementById('layerDrawer');
const drawerBackdrop = document.getElementById('drawerBackdrop');
const drawerCloseBtn = document.getElementById('drawerCloseBtn');

function badgeHtml(badge) {{
  if (!badge) return '—';
  const color = BADGE_COLORS[badge] || '#6b7280';
  const label = BADGE_LABELS[badge] || badge.replace(/_/g, ' ');
  return `<span class="badge" style="--bc:${{color}}">${{escapeHtml(label)}}</span>`;
}}

function renderTable(data) {{
  tableBody.innerHTML = '';
  data.forEach(row => {{
    const tr = document.createElement('tr');
    if (selectedLayerName && row.name === selectedLayerName) {{
      tr.classList.add('selected');
    }}
    tr.onclick = () => openDrawer(row.name);

    const dfVal = parseNum(row.rel_frobenius);
    let deltaHtml = '—';
    if (!isNaN(dfVal)) {{
      const pct = Math.min(100, Math.max(3, Math.sqrt(dfVal / 0.15) * 100));
      const color = (row.badge && BADGE_COLORS[row.badge]) || 'var(--cyan)';
      deltaHtml = `<div class="delta-meter"><div class="mini-bar"><div class="mini-bar-fill" style="width:${{pct}}%;background:${{color}}"></div></div><span>${{dfVal.toFixed(4)}}</span></div>`;
    }}

    let lensHtml = '—';
    if (row.lens) {{
      const pos = (row.lens.positive_tokens || []).slice(0, 2).map(tc => `<span class="lens-tag lens-pos">+${{escapeHtml(tc.token)}}</span>`).join('');
      const neg = (row.lens.negative_tokens || []).slice(0, 1).map(tc => `<span class="lens-tag lens-neg">-${{escapeHtml(tc.token)}}</span>`).join('');
      lensHtml = `<div class="lens-cell">${{pos}}${{neg}}</div>`;
    }}

    let sRank = '—', eRank = '—', theta = '—', fid = '—';
    if (row.signal_rank !== null && row.signal_rank !== undefined) sRank = row.signal_rank;
    if (row.effective_rank !== null && row.effective_rank !== undefined) eRank = parseNum(row.effective_rank).toFixed(1);
    if (row.max_principal_angle !== null && row.max_principal_angle !== undefined) {{
      const tVal = parseNum(row.max_principal_angle);
      const isDrift = tVal > 25.0;
      theta = `<span style="color:${{isDrift ? 'var(--rose)' : 'inherit'}};font-weight:${{isDrift ? '700' : 'normal'}}">${{tVal.toFixed(1)}}°</span>`;
    }}
    if (row.uhlmann_fidelity !== null && row.uhlmann_fidelity !== undefined) fid = parseNum(row.uhlmann_fidelity).toFixed(2);

    const stClass = `status-${{String(row.status || '').toLowerCase().replace(/[^a-z0-9_]/g, '_')}}`;

    tr.innerHTML = `
      <td class="td-name"><code>${{escapeHtml(row.name)}}</code></td>
      <td class="td-shape">${{escapeHtml(row.shape)}}</td>
      <td class="td-num">${{deltaHtml}}</td>
      <td class="td-num">${{escapeHtml(row.cosine_sim)}}</td>
      <td class="td-num">${{escapeHtml(row.max_delta)}}</td>
      <td class="td-num">${{sRank}}</td>
      <td class="td-num">${{eRank}}</td>
      <td class="td-num">${{theta}}</td>
      <td class="td-num">${{fid}}</td>
      <td>${{badgeHtml(row.badge)}}</td>
      <td>${{lensHtml}}</td>
      <td><span class="status-pill ${{stClass}}">${{escapeHtml(row.status)}}</span></td>
    `;
    tableBody.appendChild(tr);
  }});
  document.getElementById('tableMeta').textContent = `Displaying ${{data.length}} of ${{TABLE_DATA.length}} inspected layers`;
}}

function sortRows(data) {{
  const s = sortSelect.value;
  return [...data].sort((a, b) => {{
    if (s === 'delta_desc') return (parseNum(b.rel_frobenius) || -1) - (parseNum(a.rel_frobenius) || -1);
    if (s === 'delta_asc') return (parseNum(a.rel_frobenius) || 999) - (parseNum(b.rel_frobenius) || 999);
    if (s === 'cosine_asc') return (parseNum(a.cosine_sim) || 999) - (parseNum(b.cosine_sim) || 999);
    if (s === 'theta_desc') return (parseNum(b.max_principal_angle) || -1) - (parseNum(a.max_principal_angle) || -1);
    if (s === 'name') return String(a.name).localeCompare(String(b.name));
    return 0;
  }});
}}

function filterTable() {{
  const q = (searchInput.value || '').toLowerCase();
  const st = statusFilter.value;
  const bg = badgeFilter.value;
  const ln = lensFilter.value;
  const filtered = TABLE_DATA.filter(r => {{
    const matchesQ = !q || String(r.name).toLowerCase().includes(q);
    const matchesSt = !st || r.status === st;
    const matchesBg = !bg || r.badge === bg;
    const matchesLn = !ln || (ln === 'has_lens' && r.lens) || (ln === 'no_lens' && !r.lens);
    return matchesQ && matchesSt && matchesBg && matchesLn;
  }});
  renderTable(sortRows(filtered));
}}

function drawChart() {{
  const svg = document.getElementById('deltaChart');
  const rows = TABLE_DATA;
  if (!rows.length) {{
    svg.innerHTML = '<text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" fill="#64748b" font-size="14">No tensor data available</text>';
    return;
  }}

  let vals = [];
  if (currentChartMode === 'delta') {{
    vals = rows.map(r => parseNum(r.rel_frobenius)).filter(v => !isNaN(v));
  }} else if (currentChartMode === 'cosine') {{
    vals = rows.map(r => parseNum(r.cosine_sim)).filter(v => !isNaN(v));
  }} else if (currentChartMode === 'theta') {{
    vals = rows.map(r => parseNum(r.max_principal_angle)).filter(v => !isNaN(v));
  }}

  if (!vals.length) {{
    svg.innerHTML = `<text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" fill="#64748b" font-size="14">Run with --deep to unlock ${{currentChartMode}} landscape</text>`;
    return;
  }}

  const width = 1000, height = 260;
  const margin = {{top: 20, right: 24, bottom: 44, left: 60}};
  const cw = width - margin.left - margin.right;
  const ch = height - margin.top - margin.bottom;

  let minV = Math.min(...vals, 0);
  let maxV = Math.max(...vals, currentChartMode === 'cosine' ? 1.0 : 0.01);
  if (currentChartMode === 'theta') maxV = Math.max(...vals, 45.0);
  const span = (maxV - minV) || 1.0;

  const getX = (i) => margin.left + (rows.length > 1 ? (i / (rows.length - 1)) * cw : cw / 2);
  const getY = (v) => margin.top + (1 - (v - minV) / span) * ch;

  let grid = '';
  for (let i = 0; i <= 4; i++) {{
    const gv = minV + span * (i / 4);
    const gy = getY(gv);
    grid += `<line x1="${{margin.left}}" y1="${{gy}}" x2="${{width - margin.right}}" y2="${{gy}}" stroke="#1a2538" stroke-width="1"/>`;
    grid += `<text x="${{margin.left - 10}}" y="${{gy + 4}}" text-anchor="end" font-size="10" font-family="monospace" fill="#64748b">${{gv.toFixed(currentChartMode === 'theta' ? 0 : 3)}}</text>`;
  }}

  let bars = '';
  const barW = Math.max(1, cw / rows.length - 2);
  rows.forEach((r, i) => {{
    let v = NaN;
    if (currentChartMode === 'delta') v = parseNum(r.rel_frobenius);
    else if (currentChartMode === 'cosine') v = parseNum(r.cosine_sim);
    else if (currentChartMode === 'theta') v = parseNum(r.max_principal_angle);

    if (isNaN(v)) return;
    const bx = getX(i) - barW / 2;
    const by = getY(v);
    const bh = Math.max(2, (height - margin.bottom) - by);
    const color = (r.badge && BADGE_COLORS[r.badge]) || 'var(--cyan)';
    bars += `<rect x="${{bx}}" y="${{by}}" width="${{barW}}" height="${{bh}}" fill="${{color}}" rx="1" style="cursor:pointer;" onclick="openDrawer('${{escapeHtml(r.name)}}')">
      <title>${{escapeHtml(r.name)}} · ${{currentChartMode}}=${{v.toFixed(4)}}</title>
    </rect>`;
  }});

  // X Labels
  let labels = '';
  const step = Math.max(1, Math.floor(rows.length / 8));
  rows.forEach((r, i) => {{
    if (i % step === 0 || i === rows.length - 1) {{
      const lx = getX(i);
      labels += `<text x="${{lx}}" y="${{height - margin.bottom + 18}}" text-anchor="middle" font-size="9" font-family="monospace" fill="#64748b">${{escapeHtml(String(r.name).slice(0, 16))}}</text>`;
    }}
  }});

  svg.innerHTML = grid + bars + labels;
}}

function buildTopology() {{
  const container = document.getElementById('topologyFlow');
  if (!container) return;

  const names = TABLE_DATA.map(r => String(r.name).toLowerCase());
  const hasEmbed = names.some(n => n.includes('embed') || n.includes('wte') || n.includes('token'));
  const hasHead = names.some(n => n.includes('head') || n.includes('output'));

  const layerIndices = new Set();
  TABLE_DATA.forEach(r => {{
    const m = String(r.name).match(/(?:layers?|blk|h)\\.(\\d+)\\./i);
    if (m) layerIndices.add(parseInt(m[1], 10));
  }});

  let blocksHtml = '';
  if (hasEmbed) {{
    blocksHtml += `
      <div class="topo-node" onclick="filterByComponent('embed')">
        <div class="topo-node-title">Embedding</div>
        <div class="topo-node-sub">Token Vector Space</div>
      </div>
      <div class="topo-connector">──►</div>
    `;
  }}

  const sortedIndices = Array.from(layerIndices).sort((a, b) => a - b);
  const displayBlocks = sortedIndices.length > 0 ? sortedIndices : [0];

  displayBlocks.forEach(idx => {{
    const blockRows = TABLE_DATA.filter(r => String(r.name).includes(`.${{idx}}.`));
    const maxDelta = Math.max(...blockRows.map(r => parseNum(r.rel_frobenius) || 0), 0);
    const hasAblation = blockRows.some(r => r.badge === 'RANK1_ABLATION');
    const badgeColor = hasAblation ? 'var(--rose)' : (maxDelta > 0.05 ? 'var(--amber)' : (maxDelta > 0.001 ? 'var(--cyan)' : 'var(--slate)'));

    blocksHtml += `
      <div class="topo-block" onclick="filterByBlock(${{idx}})">
        <div class="topo-block-header">
          <span class="topo-block-name">Layer ${{idx}}</span>
          <span class="topo-block-badge" style="background:${{badgeColor}}"></span>
        </div>
        <div class="topo-subcells">
          <div class="topo-cell ${{blockRows.some(r => r.name.includes('attn_q')) ? 'lora' : ''}}"><span>Q/K/V</span></div>
          <div class="topo-cell ${{blockRows.some(r => r.name.includes('attn_out')) ? 'lora' : ''}}"><span>Out</span></div>
          <div class="topo-cell ${{blockRows.some(r => r.name.includes('ffn_gate') || r.name.includes('mlp')) ? 'hot' : ''}}"><span>Gate</span></div>
          <div class="topo-cell"><span>Down</span></div>
        </div>
      </div>
      <div class="topo-connector">─►</div>
    `;
  }});

  if (hasHead) {{
    blocksHtml += `
      <div class="topo-node" onclick="filterByComponent('head')">
        <div class="topo-node-title">Unembedding</div>
        <div class="topo-node-sub">Logit Projection</div>
      </div>
    `;
  }} else {{
    blocksHtml += `
      <div class="topo-node">
        <div class="topo-node-title">Final Norm</div>
        <div class="topo-node-sub">Output Layer</div>
      </div>
    `;
  }}

  container.innerHTML = blocksHtml;
}}

function filterByBlock(idx) {{
  searchInput.value = `.${{idx}}.`;
  filterTable();
}}

function filterByComponent(comp) {{
  if (comp === 'embed') searchInput.value = 'embed';
  else if (comp === 'head') searchInput.value = 'head';
  filterTable();
}}

function buildVerdict() {{
  const vHead = document.getElementById('verdictHeadline');
  const vDet = document.getElementById('verdictDetail');
  const max = Number(SUMMARY.max_frobenius || 0);
  const mean = Number(SUMMARY.mean_frobenius || 0);
  const minCos = Number(SUMMARY.min_cosine_value || 1);
  const ablations = TABLE_DATA.filter(r => r.badge === 'RANK1_ABLATION').length;
  const loras = TABLE_DATA.filter(r => r.badge === 'LOW_RANK_ADAPTER').length;
  const drifted = TABLE_DATA.filter(r => r.max_principal_angle !== null && Number(r.max_principal_angle) > 25.0).length;

  if (ablations > 0) {{
    vHead.textContent = 'Rank-1 Ablation Detected';
    vHead.style.color = 'var(--rose)';
    vDet.textContent = `${{ablations}} layer(s) exhibit targeted rank-1 suppression (e.g. abliteration). Residual subspace alignment broken.`;
  }} else if (drifted > 0) {{
    vHead.textContent = 'Representation Drift';
    vHead.style.color = 'var(--rose)';
    vDet.textContent = `${{drifted}} layer(s) exhibit principal angle drift θmax > 25°. Latent feature orientation diverges from base.`;
  }} else if (loras > 0) {{
    vHead.textContent = 'Low-Rank Adapter Applied';
    vHead.style.color = 'var(--cyan)';
    vDet.textContent = `${{loras}} layers match low-rank adapter signature (rank < 32, signal energy > 70%). Base structure preserved.`;
  }} else if (max > 0.1 || minCos < 0.8) {{
    vHead.textContent = 'Significant Parameter Divergence';
    vHead.style.color = 'var(--amber)';
    vDet.textContent = `Substantial weight differences detected (max ΔF: ${{max.toFixed(4)}}, min cosine: ${{minCos.toFixed(4)}}).`;
  }} else {{
    vHead.textContent = 'Subtle / Aligned Checkpoint';
    vHead.style.color = 'var(--emerald)';
    vDet.textContent = `Weights remain in close geometric alignment (mean ΔF: ${{mean.toFixed(4)}}, min cosine: ${{minCos.toFixed(4)}}).`;
  }}
}}

function buildLegend() {{
  const el = document.getElementById('legend');
  const counts = {{}};
  TABLE_DATA.forEach(r => {{ if (r.badge) counts[r.badge] = (counts[r.badge] || 0) + 1; }});
  const keys = Object.keys(counts).sort((a, b) => counts[b] - counts[a]);
  if (!keys.length) {{
    el.innerHTML = '<span class="leg-item" style="color:var(--fg-dim);">Run with --deep for spectral taxonomy badges</span>';
    return;
  }}
  el.innerHTML = keys.map(k => `
    <span class="leg-item">
      <span class="leg-dot" style="--lc:${{BADGE_COLORS[k] || '#64748b'}}"></span>
      <span>${{escapeHtml(BADGE_LABELS[k] || k)}}</span>
      <strong style="color:#fff;margin-left:2px">${{counts[k]}}</strong>
    </span>
  `).join('');
}}

function openDrawer(layerName) {{
  const row = TABLE_DATA.find(r => r.name === layerName);
  if (!row) return;

  selectedLayerName = layerName;
  document.getElementById('drawerLayerName').textContent = row.name;

  let bodyHtml = `
    <div class="drawer-card">
      <div class="drawer-card-title">Scalar Forensics</div>
      <div class="metric-pair-grid">
        <div class="metric-pair"><span class="metric-pair-k">Shape</span><span class="metric-pair-v">${{escapeHtml(row.shape)}}</span></div>
        <div class="metric-pair"><span class="metric-pair-k">Status</span><span class="metric-pair-v">${{escapeHtml(row.status)}}</span></div>
        <div class="metric-pair"><span class="metric-pair-k">Rel. ΔF</span><span class="metric-pair-v">${{escapeHtml(row.rel_frobenius)}}</span></div>
        <div class="metric-pair"><span class="metric-pair-k">Cosine Sim</span><span class="metric-pair-v">${{escapeHtml(row.cosine_sim)}}</span></div>
        <div class="metric-pair"><span class="metric-pair-k">Max Element |Δ|</span><span class="metric-pair-v">${{escapeHtml(row.max_delta)}}</span></div>
        <div class="metric-pair"><span class="metric-pair-k">Classification</span><span class="metric-pair-v">${{badgeHtml(row.badge)}}</span></div>
      </div>
    </div>
  `;

  // SVD Spectrum Card
  if (row.spectral && row.spectral.singular_values && row.spectral.singular_values.length) {{
    const sVals = row.spectral.singular_values;
    const mpEdge = row.spectral.marcenko_pastur_edge;
    const maxS = Math.max(...sVals, mpEdge * 1.2, 1e-6);

    let barsSvg = '';
    const w = 480, h = 140, mTop = 15, mBot = 25, mL = 40, mR = 15;
    const plotW = w - mL - mR;
    const plotH = h - mTop - mBot;
    const barWidth = Math.max(2, plotW / sVals.length - 4);

    sVals.forEach((s, idx) => {{
      const bx = mL + idx * (plotW / sVals.length) + 2;
      const by = mTop + (1 - s / maxS) * plotH;
      const bHeight = Math.max(1, (h - mBot) - by);
      const isSignal = s > mpEdge;
      const color = isSignal ? 'var(--cyan)' : 'var(--slate)';
      barsSvg += `<rect x="${{bx}}" y="${{by}}" width="${{barWidth}}" height="${{bHeight}}" fill="${{color}}" rx="1">
        <title>σ_${{idx + 1}} = ${{s.toFixed(4)}} (${{isSignal ? 'Signal' : 'Noise'}})</title>
      </rect>`;
    }});

    // Draw MP Edge line
    const mpY = mTop + (1 - mpEdge / maxS) * plotH;
    const mpLine = `<line x1="${{mL}}" y1="${{mpY}}" x2="${{w - mR}}" y2="${{mpY}}" stroke="var(--amber)" stroke-dasharray="4,3" stroke-width="1.5"/>
      <text x="${{w - mR}}" y="${{mpY - 4}}" text-anchor="end" font-size="9" font-family="monospace" fill="var(--amber)">MP Noise Floor (${{mpEdge.toFixed(3)}})</text>`;

    bodyHtml += `
      <div class="drawer-card">
        <div class="drawer-card-title">
          <span>SVD Spectrum vs Noise Floor</span>
          <span style="font-size:10px;color:var(--cyan);font-family:monospace;">${{row.spectral.signal_rank}} Signal Singular Values</span>
        </div>
        <div class="spec-plot">
          <svg viewBox="0 0 ${{w}} ${{h}}" preserveAspectRatio="none">
            ${{mpLine}}
            ${{barsSvg}}
            <line x1="${{mL}}" y1="${{h - mBot}}" x2="${{w - mR}}" y2="${{h - mBot}}" stroke="#1e293b"/>
          </svg>
        </div>
        <div class="metric-pair-grid" style="margin-top:12px;">
          <div class="metric-pair"><span class="metric-pair-k">Effective Rank</span><span class="metric-pair-v">${{row.spectral.effective_rank.toFixed(2)}}</span></div>
          <div class="metric-pair"><span class="metric-pair-k">Stable Rank</span><span class="metric-pair-v">${{row.spectral.stable_rank.toFixed(2)}}</span></div>
          <div class="metric-pair"><span class="metric-pair-k">Primary Energy (E1)</span><span class="metric-pair-v">${{(row.spectral.primary_direction_energy * 100).toFixed(1)}}%</span></div>
          <div class="metric-pair"><span class="metric-pair-k">Signal Energy Ratio</span><span class="metric-pair-v">${{(row.spectral.signal_energy_ratio * 100).toFixed(1)}}%</span></div>
        </div>
      </div>
    `;
  }}

  // Subspace Geometry Card
  if (row.subspace) {{
    const sub = row.subspace;
    const isDrift = sub.max_principal_angle > 25.0;
    bodyHtml += `
      <div class="drawer-card">
        <div class="drawer-card-title">
          <span>Grassmannian Subspace Geometry</span>
          ${{isDrift ? '<span style="font-size:10px;color:var(--rose);font-weight:700;">⚠ Representation Drift</span>' : ''}}
        </div>
        <div class="metric-pair-grid">
          <div class="metric-pair"><span class="metric-pair-k">Max Angle (θmax)</span><span class="metric-pair-v" style="color:${{isDrift ? 'var(--rose)' : '#fff'}}">${{sub.max_principal_angle.toFixed(1)}}°</span></div>
          <div class="metric-pair"><span class="metric-pair-k">Uhlmann Fidelity</span><span class="metric-pair-v">${{sub.uhlmann_fidelity.toFixed(4)}}</span></div>
          <div class="metric-pair"><span class="metric-pair-k">Mean Subspace Cosine</span><span class="metric-pair-v">${{sub.mean_cosine.toFixed(4)}}</span></div>
          <div class="metric-pair"><span class="metric-pair-k">Geodesic Distance</span><span class="metric-pair-v">${{sub.geodesic_distance.toFixed(4)}}</span></div>
        </div>
      </div>
    `;
  }}

  // Concept Lens Card
  if (row.lens) {{
    const pos = (row.lens.positive_tokens || []).map(tc => `
      <div style="display:flex;justify-content:space-between;padding:3px 0;border-bottom:1px solid rgba(255,255,255,0.04);font-family:monospace;font-size:12px;">
        <span style="color:var(--emerald);">+ ${{escapeHtml(tc.token)}}</span>
        <span style="color:var(--fg-dim);">${{tc.logit.toFixed(2)}}</span>
      </div>
    `).join('');

    const neg = (row.lens.negative_tokens || []).map(tc => `
      <div style="display:flex;justify-content:space-between;padding:3px 0;border-bottom:1px solid rgba(255,255,255,0.04);font-family:monospace;font-size:12px;">
        <span style="color:var(--rose);">- ${{escapeHtml(tc.token)}}</span>
        <span style="color:var(--fg-dim);">${{tc.logit.toFixed(2)}}</span>
      </div>
    `).join('');

    bodyHtml += `
      <div class="drawer-card">
        <div class="drawer-card-title">Logit Lens Concept Projections</div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">
          <div>
            <div style="font-size:10px;text-transform:uppercase;color:var(--emerald);font-weight:700;margin-bottom:6px;">Promoted Concepts</div>
            ${{pos || '<span style="color:var(--fg-dim);font-size:11px;">None</span>'}}
          </div>
          <div>
            <div style="font-size:10px;text-transform:uppercase;color:var(--rose);font-weight:700;margin-bottom:6px;">Suppressed Concepts</div>
            ${{neg || '<span style="color:var(--fg-dim);font-size:11px;">None</span>'}}
          </div>
        </div>
      </div>
    `;
  }}

  // Partial Match / Vocab Expansion Card
  if (row.partial_info) {{
    const pi = row.partial_info;
    bodyHtml += `
      <div class="drawer-card">
        <div class="drawer-card-title">Vocabulary Expansion Telemetry</div>
        <div class="metric-pair-grid">
          <div class="metric-pair"><span class="metric-pair-k">Added Tokens</span><span class="metric-pair-v">+${{pi.extra_count.toLocaleString()}}</span></div>
          <div class="metric-pair"><span class="metric-pair-k">Source Model</span><span class="metric-pair-v">${{escapeHtml(pi.extra_source)}}</span></div>
          <div class="metric-pair"><span class="metric-pair-k">New Embedding Norm</span><span class="metric-pair-v">${{pi.extra_norm.toExponential(2)}}</span></div>
          <div class="metric-pair"><span class="metric-pair-k">Mean / Std</span><span class="metric-pair-v">${{pi.extra_mean.toFixed(4)}} / ${{pi.extra_std.toFixed(4)}}</span></div>
        </div>
      </div>
    `;
  }}

  // Taxonomy Explanation Card
  if (row.taxonomy && row.taxonomy.explanation) {{
    bodyHtml += `
      <div class="drawer-card">
        <div class="drawer-card-title">Taxonomy Evidence & Rationale</div>
        <p style="font-size:13px;color:var(--fg-muted);line-height:1.5;">${{escapeHtml(row.taxonomy.explanation)}}</p>
      </div>
    `;
  }}

  document.getElementById('drawerBody').innerHTML = bodyHtml;
  layerDrawer.classList.add('open');
  drawerBackdrop.classList.add('open');
}}

function closeDrawer() {{
  layerDrawer.classList.remove('open');
  drawerBackdrop.classList.remove('open');
  selectedLayerName = null;
}}

drawerCloseBtn.onclick = closeDrawer;
drawerBackdrop.onclick = closeDrawer;
document.addEventListener('keydown', (e) => {{
  if (e.key === 'Escape') closeDrawer();
  if (e.key === '/' && document.activeElement !== searchInput) {{
    e.preventDefault();
    searchInput.focus();
  }}
}});

// Chart Tab Handlers
document.getElementById('btnTabDelta').onclick = () => {{
  currentChartMode = 'delta';
  document.querySelectorAll('.chart-tabs .tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('btnTabDelta').classList.add('active');
  document.getElementById('chartTitle').textContent = 'Layer Delta Profile';
  document.getElementById('chartDesc').textContent = 'Relative Frobenius delta across network layers, color-coded by spectral taxonomy.';
  drawChart();
}};
document.getElementById('btnTabCosine').onclick = () => {{
  currentChartMode = 'cosine';
  document.querySelectorAll('.chart-tabs .tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('btnTabCosine').classList.add('active');
  document.getElementById('chartTitle').textContent = 'Cosine Similarity Profile';
  document.getElementById('chartDesc').textContent = 'Directional alignment between checkpoints (1.0 = identical orientation).';
  drawChart();
}};
document.getElementById('btnTabTheta').onclick = () => {{
  currentChartMode = 'theta';
  document.querySelectorAll('.chart-tabs .tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('btnTabTheta').classList.add('active');
  document.getElementById('chartTitle').textContent = 'Subspace Drift Profile (θmax)';
  document.getElementById('chartDesc').textContent = 'Maximum principal angle (degrees) between top-k subspaces (>25° indicates representation drift).';
  drawChart();
}};

[searchInput, statusFilter, badgeFilter, lensFilter, sortSelect].forEach(el => {{
  if (!el) return;
  el.addEventListener(el.tagName === 'INPUT' ? 'input' : 'change', filterTable);
}});

buildVerdict();
buildLegend();
buildTopology();
filterTable();
drawChart();
</script>
</body>
</html>"""
    return html


# -----------------------------------------------------------------------------
# Inspection report (kept self-contained, reusing same aesthetic)
# -----------------------------------------------------------------------------


def _generate_inspection_html_content(
    result: ModelInspectionResult, title: str | None = None
) -> str:
    table_data = _prepare_inspection_table_data(result)
    summary_data = _prepare_inspection_summary_data(result)
    has_deep = any(t.spectral is not None for t in result.layers)
    table_data_json = json.dumps(table_data).replace("</", "<\\/").replace("<script", "<\\script")

    if title is None:
        title = f"wdiff inspect · {Path(result.model_path).name}"
    safe_title = html_lib.escape(title)

    mean_f = summary_data.get("mean_frobenius", 0.0)
    min_c = summary_data.get("min_cosine_value", 1.0)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{safe_title}</title>
<style>
{_get_embedded_css()}
</style>
</head>
<body>
<div class="shell">
  <nav class="top-nav">
    <div class="brand-group">
      <div class="brand-logo">W</div>
      <div>
        <div class="brand-title">wdiff inspect</div>
        <div style="font-size:11px;color:var(--fg-dim);">Model Health & Quantization Readiness Audit</div>
      </div>
      <span class="brand-tag">v0.3.0</span>
    </div>
    <div class="header-meta">
      <span>Duration: <strong style="color:#fff">{result.summary.duration_seconds:.2f}s</strong></span>
      <span>Format: <strong style="color:#fff">{result.format}</strong></span>
    </div>
  </nav>

  <header class="masthead">
    <div class="masthead-info">
      <div>
        <h1>{safe_title}</h1>
        <p class="deck">wdiff Report · Model inspection report</p>
      </div>
      <div class="model-spec-grid">
        <div class="model-box">
          <span class="model-box-label">Inspected Checkpoint</span>
          <span class="model-box-path"><code>{html_lib.escape(str(result.model_path))}</code><span class="format-pill">{html_lib.escape(str(result.format))}</span></span>
        </div>
        <div class="model-box">
          <span class="model-box-label">Inferred Architecture</span>
          <span class="model-box-path"><code>{html_lib.escape(str(result.summary.architecture))}</code></span>
        </div>
      </div>
    </div>

    <div class="verdict-card">
      <div>
        <div class="verdict-eyebrow">Quantization Readiness</div>
        <div class="verdict-headline">{html_lib.escape(str(result.summary.quantization_readiness))}</div>
      </div>
      <div class="verdict-detail">{result.summary.outlier_risk_layer_count} outlier-risk layers, {result.summary.dead_neurons_total} dead neurons detected.</div>
    </div>
  </header>

  <section class="kpi-row">
    <div class="kpi-card hero"><div class="kpi-val">{result.summary.total_params:,}</div><div class="kpi-lbl">Total Parameters</div></div>
    <div class="kpi-card"><div class="kpi-val">{result.summary.trainable_params_mb:.2f} MB</div><div class="kpi-lbl">Model Size</div></div>
    <div class="kpi-card"><div class="kpi-val">{result.summary.tensor_count}</div><div class="kpi-lbl">Tensors</div></div>
    <div class="kpi-card"><div class="kpi-val">{result.summary.duration_seconds:.2f}s</div><div class="kpi-lbl">Duration</div></div>
    <div class="kpi-card"><div class="kpi-val">{result.summary.dead_neurons_total}</div><div class="kpi-lbl">Dead Neurons</div></div>
    <div class="kpi-card"><div class="kpi-val">{result.summary.outlier_risk_layer_count}</div><div class="kpi-lbl">Outlier Risk Layers</div></div>
    <div class="kpi-card"><div class="kpi-val">{mean_f:.4f}</div><div class="kpi-lbl">Mean Rel. ΔF</div></div>
    <div class="kpi-card"><div class="kpi-val">{min_c:.4f}</div><div class="kpi-lbl">Min Cosine</div></div>
  </section>

  <section class="panel chart-panel">
    <div class="panel-header">
      <div class="panel-title-group">
        <h2>Layer Delta Profile</h2>
        <p>L2 Norm magnitude across layers, highlighting outlier spikes and dead neurons.</p>
      </div>
      <div class="legend-bar" id="legend"></div>
    </div>
    <div class="chart-body">
      <div class="chart-wrap">
        <svg id="deltaChart" viewBox="0 0 1000 260" preserveAspectRatio="none" role="img" aria-label="Inspection norm chart"></svg>
      </div>
    </div>
  </section>

  <section class="panel table-panel">
    <div class="panel-header">
      <div class="panel-title-group">
        <h2>Layer Details</h2>
        <p id="tableMeta"></p>
      </div>
    </div>
    <div class="controls-bar">
      <div class="search-box">
        <input type="text" id="searchInput" placeholder="Search layers… [/]">
      </div>
      <select class="filter-select" id="statusFilter">
        <option value="">All Status</option>
        <option value="HEALTHY">HEALTHY</option>
        <option value="DEAD_NEURONS">DEAD_NEURONS</option>
        <option value="OUTLIER_SPIKE">OUTLIER_SPIKE</option>
      </select>
      <select class="filter-select" id="badgeFilter">
        <option value="">All Classifications</option>
        <option value="RANK1_ABLATION">Rank-1 Ablation</option>
        <option value="LOW_RANK_ADAPTER">Low-Rank Adapter</option>
        <option value="SPARSE_UPDATE">Sparse Update</option>
        <option value="FULL_RANK_CPT">Full-Rank CPT</option>
        <option value="STOCHASTIC_NOISE">Stochastic SGD Noise</option>
        <option value="UNMODIFIED">Unmodified</option>
      </select>
    </div>
    <div class="table-scroll">
      <table class="forensic-table" id="dataTable">
        <thead>
          <tr>
            <th>Layer Name</th>
            <th>Shape</th>
            <th>Dtype</th>
            <th>Params</th>
            <th>L2 Norm</th>
            <th>Outlier</th>
            <th>Dead</th>
            <th>Status</th>
            <th>Rank</th>
          </tr>
        </thead>
        <tbody id="tableBody"></tbody>
      </table>
    </div>
  </section>

  <footer class="footer-bar">
    <p>Generated by <strong>wdiff</strong> — git diff, but for neural network weights. Self-contained report, no external assets.</p>
  </footer>
</div>

<script>
const TABLE_DATA = {table_data_json};
const HAS_DEEP = {json.dumps(has_deep)};
const BADGE_COLORS = {json.dumps(BADGE_COLORS)};
const BADGE_LABELS = {json.dumps(BADGE_LABELS)};

function escapeHtml(str) {{
  if (str === null || str === undefined) return '';
  const p = document.createElement('p');
  p.textContent = String(str);
  return p.innerHTML;
}}

function parseNum(v) {{
  if (v === null || v === undefined || v === '—') return NaN;
  const n = Number(v);
  return Number.isFinite(n) ? n : NaN;
}}

const tableBody = document.getElementById('tableBody');
const searchInput = document.getElementById('searchInput');
const statusFilter = document.getElementById('statusFilter');
const badgeFilter = document.getElementById('badgeFilter');

function badgeHtml(badge) {{
  if (!badge) return '—';
  const color = BADGE_COLORS[badge] || '#6b7280';
  const label = BADGE_LABELS[badge] || badge.replace(/_/g, ' ');
  return `<span class="badge" style="--bc:${{color}}">${{escapeHtml(label)}}</span>`;
}}

function renderTable(data) {{
  tableBody.innerHTML = '';
  data.forEach(row => {{
    const tr = document.createElement('tr');
    const health = row.dead_neuron_count > 0 ? 'DEAD_NEURONS' : (row.is_outlier_risk ? 'OUTLIER_SPIKE' : 'HEALTHY');
    const hclass = health === 'HEALTHY' ? 'status-match' : (health === 'DEAD_NEURONS' ? 'status-only_in_a' : 'status-shape_mismatch');
    let rankStr = '—';
    if (row.signal_rank !== null && row.signal_rank !== undefined) {{
      rankStr = `s=${{row.signal_rank}} e=${{Number(row.effective_rank).toFixed(1)}}`;
      if (row.badge) rankStr += ' ' + badgeHtml(row.badge);
    }}
    tr.innerHTML = `
      <td class="td-name"><code>${{escapeHtml(row.name)}}</code></td>
      <td class="td-shape">${{escapeHtml(row.shape)}}</td>
      <td style="font-family:var(--mono);font-size:11px;">${{escapeHtml(row.dtype)}}</td>
      <td class="td-num">${{Number(row.param_count || 0).toLocaleString()}}</td>
      <td class="td-num" style="color:var(--violet)">${{Number(row.norm || 0).toExponential(2)}}</td>
      <td class="td-num" style="color:${{row.is_outlier_risk ? 'var(--rose)' : 'inherit'}}">${{Number(row.outlier_score || 0).toFixed(2)}}</td>
      <td class="td-num" style="color:${{row.dead_neuron_count > 0 ? 'var(--amber)' : 'inherit'}}">${{row.dead_neuron_count || 0}}</td>
      <td><span class="status-pill ${{hclass}}">${{escapeHtml(row.status || health)}}</span></td>
      <td class="td-num">${{rankStr}}</td>`;
    tableBody.appendChild(tr);
  }});
  document.getElementById('tableMeta').textContent = `Showing ${{data.length}} of ${{TABLE_DATA.length}} layers`;
}}

function filterTable() {{
  const search = (searchInput.value || '').toLowerCase();
  const status = statusFilter.value;
  const badge = badgeFilter.value;
  const filtered = TABLE_DATA.filter(row => {{
    const health = row.dead_neuron_count > 0 ? 'DEAD_NEURONS' : (row.is_outlier_risk ? 'OUTLIER_SPIKE' : 'HEALTHY');
    const matchesSearch = String(row.name || '').toLowerCase().includes(search);
    const matchesStatus = !status || row.status === status || health === status;
    const matchesBadge = !badge || row.badge === badge;
    return matchesSearch && matchesStatus && matchesBadge;
  }});
  renderTable(filtered);
}}

function drawChart() {{
  const svg = document.getElementById('deltaChart');
  const vals = TABLE_DATA.map(r => Number(r.norm || 0)).filter(v => Number.isFinite(v));
  if (!vals.length) {{
    svg.innerHTML = '<text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" fill="#64748b" font-size="14">Run with --deep to see delta profile chart</text>';
    return;
  }}
  const rows = TABLE_DATA;
  const width = 1000, height = 260, margin = {{top: 18, right: 16, bottom: 46, left: 56}};
  const cw = width - margin.left - margin.right, ch = height - margin.top - margin.bottom;
  const maxV = Math.max(...vals, 1e-9), minV = 0, span = maxV - minV || 1;
  const x = (i) => margin.left + (rows.length > 1 ? (i / (rows.length - 1)) * cw : cw / 2);
  const y = (v) => margin.top + (1 - (v - minV) / span) * ch;

  let bars = '';
  const barW = Math.max(1, cw / rows.length - 1);
  rows.forEach((row, i) => {{
    const v = Number(row.norm || 0);
    const bx = x(i) - barW / 2;
    const by = y(v);
    const bh = Math.max(1, (height - margin.bottom) - by);
    const color = row.is_outlier_risk ? 'var(--rose)' : (row.dead_neuron_count > 0 ? 'var(--amber)' : 'var(--cyan)');
    bars += `<rect x="${{bx}}" y="${{by}}" width="${{barW}}" height="${{bh}}" fill="${{color}}" rx="1"><title>${{escapeHtml(row.name)}} · norm=${{v.toExponential(2)}}</title></rect>`;
  }});

  let grid = '';
  for (let i = 0; i <= 4; i++) {{
    const gv = minV + span * (i / 4), gy = y(gv);
    grid += `<line x1="${{margin.left}}" y1="${{gy}}" x2="${{width - margin.right}}" y2="${{gy}}" stroke="#1a2538"/>`;
    grid += `<text x="${{margin.left - 8}}" y="${{gy + 4}}" text-anchor="end" font-size="10" font-family="monospace" fill="#64748b">${{gv.toExponential(1)}}</text>`;
  }}
  const axes = `<line x1="${{margin.left}}" y1="${{margin.top}}" x2="${{margin.left}}" y2="${{height - margin.bottom}}" stroke="#334155"/><line x1="${{margin.left}}" y1="${{height - margin.bottom}}" x2="${{width - margin.right}}" y2="${{height - margin.bottom}}" stroke="#334155"/>`;
  svg.innerHTML = grid + axes + bars;
  if (!HAS_DEEP) {{
    svg.innerHTML += `<text x="${{width - margin.right}}" y="${{margin.top + 10}}" text-anchor="end" fill="#9ca3af" font-size="11">Run with --deep to see delta profile chart</text>`;
  }}
}}

function buildLegend() {{
  const el = document.getElementById('legend');
  el.innerHTML = '<span class="leg-item"><span class="leg-dot" style="--lc:var(--cyan)"></span>Healthy</span><span class="leg-item"><span class="leg-dot" style="--lc:var(--amber)"></span>Dead neurons</span><span class="leg-item"><span class="leg-dot" style="--lc:var(--rose)"></span>Outlier risk</span>';
}}

[searchInput, statusFilter, badgeFilter].forEach(el => {{
  if (!el) return;
  el.addEventListener(el.tagName === 'INPUT' ? 'input' : 'change', filterTable);
}});

buildLegend();
filterTable();
drawChart();
</script>
</body>
</html>"""
    return html
