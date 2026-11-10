from __future__ import annotations

import html as html_lib
import json
from pathlib import Path
from typing import Any

from wdiff.engine.diff import DiffResult
from wdiff.engine.inspect import ModelInspectionResult

DiffOrInspection = DiffResult | ModelInspectionResult

# Consistent taxonomy palette (matches terminal)
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
    "STOCHASTIC_NOISE": "Stochastic Noise",
    "UNMODIFIED": "Unmodified",
}


def generate_html_report(
    result: DiffOrInspection,
    output_path: str | Path,
    title: str | None = None,
) -> Path:
    """
    Generate a standalone, self-contained interactive HTML report.

    Parameters:
        result: DiffResult from diff_models() or ModelInspectionResult from inspect_model()
        output_path: Path to write the HTML file
        title: Optional custom title for the report

    Returns:
        Path to the generated HTML file
    """
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


def _generate_diff_html_content(result: DiffResult, title: str | None = None) -> str:
    table_data = _prepare_table_data(result)
    summary_data = _prepare_summary_data(result)
    has_deep = any(t.spectral is not None or t.taxonomy is not None for t in result.tensor_results)
    has_lens = any(t.lens is not None for t in result.tensor_results)

    if title is None:
        title = f"wdiff · {Path(result.model_a_path).name} ↔ {Path(result.model_b_path).name}"
    safe_title = html_lib.escape(title)

    # Precomputed, honest values
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
  <header class="masthead">
    <div class="brand">
      <div class="eyebrow">weight forensics</div>
      <h1>{safe_title}</h1>
      <p class="deck">wdiff Report · Differential analysis report</p>
    </div>
    <div class="verdict" id="verdictBox">
      <div class="verdict-label">read</div>
      <div class="verdict-value" id="verdictValue">—</div>
      <div class="verdict-sub" id="verdictSub"></div>
    </div>
  </header>

  <section class="panel meta-panel">
    <div class="meta-grid">
      <div class="meta-item"><span class="meta-k">Model A</span><span class="meta-v"><code>{html_lib.escape(str(result.model_a_path))}</code> <span class="pill">{html_lib.escape(str(result.format_a))}</span></span></div>
      <div class="meta-item"><span class="meta-k">Model B</span><span class="meta-v"><code>{html_lib.escape(str(result.model_b_path))}</code> <span class="pill">{html_lib.escape(str(result.format_b))}</span></span></div>
      <div class="meta-item"><span class="meta-k">Duration</span><span class="meta-v"><strong>{summary_data["duration_seconds"]:.2f}s</strong></span></div>
      <div class="meta-item"><span class="meta-k">Peak RSS</span><span class="meta-v"><strong>{html_lib.escape(peak_label)}</strong></span></div>
    </div>
  </section>

  <section class="kpi-grid" aria-label="Summary">
    <div class="kpi"><div class="kpi-n">{matched}</div><div class="kpi-l">Matched Layers</div></div>
    <div class="kpi"><div class="kpi-n">{summary_data.get("partial_match_count", 0)}</div><div class="kpi-l">Partial Matches</div></div>
    <div class="kpi"><div class="kpi-n">{summary_data.get("lens_analyzed_count", 0)}</div><div class="kpi-l">Decoded Concepts</div></div>
    <div class="kpi"><div class="kpi-n">{summary_data["mismatched_shape_count"]}</div><div class="kpi-l">Shape Mismatches</div></div>
    <div class="kpi"><div class="kpi-n">{summary_data["only_in_a_count"]}</div><div class="kpi-l">Only in A</div></div>
    <div class="kpi"><div class="kpi-n">{summary_data["only_in_b_count"]}</div><div class="kpi-l">Only in B</div></div>
    <div class="kpi accent"><div class="kpi-n">{summary_data["mean_frobenius"]:.4f}</div><div class="kpi-l">Mean Rel. ΔF</div></div>
    <div class="kpi accent"><div class="kpi-n">{max_df:.4f}</div><div class="kpi-l">Max Rel. ΔF</div><div class="kpi-s" title="{max_df_tensor}">{max_df_tensor}</div></div>
    <div class="kpi"><div class="kpi-n">{summary_data["min_cosine_value"]:.4f}</div><div class="kpi-l">Min Cosine</div></div>
    <div class="kpi"><div class="kpi-n">{summary_data["duration_seconds"]:.2f}s</div><div class="kpi-l">Duration</div></div>
  </section>

  <section class="panel chart-panel">
    <div class="panel-head">
      <div>
        <h2>Layer Delta Profile</h2>
        <p class="panel-sub">Relative Frobenius change by layer, colored by spectral taxonomy (when available).</p>
      </div>
      <div class="legend" id="legend"></div>
    </div>
    <div class="chart-wrap">
      <svg id="deltaChart" viewBox="0 0 1000 300" preserveAspectRatio="none" role="img" aria-label="Layer delta profile"></svg>
    </div>
  </section>

  <section class="panel table-panel">
    <div class="panel-head">
      <div>
        <h2>Layer Details</h2>
        <p class="panel-sub" id="tableMeta"></p>
      </div>
    </div>
    <div class="controls">
      <input type="text" id="searchInput" placeholder="Search layers…" aria-label="Search layers">
      <select id="statusFilter" aria-label="Filter by status">
        <option value="">All Status</option>
        <option value="MATCH">MATCH</option>
        <option value="PARTIAL_MATCH">PARTIAL_MATCH</option>
        <option value="SHAPE_MISMATCH">SHAPE_MISMATCH</option>
        <option value="ONLY_IN_A">ONLY_IN_A</option>
        <option value="ONLY_IN_B">ONLY_IN_B</option>
      </select>
      <select id="badgeFilter" aria-label="Filter by classification">
        <option value="">All Classifications</option>
        <option value="RANK1_ABLATION">Rank-1 Ablation</option>
        <option value="LOW_RANK_ADAPTER">Low-Rank Adapter</option>
        <option value="SPARSE_UPDATE">Sparse Update</option>
        <option value="FULL_RANK_CPT">Full-Rank CPT</option>
        <option value="STOCHASTIC_NOISE">Stochastic Noise</option>
        <option value="UNMODIFIED">Unmodified</option>
      </select>
      <select id="lensFilter" aria-label="Filter by lens">
        <option value="">All Lens</option>
        <option value="has_lens">Has Concept Lens</option>
        <option value="no_lens">No Concept Lens</option>
      </select>
      <select id="sortSelect" aria-label="Sort">
        <option value="delta_desc">Sort: ΔF high→low</option>
        <option value="delta_asc">Sort: ΔF low→high</option>
        <option value="cosine_asc">Sort: Cosine low→high</option>
        <option value="name">Sort: Name</option>
      </select>
    </div>
    <div class="table-wrap">
      <table id="dataTable">
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

  <footer class="foot">
    <p>Generated by <strong>wdiff</strong> — git diff, but for neural network weights. Self-contained report, no external assets.</p>
  </footer>
</div>

<script>
const TABLE_DATA = {json.dumps(table_data)};
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

const tableBody = document.getElementById('tableBody');
const searchInput = document.getElementById('searchInput');
const statusFilter = document.getElementById('statusFilter');
const badgeFilter = document.getElementById('badgeFilter');
const lensFilter = document.getElementById('lensFilter');
const sortSelect = document.getElementById('sortSelect');

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
    let lensHtml = '—';
    if (row.lens) {{
      const pos = (row.lens.positive_tokens || []).map(tc => `${{escapeHtml(tc.token)}} <span class="logit">(${{Number(tc.logit).toFixed(2)}})</span>`).join(', ');
      const neg = (row.lens.negative_tokens || []).map(tc => `${{escapeHtml(tc.token)}} <span class="logit">(${{Number(tc.logit).toFixed(2)}})</span>`).join(', ');
      lensHtml = `<div class="lens"><div class="lens-pos"><strong>+</strong> ${{pos || '—'}}</div><div class="lens-neg"><strong>−</strong> ${{neg || '—'}}</div></div>`;
    }}
    const sr = row.signal_rank !== null && row.signal_rank !== undefined ? row.signal_rank : '—';
    const er = row.effective_rank !== null && row.effective_rank !== undefined ? Number(row.effective_rank).toFixed(1) : '—';
    const theta = row.max_principal_angle !== null && row.max_principal_angle !== undefined ? Number(row.max_principal_angle).toFixed(1) : '—';
    const fid = row.uhlmann_fidelity !== null && row.uhlmann_fidelity !== undefined ? Number(row.uhlmann_fidelity).toFixed(3) : '—';
    const drift = row.max_principal_angle !== null && row.max_principal_angle !== undefined && Number(row.max_principal_angle) > 25;
    tr.innerHTML = `
      <td class="c-name"><code>${{escapeHtml(row.name)}}</code></td>
      <td class="c-shape">${{escapeHtml(row.shape)}}</td>
      <td class="num">${{escapeHtml(row.rel_frobenius)}}</td>
      <td class="num">${{escapeHtml(row.cosine_sim)}}</td>
      <td class="num">${{escapeHtml(row.max_delta)}}</td>
      <td class="num">${{sr}}</td>
      <td class="num">${{er}}</td>
      <td class="num ${{drift ? 'drift' : ''}}">${{theta}}</td>
      <td class="num ${{drift ? 'drift' : ''}}">${{fid}}</td>
      <td>${{badgeHtml(row.badge)}}</td>
      <td>${{lensHtml}}</td>
      <td><span class="status status-${{escapeHtml(String(row.status || '').toLowerCase())}}">${{escapeHtml(row.status)}}</span></td>`;
    tableBody.appendChild(tr);
  }});
  document.getElementById('tableMeta').textContent = `Showing ${{data.length}} of ${{TABLE_DATA.length}} layers` + (HAS_DEEP ? ' · deep metrics present' : '');
}}

function sortRows(rows) {{
  const mode = sortSelect ? sortSelect.value : 'delta_desc';
  const copy = rows.slice();
  if (mode === 'name') return copy.sort((a,b) => String(a.name).localeCompare(String(b.name)));
  if (mode === 'cosine_asc') return copy.sort((a,b) => (parseNum(a.cosine_sim)||2) - (parseNum(b.cosine_sim)||2));
  if (mode === 'delta_asc') return copy.sort((a,b) => (parseNum(a.rel_frobenius)||0) - (parseNum(b.rel_frobenius)||0));
  return copy.sort((a,b) => (parseNum(b.rel_frobenius)||0) - (parseNum(a.rel_frobenius)||0));
}}

function filterTable() {{
  const search = (searchInput.value || '').toLowerCase();
  const status = statusFilter.value;
  const badge = badgeFilter.value;
  const lens = lensFilter.value;
  const filtered = TABLE_DATA.filter(row => {{
    const matchesSearch = String(row.name || '').toLowerCase().includes(search);
    const matchesStatus = !status || row.status === status;
    const matchesBadge = !badge || row.badge === badge;
    const matchesLens = !lens || (lens === 'has_lens' && row.lens) || (lens === 'no_lens' && !row.lens);
    return matchesSearch && matchesStatus && matchesBadge && matchesLens;
  }});
  renderTable(sortRows(filtered));
}}

function drawChart() {{
  const svg = document.getElementById('deltaChart');
  const rows = TABLE_DATA.filter(r => parseNum(r.rel_frobenius) === parseNum(r.rel_frobenius));
  if (rows.length === 0) {{
    svg.innerHTML = '<text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" fill="#9ca3af" font-size="14">No metric data available</text>';
    return;
  }}
  const width = 1000, height = 300;
  const margin = {{top: 18, right: 16, bottom: 46, left: 56}};
  const cw = width - margin.left - margin.right;
  const ch = height - margin.top - margin.bottom;
  const vals = rows.map(r => parseNum(r.rel_frobenius));
  const maxV = Math.max(...vals, 0.0001);
  const minV = Math.min(...vals, 0);
  const span = (maxV - minV) || 1;
  const x = (i) => margin.left + (rows.length > 1 ? (i/(rows.length-1))*cw : cw/2);
  const y = (v) => margin.top + (1 - (v - minV)/span) * ch;
  let bars = '';
  const barW = Math.max(1, cw / rows.length - 1);
  rows.forEach((row, i) => {{
    const v = parseNum(row.rel_frobenius);
    const bx = x(i) - barW/2;
    const by = y(v);
    const bh = Math.max(1, (height - margin.bottom) - by);
    const color = (row.badge && BADGE_COLORS[row.badge]) || '#64748b';
    bars += `<rect x="${{bx}}" y="${{by}}" width="${{barW}}" height="${{bh}}" fill="${{color}}" rx="1"><title>${{escapeHtml(row.name)}} · ΔF=${{v.toFixed(6)}}${{row.badge ? ' · '+(BADGE_LABELS[row.badge]||row.badge) : ''}}</title></rect>`;
  }});
  // axes & grid
  let grid = '';
  for (let i=0;i<=4;i++) {{
    const gv = minV + span*(i/4);
    const gy = y(gv);
    grid += `<line x1="${{margin.left}}" y1="${{gy}}" x2="${{width-margin.right}}" y2="${{gy}}" stroke="#2b3140" stroke-width="1"/>`;
    grid += `<text x="${{margin.left-8}}" y="${{gy+4}}" text-anchor="end" font-size="10" fill="#9ca3af">${{gv.toFixed(3)}}</text>`;
  }}
  const axes = `<line x1="${{margin.left}}" y1="${{margin.top}}" x2="${{margin.left}}" y2="${{height-margin.bottom}}" stroke="#475569"/><line x1="${{margin.left}}" y1="${{height-margin.bottom}}" x2="${{width-margin.right}}" y2="${{height-margin.bottom}}" stroke="#475569"/>`;
  // sparse x labels
  let labels = '';
  const step = Math.max(1, Math.floor(rows.length/8));
  rows.forEach((row,i)=>{{
    if (i%step===0 || i===rows.length-1) {{
      const lx = x(i);
      labels += `<text x="${{lx}}" y="${{height-margin.bottom+16}}" text-anchor="middle" font-size="9" fill="#9ca3af">${{escapeHtml(String(row.name).slice(0,18))}}</text>`;
    }}
  }});
  svg.innerHTML = grid + axes + bars + labels;
  if (!HAS_DEEP) {{
    svg.innerHTML += `<text x="${{width-margin.right}}" y="${{margin.top+10}}" text-anchor="end" fill="#9ca3af" font-size="11">Run with --deep to see delta profile chart</text>`;
  }}
}}

function buildVerdict() {{
  const v = document.getElementById('verdictValue');
  const s = document.getElementById('verdictSub');
  const max = Number(SUMMARY.max_frobenius || 0);
  const mean = Number(SUMMARY.mean_frobenius || 0);
  const minCos = Number(SUMMARY.min_cosine_value || 1);
  const ablations = TABLE_DATA.filter(r => r.badge === 'RANK1_ABLATION').length;
  const drifted = TABLE_DATA.filter(r => r.max_principal_angle !== null && Number(r.max_principal_angle) > 25).length;
  let label = 'Subtle / aligned';
  let sub = `mean ΔF ${{mean.toFixed(4)}}, min cosine ${{minCos.toFixed(4)}}`;
  if (ablations > 0) {{ label = 'Targeted low-rank shifts'; sub = `${{ablations}} rank-1 ablation layer(s) detected`; }}
  else if (drifted > 0) {{ label = 'Representation drift'; sub = `${{drifted}} layer(s) with θmax > 25°`; }}
  else if (max > 0.1 || minCos < 0.8) {{ label = 'Significant divergence'; sub = `max ΔF ${{max.toFixed(4)}}, min cosine ${{minCos.toFixed(4)}}`; }}
  else if (mean > 0.01) {{ label = 'Moderate fine-tune delta'; }}
  v.textContent = label; s.textContent = sub;
}}

function buildLegend() {{
  const el = document.getElementById('legend');
  const counts = {{}};
  TABLE_DATA.forEach(r => {{ if (r.badge) counts[r.badge] = (counts[r.badge]||0)+1; }});
  const keys = Object.keys(counts).sort((a,b)=>counts[b]-counts[a]);
  if (!keys.length) {{ el.innerHTML = '<span class="leg dim">No taxonomy (run with --deep)</span>'; return; }}
  el.innerHTML = keys.map(k => `<span class="leg"><i style="--c:${{BADGE_COLORS[k]||'#64748b'}}"></i>${{escapeHtml(BADGE_LABELS[k]||k)}} <b>${{counts[k]}}</b></span>`).join('');
}}

[searchInput, statusFilter, badgeFilter, lensFilter, sortSelect].forEach(el => {{
  if (!el) return;
  const ev = el.tagName === 'INPUT' ? 'input' : 'change';
  el.addEventListener(ev, filterTable);
}});

buildVerdict();
buildLegend();
filterTable();
drawChart();
</script>
</body>
</html>"""
    return html


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

        rows.append(
            {
                "name": t.name,
                "shape": shape_display,
                "rel_frobenius": f"{t.metrics.relative_frobenius:.4f}" if t.metrics else "—",
                "cosine_sim": f"{t.metrics.cosine_similarity:.4f}" if t.metrics else "—",
                "max_delta": f"{t.metrics.max_element_delta:.4f}" if t.metrics else "—",
                "signal_rank": t.spectral.signal_rank if t.spectral else None,
                "effective_rank": t.spectral.effective_rank if t.spectral else None,
                "max_principal_angle": (
                    t.subspace.max_principal_angle if getattr(t, "subspace", None) else None
                ),
                "uhlmann_fidelity": (
                    t.subspace.uhlmann_fidelity if getattr(t, "subspace", None) else None
                ),
                "badge": badge,
                "status": t.status,
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


def _format_shape(shape: tuple[int, ...]) -> str:
    if not shape:
        return "—"
    return " × ".join(str(d) for d in shape)


def _get_embedded_css() -> str:
    # Instrument/editorial palette — dark slate, spectral accents, no generic SaaS tells
    return r"""
:root{
  --bg:#0b0f14; --bg2:#0f141b; --panel:#121922; --panel2:#0f1620;
  --ink:#e6edf6; --muted:#93a0b4; --dim:#6b778c; --line:#223043;
  --accent:#7dd3fc; --accent2:#a78bfa; --good:#34d399; --warn:#fbbf24; --bad:#f87171;
  --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
  --sans: "Avenir Next", "Segoe UI Variable", "Segoe UI", "Helvetica Neue", sans-serif;
}
*{box-sizing:border-box;margin:0;padding:0}
html,body{background:var(--bg);color:var(--ink);font-family:var(--sans);line-height:1.5}
body{background-image:radial-gradient(1200px 400px at 10% -10%, #132033 0%, transparent 60%),radial-gradient(900px 300px at 100% 0%, #151a2e 0%, transparent 55%)}
.shell{max-width:1280px;margin:0 auto;padding:28px 22px 40px}
.masthead{display:flex;gap:20px;align-items:flex-end;justify-content:space-between;margin-bottom:18px}
.eyebrow{color:var(--accent);font-size:12px;letter-spacing:.12em;text-transform:uppercase}
h1{font-size:30px;font-weight:650;letter-spacing:-.02em;margin:2px 0 4px}
.deck{color:var(--muted);font-size:14px;max-width:70ch;word-break:break-all}
.verdict{min-width:260px;background:linear-gradient(180deg,var(--panel),var(--panel2));border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.verdict-label{font-size:11px;color:var(--dim);text-transform:uppercase;letter-spacing:.1em}
.verdict-value{font-size:18px;font-weight:650;margin-top:2px;color:var(--accent)}
.verdict-sub{font-size:12px;color:var(--muted);margin-top:4px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;margin-bottom:16px}
.meta-panel{padding:10px 14px}
.meta-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px 18px}
.meta-item{display:flex;flex-direction:column;gap:2px;min-width:0}
.meta-k{font-size:11px;color:var(--dim);text-transform:uppercase;letter-spacing:.08em}
.meta-v{font-size:13px;color:var(--ink);overflow:hidden;text-overflow:ellipsis}
.meta-v code{font-family:var(--mono);font-size:12px;color:var(--muted)}
.pill{display:inline-block;margin-left:6px;padding:1px 7px;border:1px solid var(--line);border-radius:999px;font-size:11px;color:var(--accent);vertical-align:middle}
.kpi-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-bottom:16px}
.kpi{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 14px 12px}
.kpi.accent{border-color:#2a3b52;background:linear-gradient(180deg,#141d29,#121922)}
.kpi-n{font-size:26px;font-weight:650;letter-spacing:-.02em;color:var(--ink)}
.kpi-l{font-size:11px;color:var(--dim);text-transform:uppercase;letter-spacing:.08em;margin-top:2px}
.kpi-s{font-size:11px;color:var(--muted);margin-top:4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.panel-head{display:flex;justify-content:space-between;gap:12px;align-items:flex-end;padding:14px 16px 10px;border-bottom:1px solid var(--line)}
.panel-head h2{font-size:15px;font-weight:600}
.panel-sub{font-size:12px;color:var(--muted);margin-top:2px}
.legend{display:flex;flex-wrap:wrap;gap:10px;justify-content:flex-end}
.leg{font-size:11px;color:var(--muted);display:inline-flex;align-items:center;gap:6px}
.leg i{width:10px;height:10px;border-radius:2px;background:var(--c);display:inline-block}
.leg b{color:var(--ink);font-weight:600}
.chart-wrap{padding:10px 10px 6px;background:var(--bg2);border-radius:0 0 10px 10px}
.chart-wrap svg{width:100%;height:300px;display:block}
.controls{display:flex;flex-wrap:wrap;gap:8px;padding:12px 16px;border-bottom:1px solid var(--line);background:var(--panel2)}
.controls input,.controls select{background:var(--bg2);color:var(--ink);border:1px solid var(--line);border-radius:7px;padding:8px 10px;font:13px var(--sans)}
.controls input{flex:1 1 220px;min-width:180px}
.controls select{flex:0 0 auto}
.controls input:focus,.controls select:focus{outline:none;border-color:var(--accent)}
.table-wrap{overflow:auto;max-height:70vh}
table{width:100%;border-collapse:separate;border-spacing:0;font-size:13px}
thead th{position:sticky;top:0;background:#0e1520;color:var(--muted);font-weight:600;text-align:left;padding:10px 12px;border-bottom:1px solid var(--line);white-space:nowrap;z-index:1}
tbody td{padding:9px 12px;border-bottom:1px solid #182230;vertical-align:top}
tbody tr:hover td{background:#131b26}
.c-name code{font-family:var(--mono);font-size:12px;color:#d7e3f4;word-break:break-all}
.c-shape{color:var(--muted);white-space:nowrap}
.num{font-family:var(--mono);font-size:12px;text-align:right;white-space:nowrap}
.num.drift{color:var(--bad);font-weight:600}
.badge{display:inline-block;padding:2px 8px;border-radius:999px;font-size:11px;font-weight:600;color:#071018;background:var(--bc);white-space:nowrap}
.status{display:inline-block;padding:2px 8px;border-radius:6px;font-size:11px;font-weight:600;border:1px solid var(--line);color:var(--muted)}
.status-match{color:var(--good);border-color:#1f3b33;background:#0e1f1a}
.status-partial_match{color:#fdba74;border-color:#4a341f;background:#1f1710}
.status-shape_mismatch{color:var(--bad);border-color:#4a2323;background:#1f1214}
.status-only_in_a,.status-only_in_b{color:var(--warn);border-color:#4a3a1f;background:#1f1a10}
.lens{font-size:12px;line-height:1.35;min-width:220px}
.lens-pos{color:#86efac}.lens-neg{color:#fca5a5;margin-top:3px}
.logit{color:var(--dim)}
.foot{color:var(--dim);font-size:12px;text-align:center;margin-top:18px}
.dim{color:var(--dim)}
@media (max-width: 900px){
  .masthead{flex-direction:column;align-items:stretch}
  .meta-grid{grid-template-columns:1fr}
  .verdict{min-width:0}
}
"""


# -----------------------------------------------------------------------------
# Inspection report (kept self-contained, reusing same aesthetic)
# -----------------------------------------------------------------------------


def _generate_inspection_html_content(
    result: ModelInspectionResult, title: str | None = None
) -> str:
    table_data = _prepare_inspection_table_data(result)
    summary_data = _prepare_inspection_summary_data(result)
    has_deep = any(t.spectral is not None for t in result.layers)

    if title is None:
        title = f"wdiff inspect · {Path(result.model_path).name}"
    safe_title = html_lib.escape(title)

    # Keys used by existing report/test expectations
    # (Mean Rel. ΔF, Min Cosine, Duration labels must appear)
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
{_inspection_extra_css()}
</style>
</head>
<body>
<div class="shell">
  <header class="masthead">
    <div class="brand">
      <div class="eyebrow">model health</div>
      <h1>{safe_title}</h1>
      <p class="deck">wdiff Report · Model inspection report</p>
    </div>
    <div class="verdict">
      <div class="verdict-label">quantization readiness</div>
      <div class="verdict-value">{html_lib.escape(str(result.summary.quantization_readiness))}</div>
      <div class="verdict-sub">{html_lib.escape(str(result.summary.architecture))} · {result.format}</div>
    </div>
  </header>

  <section class="kpi-grid">
    <div class="kpi accent"><div class="kpi-n">{result.summary.total_params:,}</div><div class="kpi-l">Total Parameters</div></div>
    <div class="kpi"><div class="kpi-n">{result.summary.trainable_params_mb:.2f} MB</div><div class="kpi-l">Model Size</div></div>
    <div class="kpi"><div class="kpi-n">{result.summary.tensor_count}</div><div class="kpi-l">Tensors</div></div>
    <div class="kpi"><div class="kpi-n">{result.summary.duration_seconds:.2f}s</div><div class="kpi-l">Duration</div></div>
    <div class="kpi"><div class="kpi-n">{result.summary.dead_neurons_total}</div><div class="kpi-l">Dead Neurons</div></div>
    <div class="kpi"><div class="kpi-n">{result.summary.outlier_risk_layer_count}</div><div class="kpi-l">Outlier Risk Layers</div></div>
    <div class="kpi"><div class="kpi-n">{mean_f:.4f}</div><div class="kpi-l">Mean Rel. ΔF</div></div>
    <div class="kpi"><div class="kpi-n">{min_c:.4f}</div><div class="kpi-l">Min Cosine</div></div>
  </section>

  <section class="panel chart-panel">
    <div class="panel-head"><div><h2>Layer Delta Profile</h2><p class="panel-sub">Norm / outlier profile by layer (inspection view).</p></div><div class="legend" id="legend"></div></div>
    <div class="chart-wrap"><svg id="deltaChart" viewBox="0 0 1000 300" preserveAspectRatio="none"></svg></div>
  </section>

  <section class="panel table-panel">
    <div class="panel-head"><div><h2>Layer Details</h2><p class="panel-sub" id="tableMeta"></p></div></div>
    <div class="controls">
      <input type="text" id="searchInput" placeholder="Search layers…">
      <select id="statusFilter">
        <option value="">All Status</option>
        <option value="HEALTHY">HEALTHY</option>
        <option value="DEAD_NEURONS">DEAD_NEURONS</option>
        <option value="OUTLIER_SPIKE">OUTLIER_SPIKE</option>
      </select>
      <select id="badgeFilter">
        <option value="">All Classifications</option>
        <option value="RANK1_ABLATION">Rank-1 Ablation</option>
        <option value="LOW_RANK_ADAPTER">Low-Rank Adapter</option>
        <option value="SPARSE_UPDATE">Sparse Update</option>
        <option value="FULL_RANK_CPT">Full-Rank CPT</option>
        <option value="STOCHASTIC_NOISE">Stochastic Noise</option>
        <option value="UNMODIFIED">Unmodified</option>
      </select>
    </div>
    <div class="table-wrap">
      <table id="dataTable">
        <thead><tr>
          <th>Layer Name</th><th>Shape</th><th>Dtype</th><th>Params</th>
          <th>L2 Norm</th><th>Outlier</th><th>Dead</th><th>Status</th><th>Rank</th>
        </tr></thead>
        <tbody id="tableBody"></tbody>
      </table>
    </div>
  </section>

  <footer class="foot"><p>Generated by <strong>wdiff</strong> — git diff, but for neural network weights. Self-contained report, no external assets.</p></footer>
</div>

<script>
const TABLE_DATA = {json.dumps(table_data)};
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
      <td class="c-name"><code>${{escapeHtml(row.name)}}</code></td>
      <td class="c-shape">${{escapeHtml(row.shape)}}</td>
      <td>${{escapeHtml(row.dtype)}}</td>
      <td class="num">${{Number(row.param_count||0).toLocaleString()}}</td>
      <td class="num">${{Number(row.norm||0).toExponential(2)}}</td>
      <td class="num">${{Number(row.outlier_score||0).toFixed(2)}}</td>
      <td class="num">${{row.dead_neuron_count||0}}</td>
      <td><span class="status ${{hclass}}">${{escapeHtml(row.status || health)}}</span></td>
      <td class="num">${{rankStr}}</td>`;
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
  // For inspection we chart L2 norm magnitude as a profile (rel_frobenius holds norm string)
  const vals = TABLE_DATA.map(r => Number(r.norm || 0)).filter(v => Number.isFinite(v));
  if (!vals.length) {{
    svg.innerHTML = '<text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" fill="#9ca3af" font-size="14">Run with --deep to see delta profile chart</text>';
    return;
  }}
  const rows = TABLE_DATA;
  const width=1000,height=300,margin={{top:18,right:16,bottom:46,left:56}};
  const cw=width-margin.left-margin.right,ch=height-margin.top-margin.bottom;
  const maxV=Math.max(...vals,1e-9), minV=0, span=maxV-minV||1;
  const x=(i)=>margin.left+(rows.length>1?(i/(rows.length-1))*cw:cw/2);
  const y=(v)=>margin.top+(1-(v-minV)/span)*ch;
  let bars=''; const barW=Math.max(1,cw/rows.length-1);
  rows.forEach((row,i)=>{{
    const v=Number(row.norm||0); const bx=x(i)-barW/2; const by=y(v); const bh=Math.max(1,(height-margin.bottom)-by);
    const color=row.is_outlier_risk?'#f87171':(row.dead_neuron_count>0?'#fbbf24':'#60a5fa');
    bars+=`<rect x="${{bx}}" y="${{by}}" width="${{barW}}" height="${{bh}}" fill="${{color}}" rx="1"><title>${{escapeHtml(row.name)}} · norm=${{v.toExponential(2)}}</title></rect>`;
  }});
  let grid=''; for(let i=0;i<=4;i++){{const gv=minV+span*(i/4),gy=y(gv);grid+=`<line x1="${{margin.left}}" y1="${{gy}}" x2="${{width-margin.right}}" y2="${{gy}}" stroke="#2b3140"/>`;grid+=`<text x="${{margin.left-8}}" y="${{gy+4}}" text-anchor="end" font-size="10" fill="#9ca3af">${{gv.toExponential(1)}}</text>`;}}
  const axes=`<line x1="${{margin.left}}" y1="${{margin.top}}" x2="${{margin.left}}" y2="${{height-margin.bottom}}" stroke="#475569"/><line x1="${{margin.left}}" y1="${{height-margin.bottom}}" x2="${{width-margin.right}}" y2="${{height-margin.bottom}}" stroke="#475569"/>`;
  svg.innerHTML = grid + axes + bars;
  if (!HAS_DEEP) {{
    svg.innerHTML += `<text x="${{width-margin.right}}" y="${{margin.top+10}}" text-anchor="end" fill="#9ca3af" font-size="11">Run with --deep to see delta profile chart</text>`;
  }}
}}

function buildLegend() {{
  const el = document.getElementById('legend');
  el.innerHTML = '<span class="leg"><i style="--c:#60a5fa"></i>Healthy</span><span class="leg"><i style="--c:#fbbf24"></i>Dead neurons</span><span class="leg"><i style="--c:#f87171"></i>Outlier risk</span>';
}}

[searchInput,statusFilter,badgeFilter].forEach(el=>{{if(!el)return; el.addEventListener(el.tagName==='INPUT'?'input':'change', filterTable);}});
buildLegend();
filterTable();
drawChart();
</script>
</body>
</html>"""
    return html


def _inspection_extra_css() -> str:
    return ""  # shares main CSS


def _prepare_inspection_table_data(result: ModelInspectionResult) -> list[dict]:
    rows = []
    for t in result.layers:
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
                # keep keys some consumers might expect
                "rel_frobenius": f"{t.norm:.2e}" if t.norm else "—",
                "max_delta": f"{t.outlier_score:.2f}" if t.outlier_score else "—",
            }
        )
    return rows


def _prepare_inspection_summary_data(result: ModelInspectionResult) -> dict:
    s = result.summary
    # include keys expected/used broadly
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
