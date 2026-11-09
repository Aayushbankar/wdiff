from __future__ import annotations

import json
from pathlib import Path

from wdiff.engine.diff import DiffResult
from wdiff.engine.inspect import ModelInspectionResult

DiffOrInspection = DiffResult | ModelInspectionResult


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

    output_path.write_text(html)
    return output_path


def _generate_diff_html_content(result: DiffResult, title: str | None = None) -> str:
    """Generate the complete HTML content as a string."""

    # Prepare data for JavaScript
    table_data = _prepare_table_data(result)
    summary_data = _prepare_summary_data(result)
    has_deep = any(t.spectral is not None or t.taxonomy is not None for t in result.tensor_results)

    # Taxonomy badge colors
    badge_colors = {
        "RANK1_ABLATION": "#e056fd",
        "LOW_RANK_ADAPTER": "#0984e3",
        "SPARSE_UPDATE": "#f1c40f",
        "FULL_RANK_CPT": "#00b894",
        "STOCHASTIC_NOISE": "#d63031",
        "UNMODIFIED": "#636e72",
    }

    badge_colors_json = json.dumps(badge_colors)
    table_data_json = json.dumps(table_data)
    json.dumps(summary_data)
    has_deep_json = json.dumps(has_deep)

    if title is None:
        title = (
            f"wdiff Report: {Path(result.model_a_path).name} vs {Path(result.model_b_path).name}"
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        {_get_embedded_css()}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{title}</h1>
            <p class="subtitle">Interactive differential analysis report</p>
        </header>
        
        <section class="kpi-cards">
            <div class="kpi-card">
                <div class="kpi-value">{summary_data["matched_count"]}</div>
                <div class="kpi-label">Matched Layers</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-value">{summary_data.get("partial_match_count", 0)}</div>
                <div class="kpi-label">Partial Matches</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-value">{summary_data.get("lens_analyzed_count", 0)}</div>
                <div class="kpi-label">Decoded Concepts</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-value">{summary_data["mismatched_shape_count"]}</div>
                <div class="kpi-label">Shape Mismatches</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-value">{summary_data["only_in_a_count"]}</div>
                <div class="kpi-label">Only in A</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-value">{summary_data["only_in_b_count"]}</div>
                <div class="kpi-label">Only in B</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-value">{summary_data["duration_seconds"]:.2f}s</div>
                <div class="kpi-label">Duration</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-value">{summary_data["mean_frobenius"]:.4f}</div>
                <div class="kpi-label">Mean Rel. ΔF</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-value">{summary_data["min_cosine_value"]:.4f}</div>
                <div class="kpi-label">Min Cosine</div>
            </div>
        </section>
        
        <section class="controls">
            <div class="search-box">
                <input type="text" id="searchInput" placeholder="Search layers..." onkeyup="filterTable()">
            </div>
            <div class="filter-box">
                <select id="statusFilter" onchange="filterTable()">
                    <option value="">All Status</option>
                    <option value="MATCH">MATCH</option>
                    <option value="PARTIAL_MATCH">PARTIAL_MATCH</option>
                    <option value="SHAPE_MISMATCH">SHAPE_MISMATCH</option>
                    <option value="ONLY_IN_A">ONLY_IN_A</option>
                    <option value="ONLY_IN_B">ONLY_IN_B</option>
                </select>
                <select id="badgeFilter" onchange="filterTable()">
                    <option value="">All Classifications</option>
                    <option value="RANK1_ABLATION">Surgical Rank-1 Ablation</option>
                    <option value="LOW_RANK_ADAPTER">Low-Rank Adapter</option>
                    <option value="SPARSE_UPDATE">Sparse Update</option>
                    <option value="FULL_RANK_CPT">Full-Rank CPT</option>
                    <option value="STOCHASTIC_NOISE">Stochastic Noise</option>
                    <option value="UNMODIFIED">Unmodified</option>
                </select>
                <select id="lensFilter" onchange="filterTable()">
                    <option value="">All Lens</option>
                    <option value="has_lens">Has Concept Lens</option>
                    <option value="no_lens">No Concept Lens</option>
                </select>
            </div>
        </section>
        
        <section class="chart-section">
            <h2>Layer Delta Profile</h2>
            <div class="chart-container">
                <svg id="deltaChart" viewBox="0 0 1000 300" preserveAspectRatio="none"></svg>
            </div>
        </section>
        
        <section class="table-section">
            <h2>Layer Details</h2>
            <div class="table-wrapper">
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
                            <th>Classification</th>
                            <th>Concept Lens</th>
                            <th>Status</th>
                        </tr>
                        </thead>
                        <tbody id="tableBody">
                        </tbody>
                    </table>
                </div>
        </section>
        
        <footer>
            <p>Generated by <strong>wdiff</strong> — git diff, but for neural network weights</p>
        </footer>
    </div>
    
    <script>
        const TABLE_DATA = {table_data_json};
        const HAS_DEEP = {has_deep_json};
        const BADGE_COLORS = {badge_colors_json};
        
        const tableBody = document.getElementById('tableBody');
        const searchInput = document.getElementById('searchInput');
        const statusFilter = document.getElementById('statusFilter');
        const badgeFilter = document.getElementById('badgeFilter');
        
        function renderTable(data) {{
            tableBody.innerHTML = '';
            data.forEach(row => {{
                const tr = document.createElement('tr');
                let lensHtml = '—';
                if (row.lens) {{
                    const pos = row.lens.positive_tokens.map(tc => `${{tc.token}} (${{tc.logit.toFixed(2)}})`).join(', ');
                    const neg = row.lens.negative_tokens.map(tc => `${{tc.token}} (${{tc.logit.toFixed(2)}})`).join(', ');
                    lensHtml = `<div style="font-size:0.8em;"><div style="color:#00b894;"><strong>(+)</strong> ${{pos || '—'}}</div><div style="color:#ff6b6b;"><strong>(-)</strong> ${{neg || '—'}}</div></div>`;
                }}
                tr.innerHTML = `
                    <td><code>${{row.name}}</code></td>
                    <td>${{row.shape}}</td>
                    <td>${{row.rel_frobenius}}</td>
                    <td>${{row.cosine_sim}}</td>
                    <td>${{row.max_delta}}</td>
                    <td>${{row.signal_rank !== null ? row.signal_rank : '—'}}</td>
                    <td>${{row.effective_rank !== null ? row.effective_rank.toFixed(1) : '—'}}</td>
                    <td>${{row.badge ? '<span class="badge" style="background-color:' + BADGE_COLORS[row.badge] + ';color:white;padding:2px 6px;border-radius:3px;font-size:0.8em;">' + (row.badge && row.badge.replace ? row.badge.replace(/_/g, ' ') : '') + '</span>' : '—'}}</td>
                    <td>${{lensHtml}}</td>
                    <td><span class="status-badge status-${{row.status.toLowerCase()}}">${{row.status}}</span></td>
                `;
                tableBody.appendChild(tr);
            }});
        }}
        
        function filterTable() {{
            const search = searchInput.value.toLowerCase();
            const status = statusFilter.value;
            const badge = badgeFilter.value;
            const lens = lensFilter.value;
            
            const filtered = TABLE_DATA.filter(row => {{
                const matchesSearch = row.name.toLowerCase().includes(search);
                const matchesStatus = !status || row.status === status;
                const matchesBadge = !badge || row.badge === badge;
                const matchesLens = !lens || (lens === 'has_lens' && row.lens) || (lens === 'no_lens' && !row.lens);
                return matchesSearch && matchesStatus && matchesBadge && matchesLens;
            }});
            renderTable(filtered);
        }}
        
        function drawChart() {{
            const svg = document.getElementById('deltaChart');
            if (!HAS_DEEP) {{
                svg.innerHTML = '<text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" fill="#636e72" font-size="14">Run with --deep to see delta profile chart</text>';
                return;
            }}
            
            const deepRows = TABLE_DATA.filter(r => r.signal_rank !== null);
            if (deepRows.length === 0) {{
                svg.innerHTML = '<text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" fill="#636e72" font-size="14">No deep data available</text>';
                return;
            }}
            
            const width = 1000;
            const height = 300;
            const margin = {{top: 20, right: 20, bottom: 40, left: 60}};
            const chartWidth = width - margin.left - margin.right;
            const chartHeight = height - margin.top - margin.bottom;
            
            const maxDelta = Math.max(...deepRows.map(r => r.rel_frobenius));
            const minDelta = Math.min(...deepRows.map(r => r.rel_frobenius));
            
            const xScale = (i) => margin.left + (deepRows.length > 1 ? (i / (deepRows.length - 1)) * chartWidth : chartWidth / 2);
            const yScale = (val) => margin.top + (1 - (val - minDelta) / (maxDelta - minDelta || 1)) * chartHeight;
            
            let bars = '';
            deepRows.forEach((row, i) => {{
                const x = xScale(i);
                const barWidth = Math.max(1, chartWidth / deepRows.length - 2);
                const barHeight = chartHeight - (yScale(row.rel_frobenius) - margin.top);
                const y = yScale(row.rel_frobenius);
                const color = row.badge ? BADGE_COLORS[row.badge] : '#636e72';
                
                bars += `<rect x="${{x}}" y="${{y}}" width="${{barWidth}}" height="${{barHeight}}" fill="${{color}}" rx="1" title="${{row.name}}: ${{row.rel_frobenius.toFixed(4)}}" />`;
            }});
            
            // Axes
            const axes = `
                <line x1="${{margin.left}}" y1="${{margin.top}}" x2="${{margin.left}}" y2="${{height - margin.bottom}}" stroke="#636e72" stroke-width="1"/>
                <line x1="${{margin.left}}" y1="${{height - margin.bottom}}" x2="${{width - margin.right}}" y2="${{height - margin.bottom}}" stroke="#636e72" stroke-width="1"/>
            `;
            
            // Y-axis labels
            let yLabels = '';
            for (let i = 0; i <= 4; i++) {{
                const val = minDelta + (maxDelta - minDelta) * (1 - i/4);
                const y = margin.top + (i/4) * chartHeight;
                yLabels += `<text x="${{margin.left - 10}}" y="${{y + 4}}" text-anchor="end" font-size="10" fill="#636e72">${{val.toFixed(3)}}</text>`;
            }}
            
            // X-axis labels (every nth layer)
            let xLabels = '';
            const step = Math.max(1, Math.floor(deepRows.length / 10));
            deepRows.forEach((row, i) => {{
                if (i % step === 0 || i === deepRows.length - 1) {{
                    const x = xScale(i) + 1;
                    xLabels += `<text x="${{x}}" y="${{height - margin.bottom + 20}}" text-anchor="middle" font-size="9" fill="#636e72" transform="rotate(-45, ${{x}}, ${{height - margin.bottom + 20}})">${{row.name.substring(0, 20)}}</text>`;
                }}
            }});
            
            svg.innerHTML = axes + yLabels + xLabels + bars;
        }}
        
        // Initialize
        renderTable(TABLE_DATA);
        drawChart();
    </script>
</body>
</html>"""
    return html


def _prepare_table_data(result: DiffResult) -> list[dict]:
    """Prepare table row data for JavaScript."""
    rows = []
    for t in result.tensor_results:
        badge = None

        if t.taxonomy:
            badge = t.taxonomy.badge.name

        if t.status == "PARTIAL_MATCH" and t.partial_info:
            pi = t.partial_info
            shape_display = f"{pi.common_shape[0]:,}→{t.shape_b[0]:,} × {' × '.join(str(d) for d in pi.common_shape[1:])}"
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
    """Prepare summary data for KPI cards."""
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
        "min_cosine_value": s.min_cosine_value,
    }


def _format_shape(shape: tuple[int, ...]) -> str:
    if not shape:
        return "—"
    return " × ".join(str(d) for d in shape)


def _get_embedded_css() -> str:
    return """
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #1e1e2e; color: #cdd6f4; line-height: 1.6; }
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
        header { text-align: center; margin-bottom: 30px; }
        header h1 { font-size: 2.5em; font-weight: 700; background: linear-gradient(90deg, #e056fd, #0984e3); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .subtitle { color: #6c7086; margin-top: 8px; font-size: 1.1em; }
        
        .kpi-cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin-bottom: 30px; }
        .kpi-card { background: #282a36; border: 1px solid #44475a; border-radius: 12px; padding: 20px; text-align: center; transition: transform 0.2s, box-shadow 0.2s; }
        .kpi-card:hover { transform: translateY(-2px); box-shadow: 0 8px 24px rgba(0,0,0,0.3); }
        .kpi-value { font-size: 2.2em; font-weight: 700; color: #f8f8f2; }
        .kpi-label { font-size: 0.85em; color: #6272a4; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.5px; }
        
        .controls { display: flex; gap: 16px; margin-bottom: 30px; flex-wrap: wrap; }
        .search-box, .filter-box { flex: 1; min-width: 200px; }
        .search-box input, .filter-box select { width: 100%; padding: 12px 16px; background: #282a36; border: 1px solid #44475a; border-radius: 8px; color: #f8f8f2; font-size: 1em; }
        .search-box input:focus, .filter-box select:focus { outline: none; border-color: #e056fd; }
        
        .chart-section { background: #282a36; border: 1px solid #44475a; border-radius: 12px; padding: 24px; margin-bottom: 30px; }
        .chart-section h2 { margin-bottom: 16px; font-size: 1.3em; }
        .chart-container { height: 300px; background: #1e1e2e; border-radius: 8px; }
        
        .table-section { background: #282a36; border: 1px solid #44475a; border-radius: 12px; overflow: hidden; }
        .table-section h2 { padding: 20px 24px; margin: 0; border-bottom: 1px solid #44475a; }
        .table-wrapper { overflow-x: auto; }
        table { width: 100%; border-collapse: collapse; font-size: 0.9em; }
        th, td { padding: 12px 16px; text-align: left; border-bottom: 1px solid #44475a; }
        th { background: #1e1e2e; font-weight: 600; color: #f8f8f2; position: sticky; top: 0; }
        td code { background: #1e1e2e; padding: 2px 6px; border-radius: 4px; font-family: 'Monaco', 'Menlo', monospace; font-size: 0.85em; }
        tr:hover td { background: #2d2f3e; }
        
        .badge { display: inline-block; padding: 3px 8px; border-radius: 4px; font-weight: 500; font-size: 0.75em; }
        .status-badge { display: inline-block; padding: 3px 8px; border-radius: 4px; font-size: 0.75em; font-weight: 500; }
        .status-match { background: #00b894; color: white; }
        .status-shape_mismatch { background: #ff6b6b; color: white; }
        .status-only_in_a, .status-only_in_b { background: #f1c40f; color: #1e1e2e; }
        .status-partial_match { background: #fdcb6e; color: #1e1e2e; }
        
        footer { text-align: center; margin-top: 40px; padding: 20px; color: #6272a4; font-size: 0.9em; }
        
        /* Scrollbar styling */
        ::-webkit-scrollbar { width: 8px; height: 8px; }
        ::-webkit-scrollbar-track { background: #1e1e2e; }
        ::-webkit-scrollbar-thumb { background: #44475a; border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: #6272a4; }
    """
    return """* { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #1e1e2e; color: #cdd6f4; line-height: 1.6; }
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
        header { text-align: center; margin-bottom: 30px; }
        header h1 { font-size: 2.5em; font-weight: 700; background: linear-gradient(90deg, #e056fd, #0984e3); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .subtitle { color: #6c7086; margin-top: 8px; font-size: 1.1em; }
        
        .kpi-cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin-bottom: 30px; }
        .kpi-card { background: #282a36; border: 1px solid #44475a; border-radius: 12px; padding: 20px; text-align: center; transition: transform 0.2s, box-shadow 0.2s; }
        .kpi-card:hover { transform: translateY(-2px); box-shadow: 0 8px 24px rgba(0,0,0,0.3); }
        .kpi-value { font-size: 2.2em; font-weight: 700; color: #f8f8f2; }
        .kpi-label { font-size: 0.85em; color: #6272a4; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.5px; }
        
        .controls { display: flex; gap: 16px; margin-bottom: 30px; flex-wrap: wrap; }
        .search-box, .filter-box { flex: 1; min-width: 200px; }
        .search-box input, .filter-box select { width: 100%; padding: 12px 16px; background: #282a36; border: 1px solid #44475a; border-radius: 8px; color: #f8f8f2; font-size: 1em; }
        .search-box input:focus, .filter-box select:focus { outline: none; border-color: #e056fd; }
        
        .chart-section { background: #282a36; border: 1px solid #44475a; border-radius: 12px; padding: 24px; margin-bottom: 30px; }
        .chart-section h2 { margin-bottom: 16px; font-size: 1.3em; }
        .chart-container { height: 300px; background: #1e1e2e; border-radius: 8px; }
        
        .table-section { background: #282a36; border: 1px solid #44475a; border-radius: 12px; overflow: hidden; }
        .table-section h2 { padding: 20px 24px; margin: 0; border-bottom: 1px solid #44475a; }
        .table-wrapper { overflow-x: auto; }
        table { width: 100%; border-collapse: collapse; font-size: 0.9em; }
        th, td { padding: 12px 16px; text-align: left; border-bottom: 1px solid #44475a; }
        th { background: #1e1e2e; font-weight: 600; color: #f8f8f2; position: sticky; top: 0; }
        td code { background: #1e1e2e; padding: 2px 6px; border-radius: 4px; font-family: 'Monaco', 'Menlo', monospace; font-size: 0.85em; }
        tr:hover td { background: #2d2f3e; }
        
        .badge { display: inline-block; padding: 3px 8px; border-radius: 4px; font-weight: 500; font-size: 0.75em; }
        .status-badge { display: inline-block; padding: 3px 8px; border-radius: 4px; font-size: 0.75em; font-weight: 500; }
        .status-match { background: #00b894; color: white; }
        .status-shape_mismatch { background: #ff6b6b; color: white; }
        .status-only_in_a, .status-only_in_b { background: #f1c40f; color: #1e1e2e; }
        .status-partial_match { background: #fdcb6e; color: #1e1e2e; }
        
        footer { text-align: center; margin-top: 40px; padding: 20px; color: #6272a4; font-size: 0.9em; }
        
        /* Scrollbar styling */
        ::-webkit-scrollbar { width: 8px; height: 8px; }
        ::-webkit-scrollbar-track { background: #1e1e2e; }
        ::-webkit-scrollbar-thumb { background: #44475a; border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: #6272a4; }
    """


def _generate_inspection_html_content(
    result: ModelInspectionResult, title: str | None = None
) -> str:
    """Generate the complete HTML content for an inspection report as a string."""

    # Prepare data for JavaScript
    table_data = _prepare_inspection_table_data(result)
    summary_data = _prepare_inspection_summary_data(result)
    has_deep = any(t.spectral is not None for t in result.layers)

    # Taxonomy badge colors
    badge_colors = {
        "RANK1_ABLATION": "#e056fd",
        "LOW_RANK_ADAPTER": "#0984e3",
        "SPARSE_UPDATE": "#f1c40f",
        "FULL_RANK_CPT": "#00b894",
        "STOCHASTIC_NOISE": "#d63031",
        "UNMODIFIED": "#636e72",
    }

    json.dumps(badge_colors)
    json.dumps(table_data)
    json.dumps(summary_data)
    json.dumps(has_deep)

    if title is None:
        title = f"wdiff Inspection: {Path(result.model_path).name}"

    # Use a template with placeholders for Python variables
    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        {css}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>{title}</h1>
            <p class="subtitle">Interactive model inspection report</p>
        </header>

        <section class="kpi-cards">
            <div class="kpi-card">
                <div class="kpi-value">{total_params}</div>
                <div class="kpi-label">Total Parameters</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-value">{trainable_params_mb:.2f} MB</div>
                <div class="kpi-label">Model Size</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-value">{tensor_count}</div>
                <div class="kpi-label">Tensors</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-value">{duration_seconds:.2f}s</div>
                <div class="kpi-label">Duration</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-value">{dead_neurons_total}</div>
                <div class="kpi-label">Dead Neurons</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-value">{outlier_risk_layer_count}</div>
                <div class="kpi-label">Outlier Risk Layers</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-value">{mean_frobenius:.4f}</div>
                <div class="kpi-label">Mean Rel. ΔF</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-value">{min_cosine_value:.4f}</div>
                <div class="kpi-label">Min Cosine</div>
            </div>
        </section>

        <section class="controls">
            <div class="search-box">
                <input type="text" id="searchInput" placeholder="Search layers..." onkeyup="filterTable()">
            </div>
            <div class="filter-box">
                <select id="statusFilter" onchange="filterTable()">
                    <option value="">All Status</option>
                    <option value="HEALTHY">HEALTHY</option>
                    <option value="DEAD_NEURONS">DEAD_NEURONS</option>
                    <option value="OUTLIER_SPIKE">OUTLIER_SPIKE</option>
                </select>
                <select id="badgeFilter" onchange="filterTable()">
                    <option value="">All Classifications</option>
                    <option value="RANK1_ABLATION">Surgical Rank-1 Ablation</option>
                    <option value="LOW_RANK_ADAPTER">Low-Rank Adapter</option>
                    <option value="SPARSE_UPDATE">Sparse Update</option>
                    <option value="FULL_RANK_CPT">Full-Rank CPT</option>
                    <option value="STOCHASTIC_NOISE">Stochastic Noise</option>
                    <option value="UNMODIFIED">Unmodified</option>
                </select>
            </div>
        </section>

        <section class="chart-section">
            <h2>Layer Delta Profile</h2>
            <div class="chart-container">
                <svg id="deltaChart" viewBox="0 0 1000 300" preserveAspectRatio="none"></svg>
            </div>
        </section>

        <section class="table-section">
            <h2>Layer Details</h2>
            <div class="table-wrapper">
                <table id="dataTable">
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
                        </thead>
                        <tbody id="tableBody">
                        </tbody>
                    </table>
                </div>
            </section>

        <footer>
            <p>Generated by <strong>wdiff</strong> — git diff, but for neural network weights</p>
        </footer>
    </div>

    <script>
        const TABLE_DATA = {table_data_json};
        const HAS_DEEP = {has_deep_json};
        const BADGE_COLORS = {badge_colors_json};

        const tableBody = document.getElementById('tableBody');
        const searchInput = document.getElementById('searchInput');
        const statusFilter = document.getElementById('statusFilter');
        const badgeFilter = document.getElementById('badgeFilter');

        function renderTable(data) {{
            tableBody.innerHTML = '';
            data.forEach(row => {{
                const healthStyle = row.dead_neuron_count > 0 ? 'color: #f1c40f' : (row.is_outlier_risk ? 'color: #d63031' : 'color: #00b894');
                const statusStyle = row.status === 'MATCH' ? 'color: #00b894' : (row.status === 'SHAPE_MISMATCH' ? 'color: #ff6b6b' : 'color: #f1c40f');
                let rankStr = '—';
                let classStr = '—';
                if (row.signal_rank !== null) {{
                    rankStr = `s=${{row.signal_rank}} e=${{row.effective_rank.toFixed(1)}}`;
if (row.badge) {{
                    classStr = `<span class="badge" style="background-color:{{BADGE_COLORS[row.badge]}};color:white;padding:2px 6px;border-radius:3px;font-size:0.8em;">${{row.badge.replace(/_/g, ' ')}}</span>`;
                    }} else {{
                        classStr = '—';
                    }}
                }} else {{
                    rankStr = '—';
                    classStr = '—';
                }}
                tr.innerHTML = `
                    <td><code>${{row.name}}</code></td>
                    <td>${{row.shape}}</td>
                    <td>${{row.dtype}}</td>
                    <td>${{row.param_count.toLocaleString()}}</td>
                    <td>${{row.norm.toExponential(2)}}</td>
                    <td>${{row.outlier_score.toFixed(2)}}</td>
                    <td>${{row.dead_neuron_count}}</td>
                    <td><span style="${{healthStyle}}">${{row.status}}</span></td>
                    <td>${{rankStr}}</td>
                `;
                tableBody.appendChild(tr);
            }});
        }}

        function filterTable() {{
            const search = searchInput.value.toLowerCase();
            const status = statusFilter.value;
            const badge = badgeFilter.value;

            const filtered = TABLE_DATA.filter(row => {{
                const matchesSearch = row.name.toLowerCase().includes(search);
                const matchesStatus = !status || row.status === status;
                const matchesBadge = !badge || row.badge === badge;
                return matchesSearch && matchesStatus && matchesBadge;
            }});
            renderTable(filtered);
        }}

        function drawChart() {{
            const svg = document.getElementById('deltaChart');
            if (!HAS_DEEP) {{
                svg.innerHTML = '<text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" fill="#636e72" font-size="14">Run with --deep to see delta profile chart</text>';
                return;
            }}

            const deepRows = TABLE_DATA.filter(r => r.signal_rank !== null);
            if (deepRows.length === 0) {{
                svg.innerHTML = '<text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" fill="#636e72" font-size="14">No deep data available</text>';
                return;
            }}

            const width = 1000;
            const height = 300;
            const margin = {{top: 20, right: 20, bottom: 40, left: 60}};
            const chartWidth = width - margin.left - margin.right;
            const chartHeight = height - margin.top - margin.bottom;

            const maxDelta = Math.max(...deepRows.map(r => r.rel_frobenius));
            const minDelta = Math.min(...deepRows.map(r => r.rel_frobenius));

            const xScale = (i) => margin.left + (deepRows.length > 1 ? (i / (deepRows.length - 1)) * chartWidth : chartWidth / 2);
            const yScale = (val) => margin.top + (1 - (val - minDelta) / (maxDelta - minDelta || 1)) * chartHeight;

            let bars = '';
            deepRows.forEach((row, i) => {{
                const x = xScale(i);
                const barWidth = Math.max(1, chartWidth / deepRows.length - 2);
                const barHeight = chartHeight - (yScale(row.rel_frobenius) - margin.top);
                const y = yScale(row.rel_frobenius);
                const color = row.badge ? BADGE_COLORS[row.badge] : '#636e72';

                bars += `<rect x="${{x}}" y="${{y}}" width="${{barWidth}}" height="${{barHeight}}" fill="${{color}}" rx="1" title="${{row.name}}: ${{row.rel_frobenius.toFixed(4)}}" />`;
            }});

            // Axes
            const axes = `
                <line x1="${{margin.left}}" y1="${{margin.top}}" x2="${{margin.left}}" y2="${{height - margin.bottom}}" stroke="#636e72" stroke-width="1"/>
                <line x1="${{margin.left}}" y1="${{height - margin.bottom}}" x2="${{width - margin.right}}" y2="${{height - margin.bottom}}" stroke="#636e72" stroke-width="1"/>
            `;

            // Y-axis labels
            let yLabels = '';
            for (let i = 0; i <= 4; i++) {{
                const val = minDelta + (maxDelta - minDelta) * (1 - i/4);
                const y = margin.top + (i/4) * chartHeight;
                yLabels += `<text x="${{margin.left - 10}}" y="${{y + 4}}" text-anchor="end" font-size="10" fill="#636e72">${{val.toFixed(3)}}</text>`;
            }}

            // X-axis labels (every nth layer)
            let xLabels = '';
            const step = Math.max(1, Math.floor(deepRows.length / 10));
            deepRows.forEach((row, i) => {{
                if (i % step === 0 || i === deepRows.length - 1) {{
                    const x = xScale(i) + 1;
                    xLabels += `<text x="${{x}}" y="${{height - margin.bottom + 20}}" text-anchor="middle" font-size="9" fill="#636e72" transform="rotate(-45, ${{x}}, ${{height - margin.bottom + 20}})">${{row.name.substring(0, 20)}}</text>`;
                }}
            }});

            svg.innerHTML = axes + yLabels + xLabels + bars;
        }}

        // Initialize
        renderTable(TABLE_DATA);
        drawChart();
    </script>
</body>
</html>"""

    # Prepare the CSS and JSON data
    _get_embedded_css()
    badge_colors = {
        "RANK1_ABLATION": "#e056fd",
        "LOW_RANK_ADAPTER": "#0984e3",
        "SPARSE_UPDATE": "#f1c40f",
        "FULL_RANK_CPT": "#00b894",
        "STOCHASTIC_NOISE": "#d63031",
        "UNMODIFIED": "#636e72",
    }
    json.dumps(badge_colors)
    json.dumps(_prepare_inspection_table_data(result))
    summary_data = _prepare_inspection_summary_data(result)
    has_deep = any(t.spectral is not None for t in result.layers)
    json.dumps(any(t.spectral is not None for t in result.layers))

    if title is None:
        title = f"wdiff Inspection: {Path(result.model_path).name}"

    # Format the template with Python variables
    html = html_template.format(
        title=title,
        css=_get_embedded_css(),
        badge_colors_json=json.dumps(
            {
                "RANK1_ABLATION": "#e056fd",
                "LOW_RANK_ADAPTER": "#0984e3",
                "SPARSE_UPDATE": "#f1c40f",
                "FULL_RANK_CPT": "#00b894",
                "STOCHASTIC_NOISE": "#d63031",
                "UNMODIFIED": "#636e72",
            }
        ),
        table_data_json=json.dumps(_prepare_inspection_table_data(result)),
        **summary_data,
        has_deep_json=json.dumps(any(t.spectral is not None for t in result.layers)),
    )

    return html


def _prepare_inspection_table_data(result: ModelInspectionResult) -> list[dict]:
    """Prepare table row data for JavaScript."""
    rows = []
    for t in result.layers:
        if t.taxonomy:
            pass

        shape_display = _format_shape(t.shape)

        rows.append(
            {
                "name": t.name,
                "shape": shape_display,
                "dtype": t.dtype,
                "param_count": t.param_count,
                "rel_frobenius": f"{t.norm:.2e}" if t.norm else "—",
                "cosine_sim": "—",
                "max_delta": f"{t.outlier_score:.2f}" if t.outlier_score else "—",
                "signal_rank": t.spectral.signal_rank if t.spectral else None,
                "effective_rank": t.spectral.effective_rank if t.spectral else None,
                "badge": t.taxonomy.badge.name if t.taxonomy else None,
                "status": t.status,
                "dead_neuron_count": t.dead_neuron_count,
                "outlier_score": t.outlier_score,
                "is_outlier_risk": t.is_outlier_risk,
            }
        )
    return rows


def _prepare_inspection_summary_data(result: ModelInspectionResult) -> dict:
    """Prepare summary data for KPI cards."""
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
        "mean_frobenius": s.mean_frobenius,
        "min_cosine_value": s.min_cosine_value,
    }
