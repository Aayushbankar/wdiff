from __future__ import annotations

from math import ceil

from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from wdiff.engine.diff import DiffResult, DiffSummary
from wdiff.engine.inspect import LayerInspectionResult, ModelInspectionResult
from wdiff.engine.taxonomy import TaxonomyBadge

# Spectral forensics palette: distinct hues per structural class
TAXONOMY_STYLE = {
    TaxonomyBadge.UNMODIFIED: "dim #6b7280",
    TaxonomyBadge.RANK1_ABLATION: "bold #e879f9",
    TaxonomyBadge.LOW_RANK_ADAPTER: "bold #60a5fa",
    TaxonomyBadge.SPARSE_UPDATE: "bold #fbbf24",
    TaxonomyBadge.FULL_RANK_CPT: "bold #34d399",
    TaxonomyBadge.STOCHASTIC_NOISE: "dim #f87171",
}

TAXONOMY_BAR = {
    TaxonomyBadge.UNMODIFIED: "#374151",
    TaxonomyBadge.RANK1_ABLATION: "#e879f9",
    TaxonomyBadge.LOW_RANK_ADAPTER: "#60a5fa",
    TaxonomyBadge.SPARSE_UPDATE: "#fbbf24",
    TaxonomyBadge.FULL_RANK_CPT: "#34d399",
    TaxonomyBadge.STOCHASTIC_NOISE: "#f87171",
}

STATUS_STYLE = {
    "MATCH": "#34d399",
    "SHAPE_MISMATCH": "#f87171",
    "ONLY_IN_A": "#fbbf24",
    "ONLY_IN_B": "#fbbf24",
    "PARTIAL_MATCH": "bold #fb923c",
}


def _format_shape(shape: tuple[int, ...]) -> str:
    if not shape:
        return "—"
    return " × ".join(str(d) for d in shape)


def _format_partial_shape(shape_a: tuple[int, ...], shape_b: tuple[int, ...], partial_info) -> str:
    if partial_info is None or partial_info.differing_axis is None:
        return f"{_format_shape(shape_a)} → {_format_shape(shape_b)}"

    axis = partial_info.differing_axis
    size_a = shape_a[axis] if axis < len(shape_a) else 0
    size_b = shape_b[axis] if axis < len(shape_b) else 0

    if len(shape_a) == len(shape_b):
        dims = []
        for i in range(len(shape_a)):
            if i == axis:
                dims.append(f"{size_a:,}→{size_b:,}")
            else:
                dims.append(f"{shape_a[i]:,}")
        return " × ".join(dims)

    return f"{_format_shape(shape_a)} → {_format_shape(shape_b)}"


def _delta_bar(value: float, width: int = 10) -> Text:
    """Compact heat bar for relative Frobenius delta."""
    if value <= 0:
        return Text("▁" * width, style="dim #374151")
    # log-ish scaling so small-but-real deltas stay visible
    scaled = min(1.0, (value / 0.20) ** 0.5) if value < 0.20 else min(1.0, 0.7 + value)
    filled = max(1, min(width, ceil(scaled * width)))
    if value < 0.01:
        color = "#34d399"
    elif value < 0.05:
        color = "#60a5fa"
    elif value < 0.10:
        color = "#fbbf24"
    else:
        color = "#f87171"
    t = Text()
    t.append("█" * filled, style=color)
    t.append("▁" * (width - filled), style="dim #374151")
    return t


def _get_status_style(status: str) -> str:
    return STATUS_STYLE.get(status, "white")


def _get_taxonomy_style(badge: TaxonomyBadge) -> str:
    return TAXONOMY_STYLE.get(badge, "white")


def _get_health_style(status: str) -> str:
    styles = {
        "HEALTHY": "#34d399",
        "DEAD_NEURONS": "#fbbf24",
        "OUTLIER_SPIKE": "bold #f87171",
    }
    return styles.get(status, "white")


def _has_deep_data(result: DiffResult) -> bool:
    return any(t.spectral is not None or t.taxonomy is not None for t in result.tensor_results)


def _format_peak_ram(mb: float) -> str:
    if not mb:
        return "—"
    if mb >= 1024:
        return f"{mb / 1024:.2f} GB"
    return f"{mb:.0f} MB"


# -----------------------------------------------------------------------------
# Diff rendering
# -----------------------------------------------------------------------------


def render_diff_table(
    result: DiffResult, max_rows: int | None = None, show_geometry: bool = True
) -> Table:
    has_deep = _has_deep_data(result)

    table = Table(
        title="Layer-by-layer delta",
        title_style="bold #e5e7eb",
        show_header=True,
        header_style="bold #93c5fd",
        box=box.SIMPLE_HEAVY,
        show_lines=False,
        expand=True,
        row_styles=["", "dim"],
        pad_edge=False,
    )
    table.add_column(
        "Layer", style="#d1d5db", min_width=18, ratio=3, overflow="fold", no_wrap=False
    )
    table.add_column("Shape", justify="right", style="#9ca3af", min_width=10, ratio=2)
    table.add_column("Rel. ΔF", justify="left", style="#e5e7eb", min_width=18, ratio=2)
    table.add_column("Cosine Sim", justify="right", style="#93c5fd", width=10)
    table.add_column("Max |Δ|", justify="right", style="#fcd34d", width=10)

    if has_deep:
        table.add_column("Rank", justify="center", width=10)
        if show_geometry:
            table.add_column("Geometry", justify="center", width=14)
        table.add_column("Classification", justify="center", width=16)

    table.add_column("Status", justify="center", width=10)

    rows_to_show = result.tensor_results
    if max_rows and len(rows_to_show) > max_rows:
        rows_to_show = rows_to_show[:max_rows]

    for t in rows_to_show:
        status_style = _get_status_style(t.status)

        if t.metrics:
            frob_cell = Text()
            frob_cell.append_text(_delta_bar(t.metrics.relative_frobenius))
            frob_cell.append(f" {t.metrics.relative_frobenius:.4f}", style="#e5e7eb")
            cosine_val = f"{t.metrics.cosine_similarity:.4f}"
            max_delta = f"{t.metrics.max_element_delta:.4f}"
        else:
            frob_cell = Text("—", style="dim")
            cosine_val = "—"
            max_delta = "—"

        if t.status == "PARTIAL_MATCH" and t.partial_info:
            shape_display = _format_partial_shape(t.shape_a, t.shape_b, t.partial_info)
        else:
            shape_display = (
                _format_shape(t.shape_a)
                if t.shape_a == t.shape_b
                else f"{_format_shape(t.shape_a)} → {_format_shape(t.shape_b)}"
            )

        row_data: list = [
            t.name,
            shape_display,
            frob_cell,
            cosine_val,
            max_delta,
        ]

        if has_deep:
            if t.spectral:
                rank_str = f"s={t.spectral.signal_rank} e={t.spectral.effective_rank:.1f}"
                if t.taxonomy:
                    class_text = Text(
                        t.taxonomy.badge.value,
                        style=_get_taxonomy_style(t.taxonomy.badge),
                    )
                else:
                    class_text = Text("—", style="dim")
            else:
                rank_str = "—"
                class_text = Text("—", style="dim")

            if show_geometry:
                subspace = getattr(t, "subspace", None)
                if subspace:
                    drift = subspace.max_principal_angle > 25.0
                    gstyle = "bold #f87171" if drift else "#a7f3d0"
                    geom_text = Text(
                        f"θ={subspace.max_principal_angle:.0f}° F={subspace.uhlmann_fidelity:.2f}",
                        style=gstyle,
                    )
                else:
                    geom_text = Text("—", style="dim")
                row_data.extend([rank_str, geom_text, class_text])
            else:
                row_data.extend([rank_str, class_text])

        row_data.append(Text(t.status, style=status_style))
        table.add_row(*row_data)

    return table


def render_summary_panel(result: DiffResult) -> Panel:
    summary = result.summary
    peak = getattr(result, "peak_ram_mb", 0.0) or 0.0

    # Plain-text summary line (keeps legacy tokens like "Matched: 3" and format names
    # discoverable via str(panel.renderable) for tests/consumers)
    legacy = (
        f"Models: {result.model_a_path} ({result.format_a}) vs "
        f"{result.model_b_path} ({result.format_b})\n"
        f"Matched: {summary.matched_count}  Partial: {summary.partial_match_count}  "
        f"Mismatched: {summary.mismatched_shape_count}  "
        f"Only A: {summary.only_in_a_count}  Only B: {summary.only_in_b_count}\n"
        f"Mean ΔF: {summary.mean_frobenius:.6f}  "
        f"Max ΔF: {summary.max_frobenius_value:.6f}  "
        f"Min Cosine: {summary.min_cosine_value:.6f}  "
        f"Peak RSS: {_format_peak_ram(peak)}"
    )

    header = Table.grid(expand=True)
    header.add_column(ratio=3)
    header.add_column(ratio=2, justify="right")
    header.add_row(
        Text.assemble(
            ("wdiff  ", "bold #93c5fd"),
            ("spectral checkpoint diff", "dim #9ca3af"),
        ),
        Text(f"{result.duration_seconds:.2f}s", style="bold #e5e7eb"),
    )

    models = Table.grid(padding=(0, 1))
    models.add_column(style="dim #9ca3af", width=6)
    models.add_column()
    models.add_row("A", f"{result.model_a_path} ({result.format_a})")
    models.add_row("B", f"{result.model_b_path} ({result.format_b})")

    stats = Table(box=None, show_header=False, pad_edge=False, padding=(0, 2))
    for _ in range(6):
        stats.add_column()
    stats.add_row(
        Text.assemble(("Matched: ", "dim"), (str(summary.matched_count), "bold #34d399")),
        Text.assemble(("Partial: ", "dim"), (str(summary.partial_match_count), "bold #fb923c")),
        Text.assemble(
            ("Mismatched: ", "dim"), (str(summary.mismatched_shape_count), "bold #f87171")
        ),
        Text.assemble(("Only A: ", "dim"), (str(summary.only_in_a_count), "bold #fbbf24")),
        Text.assemble(("Only B: ", "dim"), (str(summary.only_in_b_count), "bold #fbbf24")),
        Text.assemble(("Peak RSS: ", "dim"), (_format_peak_ram(peak), "bold #c4b5fd")),
    )

    metrics = Table(box=None, show_header=False, pad_edge=False, padding=(0, 2))
    for _ in range(3):
        metrics.add_column()
    metrics.add_row(
        Text.assemble(("Mean ΔF: ", "dim"), (f"{summary.mean_frobenius:.6f}", "bold #e5e7eb")),
        Text.assemble(
            ("Max ΔF: ", "dim"),
            (
                f"{summary.max_frobenius_value:.6f}",
                "bold #f87171" if summary.max_frobenius_value > 0.1 else "bold #e5e7eb",
            ),
            ("  ", ""),
            (str(summary.max_frobenius_tensor or "—"), "dim #9ca3af"),
        ),
        Text.assemble(("Min Cosine: ", "dim"), (f"{summary.min_cosine_value:.6f}", "bold #93c5fd")),
    )

    inner = Group(header, Text(""), models, Text(""), stats, metrics)

    class _SummaryRenderable:
        def __init__(self, renderable, text: str) -> None:
            self._renderable = renderable
            self._text = text

        def __rich_console__(self, console, options):
            yield from self._renderable.__rich_console__(console, options)

        def __str__(self) -> str:
            return self._text

    return Panel(
        _SummaryRenderable(inner, legacy),
        title="[bold #93c5fd]Summary[/]",
        border_style="#374151",
        padding=(1, 2),
    )


def _taxonomy_distribution(summary: DiffSummary, result: DiffResult) -> str:
    counts: dict[str, int] = {}
    for t in result.tensor_results:
        if t.taxonomy:
            key = t.taxonomy.badge.value
            counts[key] = counts.get(key, 0) + 1
    if not counts:
        return ""
    parts = [f"{k} {v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])]
    return " · ".join(parts)


def detect_anomalies(result: DiffResult) -> list[str]:
    anomalies = []

    for t in result.tensor_results:
        if t.metrics:
            if t.metrics.cosine_similarity < 0.1:
                anomalies.append(
                    f"[red]⚠ Near-orthogonal shift[/red]: {t.name} "
                    f"(cosine={t.metrics.cosine_similarity:.4f})"
                )
            if t.metrics.relative_frobenius > 0.10:
                anomalies.append(
                    f"[yellow]⚠ Significant change[/yellow]: {t.name} "
                    f"(ΔF={t.metrics.relative_frobenius:.2%})"
                )

            # Deep anomalies
            if t.taxonomy:
                if t.taxonomy.badge == TaxonomyBadge.RANK1_ABLATION:
                    anomalies.append(
                        f"[bold magenta]⚠ Rank-1 Ablation Detected[/bold magenta]: {t.name} "
                        f"({t.taxonomy.confidence:.0%} confidence)"
                    )
                elif t.taxonomy.badge == TaxonomyBadge.STOCHASTIC_NOISE:
                    anomalies.append(
                        f"[dim red]⚠ Stochastic Noise[/dim red]: {t.name} "
                        f"classified as pure SGD noise"
                    )

            # Subspace geometry anomalies (learning/04 thresholds)
            subspace = getattr(t, "subspace", None)
            if subspace and subspace.max_principal_angle > 25.0:
                anomalies.append(
                    f"[bold red]⚠ Representation Drift[/bold red]: {t.name} "
                    f"(θmax={subspace.max_principal_angle:.1f}°, "
                    f"fid={subspace.uhlmann_fidelity:.3f})"
                )

        if t.status == "SHAPE_MISMATCH":
            anomalies.append(f"[red]⚠ Shape mismatch[/red]: {t.name} ({t.shape_a} vs {t.shape_b})")
        if t.status in ("ONLY_IN_A", "ONLY_IN_B"):
            anomalies.append(f"[yellow]⚠ Asymmetric key[/yellow]: {t.name} ({t.status})")
        if t.status == "PARTIAL_MATCH" and t.partial_info:
            pi = t.partial_info
            anomalies.append(
                f"[yellow]⚠ Vocabulary expansion[/yellow]: {t.name} "
                f"(+{pi.extra_count:,} new tokens, norm={pi.extra_norm:.2e})"
            )

    return anomalies


def render_architecture_topology(result: DiffResult) -> Panel | None:
    """Render a clean Unicode architecture topology pipeline."""
    if not result.tensor_results:
        return None

    import re

    names = [t.name.lower() for t in result.tensor_results]

    has_embed = any("embed" in n or "wte" in n or "token" in n for n in names)
    has_head = any("head" in n or "output" in n for n in names)

    layer_indices = set()
    for n in names:
        m = re.search(r"(?:layers?|blk|h)\.(\d+)\.", n)
        if m:
            layer_indices.add(int(m.group(1)))

    num_blocks = (
        len(layer_indices) if layer_indices else (1 if any("attn" in n for n in names) else 0)
    )
    has_attn = any("attn" in n or "_q" in n or "_k" in n or "_v" in n for n in names)
    has_mlp = any(
        "ffn" in n or "mlp" in n or "gate" in n or "up" in n or "down" in n for n in names
    )

    flow = Text()
    flow.append("  ", "")
    if has_embed:
        flow.append("[Input Embedding]", "bold #38bdf8")
        flow.append(" ──► ", "#475569")

    block_desc = []
    if has_attn:
        block_desc.append("Attn (Q, K, V, O)")
    if has_mlp:
        block_desc.append("MLP (Gate, Up, Down)")
    sub_desc = " ─► ".join(block_desc) if block_desc else "Transform"

    block_range = (
        f"0..{max(layer_indices)}" if len(layer_indices) > 1 else ("0" if layer_indices else "0..N")
    )
    flow.append(f"[Block {block_range}: {sub_desc}]", "bold #a855f7")

    if has_head:
        flow.append(" ──► ", "#475569")
        flow.append("[Unembedding Head]", "bold #10b981")

    sub_text = Text()
    sub_text.append(f"  {result.summary.matched_count} layers matched", "dim #94a3b8")
    if num_blocks > 0:
        sub_text.append(
            f"  ·  {num_blocks} transformer block{'s' if num_blocks > 1 else ''}", "dim #94a3b8"
        )
    if _has_deep_data(result):
        sub_text.append("  ·  spectral forensics & subspace geometry active", "dim #38bdf8")

    body = Group(flow, sub_text)
    return Panel(
        body,
        title="[bold #38bdf8]Architecture Topology[/]",
        border_style="#334155",
        padding=(0, 1),
    )


def render_diff(result: DiffResult, console: Console | None = None) -> None:
    if console is None:
        console = Console()

    console.print()
    console.print(render_summary_panel(result))

    topo = render_architecture_topology(result)
    if topo:
        console.print(topo)

    dist = _taxonomy_distribution(result.summary, result)
    if dist:
        console.print(
            Panel(
                dist,
                title="[dim]Taxonomy mix · Rank & Classification (deep)[/]",
                border_style="#374151",
            )
        )

    console.print()
    console.print(render_diff_table(result, show_geometry=console.width >= 110))

    # Render lens analysis if present
    lens_layers = [t for t in result.tensor_results if t.lens is not None]
    if lens_layers:
        console.print()
        console.print(render_lens_panel(lens_layers))

    anomalies = detect_anomalies(result)
    if anomalies:
        console.print()
        lines = [Text.from_markup(a) for a in anomalies]
        console.print(
            Panel(
                Group(*lines),
                title="[bold #f87171]Anomalies[/]",
                border_style="#7f1d1d",
                padding=(0, 1),
            )
        )
    console.print()


def render_lens_panel(lens_layers: list) -> Panel:
    """Render a panel showing logit lens concept decoding results."""

    table = Table(
        show_header=True,
        header_style="bold #93c5fd",
        box=box.SIMPLE_HEAVY,
        show_lines=False,
        expand=True,
        pad_edge=False,
    )
    table.add_column("Layer", style="#d1d5db", min_width=16, ratio=2, overflow="fold")
    table.add_column("σ₁", justify="right", style="#e879f9", width=8)
    table.add_column("Energy", justify="right", style="#93c5fd", width=8)
    table.add_column("Promoted (+)", style="#34d399", min_width=20, ratio=3, overflow="fold")
    table.add_column("Suppressed (−)", style="#f87171", min_width=20, ratio=3, overflow="fold")

    for layer in lens_layers:
        lens = layer.lens
        if not lens:
            continue

        pos_tokens = ", ".join(f"{tc.token} ({tc.logit:+.2f})" for tc in lens.positive_tokens)
        neg_tokens = ", ".join(f"{tc.token} ({tc.logit:+.2f})" for tc in lens.negative_tokens)

        table.add_row(
            lens.layer_name,
            f"{lens.singular_value:.4f}",
            f"{lens.energy_ratio:.1%}",
            pos_tokens or "—",
            neg_tokens or "—",
        )

    return Panel(
        table,
        title="[bold #93c5fd]Logit Lens Concept Decoding[/]",
        border_style="#374151",
        padding=(0, 1),
    )


# -----------------------------------------------------------------------------
# Inspection rendering
# -----------------------------------------------------------------------------


def _get_health_status(layer: LayerInspectionResult) -> str:
    if layer.dead_neuron_count > 0:
        return "DEAD_NEURONS"
    elif layer.is_outlier_risk:
        return "OUTLIER_SPIKE"
    return "HEALTHY"


def _get_readiness_style(readiness: str) -> str:
    if "EXCELLENT" in readiness:
        return "bold #34d399"
    elif "MODERATE" in readiness:
        return "bold #fbbf24"
    elif "POOR" in readiness:
        return "bold #f87171"
    return "white"


def render_inspection_table(result: ModelInspectionResult, max_rows: int | None = None) -> Table:
    has_deep = any(l.spectral is not None for l in result.layers)

    table = Table(
        title="Layer health",
        title_style="bold #e5e7eb",
        show_header=True,
        header_style="bold #93c5fd",
        box=box.SIMPLE_HEAVY,
        show_lines=False,
        expand=True,
        row_styles=["", "dim"],
        pad_edge=False,
    )
    table.add_column("Layer", style="#d1d5db", min_width=20, ratio=2, overflow="fold")
    table.add_column("Shape", justify="right", style="#9ca3af", min_width=8, ratio=1)
    table.add_column("Dtype", justify="center")
    table.add_column("Params", justify="right")
    table.add_column("L2 Norm", justify="right", style="#e879f9")
    table.add_column("Outlier", justify="right", style="#fcd34d")
    table.add_column("Dead", justify="center")

    if has_deep:
        table.add_column("Rank", justify="center")

    table.add_column("Status", justify="center")

    rows_to_show = result.layers
    if max_rows and len(rows_to_show) > max_rows:
        rows_to_show = rows_to_show[:max_rows]

    for t in rows_to_show:
        health = _get_health_status(t)

        dtype_display = str(t.dtype)
        if dtype_display == "float32":
            dtype_display = "f32"
        elif dtype_display == "float16":
            dtype_display = "f16"
        elif dtype_display == "bfloat16":
            dtype_display = "bf16"

        row_data: list = [
            t.name,
            _format_shape(t.shape),
            dtype_display,
            f"{t.param_count:,}",
            f"{t.norm:.2e}",
            f"{t.outlier_score:.2f}",
            str(t.dead_neuron_count),
        ]

        if has_deep:
            if t.spectral:
                rank_str = f"s={t.spectral.signal_rank} e={t.spectral.effective_rank:.1f}"
            else:
                rank_str = "—"
            row_data.append(rank_str)

        row_data.append(Text(health, style=_get_health_style(health)))
        table.add_row(*row_data)

    return table


def render_inspection_summary_panel(result: ModelInspectionResult) -> Panel:
    summary = result.summary
    readiness_style = _get_readiness_style(summary.quantization_readiness)

    header = Table.grid(expand=True)
    header.add_column(ratio=3)
    header.add_column(ratio=2, justify="right")
    header.add_row(
        Text.assemble(("wdiff inspect  ", "bold #93c5fd"), ("model health", "dim #9ca3af")),
        Text(f"{summary.duration_seconds:.2f}s", style="bold #e5e7eb"),
    )

    meta = Table.grid(padding=(0, 1))
    meta.add_column(style="dim #9ca3af", width=12)
    meta.add_column()
    meta.add_row("Model", str(result.model_path))
    meta.add_row("Format", str(result.format))
    meta.add_row("Architecture", str(summary.architecture))

    stats = Table(box=None, show_header=False, pad_edge=False, padding=(0, 2))
    for _ in range(4):
        stats.add_column()
    stats.add_row(
        Text.assemble(("Parameters: ", "dim"), (f"{summary.total_params:,}", "bold #34d399")),
        Text.assemble(("Size: ", "dim"), (f"{summary.trainable_params_mb:.1f} MB", "bold #e5e7eb")),
        Text.assemble(("Tensors: ", "dim"), (str(summary.tensor_count), "bold #e5e7eb")),
        Text.assemble(("Dead neurons: ", "dim"), (str(summary.dead_neurons_total), "bold #fbbf24")),
    )

    risk = Text.assemble(
        ("Outlier-risk layers: ", "dim"),
        (str(summary.outlier_risk_layer_count), "bold #f87171"),
        ("   Quantization readiness: ", "dim"),
        (summary.quantization_readiness, readiness_style),
    )

    body = Group(header, Text(""), meta, Text(""), stats, risk)
    return Panel(
        body,
        title="[bold #93c5fd]Inspection Summary[/]",
        border_style="#374151",
        padding=(1, 2),
    )


def detect_inspection_anomalies(result: ModelInspectionResult) -> list[str]:
    anomalies = []

    for t in result.layers:
        if t.dead_neuron_count > 0:
            anomalies.append(
                f"[yellow]⚠ Dead neurons[/yellow]: {t.name} has {t.dead_neuron_count} dead neuron(s)"
            )
        if t.is_outlier_risk:
            anomalies.append(
                f"[bold red]⚠ Outlier spike[/bold red]: {t.name} (score={t.outlier_score:.2f})"
            )
        if t.spectral and t.spectral.primary_direction_energy > 0.90 and len(t.shape) >= 2:
            anomalies.append(
                f"[bold magenta]⚠ Rank-1 dominant[/bold magenta]: {t.name} "
                f"(E1={t.spectral.primary_direction_energy:.1%})"
            )

    if result.summary.outlier_risk_layer_count > 3:
        anomalies.append(
            f"[bold red]⚠ High quantization risk[/bold red]: "
            f"{result.summary.outlier_risk_layer_count} layers with outlier spikes"
        )

    return anomalies


def render_inspection(result: ModelInspectionResult, console: Console | None = None) -> None:
    if console is None:
        console = Console()

    console.print()
    console.print(render_inspection_summary_panel(result))
    console.print()
    console.print(render_inspection_table(result))

    anomalies = detect_inspection_anomalies(result)
    if anomalies:
        console.print()
        lines = [Text.from_markup(a) for a in anomalies]
        console.print(
            Panel(
                Group(*lines),
                title="[bold #f87171]Anomalies[/]",
                border_style="#7f1d1d",
                padding=(0, 1),
            )
        )
    console.print()
