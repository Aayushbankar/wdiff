from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from wdiff.engine.diff import DiffResult
from wdiff.engine.inspect import LayerInspectionResult, ModelInspectionResult
from wdiff.engine.taxonomy import TaxonomyBadge


def _format_shape(shape: tuple[int, ...]) -> str:
    if not shape:
        return "—"
    return " × ".join(str(d) for d in shape)


def _format_metric_bar(value: float, max_val: float = 1.0, width: int = 8) -> str:
    """Create a visual bar representation of a metric."""
    if value <= 0:
        return "░" * width
    filled = min(int((value / max_val) * width), width)
    return "█" * filled + "░" * (width - filled)


def _get_status_style(status: str) -> str:
    styles = {
        "MATCH": "green",
        "SHAPE_MISMATCH": "red",
        "ONLY_IN_A": "yellow",
        "ONLY_IN_B": "yellow",
        "PARTIAL_MATCH": "bold yellow",
    }
    return styles.get(status, "white")


def _get_taxonomy_style(badge: TaxonomyBadge) -> str:
    """Return Rich style for taxonomy badge."""
    styles = {
        TaxonomyBadge.UNMODIFIED: "dim",
        TaxonomyBadge.RANK1_ABLATION: "bold magenta",
        TaxonomyBadge.LOW_RANK_ADAPTER: "bold blue",
        TaxonomyBadge.SPARSE_UPDATE: "bold yellow",
        TaxonomyBadge.FULL_RANK_CPT: "bold green",
        TaxonomyBadge.STOCHASTIC_NOISE: "dim red",
    }
    return styles.get(badge, "white")


def _get_health_style(status: str) -> str:
    """Return Rich style for health status."""
    styles = {
        "HEALTHY": "green",
        "DEAD_NEURONS": "yellow",
        "OUTLIER_SPIKE": "bold red",
    }
    return styles.get(status, "white")


def _has_deep_data(result: DiffResult) -> bool:
    return any(t.spectral is not None or t.taxonomy is not None for t in result.tensor_results)


def _format_shape(shape: tuple[int, ...]) -> str:
    if not shape:
        return "—"
    return " × ".join(str(d) for d in shape)


def _format_partial_shape(shape_a: tuple[int, ...], shape_b: tuple[int, ...], partial_info) -> str:
    """Format shape display for partial match showing vocabulary expansion."""
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


def render_diff_table(result: DiffResult, max_rows: int | None = None) -> Table:
    _has_deep_data(result)

    table = Table(
        title=f"wdiff: {result.model_a_path} ↔ {result.model_b_path}",
        show_header=True,
        header_style="bold cyan",
        show_lines=False,
        expand=True,
    )
    table.add_column("Layer", style="dim", min_width=16, ratio=3, overflow="fold")
    table.add_column("Shape", justify="right", style="dim", min_width=10, ratio=2)
    table.add_column("Rel. ΔF", justify="right", style="magenta", width=9)
    table.add_column("Cosine Sim", justify="right", style="blue", width=10)
    table.add_column("Max |Δ|", justify="right", style="yellow", width=8)

    if _has_deep_data(result):
        table.add_column("Rank", justify="center", width=8)
        table.add_column("Class", justify="center", width=14)

    table.add_column("Status", justify="center", width=8)

    rows_to_show = result.tensor_results
    if max_rows and len(rows_to_show) > max_rows:
        rows_to_show = rows_to_show[:max_rows]

    for t in rows_to_show:
        status_style = _get_status_style(t.status)

        if t.metrics:
            frob_bar = _format_metric_bar(t.metrics.relative_frobenius)
            cosine_val = f"{t.metrics.cosine_similarity:.4f}"
            max_delta = f"{t.metrics.max_element_delta:.4f}"
        else:
            frob_bar = "—"
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

        row_data = [
            t.name,
            shape_display,
            frob_bar + f" {t.metrics.relative_frobenius:.4f}" if t.metrics else "—",
            cosine_val,
            max_delta,
        ]

        if _has_deep_data(result):
            if t.spectral:
                rank_str = f"s={t.spectral.signal_rank} e={t.spectral.effective_rank:.1f}"
                if t.taxonomy:
                    _get_taxonomy_style(t.taxonomy.badge)
                    class_str = f"[{_get_taxonomy_style(t.taxonomy.badge)}]{t.taxonomy.badge.value}[/{_get_taxonomy_style(t.taxonomy.badge)}]"
                else:
                    class_str = "—"
            else:
                rank_str = "—"
                class_str = "—"

            row_data.extend([rank_str, class_str])

        row_data.append(Text(t.status, style=status_style))

        table.add_row(*row_data)

    return table


def render_summary_panel(result: DiffResult) -> Panel:
    summary = result.summary
    content = (
        f"[bold]Models:[/bold] {result.model_a_path} ({result.format_a}) vs "
        f"{result.model_b_path} ({result.format_b})\n"
        f"[bold]Duration:[/bold] {result.duration_seconds:.2f}s\n\n"
        f"[bold green]Matched:[/bold green] {summary.matched_count}  "
        f"[bold yellow]Partial:[/bold yellow] {summary.partial_match_count}  "
        f"[bold red]Shape Mismatches:[/bold red] {summary.mismatched_shape_count}  "
        f"[bold yellow]Only in A:[/bold yellow] {summary.only_in_a_count}  "
        f"[bold yellow]Only in B:[/bold yellow] {summary.only_in_b_count}\n\n"
        f"[bold]Mean ΔF:[/bold] {summary.mean_frobenius:.6f}  "
        f"[bold]Max ΔF:[/bold] {summary.max_frobenius_value:.6f} "
        f"([dim]{summary.max_frobenius_tensor}[/dim])  "
        f"[bold]Min Cosine:[/bold] {summary.min_cosine_value:.6f}"
    )
    return Panel(content, title="Summary", border_style="cyan", expand=False)


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


def render_diff(result: DiffResult, console: Console | None = None) -> None:
    if console is None:
        console = Console()

    console.print(render_summary_panel(result))
    console.print()
    console.print(render_diff_table(result))

    # Render lens analysis if present
    lens_layers = [t for t in result.tensor_results if t.lens is not None]
    if lens_layers:
        console.print()
        console.print(render_lens_panel(lens_layers))

    anomalies = detect_anomalies(result)
    if anomalies:
        console.print()
        console.print("[bold red]Anomalies Detected:[/bold red]")
        for anomaly in anomalies:
            console.print(f"  {anomaly}")


def render_lens_panel(lens_layers: list) -> Panel:
    """Render a panel showing logit lens concept decoding results."""
    from rich.table import Table

    table = Table(
        title="[bold cyan]Semantic Concept Lens (Logit Lens)[/bold cyan]",
        show_header=True,
        header_style="bold cyan",
        show_lines=True,
        expand=True,
    )
    table.add_column("Layer", style="dim", min_width=16, ratio=2, overflow="fold")
    table.add_column("σ₁", justify="right", style="magenta", width=8)
    table.add_column("Energy", justify="right", style="blue", width=8)
    table.add_column("Positive Concepts (+)", style="green", min_width=20, ratio=3, overflow="fold")
    table.add_column("Negative Concepts (-)", style="red", min_width=20, ratio=3, overflow="fold")

    for layer in lens_layers:
        lens = layer.lens
        if not lens:
            continue

        pos_tokens = ", ".join(f"{tc.token} ({tc.logit:+.2f})" for tc in lens.positive_tokens)
        neg_tokens = ", ".join(f"{tc.token} ({tc.logit:+.2f})" for tc in lens.negative_tokens)

        table.add_row(
            lens.layer_name,
            f"{lens.singular_value:.4f}",
            f"{lens.energy_ratio:.2%}",
            pos_tokens or "—",
            neg_tokens or "—",
        )

    return Panel(table, title="Logit Lens Concept Decoding", border_style="cyan", expand=False)


# =============================================================================
# Inspection Display
# =============================================================================


def _get_health_status(layer: LayerInspectionResult) -> str:
    """Get health status for a layer."""
    if layer.dead_neuron_count > 0:
        return "DEAD_NEURONS"
    elif layer.is_outlier_risk:
        return "OUTLIER_SPIKE"
    return "HEALTHY"


def _get_readiness_style(readiness: str) -> str:
    """Return Rich style for quantization readiness."""
    if "EXCELLENT" in readiness:
        return "bold green"
    elif "MODERATE" in readiness:
        return "bold yellow"
    elif "POOR" in readiness:
        return "bold red"
    return "white"


def render_inspection_table(result: ModelInspectionResult, max_rows: int | None = None) -> Table:
    has_deep = any(l.spectral is not None for l in result.layers)

    table = Table(
        title=f"winspect: {result.model_path}",
        show_header=True,
        header_style="bold cyan",
        show_lines=False,
    )
    table.add_column("Layer", style="dim", min_width=20, ratio=2, overflow="fold")
    table.add_column("Shape", justify="right", style="dim", min_width=8, ratio=1)
    table.add_column("Dtype", justify="center")
    table.add_column("Params", justify="right")
    table.add_column("L2 Norm", justify="right", style="magenta")
    table.add_column("Outlier", justify="right", style="yellow")
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

        row_data = [
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

    content = (
        f"[bold]Model:[/bold] {result.model_path}\n"
        f"[bold]Format:[/bold] {result.format}\n"
        f"[bold]Architecture:[/bold] {summary.architecture}\n"
        f"[bold]Duration:[/bold] {summary.duration_seconds:.2f}s\n\n"
        f"[bold green]Total Parameters:[/bold green] {summary.total_params:,} "
        f"({summary.trainable_params_mb:.1f} MB)\n"
        f"[bold]Tensors:[/bold] {summary.tensor_count}\n\n"
        f"[bold yellow]Dead Neurons:[/bold yellow] {summary.dead_neurons_total}\n"
        f"[bold red]Outlier Risk Layers:[/bold red] {summary.outlier_risk_layer_count}\n\n"
        f"[bold]Quantization Readiness:[/bold] "
        f"[{readiness_style}]{summary.quantization_readiness}[/{readiness_style}]"
    )
    return Panel(content, title="Inspection Summary", border_style="cyan", expand=False)


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

    console.print(render_inspection_summary_panel(result))
    console.print()
    console.print(render_inspection_table(result))

    anomalies = detect_inspection_anomalies(result)
    if anomalies:
        console.print()
        console.print("[bold red]Anomalies Detected:[/bold red]")
        for anomaly in anomalies:
            console.print(f"  {anomaly}")
