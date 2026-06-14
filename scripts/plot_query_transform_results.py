from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DEFAULT_REPORT_DIR = Path("reports/model-matrix-crud-rag-compact-e2e-structured-judge")
DEFAULT_OUTPUT_DIR = Path("docs/assets/query-transformations")

METHOD_ORDER = [
    "baseline",
    "rewrite",
    "expand",
    "multi_query",
    "child_query",
    "hyde",
    "step_back",
]

PROVIDER_ORDER = ["gemini", "openrouter", "nvidia_nim", "openai"]

METHOD_LABELS = {
    "baseline": "Baseline",
    "rewrite": "Rewrite",
    "expand": "Expand",
    "multi_query": "Multi Query",
    "child_query": "Child Query",
    "hyde": "HyDE",
    "step_back": "Step-Back",
}

PROVIDER_LABELS = {
    "gemini": "Gemini",
    "openrouter": "OpenRouter",
    "nvidia_nim": "NVIDIA NIM",
    "openai": "OpenAI",
}

METRIC_LABELS = {
    "intent_preservation": "Intent",
    "clarity_enhancement": "Clarity",
    "answer_preference": "Answer Preference",
    "faithfulness": "Faithfulness",
    "recall_at_5": "Recall@5",
    "parse_ok": "Parse OK",
    "total_latency_ms": "Latency (ms)",
    "estimated_cost_usd": "Estimated Cost (USD)",
    "query_count": "Query Count",
}

IBM_COLORS = {
    "blue": "#0f62fe",
    "cyan": "#1192e8",
    "teal": "#009d9a",
    "green": "#24a148",
    "purple": "#8a3ffc",
    "magenta": "#ee538b",
    "red": "#da1e28",
    "orange": "#ff832b",
    "gray": "#6f6f6f",
}

METHOD_PALETTE = {
    "baseline": IBM_COLORS["gray"],
    "rewrite": IBM_COLORS["blue"],
    "expand": IBM_COLORS["cyan"],
    "multi_query": IBM_COLORS["teal"],
    "child_query": IBM_COLORS["purple"],
    "hyde": IBM_COLORS["orange"],
    "step_back": IBM_COLORS["red"],
}

PROVIDER_PALETTE = {
    "gemini": IBM_COLORS["blue"],
    "openrouter": IBM_COLORS["teal"],
    "nvidia_nim": IBM_COLORS["purple"],
    "openai": IBM_COLORS["orange"],
}

ANNOTATION_BOX = {
    "boxstyle": "round,pad=0.18",
    "facecolor": "white",
    "edgecolor": "#e0e0e0",
    "alpha": 0.92,
}

ANNOTATION_ARROW = {
    "arrowstyle": "-",
    "color": "#6f6f6f",
    "linewidth": 0.8,
    "shrinkA": 2,
    "shrinkB": 5,
}

LATENCY_LABEL_OFFSETS = {
    "baseline": (12, 8),
    "rewrite": (-58, 14),
    "expand": (-46, 28),
    "multi_query": (22, 12),
    "child_query": (14, 8),
    "hyde": (12, -20),
    "step_back": (12, -20),
}

COST_LABEL_OFFSETS = {
    "baseline": (12, 10),
    "rewrite": (-56, 20),
    "expand": (-34, 34),
    "multi_query": (22, 8),
    "child_query": (14, 12),
    "hyde": (12, -6),
    "step_back": (12, -18),
}


ARTICLE_METHOD_VALUES = {
    "child_query": {
        "intent_preservation": 0.940,
        "clarity_enhancement": 0.898,
        "answer_preference": 0.615,
        "faithfulness": 0.985,
        "recall_at_5": 1.000,
        "query_count": 10.84,
        "total_latency_ms": 4210,
    },
    "expand": {
        "intent_preservation": 0.740,
        "clarity_enhancement": 0.652,
        "answer_preference": 0.612,
        "faithfulness": 0.978,
        "recall_at_5": 0.996,
        "query_count": 6.01,
        "total_latency_ms": 3005,
    },
    "multi_query": {
        "intent_preservation": 0.902,
        "clarity_enhancement": 0.885,
        "answer_preference": 0.610,
        "faithfulness": 0.985,
        "recall_at_5": 1.000,
        "query_count": 4.18,
        "total_latency_ms": 2813,
    },
    "rewrite": {
        "intent_preservation": 0.975,
        "clarity_enhancement": 0.910,
        "answer_preference": 0.610,
        "faithfulness": 0.982,
        "recall_at_5": 1.000,
        "query_count": 1.00,
        "total_latency_ms": 2455,
    },
    "baseline": {
        "intent_preservation": 1.000,
        "clarity_enhancement": 0.600,
        "answer_preference": 0.605,
        "faithfulness": 0.985,
        "recall_at_5": 1.000,
        "query_count": 1.00,
        "total_latency_ms": 1602,
    },
    "hyde": {
        "intent_preservation": 0.782,
        "clarity_enhancement": 0.715,
        "answer_preference": 0.590,
        "faithfulness": 0.945,
        "recall_at_5": 1.000,
        "query_count": 4.01,
        "total_latency_ms": 4347,
    },
    "step_back": {
        "intent_preservation": 0.555,
        "clarity_enhancement": 0.572,
        "answer_preference": 0.590,
        "faithfulness": 0.928,
        "recall_at_5": 0.977,
        "query_count": 2.96,
        "total_latency_ms": 3467,
    },
}

ARTICLE_BASELINE_DELTA_VALUES = {
    "child_query": {
        "answer_preference_delta": 0.010,
        "faithfulness_delta": 0.000,
        "latency_delta_ms": 2608,
        "query_count_delta": 9.84,
    },
    "expand": {
        "answer_preference_delta": 0.007,
        "faithfulness_delta": -0.007,
        "latency_delta_ms": 1403,
        "query_count_delta": 5.01,
    },
    "multi_query": {
        "answer_preference_delta": 0.005,
        "faithfulness_delta": 0.000,
        "latency_delta_ms": 1211,
        "query_count_delta": 3.18,
    },
    "rewrite": {
        "answer_preference_delta": 0.005,
        "faithfulness_delta": -0.003,
        "latency_delta_ms": 853,
        "query_count_delta": 0.00,
    },
    "hyde": {
        "answer_preference_delta": -0.015,
        "faithfulness_delta": -0.040,
        "latency_delta_ms": 2745,
        "query_count_delta": 3.01,
    },
    "step_back": {
        "answer_preference_delta": -0.015,
        "faithfulness_delta": -0.057,
        "latency_delta_ms": 1865,
        "query_count_delta": 1.96,
    },
}

ARTICLE_PROVIDER_VALUES = {
    "gemini": {
        "model_id": "gemini_fast",
        "answer_preference": 0.613,
        "faithfulness": 0.974,
        "parse_ok": 0.993,
        "total_latency_ms": 2460,
    },
    "openrouter": {
        "model_id": "openrouter_fast",
        "answer_preference": 0.607,
        "faithfulness": 0.960,
        "parse_ok": 0.979,
        "total_latency_ms": 3127,
    },
    "nvidia_nim": {
        "model_id": "nvidia_nim_fast",
        "answer_preference": 0.604,
        "faithfulness": 0.980,
        "parse_ok": 0.986,
        "total_latency_ms": 3816,
    },
    "openai": {
        "model_id": "openai_fast",
        "answer_preference": 0.594,
        "faithfulness": 0.964,
        "parse_ok": 0.971,
        "total_latency_ms": 3110,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate article-ready seaborn charts for query transformation results."
    )
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def configure_theme() -> None:
    sns.set_theme(
        context="talk",
        style="whitegrid",
        font="sans-serif",
        rc={
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#525252",
            "axes.labelcolor": "#161616",
            "axes.titlecolor": "#161616",
            "grid.color": "#e0e0e0",
            "grid.linewidth": 0.8,
            "xtick.color": "#161616",
            "ytick.color": "#161616",
            "font.sans-serif": [
                "IBM Plex Sans",
                "PingFang TC",
                "Arial Unicode MS",
                "Helvetica",
                "DejaVu Sans",
            ],
        },
    )


def read_records(report_dir: Path) -> pd.DataFrame:
    records_path = report_dir / "records.jsonl"
    if not records_path.exists():
        raise FileNotFoundError(f"Missing records file: {records_path}")
    records = pd.read_json(records_path, lines=True)
    required_columns = {
        "provider",
        "model_id",
        "method",
        "intent_preservation",
        "clarity_enhancement",
        "recall_at_5",
        "gold_doc_hit_count",
        "answer_preference",
        "faithfulness",
        "total_latency_ms",
        "estimated_cost_usd",
        "billed_cost_usd",
        "parse_ok",
        "query_count",
    }
    missing = required_columns - set(records.columns)
    if missing:
        raise ValueError(f"Missing columns in {records_path}: {sorted(missing)}")
    return records


def summarize_by_method(records: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, group in records.groupby("method", sort=False):
        rows.append(
            {
                "method": method,
                "method_label": METHOD_LABELS[method],
                "cases": int(len(group)),
                "intent_preservation": float(group["intent_preservation"].mean()),
                "clarity_enhancement": float(group["clarity_enhancement"].mean()),
                "answer_preference": float(group["answer_preference"].mean()),
                "faithfulness": float(group["faithfulness"].mean()),
                "recall_at_5": float(group["recall_at_5"].mean()),
                "gold_doc_hit_count": float(group["gold_doc_hit_count"].mean()),
                "parse_ok": float(group["parse_ok"].mean()),
                "query_count": float(group["query_count"].mean()),
                "total_latency_ms": float(group["total_latency_ms"].mean()),
                "estimated_cost_usd": float(group["estimated_cost_usd"].sum()),
                "billed_cost_usd": float(group["billed_cost_usd"].sum()),
            }
        )
    output = pd.DataFrame(rows)
    output["method"] = pd.Categorical(output["method"], METHOD_ORDER, ordered=True)
    return output.sort_values("method").reset_index(drop=True)


def summarize_by_provider(records: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for provider, group in records.groupby("provider", sort=False):
        rows.append(
            {
                "provider": provider,
                "provider_label": PROVIDER_LABELS[provider],
                "model_id": group["model_id"].iloc[0],
                "cases": int(len(group)),
                "intent_preservation": float(group["intent_preservation"].mean()),
                "clarity_enhancement": float(group["clarity_enhancement"].mean()),
                "answer_preference": float(group["answer_preference"].mean()),
                "faithfulness": float(group["faithfulness"].mean()),
                "parse_ok": float(group["parse_ok"].mean()),
                "total_latency_ms": float(group["total_latency_ms"].mean()),
                "estimated_cost_usd": float(group["estimated_cost_usd"].sum()),
                "billed_cost_usd": float(group["billed_cost_usd"].sum()),
            }
        )
    output = pd.DataFrame(rows)
    output["provider"] = pd.Categorical(output["provider"], PROVIDER_ORDER, ordered=True)
    return output.sort_values("provider").reset_index(drop=True)


def summarize_baseline_delta(method_summary: pd.DataFrame) -> pd.DataFrame:
    baseline = method_summary.loc[method_summary["method"] == "baseline"].iloc[0]
    rows = []
    for row in method_summary.itertuples(index=False):
        if row.method == "baseline":
            continue
        rows.append(
            {
                "method": row.method,
                "method_label": row.method_label,
                "answer_preference_delta": row.answer_preference
                - baseline.answer_preference,
                "faithfulness_delta": row.faithfulness - baseline.faithfulness,
                "latency_delta_ms": row.total_latency_ms - baseline.total_latency_ms,
                "query_count_delta": row.query_count - baseline.query_count,
            }
        )
    output = pd.DataFrame(rows)
    output["method"] = pd.Categorical(output["method"], METHOD_ORDER, ordered=True)
    return output.sort_values("method").reset_index(drop=True)


def round_for_article(metric: str, value: float) -> float:
    if metric in {"total_latency_ms", "latency_delta_ms"}:
        return float(pd.Series([value]).round(0).iloc[0])
    if metric in {"query_count", "query_count_delta"}:
        return float(pd.Series([value]).round(2).iloc[0])
    if metric == "model_id":
        return value
    return float(pd.Series([value]).round(3).iloc[0])


def tolerance_for_article(metric: str) -> float:
    if metric in {"total_latency_ms", "latency_delta_ms"}:
        return 1.0
    if metric in {"query_count", "query_count_delta"}:
        return 0.01
    if metric == "model_id":
        return 0.0
    return 0.001


def compare_values(
    table_name: str,
    article_values: dict[str, dict[str, float | str]],
    current: pd.DataFrame,
    key_column: str,
) -> pd.DataFrame:
    rows = []
    current_index = current.set_index(key_column)
    for row_key, expected_values in article_values.items():
        for metric, article_value in expected_values.items():
            current_value = current_index.loc[row_key, metric]
            if isinstance(article_value, str):
                rounded_current = str(current_value)
                diff = ""
                exact_display_match = article_value == rounded_current
                within_article_precision = exact_display_match
            else:
                rounded_current = round_for_article(metric, float(current_value))
                diff = float(rounded_current) - float(article_value)
                exact_display_match = abs(diff) <= 1e-9
                within_article_precision = abs(diff) <= tolerance_for_article(metric) + 1e-12
            rows.append(
                {
                    "table": table_name,
                    "row": row_key,
                    "metric": metric,
                    "article_value": article_value,
                    "current_value": rounded_current,
                    "diff": diff,
                    "exact_display_match": exact_display_match,
                    "within_article_precision": within_article_precision,
                }
            )
    return pd.DataFrame(rows)


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, dpi: int) -> None:
    for suffix in ("png", "svg"):
        fig.savefig(output_dir / f"{stem}.{suffix}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def add_bar_labels(ax: plt.Axes, fmt: str = "{:.3f}") -> None:
    for container in ax.containers:
        labels = []
        for value in container.datavalues:
            if abs(value) >= 100:
                labels.append(f"{value:.0f}")
            else:
                labels.append(fmt.format(value))
        ax.bar_label(container, labels=labels, fontsize=9, padding=2)


def plot_method_quality(method_summary: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    metrics = [
        "intent_preservation",
        "clarity_enhancement",
        "answer_preference",
        "faithfulness",
    ]
    plot_data = method_summary.melt(
        id_vars=["method", "method_label"],
        value_vars=metrics,
        var_name="metric",
        value_name="score",
    )
    plot_data["metric_label"] = plot_data["metric"].map(METRIC_LABELS)

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.barplot(
        data=plot_data,
        x="method_label",
        y="score",
        hue="metric_label",
        palette=[
            IBM_COLORS["blue"],
            IBM_COLORS["cyan"],
            IBM_COLORS["teal"],
            IBM_COLORS["purple"],
        ],
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Normalized score")
    ax.set_ylim(0.45, 1.05)
    ax.legend(title="", ncols=4, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    ax.tick_params(axis="x", rotation=25)
    fig.subplots_adjust(top=0.78)
    save_figure(fig, output_dir, "01_method_quality_scores", dpi)


def plot_method_tradeoff(method_summary: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.scatterplot(
        data=method_summary,
        x="total_latency_ms",
        y="answer_preference",
        hue="method",
        size="query_count",
        sizes=(120, 900),
        palette=METHOD_PALETTE,
        legend=False,
        ax=ax,
    )
    for row in method_summary.itertuples(index=False):
        xytext = LATENCY_LABEL_OFFSETS.get(str(row.method), (10, 8))
        ax.annotate(
            row.method_label,
            (row.total_latency_ms, row.answer_preference),
            xytext=xytext,
            textcoords="offset points",
            fontsize=11,
            bbox=ANNOTATION_BOX,
            arrowprops=ANNOTATION_ARROW,
        )
    ax.set_xlabel("Average E2E latency (ms)")
    ax.set_ylabel("Answer preference")
    ax.set_xlim(method_summary["total_latency_ms"].min() - 250, method_summary["total_latency_ms"].max() + 500)
    ax.set_ylim(0.58, 0.622)
    save_figure(fig, output_dir, "02_method_latency_quality_tradeoff", dpi)


def plot_baseline_delta(delta_summary: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(15, 6),
        sharey=True,
        gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.28},
    )

    sns.barplot(
        data=delta_summary,
        x="answer_preference_delta",
        y="method_label",
        hue="method",
        palette=METHOD_PALETTE,
        dodge=False,
        ax=axes[0],
    )
    axes[0].axvline(0, color="#161616", linewidth=1)
    axes[0].set_xlabel("Delta vs. baseline")
    axes[0].set_ylabel("")
    axes[0].legend_.remove()

    sns.barplot(
        data=delta_summary,
        x="latency_delta_ms",
        y="method_label",
        hue="method",
        palette=METHOD_PALETTE,
        dodge=False,
        ax=axes[1],
    )
    axes[1].set_xlabel("Additional latency (ms)")
    axes[1].set_ylabel("")
    axes[1].tick_params(axis="y", left=False, labelleft=False)
    axes[1].legend_.remove()
    add_bar_labels(axes[1], "{:.0f}")

    save_figure(fig, output_dir, "03_baseline_delta_tradeoff", dpi)


def plot_provider_quality(provider_summary: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    quality_metrics = ["answer_preference", "faithfulness", "parse_ok"]
    quality_data = provider_summary.melt(
        id_vars=["provider", "provider_label"],
        value_vars=quality_metrics,
        var_name="metric",
        value_name="score",
    )
    quality_data["metric_label"] = quality_data["metric"].map(METRIC_LABELS)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=quality_data,
        x="provider_label",
        y="score",
        hue="metric_label",
        palette=[IBM_COLORS["blue"], IBM_COLORS["teal"], IBM_COLORS["purple"]],
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Normalized score")
    ax.set_ylim(0.55, 1.02)
    ax.legend(title="", ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    ax.tick_params(axis="x", rotation=20)
    fig.subplots_adjust(top=0.80)
    save_figure(fig, output_dir, "04_provider_quality", dpi)


def plot_provider_latency(provider_summary: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(
        data=provider_summary,
        x="provider_label",
        y="total_latency_ms",
        hue="provider",
        palette=PROVIDER_PALETTE,
        dodge=False,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Average E2E latency (ms)")
    ax.tick_params(axis="x", rotation=20)
    ax.legend_.remove()
    add_bar_labels(ax, "{:.0f}")
    save_figure(fig, output_dir, "05_provider_latency", dpi)


def plot_method_cost(method_summary: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.scatterplot(
        data=method_summary,
        x="estimated_cost_usd",
        y="answer_preference",
        hue="method",
        size="total_latency_ms",
        sizes=(140, 850),
        palette=METHOD_PALETTE,
        legend=False,
        ax=ax,
    )
    for row in method_summary.itertuples(index=False):
        xytext = COST_LABEL_OFFSETS.get(str(row.method), (10, 8))
        ax.annotate(
            row.method_label,
            (row.estimated_cost_usd, row.answer_preference),
            xytext=xytext,
            textcoords="offset points",
            fontsize=11,
            bbox=ANNOTATION_BOX,
            arrowprops=ANNOTATION_ARROW,
        )
    ax.set_xlabel("Estimated API cost across method runs (USD)")
    ax.set_ylabel("Answer preference")
    ax.set_xlim(
        method_summary["estimated_cost_usd"].min() - 0.0004,
        method_summary["estimated_cost_usd"].max() + 0.0009,
    )
    ax.set_ylim(0.58, 0.622)
    save_figure(fig, output_dir, "06_method_cost_quality_tradeoff", dpi)


def plot_recall_ceiling(method_summary: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(
        data=method_summary,
        x="method_label",
        y="recall_at_5",
        hue="method",
        palette=METHOD_PALETTE,
        dodge=False,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Recall@5")
    ax.set_ylim(0.94, 1.01)
    ax.tick_params(axis="x", rotation=25)
    ax.legend_.remove()
    add_bar_labels(ax)
    save_figure(fig, output_dir, "07_method_recall_ceiling", dpi)


def plot_method_query_count(method_summary: pd.DataFrame, output_dir: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(
        data=method_summary,
        x="method_label",
        y="query_count",
        hue="method",
        palette=METHOD_PALETTE,
        dodge=False,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Average retrieval query count")
    ax.tick_params(axis="x", rotation=25)
    ax.legend_.remove()
    add_bar_labels(ax, "{:.2f}")
    save_figure(fig, output_dir, "08_method_query_count", dpi)


def write_tables(
    method_summary: pd.DataFrame,
    provider_summary: pd.DataFrame,
    delta_summary: pd.DataFrame,
    comparison: pd.DataFrame,
    output_dir: Path,
) -> None:
    method_summary.to_csv(output_dir / "method_summary_for_article.csv", index=False)
    provider_summary.to_csv(output_dir / "provider_summary_for_article.csv", index=False)
    delta_summary.to_csv(output_dir / "baseline_delta_for_article.csv", index=False)
    comparison.to_csv(output_dir / "article_result_comparison.csv", index=False)


def main() -> None:
    args = parse_args()
    configure_theme()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = read_records(args.report_dir)
    method_summary = summarize_by_method(records)
    provider_summary = summarize_by_provider(records)
    delta_summary = summarize_baseline_delta(method_summary)

    comparison = pd.concat(
        [
            compare_values(
                "method",
                ARTICLE_METHOD_VALUES,
                method_summary,
                "method",
            ),
            compare_values(
                "baseline_delta",
                ARTICLE_BASELINE_DELTA_VALUES,
                delta_summary,
                "method",
            ),
            compare_values(
                "provider",
                ARTICLE_PROVIDER_VALUES,
                provider_summary,
                "provider",
            ),
        ],
        ignore_index=True,
    )

    write_tables(method_summary, provider_summary, delta_summary, comparison, args.output_dir)
    plot_method_quality(method_summary, args.output_dir, args.dpi)
    plot_method_tradeoff(method_summary, args.output_dir, args.dpi)
    plot_baseline_delta(delta_summary, args.output_dir, args.dpi)
    plot_provider_quality(provider_summary, args.output_dir, args.dpi)
    plot_provider_latency(provider_summary, args.output_dir, args.dpi)
    plot_method_cost(method_summary, args.output_dir, args.dpi)
    plot_recall_ceiling(method_summary, args.output_dir, args.dpi)
    plot_method_query_count(method_summary, args.output_dir, args.dpi)

    exact_matches = int(comparison["exact_display_match"].sum())
    tolerant_matches = int(comparison["within_article_precision"].sum())
    total = len(comparison)
    print(f"Generated figures in: {args.output_dir}")
    print(f"Article result comparison: {exact_matches}/{total} exact display matches")
    print(f"Article result comparison: {tolerant_matches}/{total} within article precision")
    if exact_matches != total:
        mismatches = comparison.loc[~comparison["exact_display_match"]]
        print(mismatches.to_string(index=False))


if __name__ == "__main__":
    main()
