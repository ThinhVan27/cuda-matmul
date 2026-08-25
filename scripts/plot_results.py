"""Generate per-size and cross-size plots for CUDA SGEMM benchmarks.

Expected layout::

    results/
    ├── 128/tables/{k1,k2,...,cublas}.csv
    ├── 256/tables/{k1,k2,...,cublas}.csv
    ├── 512/tables/{k1,k2,...,cublas}.csv
    └── ...

For every numeric directory, plots are written to ``<size>/figures``.
One combined GFLOP/s plot is written directly under ``results``.
The cuBLAS baseline is read exclusively from ``tables/cublas.csv``.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 12

REQUIRED_NUMERIC_COLUMNS = {"M", "N", "K", "gflops"}


def kernel_sort_key(name: str) -> tuple[int, int | str]:
    """Sort k1, k2, ... numerically and place cuBLAS last."""
    lowered = str(name).lower()
    match = re.fullmatch(r"k(\d+)", lowered)
    if match:
        return (0, int(match.group(1)))
    if lowered == "cublas":
        return (2, lowered)
    return (1, lowered)


def discover_size_dirs(results_dir: Path) -> list[tuple[int, Path]]:
    """Return numeric child directories ordered by matrix size."""
    found: list[tuple[int, Path]] = []
    if not results_dir.is_dir():
        return found

    for child in results_dir.iterdir():
        if child.is_dir() and child.name.isdigit():
            found.append((int(child.name), child))
    return sorted(found, key=lambda item: item[0])


def load_csvs(directory: Path) -> pd.DataFrame:
    """Load custom kernels and the standalone ``cublas.csv`` baseline.

    The filename is the authoritative kernel name. This allows the standalone
    cuBLAS file to omit the ``kernel`` column or contain a stale value.
    """
    frames: list[pd.DataFrame] = []
    for csv_path in sorted(directory.glob("*.csv")):
        try:
            frame = pd.read_csv(csv_path)
        except Exception as exc:
            print(f"Warning: cannot read {csv_path}: {exc}")
            continue

        missing = REQUIRED_NUMERIC_COLUMNS - set(frame.columns)
        if missing:
            print(f"Warning: skip {csv_path}; missing columns: {sorted(missing)}")
            continue

        filename_kernel = csv_path.stem.lower()
        if filename_kernel == "cublas" or re.fullmatch(r"k\d+", filename_kernel):
            frame["kernel"] = filename_kernel
        elif "kernel" not in frame.columns:
            print(
                f"Warning: skip {csv_path}; cannot infer kernel name from filename"
            )
            continue

        frame["source_csv"] = str(csv_path)
        frames.append(frame)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def square_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep square matrices and normalize kernel names."""
    if df.empty:
        return df.copy()
    square = df[(df["M"] == df["N"]) & (df["N"] == df["K"])].copy()
    square["kernel"] = square["kernel"].astype(str).str.lower()
    return square


def performance_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return measured GFLOP/s rows, including explicit ``cublas.csv`` rows."""
    square = square_rows(df)
    if square.empty:
        return square

    columns = ["M", "N", "K", "kernel", "gflops"]
    return square[columns].copy()


def aggregate_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse repeated runs to median GFLOP/s per size and kernel."""
    perf = performance_rows(df)
    if perf.empty:
        return perf
    return perf.groupby(["M", "N", "K", "kernel"], as_index=False)["gflops"].median()


def plot_gflops_vs_size(
    df: pd.DataFrame,
    output_dir: Path,
    filename: str = "gflops_vs_size.png",
    linear_x: bool = False,
) -> None:
    perf = aggregate_performance(df)
    if perf.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    kernels = sorted(perf["kernel"].unique(), key=kernel_sort_key)
    for kernel in kernels:
        data = perf[perf["kernel"] == kernel].sort_values("M")
        is_cublas = kernel == "cublas"
        ax.plot(
            data["M"], data["gflops"], marker="s" if is_cublas else "o",
            linestyle="--" if is_cublas else "-", linewidth=2,
            color="black" if is_cublas else None,
            label="cuBLAS" if is_cublas else kernel.upper(),
        )

    sizes = sorted(perf["M"].astype(int).unique())
    if linear_x:
        # Preserve the true numerical distance between matrix sizes.
        # For example, 0→4096 is twice as long as 0→2048.
        ticks = [0, *sizes]
        ax.set_xscale("linear")
        ax.set_xlim(0, max(sizes) * 1.03)
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(value) for value in ticks])
    else:
        ax.set_xticks(sizes)
        ax.set_xticklabels([str(size) for size in sizes])
    if not linear_x and len(sizes) > 1:
        ax.set_xscale("log", base=2)
    ax.set_xlabel("Square matrix size (M=N=K)")
    ax.set_ylabel("GFLOP/s")
    ax.set_title("SGEMM performance")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    path = output_dir / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_percent_cublas_vs_size(df: pd.DataFrame, output_dir: Path) -> None:
    perf = aggregate_performance(df)
    if perf.empty:
        return

    cublas = perf[perf["kernel"] == "cublas"][["M", "gflops"]].rename(
        columns={"gflops": "cublas_gflops"}
    )
    custom = perf[perf["kernel"] != "cublas"].merge(cublas, on="M", how="inner")
    if custom.empty:
        return
    custom["percent_cublas"] = 100.0 * custom["gflops"] / custom["cublas_gflops"]

    fig, ax = plt.subplots(figsize=(10, 6))
    for kernel in sorted(custom["kernel"].unique(), key=kernel_sort_key):
        data = custom[custom["kernel"] == kernel].sort_values("M")
        ax.plot(data["M"], data["percent_cublas"], marker="o", linewidth=2,
                label=kernel.upper())

    sizes = sorted(custom["M"].astype(int).unique())
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(size) for size in sizes])
    if len(sizes) > 1:
        ax.set_xscale("log", base=2)
    ax.axhline(100, color="gray", linestyle=":", label="cuBLAS (100%)")
    ax.set_xlabel("Square matrix size (M=N=K)")
    ax.set_ylabel("% of cuBLAS performance")
    ax.set_title("Relative performance against cuBLAS")
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    path = output_dir / "percent_cublas_vs_size.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_speedup_progression(df: pd.DataFrame, output_dir: Path, size: int) -> None:
    perf = aggregate_performance(df)
    custom = perf[(perf["M"] == size) & (perf["kernel"] != "cublas")].copy()
    if custom.empty:
        return

    order = sorted(custom["kernel"].unique(), key=kernel_sort_key)
    custom["kernel"] = pd.Categorical(custom["kernel"], categories=order, ordered=True)
    custom = custom.sort_values("kernel")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(custom["kernel"].astype(str), custom["gflops"], marker="o",
            linewidth=2.5, color="steelblue", markersize=8,
            markerfacecolor="white", markeredgewidth=2)
    for kernel, gflops in zip(custom["kernel"].astype(str), custom["gflops"]):
        ax.annotate(f"{gflops:,.1f}", (kernel, gflops), xytext=(0, 10),
                    textcoords="offset points", ha="center", fontsize=9)

    cublas = perf[(perf["M"] == size) & (perf["kernel"] == "cublas")]
    if not cublas.empty:
        value = cublas["gflops"].iloc[0]
        ax.axhline(value, color="red", linestyle="--",
                   label=f"cuBLAS ({value:,.1f} GFLOP/s)")
        ax.legend()

    ax.set_ylabel("GFLOP/s")
    ax.set_title(f"Kernel progression at {size}×{size}×{size}")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = output_dir / "speedup_progression.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_rectangular_comparison(df: pd.DataFrame, output_dir: Path) -> None:
    rect = df[(df["M"] != df["N"]) | (df["N"] != df["K"])].copy()
    rect = rect[rect["kernel"].astype(str).str.lower() != "cublas"]
    shapes = rect[["M", "N", "K"]].drop_duplicates().head(3)
    if shapes.empty:
        return

    fig, axes = plt.subplots(1, len(shapes), figsize=(6 * len(shapes), 5), squeeze=False)
    for idx, (_, shape) in enumerate(shapes.iterrows()):
        ax = axes[0, idx]
        data = rect[(rect["M"] == shape["M"]) & (rect["N"] == shape["N"]) &
                    (rect["K"] == shape["K"])].copy()
        data = data.groupby("kernel", as_index=False)["gflops"].median()
        order = sorted(data["kernel"].astype(str).unique(), key=kernel_sort_key)
        data["kernel"] = pd.Categorical(data["kernel"], categories=order, ordered=True)
        data = data.sort_values("kernel")
        ax.bar(data["kernel"].astype(str), data["gflops"])
        ax.set_title(f"{shape['M']}×{shape['N']}×{shape['K']}")
        ax.set_ylabel("GFLOP/s" if idx == 0 else "")
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Rectangular matrix performance")
    fig.tight_layout()
    path = output_dir / "rectangular_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_latency_distribution(df: pd.DataFrame, output_dir: Path, size: int) -> None:
    square = square_rows(df)
    square = square[(square["M"] == size) & (square["kernel"] != "cublas")].copy()
    metrics = [column for column in ("min_ms", "median_ms", "p95_ms")
               if column in square.columns]
    if square.empty or not metrics:
        return

    latency = square.groupby("kernel", as_index=False)[metrics].median()
    order = sorted(latency["kernel"].unique(), key=kernel_sort_key)
    latency["kernel"] = pd.Categorical(latency["kernel"], categories=order, ordered=True)
    latency = latency.sort_values("kernel")

    x = np.arange(len(latency))
    width = 0.8 / len(metrics)
    fig, ax = plt.subplots(figsize=(10, 6))
    for index, metric in enumerate(metrics):
        offset = (index - (len(metrics) - 1) / 2) * width
        ax.bar(x + offset, latency[metric], width, label=metric.removesuffix("_ms").upper())

    ax.set_xticks(x)
    ax.set_xticklabels([str(k).upper() for k in latency["kernel"]])
    ax.set_ylabel("Latency (ms)")
    ax.set_title(f"Latency distribution at {size}×{size}×{size}")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = output_dir / "latency_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def generate_summary_table(df: pd.DataFrame, output_dir: Path) -> None:
    perf = aggregate_performance(df)
    if perf.empty:
        return

    gflops = perf.pivot_table(index="M", columns="kernel", values="gflops", aggfunc="median")
    columns = sorted(gflops.columns, key=kernel_sort_key)
    gflops = gflops.reindex(columns=columns)

    cublas = gflops["cublas"] if "cublas" in gflops.columns else None
    relative = pd.DataFrame(index=gflops.index)
    if cublas is not None:
        for kernel in columns:
            if kernel != "cublas":
                relative[kernel] = 100.0 * gflops[kernel] / cublas

    path = output_dir / "summary_table.md"
    with path.open("w", encoding="utf-8") as file:
        file.write("# Performance Summary\n\n## GFLOP/s\n\n")
        file.write(dataframe_to_markdown(gflops))
        if not relative.empty:
            file.write("\n\n## % cuBLAS\n\n")
            file.write(dataframe_to_markdown(relative))
        file.write("\n")
    print(f"Saved: {path}")


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    """Render a small numeric DataFrame without requiring ``tabulate``."""
    index_name = frame.index.name or "index"
    headers = [str(index_name), *[str(column) for column in frame.columns]]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for index, row in frame.iterrows():
        values = [str(index)]
        for value in row:
            values.append("" if pd.isna(value) else f"{float(value):.2f}")
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def generate_size_outputs(df: pd.DataFrame, size: int, size_dir: Path) -> None:
    figures_dir = size_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_gflops_vs_size(df, figures_dir)
    plot_percent_cublas_vs_size(df, figures_dir)
    plot_speedup_progression(df, figures_dir, size)
    plot_rectangular_comparison(df, figures_dir)
    plot_latency_distribution(df, figures_dir, size)
    generate_summary_table(df, figures_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"),
                        help="Root results directory (default: results)")
    args = parser.parse_args()
    results_dir = args.results_dir

    size_dirs = discover_size_dirs(results_dir)
    if not size_dirs:
        raise SystemExit(f"No numeric size directories found under {results_dir}")

    all_frames: list[pd.DataFrame] = []
    for size, size_dir in size_dirs:
        print(f"\nLoading size {size} from {size_dir}")
        frame = load_csvs(size_dir / "tables")
        if frame.empty:
            print(f"Warning: no valid CSV data for size {size}")
            continue

        matching = frame[(frame["M"] == size) & (frame["N"] == size) &
                         (frame["K"] == size)]
        if matching.empty:
            print(f"Warning: CSV data in {size_dir} has no {size}x{size}x{size} rows")
        generate_size_outputs(frame, size, size_dir)
        all_frames.append(frame)

    if not all_frames:
        raise SystemExit("No valid benchmark rows were loaded")

    combined = pd.concat(all_frames, ignore_index=True)
    # Required overview: every custom kernel plus cuBLAS across all sizes.
    overview = "gflops_all_kernels_vs_size.png"
    plot_gflops_vs_size(
        combined,
        results_dir,
        filename=overview,
        linear_x=True,
    )
    print(f"Combined overview: {results_dir / overview}")


if __name__ == "__main__":
    main()