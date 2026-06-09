# -*- coding: utf-8 -*-
"""Generate report charts for IVF MPI+OpenMP 3.2."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT_DIR = Path(__file__).resolve().parent
ROOT_DIR = OUT_DIR.parent
RAW_DATA = ROOT_DIR / "rawdata.md"


INT_COLS = {
    "np",
    "threads",
    "nodes",
    "ppn",
    "total_cores",
    "nlist",
    "nprobe",
    "k",
    "test_queries",
    "cache_hit",
}

FLOAT_COLS = {
    "recall",
    "latency_us",
    "bcast_us",
    "search_avg_us",
    "search_max_us",
    "gather_us",
    "merge_us",
    "candidate_count_avg",
    "work_min",
    "work_avg",
    "work_max",
    "imbalance_ratio",
    "batch_time_ms",
    "qps",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "legend.frameon": False,
            "font.family": "DejaVu Sans",
        }
    )


def load_raw_rows() -> pd.DataFrame:
    lines = [line.strip() for line in RAW_DATA.read_text(encoding="utf-8").splitlines() if line.strip()]
    start = next(i for i, line in enumerate(lines) if line.startswith("run_id,"))
    rows = list(csv.DictReader(lines[start:]))
    for row in rows:
        for col in INT_COLS:
            row[col] = int(row[col])
        for col in FLOAT_COLS:
            row[col] = float(row[col])
    return pd.DataFrame(rows)


def group_key(row: pd.Series) -> tuple[int, int, int, int, int, int, int]:
    return (
        int(row["nodes"]),
        int(row["ppn"]),
        int(row["np"]),
        int(row["threads"]),
        int(row["nlist"]),
        int(row["nprobe"]),
        int(row["test_queries"]),
    )


def remove_outliers(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    formal = df[df["test_queries"] == 2000].copy()
    groups: dict[tuple[int, int, int, int, int, int, int], list[int]] = defaultdict(list)
    for idx, row in formal.iterrows():
        groups[group_key(row)].append(idx)

    removed_indices: set[int] = set()
    reasons: dict[int, str] = {}
    for _, indices in groups.items():
        if len(indices) < 4:
            continue

        group = formal.loc[indices]
        med_latency = median(group["latency_us"])
        med_gather = median(group["gather_us"])
        med_search_max = median(group["search_max_us"])

        candidate_indices = []
        for idx, row in group.iterrows():
            reason = []
            if row["latency_us"] > med_latency * 1.15:
                reason.append("latency > 1.15*median")
            if med_gather > 0 and row["gather_us"] > med_gather * 4 and row["latency_us"] > med_latency * 1.10:
                reason.append("gather spike")
            if med_search_max > 0 and row["search_max_us"] > med_search_max * 1.30 and row["latency_us"] > med_latency * 1.10:
                reason.append("search_max spike")
            if reason:
                candidate_indices.append(idx)
                reasons[idx] = "; ".join(reason)

        if len(indices) - len(candidate_indices) >= 3:
            removed_indices.update(candidate_indices)

    kept = formal.drop(index=sorted(removed_indices)).copy()
    removed = formal.loc[sorted(removed_indices)].copy()
    if not removed.empty:
        removed["reason"] = [reasons[idx] for idx in removed.index]
    return kept, removed


def summarize(kept: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["nodes", "ppn", "np", "threads", "total_cores", "nlist", "nprobe", "test_queries"]
    metric_cols = [
        "recall",
        "latency_us",
        "bcast_us",
        "search_avg_us",
        "search_max_us",
        "gather_us",
        "merge_us",
        "candidate_count_avg",
        "batch_time_ms",
        "qps",
    ]

    means = kept.groupby(group_cols, as_index=False)[metric_cols].mean()
    used_counts = kept.groupby(group_cols, as_index=False).size().rename(columns={"size": "used_count"})
    raw_counts = raw[raw["test_queries"] == 2000].groupby(group_cols, as_index=False).size().rename(columns={"size": "raw_count"})
    out = means.merge(used_counts, on=group_cols).merge(raw_counts, on=group_cols)
    out["removed_count"] = out["raw_count"] - out["used_count"]
    return out


def pick(summary: pd.DataFrame, nodes: int, ppn: int, np_: int, threads: int, nlist: int, nprobe: int) -> pd.Series:
    mask = (
        (summary["nodes"] == nodes)
        & (summary["ppn"] == ppn)
        & (summary["np"] == np_)
        & (summary["threads"] == threads)
        & (summary["nlist"] == nlist)
        & (summary["nprobe"] == nprobe)
        & (summary["test_queries"] == 2000)
    )
    matched = summary[mask]
    if matched.empty:
        raise ValueError(f"missing group nodes={nodes} ppn={ppn} np={np_} threads={threads} nlist={nlist} nprobe={nprobe}")
    return matched.iloc[0]


def ordered_frames(summary: pd.DataFrame) -> dict[str, pd.DataFrame]:
    fixed_total = pd.DataFrame(
        [
            pick(summary, 2, 8, 16, 1, 512, 64),
            pick(summary, 2, 8, 8, 2, 512, 64),
            pick(summary, 2, 8, 4, 4, 512, 64),
            pick(summary, 2, 8, 2, 8, 512, 64),
        ]
    ).reset_index(drop=True)

    single_rank = pd.DataFrame(
        [
            pick(summary, 1, 1, 1, 1, 512, 64),
            pick(summary, 1, 2, 1, 2, 512, 64),
            pick(summary, 1, 4, 1, 4, 512, 64),
            pick(summary, 1, 8, 1, 8, 512, 64),
        ]
    ).reset_index(drop=True)
    single_rank["speedup"] = single_rank["latency_us"].iloc[0] / single_rank["latency_us"]
    single_rank["efficiency"] = single_rank["speedup"] / single_rank["threads"]

    np4_scan = pd.DataFrame(
        [
            pick(summary, 1, 4, 4, 1, 512, 64),
            pick(summary, 1, 8, 4, 2, 512, 64),
            pick(summary, 2, 8, 4, 4, 512, 64),
        ]
    ).reset_index(drop=True)

    recall_latency_rows = []
    for nlist in [256, 512]:
        for nprobe in [8, 16, 32, 64]:
            recall_latency_rows.append(pick(summary, 2, 8, 8, 2, nlist, nprobe))
    recall_latency = pd.DataFrame(recall_latency_rows).reset_index(drop=True)

    return {
        "fixed_total": fixed_total,
        "single_rank": single_rank,
        "np4_scan": np4_scan,
        "recall_latency": recall_latency,
    }


def save_tables(frames: dict[str, pd.DataFrame], removed: pd.DataFrame) -> None:
    for name, frame in frames.items():
        frame.to_csv(OUT_DIR / f"ivf_mpi_omp_3_2_{name}.csv", index=False)
    removed.to_csv(OUT_DIR / "ivf_mpi_omp_3_2_removed_outliers.csv", index=False)


def plot_fixed_total_cores(df: pd.DataFrame) -> None:
    labels = [f"{int(row.np)}x{int(row.threads)}" for row in df.itertuples()]
    x = np.arange(len(df))

    fig, ax1 = plt.subplots(figsize=(8.4, 4.8))
    ax1.bar(x, df["latency_us"], color="#4e79a7", width=0.58, label="Latency")
    ax1.set_xticks(x, labels)
    ax1.set_xlabel("MPI processes x OpenMP threads per rank")
    ax1.set_ylabel("Average latency (us)")
    ax1.set_title("Fixed 16 cores: pure MPI vs hybrid MPI+OpenMP")

    ax2 = ax1.twinx()
    ax2.plot(x, df["qps"], marker="o", linewidth=2.2, color="#e15759", label="QPS")
    ax2.set_ylabel("QPS")
    ax2.spines["right"].set_visible(True)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ivf_mpi_omp_fixed_16cores.png", bbox_inches="tight")
    plt.close(fig)


def plot_stage_breakdown(df: pd.DataFrame) -> None:
    labels = [f"{int(row.np)}x{int(row.threads)}" for row in df.itertuples()]
    x = np.arange(len(df))
    parts = pd.DataFrame(
        {
            "Bcast": df["bcast_us"],
            "Local search max": df["search_max_us"],
            "Gather": df["gather_us"],
            "Merge": df["merge_us"],
        }
    )
    parts["Other/wait"] = (df["latency_us"] - parts.sum(axis=1)).clip(lower=0)
    colors = ["#76b7b2", "#4e79a7", "#f28e2b", "#59a14f", "#bab0ac"]

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    bottom = np.zeros(len(df))
    for color, col in zip(colors, parts.columns):
        ax.bar(x, parts[col], bottom=bottom, color=color, width=0.6, label=col)
        bottom += parts[col].to_numpy()

    ax.plot(x, df["latency_us"], color="#111111", marker="o", linewidth=1.7, label="End-to-end")
    ax.set_xticks(x, labels)
    ax.set_xlabel("MPI processes x OpenMP threads per rank")
    ax.set_ylabel("Average time (us)")
    ax.set_title("Stage breakdown under fixed 16 cores")
    ax.legend(ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ivf_mpi_omp_stage_breakdown.png", bbox_inches="tight")
    plt.close(fig)


def plot_single_rank_scaling(df: pd.DataFrame) -> None:
    fig, ax1 = plt.subplots(figsize=(8.2, 4.8))
    x = np.arange(len(df))

    ax1.plot(x, df["latency_us"], marker="o", linewidth=2.4, color="#4e79a7", label="Latency")
    ax1.set_xticks(x, [str(int(v)) for v in df["threads"]])
    ax1.set_xlabel("OpenMP threads in one MPI rank")
    ax1.set_ylabel("Average latency (us)")
    ax1.set_title("Single-rank OpenMP scaling")

    ax2 = ax1.twinx()
    ax2.plot(x, df["speedup"], marker="s", linewidth=2.1, color="#e15759", label="Speedup")
    ax2.plot(x, df["threads"], linestyle=":", linewidth=1.7, color="#999999", label="Ideal speedup")
    ax2.set_ylabel("Speedup vs 1 thread")
    ax2.spines["right"].set_visible(True)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ivf_mpi_omp_single_rank_scaling.png", bbox_inches="tight")
    plt.close(fig)


def plot_np4_scan(df: pd.DataFrame) -> None:
    labels = [f"nodes={int(r.nodes)}\nthreads={int(r.threads)}" for r in df.itertuples()]
    x = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.bar(x - 0.2, df["latency_us"], width=0.4, color="#4e79a7", label="Latency")
    ax.bar(x + 0.2, df["search_max_us"], width=0.4, color="#f28e2b", label="Local search max")
    ax.set_xticks(x, labels)
    ax.set_xlabel("np=4 configurations")
    ax.set_ylabel("Average time (us)")
    ax.set_title("Fixed np=4: intra-rank threads and cross-node cost")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ivf_mpi_omp_np4_thread_scan.png", bbox_inches="tight")
    plt.close(fig)


def plot_recall_latency(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    colors = {256: "#4e79a7", 512: "#f28e2b"}
    markers = {256: "o", 512: "s"}

    for nlist, group in df.sort_values("nprobe").groupby("nlist"):
        ax.plot(
            group["latency_us"],
            group["recall"],
            marker=markers[nlist],
            linewidth=2.2,
            color=colors[nlist],
            label=f"nlist={nlist}",
        )
        for _, row in group.iterrows():
            ax.annotate(
                f"npb={int(row['nprobe'])}",
                (row["latency_us"], row["recall"]),
                textcoords="offset points",
                xytext=(6, 5),
                fontsize=8,
            )

    ax.set_xlabel("Average latency (us)")
    ax.set_ylabel("Recall@10")
    ax.set_title("Recall-latency trade-off under np=8, threads=2")
    ax.set_ylim(0.85, 1.005)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ivf_mpi_omp_recall_latency_tradeoff.png", bbox_inches="tight")
    plt.close(fig)


def plot_nprobe_curves(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.4), sharex=True)
    colors = {256: "#4e79a7", 512: "#f28e2b"}

    for nlist, group in df.sort_values("nprobe").groupby("nlist"):
        axes[0].plot(group["nprobe"], group["recall"], marker="o", linewidth=2.2, color=colors[nlist], label=f"nlist={nlist}")
        axes[1].plot(group["nprobe"], group["latency_us"], marker="s", linewidth=2.2, color=colors[nlist], label=f"nlist={nlist}")

    axes[0].set_title("Recall vs nprobe")
    axes[0].set_xlabel("nprobe")
    axes[0].set_ylabel("Recall@10")
    axes[1].set_title("Latency vs nprobe")
    axes[1].set_xlabel("nprobe")
    axes[1].set_ylabel("Average latency (us)")

    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks([8, 16, 32, 64], ["8", "16", "32", "64"])
        ax.legend(loc="best")

    fig.suptitle("nprobe sensitivity under np=8, threads=2", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ivf_mpi_omp_nprobe_curves.png", bbox_inches="tight")
    plt.close(fig)


def write_captions(frames: dict[str, pd.DataFrame], removed: pd.DataFrame) -> None:
    fixed = frames["fixed_total"]
    single = frames["single_rank"]
    np4 = frames["np4_scan"]
    recall = frames["recall_latency"]

    text = f"""# IVF MPI+OpenMP 3.2 图表说明

## 代码与结果核对结论

当前 3.2 代码符合 `MPI.pdf` 中 ANN 选题对 IVF MPI 的核心预期：base data 按 MPI rank 做块分片，每个 rank 维护本地 inverted lists；rank 0 广播 query 和 selected list ids；各 rank 本地搜索局部 top-k；最后由 rank 0 gather 并 merge 全局 top-k。混合并行部分也符合讲义建议，即 MPI 负责进程间数据划分，OpenMP 在每个 rank 内进一步并行 coarse search 和 selected lists 扫描。

没有发现会推翻实验结论的大 bug。需要在报告中说明的限制是：`query_bcast_us` 未单独包含 rank 0 的 coarse search 时间，coarse search 计入了端到端 `latency_us`；当前分片是按 base id 块划分，负载均衡策略尚未扩展到 list-size greedy。

## 数据处理口径

- 原始正式记录：71 条。
- 覆盖参数组：19 个，无遗漏。
- 剔除异常记录：{len(removed)} 条。
- 剔除规则：同组内端到端延迟超过中位数 15%，或 gather/search_max 明显尖峰并拉高延迟；剔除后每组仍至少保留 3 次测量。

## ivf_mpi_omp_fixed_16cores.png

固定总核心数为 16，对比 `np x threads_per_rank`。`16x1` 平均 latency 为 {fixed.loc[0, "latency_us"]:.3f} us，是该组最低；`8x2`、`4x4`、`2x8` 反而更慢，说明此数据集和实现下增加 rank 内线程会减少 MPI 进程数，使每个 rank 本地扫描量增大，并引入 OpenMP 同步开销。

## ivf_mpi_omp_stage_breakdown.png

展示固定 16 核下的阶段耗时。`16x1` 的本地搜索很短，但 gather 较高；混合配置中 local search max 明显变大，尤其 `4x4` 还出现较高 gather/等待成本。这支持“混合并行不一定优于纯 MPI，需要同时考虑通信和 rank 内线程开销”的结论。

## ivf_mpi_omp_single_rank_scaling.png

单 MPI rank 内 OpenMP 线程数从 1 增至 8 时，平均 latency 从 {single.loc[0, "latency_us"]:.3f} us 降至 {single.loc[3, "latency_us"]:.3f} us，说明 rank 内 list scan 并行本身有效。8 线程相对 1 线程加速比约为 {single.loc[3, "speedup"]:.3f}。

## ivf_mpi_omp_np4_thread_scan.png

固定 `np=4` 时，单节点 `threads=2` 的 latency 为 {np4.loc[1, "latency_us"]:.3f} us，优于 `threads=1` 的 {np4.loc[0, "latency_us"]:.3f} us；但跨 2 节点 `threads=4` latency 增至 {np4.loc[2, "latency_us"]:.3f} us，说明跨节点调度、等待和通信成本会显著影响混合并行。

## ivf_mpi_omp_recall_latency_tradeoff.png

展示 `np=8, threads=2` 下的 recall-latency 折中。`nprobe` 增大后 recall 提升；`nlist=512,nprobe=64` 达到 recall={recall[(recall["nlist"] == 512) & (recall["nprobe"] == 64)]["recall"].iloc[0]:.6f}，latency={recall[(recall["nlist"] == 512) & (recall["nprobe"] == 64)]["latency_us"].iloc[0]:.3f} us；`nlist=256,nprobe=64` recall 更高，但 latency 也更高。

## ivf_mpi_omp_nprobe_curves.png

补充展示 nprobe 参数敏感性。该图适合放在 recall-latency 散点图之后，用来说明 nprobe 是主要准确率调节旋钮。
"""
    (OUT_DIR / "chart_captions_3_2.md").write_text(text, encoding="utf-8")


def main() -> None:
    setup_style()
    raw = load_raw_rows()
    kept, removed = remove_outliers(raw)
    summary = summarize(kept, raw)
    frames = ordered_frames(summary)
    save_tables(frames, removed)

    plot_fixed_total_cores(frames["fixed_total"])
    plot_stage_breakdown(frames["fixed_total"])
    plot_single_rank_scaling(frames["single_rank"])
    plot_np4_scan(frames["np4_scan"])
    plot_recall_latency(frames["recall_latency"])
    plot_nprobe_curves(frames["recall_latency"])
    write_captions(frames, removed)

    print(f"raw_rows={len(raw)} kept_rows={len(kept)} removed_rows={len(removed)}")
    print(f"charts_dir={OUT_DIR}")


if __name__ == "__main__":
    main()
