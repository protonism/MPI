from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "charts"
DATA = ROOT / "hnsw_shard_mpi_4_1_grouped_avg.csv"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    numeric_cols = [
        "np",
        "nodes",
        "ppn",
        "M",
        "efConstruction",
        "efSearch",
        "recall_avg",
        "latency_us_avg",
        "latency_us_std",
        "bcast_us_avg",
        "search_avg_us_avg",
        "search_max_us_avg",
        "gather_us_avg",
        "merge_us_avg",
        "build_avg_ms_avg",
        "build_max_ms_avg",
        "qps_avg",
        "speedup",
        "efficiency",
        "imbalance_ratio_avg",
        "work_min_avg",
        "work_avg_avg",
        "work_max_avg",
        "used_count",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def style_axes(ax):
    ax.grid(True, axis="y", color="#D9DEE8", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=220, bbox_inches="tight")
    plt.close(fig)


def export_subset(df, name):
    df.to_csv(OUT / name, index=False)


def plot_recall_latency(df):
    a = df[df["group"] == "HNSW-A"].sort_values("efSearch")
    c = df[df["group"] == "HNSW-C"].sort_values("efSearch")
    export_subset(pd.concat([a, c], ignore_index=True), "hnsw_recall_latency_tradeoff.csv")

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.errorbar(a["latency_us_avg"], a["recall_avg"], xerr=a["latency_us_std"],
                marker="o", linewidth=2, capsize=3, label="np=1 baseline")
    ax.errorbar(c["latency_us_avg"], c["recall_avg"], xerr=c["latency_us_std"],
                marker="s", linewidth=2, capsize=3, label="np=4 shard MPI")
    for _, row in pd.concat([a, c]).iterrows():
        ax.annotate(f"ef={int(row['efSearch'])}",
                    (row["latency_us_avg"], row["recall_avg"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_title("HNSW Recall-Latency Trade-off")
    ax.set_xlabel("Average latency (us/query)")
    ax.set_ylabel("Recall@10")
    ax.set_ylim(0.89, 1.003)
    ax.legend(frameon=False)
    style_axes(ax)
    savefig(fig, "hnsw_recall_latency_tradeoff.png")


def plot_efsearch_curves(df):
    a = df[df["group"] == "HNSW-A"].sort_values("efSearch")
    c = df[df["group"] == "HNSW-C"].sort_values("efSearch")
    export_subset(pd.concat([a, c], ignore_index=True), "hnsw_efsearch_curves.csv")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharex=False)
    for label, rows, marker in [("np=1", a, "o"), ("np=4", c, "s")]:
        axes[0].plot(rows["efSearch"], rows["recall_avg"], marker=marker, linewidth=2, label=label)
        axes[1].errorbar(rows["efSearch"], rows["latency_us_avg"], yerr=rows["latency_us_std"],
                         marker=marker, linewidth=2, capsize=3, label=label)
    axes[0].set_title("Recall vs efSearch")
    axes[0].set_xlabel("efSearch")
    axes[0].set_ylabel("Recall@10")
    axes[0].set_ylim(0.89, 1.003)
    axes[1].set_title("Latency vs efSearch")
    axes[1].set_xlabel("efSearch")
    axes[1].set_ylabel("Average latency (us/query)")
    for ax in axes:
        style_axes(ax)
        ax.legend(frameon=False)
    savefig(fig, "hnsw_efsearch_curves.png")


def plot_strong_scaling(df):
    b = df[df["group"] == "HNSW-B"].sort_values("np")
    export_subset(b, "hnsw_mpi_strong_scaling.csv")

    fig, ax1 = plt.subplots(figsize=(7.2, 4.8))
    ax1.errorbar(b["np"], b["latency_us_avg"], yerr=b["latency_us_std"],
                 marker="o", linewidth=2, color="#2563EB", capsize=3, label="latency")
    ax1.set_xlabel("MPI processes (np)")
    ax1.set_ylabel("Average latency (us/query)", color="#2563EB")
    ax1.tick_params(axis="y", labelcolor="#2563EB")
    ax1.set_xticks(b["np"])
    style_axes(ax1)

    ax2 = ax1.twinx()
    ax2.plot(b["np"], b["speedup"], marker="s", linewidth=2, color="#DC2626", label="speedup")
    ax2.plot(b["np"], b["efficiency"], marker="^", linewidth=2, color="#16A34A", label="efficiency")
    ax2.set_ylabel("Speedup / efficiency")
    ax2.spines["top"].set_visible(False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="upper left")
    ax1.set_title("HNSW Shard MPI Strong Scaling")
    savefig(fig, "hnsw_mpi_strong_scaling.png")


def plot_stage_breakdown(df):
    b = df[df["group"] == "HNSW-B"].sort_values("np")
    export_subset(b, "hnsw_mpi_stage_breakdown.csv")

    x = np.arange(len(b))
    labels = [f"np={int(v)}" for v in b["np"]]
    bcast = b["bcast_us_avg"].to_numpy()
    search = b["search_max_us_avg"].to_numpy()
    gather = b["gather_us_avg"].to_numpy()
    merge = b["merge_us_avg"].to_numpy()
    residual = np.maximum(b["latency_us_avg"].to_numpy() - bcast - search - gather - merge, 0)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    bottom = np.zeros(len(b))
    parts = [
        ("bcast", bcast, "#60A5FA"),
        ("search max", search, "#2563EB"),
        ("gather", gather, "#F59E0B"),
        ("merge", merge, "#16A34A"),
        ("other/wait", residual, "#94A3B8"),
    ]
    for name, vals, color in parts:
        ax.bar(x, vals, bottom=bottom, label=name, color=color)
        bottom += vals
    ax.set_xticks(x, labels)
    ax.set_ylabel("Average latency component (us/query)")
    ax.set_title("HNSW Shard MPI Stage Breakdown")
    ax.legend(frameon=False, ncol=3, fontsize=8)
    style_axes(ax)
    savefig(fig, "hnsw_mpi_stage_breakdown.png")


def plot_partition_strategy(df):
    d = df[df["group"] == "HNSW-D"].copy()
    order = ["block-id", "cyclic", "random-hash"]
    d["partition"] = pd.Categorical(d["partition"], categories=order, ordered=True)
    d = d.sort_values("partition")
    export_subset(d, "hnsw_partition_strategy.csv")

    fig, ax1 = plt.subplots(figsize=(7.2, 4.6))
    x = np.arange(len(d))
    ax1.bar(x, d["latency_us_avg"], yerr=d["latency_us_std"], capsize=3, color="#2563EB", label="latency")
    ax1.set_xticks(x, d["partition"].astype(str), rotation=0)
    ax1.set_ylabel("Average latency (us/query)", color="#2563EB")
    ax1.tick_params(axis="y", labelcolor="#2563EB")
    style_axes(ax1)

    ax2 = ax1.twinx()
    ax2.plot(x, d["imbalance_ratio_avg"], marker="o", linewidth=2, color="#DC2626", label="imbalance")
    ax2.set_ylabel("Work imbalance ratio")
    ax2.set_ylim(0.98, max(1.03, d["imbalance_ratio_avg"].max() + 0.01))
    ax2.spines["top"].set_visible(False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="upper right")
    ax1.set_title("Partition Strategy Impact")
    savefig(fig, "hnsw_partition_strategy.png")


def plot_index_params(df):
    e = df[df["group"] == "HNSW-E"].copy()
    e["config"] = e.apply(lambda r: f"M={int(r['M'])}\nefc={int(r['efConstruction'])}", axis=1)
    e = e.sort_values(["M", "efConstruction"])
    export_subset(e, "hnsw_index_param_tradeoff.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    x = np.arange(len(e))
    axes[0].bar(x, e["latency_us_avg"], yerr=e["latency_us_std"], capsize=3, color="#2563EB")
    axes[0].set_xticks(x, e["config"], rotation=0, fontsize=8)
    axes[0].set_ylabel("Average latency (us/query)")
    axes[0].set_title("Query latency by HNSW index params")
    style_axes(axes[0])

    axes[1].scatter(e["latency_us_avg"], e["recall_avg"],
                    s=np.clip(e["build_max_ms_avg"] / 20, 40, 300),
                    color="#DC2626", alpha=0.75)
    label_offsets = {
        (24, 150): (-8, 9, "right"),
        (32, 150): (8, -2, "left"),
    }
    for _, row in e.iterrows():
        key = (int(row["M"]), int(row["efConstruction"]))
        xoff, yoff, ha = label_offsets.get(key, (5, 4, "left"))
        axes[1].annotate(f"M={int(row['M'])}, efc={int(row['efConstruction'])}",
                         (row["latency_us_avg"], row["recall_avg"]),
                         textcoords="offset points", xytext=(xoff, yoff),
                         ha=ha, fontsize=8)
    axes[1].set_xlabel("Average latency (us/query)")
    axes[1].set_ylabel("Recall@10")
    axes[1].set_title("Recall-latency-build trade-off")
    axes[1].set_xlim(e["latency_us_avg"].min() - 8, e["latency_us_avg"].max() + 18)
    style_axes(axes[1])
    savefig(fig, "hnsw_index_param_tradeoff.png")


def plot_cross_node(df):
    f = df[df["group"] == "HNSW-F"].sort_values("nodes").copy()
    f["layout"] = f.apply(lambda r: f"{int(r['nodes'])}x{int(r['ppn'])}", axis=1)
    export_subset(f, "hnsw_cross_node_breakdown.csv")

    x = np.arange(len(f))
    bcast = f["bcast_us_avg"].to_numpy()
    search = f["search_max_us_avg"].to_numpy()
    gather = f["gather_us_avg"].to_numpy()
    merge = f["merge_us_avg"].to_numpy()
    residual = np.maximum(f["latency_us_avg"].to_numpy() - bcast - search - gather - merge, 0)

    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    bottom = np.zeros(len(f))
    for name, vals, color in [
        ("bcast", bcast, "#60A5FA"),
        ("search max", search, "#2563EB"),
        ("gather", gather, "#F59E0B"),
        ("merge", merge, "#16A34A"),
        ("other/wait", residual, "#94A3B8"),
    ]:
        ax.bar(x, vals, bottom=bottom, label=name, color=color)
        bottom += vals
    ax.set_xticks(x, f["layout"])
    ax.set_xlabel("Node layout (nodes x ppn), fixed np=8")
    ax.set_ylabel("Average latency component (us/query)")
    ax.set_title("Cross-node Communication Impact")
    ax.legend(frameon=False, ncol=3, fontsize=8)
    style_axes(ax)
    savefig(fig, "hnsw_cross_node_breakdown.png")


def write_captions():
    text = """# HNSW Shard MPI 4.1 图表说明

## `hnsw_recall_latency_tradeoff.png`

展示单进程 baseline 与 `np=4` 数据分片 HNSW 的 recall-latency trade-off。`efSearch` 增大时 recall 提升，但 latency 同步上升；`np=4` 的 recall 更高，部分原因是全局 merge 汇总了多个 shard 的局部 top-k 候选。

## `hnsw_efsearch_curves.png`

分别展示 recall 和 latency 随 `efSearch` 的变化。该图适合放在“图索引调参”小节，用来说明 `efSearch` 是 HNSW 搜索阶段的主要调参旋钮。

## `hnsw_mpi_strong_scaling.png`

展示固定 `M=16,efConstruction=150,efSearch=100` 时的 MPI 强扩展。`np=1` 到 `np=8` 延迟只小幅下降，`np=16` 跨节点后延迟反升，说明 HNSW 分片的强扩展收益受到通信和同步限制。

## `hnsw_mpi_stage_breakdown.png`

展示强扩展实验中的阶段耗时。随着 `np` 增加，本地搜索最大耗时下降，但 gather 与等待开销上升，解释了跨节点或过多进程下的负优化。

## `hnsw_partition_strategy.png`

比较 `block-id`、`cyclic`、`random-hash` 三种分片方式。三者 recall 接近，端到端 latency 差异不大；`random-hash` 理论负载较均衡但波动更明显，可用于讨论负载均衡、缓存局部性和同步开销之间的关系。

## `hnsw_index_param_tradeoff.png`

展示 `M` 和 `efConstruction` 对 recall、latency 与构建时间的影响。`M=8` 延迟低但 recall 较低；`M=16` 附近较均衡；继续增大 `M` 带来的 recall 收益有限但查询延迟增加。

## `hnsw_cross_node_breakdown.png`

固定 `np=8`，比较 `(nodes,ppn)=(1,8),(2,4),(4,2)`。跨节点后 bcast/gather 明显增加，端到端 latency 上升，是 MPI 图索引阶段最直接的负优化证据之一。
"""
    (OUT / "chart_captions_4_1.md").write_text(text, encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_data()
    plot_recall_latency(df)
    plot_efsearch_curves(df)
    plot_strong_scaling(df)
    plot_stage_breakdown(df)
    plot_partition_strategy(df)
    plot_index_params(df)
    plot_cross_node(df)
    write_captions()
    print("generated charts in", OUT)


if __name__ == "__main__":
    main()
