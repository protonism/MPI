# -*- coding: utf-8 -*-
"""Generate report charts for IVF MPI 3.3."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CHART_DIR = Path(__file__).resolve().parent
ROOT_DIR = CHART_DIR.parent


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


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(ROOT_DIR / "ivf_mpi_3_3_grouped_avg.csv")
    combined = pd.read_csv(ROOT_DIR / "ivf_mpi_3_3_combined_plot_data.csv")
    return summary, combined


def save_table_slices(summary: pd.DataFrame) -> None:
    for group, name in [
        ("IVF-A", "ivf_a_correctness.csv"),
        ("IVF-B", "ivf_b_recall_latency.csv"),
        ("IVF-C", "ivf_c_strong_scaling.csv"),
        ("IVF-D", "ivf_d_cross_node.csv"),
        ("IVF-E", "ivf_e_fixed_16cores.csv"),
        ("IVF-F", "ivf_f_partition_strategy.csv"),
    ]:
        summary[summary["group"] == group].to_csv(CHART_DIR / name, index=False)


def plot_correctness_alignment(summary: pd.DataFrame) -> None:
    df = summary[summary["group"] == "IVF-A"].copy()
    df["label"] = df.apply(lambda r: f"nl={int(r.nlist)}\nnpb={int(r.nprobe)}", axis=1)

    old_latency = {
        (512, 32): 572.051,
        (512, 64): 1110.303,
        (256, 32): 1104.943,
        (256, 64): 1821.233,
    }
    old_recall = {
        (512, 32): 0.97135,
        (512, 64): 0.99055,
        (256, 32): 0.98780,
        (256, 64): 0.99710,
    }
    df["old_latency_us"] = df.apply(lambda r: old_latency[(int(r.nlist), int(r.nprobe))], axis=1)
    df["old_recall"] = df.apply(lambda r: old_recall[(int(r.nlist), int(r.nprobe))], axis=1)
    df = df.sort_values(["nlist", "nprobe"])

    x = np.arange(len(df))
    fig, ax1 = plt.subplots(figsize=(8.8, 4.9))
    ax1.bar(x - 0.18, df["old_latency_us"], width=0.36, color="#9CA3AF", label="Old IVF-SIMD latency")
    ax1.bar(x + 0.18, df["latency_us_mean"], width=0.36, color="#2563EB", label="MPI np=1 latency")
    ax1.errorbar(x + 0.18, df["latency_us_mean"], yerr=df["latency_us_std"], fmt="none", ecolor="#1F2937", capsize=3)
    ax1.set_xticks(x, df["label"])
    ax1.set_ylabel("Latency (us)")
    ax1.set_title("IVF-A correctness alignment: np=1 vs old IVF-SIMD")

    ax2 = ax1.twinx()
    ax2.plot(x, df["recall_mean"], marker="o", color="#DC2626", linewidth=2.0, label="MPI recall")
    ax2.plot(x, df["old_recall"], marker="s", color="#F59E0B", linestyle="--", linewidth=1.8, label="Old recall")
    ax2.set_ylabel("Recall@10")
    ax2.set_ylim(0.94, 1.005)
    ax2.spines["right"].set_visible(True)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "ivf_a_correctness_alignment.png", bbox_inches="tight")
    plt.close(fig)


def plot_recall_latency(summary: pd.DataFrame) -> None:
    df = summary[summary["group"] == "IVF-B"].copy().sort_values(["nlist", "nprobe"])
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    colors = {256: "#2563EB", 512: "#F97316"}
    markers = {256: "o", 512: "s"}

    for nlist, g in df.groupby("nlist"):
        ax.plot(
            g["latency_us_mean"],
            g["recall_mean"],
            color=colors[int(nlist)],
            marker=markers[int(nlist)],
            linewidth=2.2,
            label=f"nlist={int(nlist)}",
        )
        for _, row in g.iterrows():
            ax.annotate(
                f"nprobe={int(row.nprobe)}",
                (row["latency_us_mean"], row["recall_mean"]),
                textcoords="offset points",
                xytext=(6, 5),
                fontsize=8,
            )

    ax.set_xlabel("Average latency (us)")
    ax.set_ylabel("Recall@10")
    ax.set_title("IVF-B recall-latency trade-off (pure MPI, np=4)")
    ax.set_ylim(0.85, 1.005)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "ivf_b_recall_latency_tradeoff.png", bbox_inches="tight")
    plt.close(fig)


def plot_nprobe_curves(summary: pd.DataFrame) -> None:
    df = summary[summary["group"] == "IVF-B"].copy().sort_values(["nlist", "nprobe"])
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.4), sharex=True)
    colors = {256: "#2563EB", 512: "#F97316"}

    for nlist, g in df.groupby("nlist"):
        axes[0].plot(g["nprobe"], g["recall_mean"], marker="o", linewidth=2.2, color=colors[int(nlist)], label=f"nlist={int(nlist)}")
        axes[1].plot(g["nprobe"], g["latency_us_mean"], marker="s", linewidth=2.2, color=colors[int(nlist)], label=f"nlist={int(nlist)}")
        axes[1].fill_between(g["nprobe"], g["latency_us_mean"] - g["latency_us_std"], g["latency_us_mean"] + g["latency_us_std"], color=colors[int(nlist)], alpha=0.15)

    axes[0].set_title("Recall vs nprobe")
    axes[0].set_xlabel("nprobe")
    axes[0].set_ylabel("Recall@10")
    axes[1].set_title("Latency vs nprobe")
    axes[1].set_xlabel("nprobe")
    axes[1].set_ylabel("Latency (us)")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks([8, 16, 32, 64], ["8", "16", "32", "64"])
        ax.legend(loc="best")
    fig.suptitle("IVF-B nprobe sensitivity (pure MPI, np=4)", y=1.02)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "ivf_b_nprobe_curves.png", bbox_inches="tight")
    plt.close(fig)


def plot_strong_scaling(summary: pd.DataFrame) -> None:
    df = summary[summary["group"] == "IVF-C"].copy().sort_values("np")
    base_latency = df["latency_us_mean"].iloc[0]
    df["speedup"] = base_latency / df["latency_us_mean"]
    df["efficiency"] = df["speedup"] / df["np"]

    fig, ax1 = plt.subplots(figsize=(8.8, 5.4))
    ax1.errorbar(df["np"], df["latency_us_mean"], yerr=df["latency_us_std"], marker="o", linewidth=2.2, capsize=3, color="#2563EB", label="Latency")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(df["np"], [str(int(v)) for v in df["np"]])
    ax1.set_xlabel("MPI processes")
    ax1.set_ylabel("Latency (us)")
    ax1.set_title("IVF-C pure MPI strong scaling")

    ax2 = ax1.twinx()
    ax2.plot(df["np"], df["speedup"], marker="s", linewidth=2.0, color="#DC2626", label="Speedup")
    ax2.plot(df["np"], df["np"], linestyle=":", color="#6B7280", label="Ideal speedup")
    ax2.set_ylabel("Speedup vs np=1")
    ax2.spines["right"].set_visible(True)

    for row in df.itertuples():
        if int(row.np) == 1:
            offset = (10, -18)
        elif int(row.np) == 32:
            offset = (-88, 10)
        else:
            offset = (7, 8)
        ax1.annotate(
            f"gather={row.gather_us_mean:.0f}us",
            (row.np, row.latency_us_mean),
            textcoords="offset points",
            xytext=offset,
            fontsize=8,
        )

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(CHART_DIR / "ivf_c_pure_mpi_strong_scaling.png", bbox_inches="tight")
    plt.close(fig)


def plot_cross_node_breakdown(summary: pd.DataFrame) -> None:
    df = summary[summary["group"] == "IVF-D"].copy().sort_values("nodes")
    labels = [f"{int(r.nodes)} node(s)\nppn={int(r.ppn)}" for r in df.itertuples()]
    x = np.arange(len(df))
    parts = pd.DataFrame(
        {
            "Bcast": df["bcast_us_mean"],
            "Search max": df["search_max_us_mean"],
            "Gather": df["gather_us_mean"],
            "Merge": df["merge_us_mean"],
        }
    )
    parts["Other/wait"] = (df["latency_us_mean"].to_numpy() - parts.sum(axis=1).to_numpy()).clip(min=0)
    colors = ["#14B8A6", "#2563EB", "#F97316", "#22C55E", "#A3A3A3"]

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    bottom = np.zeros(len(df))
    for col, color in zip(parts.columns, colors):
        ax.bar(x, parts[col], bottom=bottom, color=color, width=0.58, label=col)
        bottom += parts[col].to_numpy()
    ax.plot(x, df["latency_us_mean"], marker="o", color="#111827", linewidth=1.8, label="End-to-end")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Average time (us)")
    ax.set_title("IVF-D cross-node cost at fixed np=8")
    ax.legend(ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "ivf_d_cross_node_breakdown.png", bbox_inches="tight")
    plt.close(fig)


def plot_fixed_16cores(summary: pd.DataFrame, combined: pd.DataFrame) -> None:
    df = summary[summary["group"] == "IVF-E"].copy().sort_values(["np"], ascending=False)
    df["label"] = df.apply(lambda r: f"{int(r.np)}x{int(r.threads)}", axis=1)
    x = np.arange(len(df))

    fig, ax1 = plt.subplots(figsize=(8.4, 4.8))
    ax1.bar(x, df["latency_us_mean"], yerr=df["latency_us_std"], color="#2563EB", width=0.58, capsize=3, label="3.3 latency")
    ax1.set_xticks(x, df["label"])
    ax1.set_xlabel("MPI processes x OpenMP threads per rank")
    ax1.set_ylabel("Latency (us)")
    ax1.set_title("IVF-E fixed 16 cores: pure MPI vs hybrid")

    ax2 = ax1.twinx()
    ax2.plot(x, df["qps_mean"], marker="o", linewidth=2.1, color="#DC2626", label="3.3 QPS")
    ax2.set_ylabel("QPS")
    ax2.spines["right"].set_visible(True)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "ivf_e_fixed_16cores_hybrid.png", bbox_inches="tight")
    plt.close(fig)

    old = combined[combined["comparison"] == "3.2_fixed_16cores_hybrid"].copy()
    if old.empty:
        return
    old["label"] = old.apply(lambda r: f"{int(r.np)}x{int(r.threads)}", axis=1)
    comp = df[["label", "latency_us_mean"]].rename(columns={"latency_us_mean": "3.3 cleaned"})
    old_comp = old[["label", "latency_us"]].rename(columns={"latency_us": "3.2 cleaned"})
    merged = comp.merge(old_comp, on="label", how="inner")
    if merged.empty:
        return
    x = np.arange(len(merged))
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.bar(x - 0.18, merged["3.2 cleaned"], width=0.36, color="#9CA3AF", label="3.2 cleaned")
    ax.bar(x + 0.18, merged["3.3 cleaned"], width=0.36, color="#2563EB", label="3.3 cleaned")
    ax.set_xticks(x, merged["label"])
    ax.set_xlabel("MPI processes x OpenMP threads per rank")
    ax.set_ylabel("Latency (us)")
    ax.set_title("Historical check: fixed 16-core hybrid results")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "ivf_e_fixed_16cores_3_2_vs_3_3.png", bbox_inches="tight")
    plt.close(fig)


def plot_partition_strategy(summary: pd.DataFrame) -> None:
    df = summary[summary["group"] == "IVF-F"].copy()
    order = ["block-id", "contiguous-list", "greedy-list"]
    df["partition"] = pd.Categorical(df["partition"], categories=order, ordered=True)
    df = df.sort_values("partition")
    x = np.arange(len(df))

    fig, ax1 = plt.subplots(figsize=(8.4, 4.8))
    ax1.bar(x, df["latency_us_mean"], yerr=df["latency_us_std"], color=["#2563EB", "#F97316", "#22C55E"], width=0.58, capsize=3, label="Latency")
    ax1.set_xticks(x, df["partition"].astype(str))
    ax1.set_ylabel("Latency (us)")
    ax1.set_title("IVF-F partition strategy comparison")

    ax2 = ax1.twinx()
    ax2.plot(x, df["imbalance_ratio_mean"], marker="o", color="#111827", linewidth=2.0, label="Imbalance ratio")
    ax2.set_ylabel("Work imbalance ratio")
    ax2.spines["right"].set_visible(True)
    ax2.set_ylim(0.9, max(1.1, df["imbalance_ratio_mean"].max() * 1.1))

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left")
    fig.tight_layout()
    fig.savefig(CHART_DIR / "ivf_f_partition_strategy.png", bbox_inches="tight")
    plt.close(fig)


def plot_combined_context(summary: pd.DataFrame, combined: pd.DataFrame) -> None:
    strong33 = summary[summary["group"] == "IVF-C"].copy().sort_values("np")
    strong31 = combined[combined["comparison"] == "3.1_pure_mpi_strong_scaling"].copy().sort_values("np")

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8))
    axes[0].plot(strong33["np"], strong33["latency_us_mean"], marker="o", linewidth=2.2, color="#2563EB", label="3.3 cleaned")
    if not strong31.empty:
        axes[0].plot(strong31["np"], strong31["latency_us"], marker="s", linestyle="--", linewidth=2.0, color="#6B7280", label="3.1 chart summary")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(strong33["np"], [str(int(v)) for v in strong33["np"]])
    axes[0].set_xlabel("MPI processes")
    axes[0].set_ylabel("Latency (us)")
    axes[0].set_title("Pure MPI strong scaling context")
    axes[0].legend(loc="best")

    scan33 = summary[summary["group"] == "IVF-B"].copy()
    scan31 = combined[combined["comparison"] == "3.1_pure_mpi_recall_latency"].copy()
    for nlist, color in [(256, "#2563EB"), (512, "#F97316")]:
        g33 = scan33[scan33["nlist"] == nlist].sort_values("nprobe")
        axes[1].plot(g33["nprobe"], g33["latency_us_mean"], marker="o", linewidth=2.1, color=color, label=f"3.3 nlist={nlist}")
        if not scan31.empty:
            g31 = scan31[scan31["nlist"] == nlist].sort_values("nprobe")
            axes[1].plot(g31["nprobe"], g31["latency_us"], marker="s", linestyle="--", linewidth=1.7, color=color, alpha=0.55, label=f"3.1 nlist={nlist}")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks([8, 16, 32, 64], ["8", "16", "32", "64"])
    axes[1].set_xlabel("nprobe")
    axes[1].set_ylabel("Latency (us)")
    axes[1].set_title("Pure MPI recall-latency context")
    axes[1].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "ivf_combined_3_1_3_3_context.png", bbox_inches="tight")
    plt.close(fig)


def write_captions(summary: pd.DataFrame) -> None:
    a = summary[summary["group"] == "IVF-A"]
    b = summary[summary["group"] == "IVF-B"]
    c = summary[summary["group"] == "IVF-C"].sort_values("np")
    d = summary[summary["group"] == "IVF-D"].sort_values("nodes")
    e = summary[summary["group"] == "IVF-E"].sort_values("np", ascending=False)
    f = summary[summary["group"] == "IVF-F"].sort_values("latency_us_mean")

    speedup_np8 = c.iloc[0]["latency_us_mean"] / c[c["np"] == 8]["latency_us_mean"].iloc[0]
    best_b = b.sort_values(["recall_mean", "latency_us_mean"], ascending=[False, True]).iloc[0]
    best_f = f.iloc[0]

    text = f"""# IVF MPI 3.3 图表说明

## 代码与 MPI.pdf 预期核对

`MPI.pdf` 的 ANN 选题要求重点探索多进程数据划分、并行搜索、负载均衡、merge/reduce 开销、MPI 与 OpenMP/SIMD 混合并行，以及 recall-latency trade-off。当前 103 代码与该预期基本一致：使用 `MPI_Init/Finalize` 管理进程，rank 0 训练或加载全局 IVF centroid 并广播，各 rank 建本地 inverted lists；每个 query 由 rank 0 选出 selected lists 后广播 query 和 list id，各 rank 扫描本地 selected lists 得到局部 top-k，最后 `MPI_Gather` 回 rank 0 做全局 top-k merge 并计算 recall。

未发现会推翻实验结论的大 bug。需要在报告中说明的限制是：`contiguous-list` 和 `greedy-list` 策略为了统计 list 大小，会让每个 rank 在构建阶段遍历全量 base 并只保存自己负责的 list，因此它更适合作为搜索阶段负载均衡策略对照，而不是严格的分布式索引构建优化。搜索计时使用缓存后的查询阶段数据，符合目标文档建议。

## ivf_a_correctness_alignment.png

用于证明 `np=1, threads=1` 时 MPI 版本与旧 IVF-SIMD 基线对齐。四组 recall 分别为 {', '.join(f'{v:.5f}' for v in a.sort_values(['nlist','nprobe'])['recall_mean'])}，与历史值一致或只有统计误差；说明 MPI top-k merge 和 recall 计算没有明显正确性问题。

## ivf_b_recall_latency_tradeoff.png

展示纯 MPI `np=4` 下的 recall-latency 折中。`nprobe` 增大后 recall 上升、latency 增大。最高 recall 组合为 `nlist={int(best_b.nlist)}, nprobe={int(best_b.nprobe)}`，recall={best_b.recall_mean:.6f}，latency={best_b.latency_us_mean:.3f} us。

## ivf_b_nprobe_curves.png

把 recall 和 latency 分成两幅曲线，便于解释 nprobe 是主要调参旋钮。`nlist=512` 在较低 nprobe 下 latency 更低；`nlist=256,nprobe=64` recall 更高但扫描代价更大。

## ivf_c_pure_mpi_strong_scaling.png

展示纯 MPI 强扩展。`np=1` 到 `np=8` latency 从 {c.iloc[0]['latency_us_mean']:.3f} us 降到 {c[c['np'] == 8]['latency_us_mean'].iloc[0]:.3f} us，速度提升约 {speedup_np8:.3f} 倍；`np=16/32` 跨节点后 gather 开销升高，整体性能回落。

## ivf_d_cross_node_breakdown.png

固定 `np=8`，比较 `(nodes,ppn)=(1,8),(2,4),(4,2)`。单节点 latency 为 {d.iloc[0]['latency_us_mean']:.3f} us，4 节点升至 {d.iloc[-1]['latency_us_mean']:.3f} us，图中 gather/bcast 明显变大，说明跨节点通信是主要负优化来源。

## ivf_e_fixed_16cores_hybrid.png

固定 16 核比较 `(np,threads)=(16,1),(8,2),(4,4),(2,8)`。`16x1` 最快，平均 latency={e.iloc[0]['latency_us_mean']:.3f} us；混合 OpenMP 配置更慢，说明本实验中减少 MPI rank 后每个 rank 扫描量增大，OpenMP 收益不足以抵消等待和线程开销。

## ivf_e_fixed_16cores_3_2_vs_3_3.png

把 3.2 已清洗的固定 16 核结果与 3.3 结果放在一起，作为历史复核。该图不用于替代 3.3 主结论，而是说明混合并行负优化现象在两轮数据中都存在。

## ivf_f_partition_strategy.png

比较 `block-id`、`contiguous-list`、`greedy-list` 三种负载均衡策略。本轮最低 latency 是 `{best_f.partition}`，为 {best_f.latency_us_mean:.3f} us。list 级策略没有带来更低端到端查询延迟，报告中可从内存布局、list 重组和同步等待角度解释。

## ivf_combined_3_1_3_3_context.png

把 3.1 和 3.3 的纯 MPI 曲线放在一起，便于说明 3.3 是对前两问的补充：3.3 不只是重复 3.1，而是补上了正确性对齐、跨节点通信、固定总核心混合并行、负载均衡策略等更完整的实验组。
"""
    (CHART_DIR / "chart_captions_3_3.md").write_text(text, encoding="utf-8")


def main() -> None:
    setup_style()
    summary, combined = load_data()
    save_table_slices(summary)
    plot_correctness_alignment(summary)
    plot_recall_latency(summary)
    plot_nprobe_curves(summary)
    plot_strong_scaling(summary)
    plot_cross_node_breakdown(summary)
    plot_fixed_16cores(summary, combined)
    plot_partition_strategy(summary)
    plot_combined_context(summary, combined)
    write_captions(summary)
    print(f"charts_dir={CHART_DIR}")


if __name__ == "__main__":
    main()
