# -*- coding: utf-8 -*-
"""Generate report charts for IVF MPI 3.1."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUT_DIR = Path(__file__).resolve().parent


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimHei",
                "Noto Sans CJK SC",
                "Arial Unicode MS",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "legend.frameon": False,
        }
    )


def strong_scaling_data() -> pd.DataFrame:
    return pd.DataFrame(
        [
            dict(nodes=1, ppn=1, np=1, threads=1, recall=0.990550, latency=997.632, bcast=0.649, search_avg=957.292, search_max=957.292, gather=1.022, merge=0.550, qps=1013.284, speedup=1.000, efficiency=1.000),
            dict(nodes=1, ppn=2, np=2, threads=1, recall=0.990550, latency=478.482, bcast=2.302, search_avg=413.241, search_max=430.859, gather=6.022, merge=0.923, qps=2082.936, speedup=2.085, efficiency=1.042),
            dict(nodes=1, ppn=4, np=4, threads=1, recall=0.990550, latency=307.413, bcast=3.010, search_avg=234.285, search_max=254.625, gather=35.296, merge=1.164, qps=3278.714, speedup=3.245, efficiency=0.811),
            dict(nodes=1, ppn=8, np=8, threads=1, recall=0.990550, latency=195.784, bcast=2.848, search_avg=120.884, search_max=146.509, gather=33.558, merge=1.434, qps=5074.111, speedup=5.096, efficiency=0.637),
            dict(nodes=2, ppn=8, np=16, threads=1, recall=0.990550, latency=337.371, bcast=19.826, search_avg=60.166, search_max=86.728, gather=188.912, merge=2.256, qps=2977.424, speedup=2.957, efficiency=0.185),
            dict(nodes=4, ppn=8, np=32, threads=1, recall=0.990550, latency=499.723, bcast=47.930, search_avg=30.685, search_max=46.307, gather=326.500, merge=3.412, qps=1997.441, speedup=1.996, efficiency=0.062),
        ]
    )


def scan_data() -> pd.DataFrame:
    return pd.DataFrame(
        [
            dict(nlist=256, nprobe=8, recall=0.915550, latency=137.729, bcast=1.783, search_avg=105.802, search_max=116.168, gather=10.552, merge=1.030, candidates=5148.187, qps=7461.034),
            dict(nlist=256, nprobe=16, recall=0.964600, latency=155.161, bcast=1.890, search_avg=120.638, search_max=131.635, gather=18.148, merge=1.031, candidates=9084.485, qps=6378.782),
            dict(nlist=256, nprobe=32, recall=0.987800, latency=302.076, bcast=2.241, search_avg=242.192, search_max=273.803, gather=43.950, merge=1.042, candidates=16025.124, qps=3325.698),
            dict(nlist=256, nprobe=64, recall=0.997100, latency=546.705, bcast=2.613, search_avg=452.182, search_max=512.396, gather=81.856, merge=1.122, candidates=28580.671, qps=1823.864),
            dict(nlist=512, nprobe=8, recall=0.869700, latency=83.159, bcast=1.903, search_avg=43.110, search_max=51.438, gather=16.045, merge=0.859, candidates=2888.335, qps=11953.033),
            dict(nlist=512, nprobe=16, recall=0.935300, latency=118.111, bcast=3.159, search_avg=68.761, search_max=80.051, gather=19.713, merge=0.901, candidates=5002.258, qps=8357.437),
            dict(nlist=512, nprobe=32, recall=0.971350, latency=177.256, bcast=2.477, search_avg=119.319, search_max=135.588, gather=25.938, merge=0.972, candidates=8724.535, qps=5594.644),
            dict(nlist=512, nprobe=64, recall=0.990550, latency=307.413, bcast=3.010, search_avg=234.285, search_max=254.625, gather=35.296, merge=1.164, candidates=15384.605, qps=3278.714),
        ]
    )


def save_data(strong: pd.DataFrame, scan: pd.DataFrame) -> None:
    strong.to_csv(OUT_DIR / "ivf_mpi_3_1_strong_scaling_data.csv", index=False)
    scan.to_csv(OUT_DIR / "ivf_mpi_3_1_scan_data.csv", index=False)


def plot_strong_scaling(df: pd.DataFrame) -> None:
    fig, ax1 = plt.subplots(figsize=(8.2, 4.8))
    x = np.arange(len(df))

    ax1.plot(x, df["latency"], marker="o", linewidth=2.4, color="#2f6f9f", label="平均延迟")
    ax1.set_xticks(x, [f"{int(v)}" for v in df["np"]])
    ax1.set_xlabel("MPI 进程数 np")
    ax1.set_ylabel("平均延迟 (us)")
    ax1.set_title("IVF-SIMD + MPI 强扩展：延迟与加速比")
    ax1.annotate(
        "最佳延迟\nnp=8",
        xy=(3, df.loc[df["np"] == 8, "latency"].iloc[0]),
        xytext=(3.35, 300),
        arrowprops=dict(arrowstyle="->", color="#2f6f9f", lw=1.2),
        fontsize=9,
    )

    ax2 = ax1.twinx()
    ax2.plot(x, df["speedup"], marker="s", linewidth=2.2, color="#c44e52", label="实测加速比")
    ax2.plot(x, df["np"], linestyle=":", linewidth=1.8, color="#999999", label="理想线性加速")
    ax2.set_ylabel("加速比")
    ax2.spines["right"].set_visible(True)

    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ivf_mpi_strong_scaling.png", bbox_inches="tight")
    plt.close(fig)


def plot_stage_breakdown(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    x = np.arange(len(df))
    width = 0.64

    parts = pd.DataFrame(
        {
            "广播": df["bcast"],
            "本地搜索(max)": df["search_max"],
            "Gather": df["gather"],
            "Merge": df["merge"],
        }
    )
    parts["其他/等待"] = (df["latency"] - parts.sum(axis=1)).clip(lower=0)
    colors = ["#76b7b2", "#4e79a7", "#f28e2b", "#59a14f", "#bab0ac"]

    bottom = np.zeros(len(df))
    for color, col in zip(colors, parts.columns):
        ax.bar(x, parts[col], width, bottom=bottom, color=color, label=col)
        bottom += parts[col].to_numpy()

    ax.plot(x, df["latency"], color="#111111", marker="o", linewidth=1.8, label="端到端延迟")
    ax.set_xticks(x, [str(int(v)) for v in df["np"]])
    ax.set_xlabel("MPI 进程数 np")
    ax.set_ylabel("平均耗时 (us)")
    ax.set_title("IVF MPI 单次查询耗时拆分")
    ax.legend(ncol=3, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ivf_mpi_stage_breakdown.png", bbox_inches="tight")
    plt.close(fig)


def plot_comm_overhead(df: pd.DataFrame) -> None:
    fig, ax1 = plt.subplots(figsize=(8.2, 4.8))
    x = np.arange(len(df))
    comm = df["bcast"] + df["gather"] + df["merge"]
    comm_ratio = comm / df["latency"] * 100.0

    ax1.bar(x - 0.18, df["search_max"], width=0.36, color="#4e79a7", label="本地搜索(max)")
    ax1.bar(x + 0.18, comm, width=0.36, color="#e15759", label="通信+merge")
    ax1.set_xticks(x, [str(int(v)) for v in df["np"]])
    ax1.set_xlabel("MPI 进程数 np")
    ax1.set_ylabel("耗时 (us)")
    ax1.set_title("进程数增加后的通信开销变化")

    ax2 = ax1.twinx()
    ax2.plot(x, comm_ratio, color="#b07aa1", marker="D", linewidth=2.0, label="通信+merge占比")
    ax2.set_ylabel("通信+merge 占端到端延迟比例 (%)")
    ax2.spines["right"].set_visible(True)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ivf_mpi_comm_overhead.png", bbox_inches="tight")
    plt.close(fig)


def plot_recall_latency(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    markers = {256: "o", 512: "s"}
    colors = {256: "#4e79a7", 512: "#f28e2b"}

    for nlist, group in df.sort_values("nprobe").groupby("nlist"):
        ax.plot(
            group["latency"],
            group["recall"],
            marker=markers[nlist],
            linewidth=2.2,
            color=colors[nlist],
            label=f"nlist={nlist}",
        )
        for _, row in group.iterrows():
            ax.annotate(
                f"nprobe={int(row['nprobe'])}",
                (row["latency"], row["recall"]),
                textcoords="offset points",
                xytext=(6, 5),
                fontsize=8,
            )

    ax.set_xlabel("平均延迟 (us)")
    ax.set_ylabel("Recall@10")
    ax.set_title("IVF MPI 参数扫描：Recall-Latency 折中")
    ax.set_ylim(0.85, 1.005)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ivf_mpi_recall_latency_tradeoff.png", bbox_inches="tight")
    plt.close(fig)


def plot_nprobe_curves(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.4), sharex=True)
    colors = {256: "#4e79a7", 512: "#f28e2b"}

    for nlist, group in df.sort_values("nprobe").groupby("nlist"):
        axes[0].plot(group["nprobe"], group["recall"], marker="o", linewidth=2.2, color=colors[nlist], label=f"nlist={nlist}")
        axes[1].plot(group["nprobe"], group["latency"], marker="s", linewidth=2.2, color=colors[nlist], label=f"nlist={nlist}")

    axes[0].set_title("Recall 随 nprobe 变化")
    axes[0].set_xlabel("nprobe")
    axes[0].set_ylabel("Recall@10")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks([8, 16, 32, 64], ["8", "16", "32", "64"])

    axes[1].set_title("Latency 随 nprobe 变化")
    axes[1].set_xlabel("nprobe")
    axes[1].set_ylabel("平均延迟 (us)")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks([8, 16, 32, 64], ["8", "16", "32", "64"])

    for ax in axes:
        ax.legend(loc="best")
    fig.suptitle("IVF MPI 参数敏感性：nprobe 越大 recall 越高但延迟上升", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ivf_mpi_nprobe_curves.png", bbox_inches="tight")
    plt.close(fig)


def write_captions() -> None:
    text = """# IVF MPI 3.1 图表说明

## ivf_mpi_strong_scaling.png

用于说明纯 MPI 强扩展结果。`np=8` 时平均延迟最低，为 `195.784 us`；继续增加到 `np=16/32` 后，本地搜索继续变快，但通信开销抵消了收益，端到端延迟回升。

## ivf_mpi_stage_breakdown.png

用于说明单次查询端到端耗时由广播、本地搜索、Gather、Merge 和其他等待构成。该图重点体现 `np=16/32` 时 Gather 开销显著增加。

## ivf_mpi_comm_overhead.png

用于说明通信与 merge 在总延迟中的占比随进程数上升而增加。它支撑“MPI 进程数不一定越多越好，需要考虑通信开销”的分析。

## ivf_mpi_recall_latency_tradeoff.png

用于说明 `nlist/nprobe` 参数对 Recall-Latency 的影响。`nprobe` 增大时 recall 提升但 latency 上升；在本实验中 `nlist=512,nprobe=32` 是较低延迟折中点，`nlist=512,nprobe=64` 是高 recall 点。

## ivf_mpi_nprobe_curves.png

用于单独展示 `nprobe` 参数敏感性，适合放在 recall-latency 图之后作为补充。
"""
    (OUT_DIR / "chart_captions.md").write_text(text, encoding="utf-8")


def main() -> None:
    setup_style()
    strong = strong_scaling_data()
    scan = scan_data()
    save_data(strong, scan)
    plot_strong_scaling(strong)
    plot_stage_breakdown(strong)
    plot_comm_overhead(strong)
    plot_recall_latency(scan)
    plot_nprobe_curves(scan)
    write_captions()


if __name__ == "__main__":
    main()
