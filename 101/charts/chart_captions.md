# IVF MPI 3.1 图表说明

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
