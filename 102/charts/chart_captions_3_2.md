# IVF MPI+OpenMP 3.2 图表说明

## ivf_mpi_omp_fixed_16cores.png

固定总核心数为 16，对比 `np x threads_per_rank`。`16x1` 平均 latency 为 369.992 us，是该组最低；`8x2`、`4x4`、`2x8` 反而更慢，说明此数据集和实现下增加 rank 内线程会减少 MPI 进程数，使每个 rank 本地扫描量增大，并引入 OpenMP 同步开销。

## ivf_mpi_omp_stage_breakdown.png

展示固定 16 核下的阶段耗时。`16x1` 的本地搜索很短，但 gather 较高；混合配置中 local search max 明显变大，尤其 `4x4` 还出现较高 gather/等待成本。这支持“混合并行不一定优于纯 MPI，需要同时考虑通信和 rank 内线程开销”的结论。

## ivf_mpi_omp_single_rank_scaling.png

单 MPI rank 内 OpenMP 线程数从 1 增至 8 时，平均 latency 从 954.697 us 降至 254.239 us，说明 rank 内 list scan 并行本身有效。8 线程相对 1 线程加速比约为 3.755。

## ivf_mpi_omp_np4_thread_scan.png

固定 `np=4` 时，单节点 `threads=2` 的 latency 为 225.056 us，优于 `threads=1` 的 259.319 us；但跨 2 节点 `threads=4` latency 增至 1698.786 us，说明跨节点调度、等待和通信成本会显著影响混合并行。

## ivf_mpi_omp_recall_latency_tradeoff.png

展示 `np=8, threads=2` 下的 recall-latency 折中。`nprobe` 增大后 recall 提升；`nlist=512,nprobe=64` 达到 recall=0.990550，latency=1407.355 us；`nlist=256,nprobe=64` recall 更高，但 latency 也更高。

## ivf_mpi_omp_nprobe_curves.png

补充展示 nprobe 参数敏感性。该图适合放在 recall-latency 散点图之后，用来说明 nprobe 是主要准确率调节旋钮。
