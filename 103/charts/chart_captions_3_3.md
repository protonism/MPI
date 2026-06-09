# IVF MPI 3.3 图表说明

## ivf_a_correctness_alignment.png

用于证明 `np=1, threads=1` 时 MPI 版本与旧 IVF-SIMD 基线对齐。四组 recall 分别为 0.98780, 0.99710, 0.97135, 0.99055，与历史值一致或只有统计误差；说明 MPI top-k merge 和 recall 计算没有明显正确性问题。

## ivf_b_recall_latency_tradeoff.png

展示纯 MPI `np=4` 下的 recall-latency 折中。`nprobe` 增大后 recall 上升、latency 增大。最高 recall 组合为 `nlist=256, nprobe=64`，recall=0.997100，latency=446.967 us。

## ivf_b_nprobe_curves.png

把 recall 和 latency 分成两幅曲线，便于解释 nprobe 是主要调参旋钮。`nlist=512` 在较低 nprobe 下 latency 更低；`nlist=256,nprobe=64` recall 更高但扫描代价更大。

## ivf_c_pure_mpi_strong_scaling.png

展示纯 MPI 强扩展。`np=1` 到 `np=8` latency 从 996.064 us 降到 211.411 us，速度提升约 4.712 倍；`np=16/32` 跨节点后 gather 开销升高，整体性能回落。

## ivf_d_cross_node_breakdown.png

固定 `np=8`，比较 `(nodes,ppn)=(1,8),(2,4),(4,2)`。单节点 latency 为 176.153 us，4 节点升至 453.930 us，图中 gather/bcast 明显变大，说明跨节点通信是主要负优化来源。

## ivf_e_fixed_16cores_hybrid.png

固定 16 核比较 `(np,threads)=(16,1),(8,2),(4,4),(2,8)`。`16x1` 最快，平均 latency=283.432 us；混合 OpenMP 配置更慢，说明本实验中减少 MPI rank 后每个 rank 扫描量增大，OpenMP 收益不足以抵消等待和线程开销。

## ivf_e_fixed_16cores_3_2_vs_3_3.png

把 3.2 已清洗的固定 16 核结果与 3.3 结果放在一起，作为历史复核。该图不用于替代 3.3 主结论，而是说明混合并行负优化现象在两轮数据中都存在。

## ivf_f_partition_strategy.png

比较 `block-id`、`contiguous-list`、`greedy-list` 三种负载均衡策略。本轮最低 latency 是 `block-id`，为 178.074 us。list 级策略没有带来更低端到端查询延迟，报告中可从内存布局、list 重组和同步等待角度解释。

## ivf_combined_3_1_3_3_context.png

把 3.1 和 3.3 的纯 MPI 曲线放在一起，便于说明 3.3 是对前两问的补充：3.3 不只是重复 3.1，而是补上了正确性对齐、跨节点通信、固定总核心混合并行、负载均衡策略等更完整的实验组。
