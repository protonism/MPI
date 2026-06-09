# HNSW Shard MPI 4.1 图表说明

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
